import logging
import pathlib
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from neo4j import Driver

from knowledge_graph.graph_loader import (setup_schema, create_document, document_sections_exist, delete_existing_document_sections,
                                          delete_orphan_concepts, create_sections_batch, link_document_sections_batch,
                                          link_parent_child_batch, link_next_batch, normalize_section_record, make_section_uid,
                                          chunked)
from knowledge_graph.add_entities import add_entities_from_sections
from knowledge_graph.entity_disambiguation import disambiguate_concepts
from knowledge_graph.add_embeddings import add_embeddings_to_sections
from knowledge_graph.umls_normalization import normalize_concepts_with_umls, SCISPACY_BACKEND, UMLS_API_BACKEND, FUZZY_ONLY_BACKEND
from knowledge_graph.add_recommendations import add_recommendations_from_extractions


logger = logging.getLogger(__name__)


@dataclass
class KGPaths:
    """Bundle of folder/file paths produced by the preprocessing managers."""

    preprocessing_output: pathlib.Path
    kg_folder: pathlib.Path
    chunks_folder: pathlib.Path

    def chunks_file(self, doc_id: str) -> pathlib.Path:
        return self.chunks_folder / f"{doc_id}.pkl"

    def acronyms_folder(self) -> pathlib.Path:
        return self.preprocessing_output / "abbreviations"

    def recommendations_file(self, doc_id: str) -> pathlib.Path:
        return self.kg_folder / "recommendations" / f"{doc_id}_recommendations.json"

    def section_linking_file(self, doc_id: str) -> Optional[pathlib.Path]:
        path = self.kg_folder / "section_linking" / f"{doc_id}_section_links.json"
        return path if path.exists() else None

    def umls_review_dir(self) -> pathlib.Path:
        return self.kg_folder / "umls_review"

    def entity_review_dir(self) -> pathlib.Path:
        return self.kg_folder / "entity_review"


def _build_breadcrumb_string(headers: Optional[Dict[str, List[str]]]) -> str:
    """Rebuild "A > B > C" from a {"Header N": [titles]} metadata dict.
    Mirrors HierarchicalChunkingManager._build_breadcrumb so the value we store on the Section matches what the chunker prepends to the text.
    """
    if not headers:
        return ""
    parts: List[str] = []
    levels = sorted(
        int(k.split()[-1]) for k in headers if k.startswith("Header ")
    )
    for level_num in levels:
        titles = headers.get(f"Header {level_num}") or []
        if titles:
            parts.append(str(titles[-1]).strip())
    return " > ".join(p for p in parts if p)


def _document_to_chunk_dict(document: Any) -> Dict[str, Any]:
    """Convert one langchain ``Document`` (as produced by the chunker) into the dict shape expected by ``graph_loader.normalize_section_record``."""
    md = document.metadata or {}
    return {
        # graph_loader-required fields
        "doc_id": md.get("filename") or md.get("doc_id"),
        "section_id": md.get("section_id"),
        "printed_section_id": md.get("printed_section_id"),
        "section_title": md.get("section_title"),
        "section_level": md.get("section_level"),
        "text": document.page_content,
        "is_empty": md.get("is_empty"),
        "embed": md.get("embed"),
        "page_start": md.get("page_start"),
        "page_end": md.get("page_end"),
        "parent_section_id": md.get("parent_section_id"),
        "part_index": md.get("part_index"),
        "part_count": md.get("part_count"),
        "quality_flags": md.get("quality_flags") or [],
        "boundary_source": md.get("boundary_source"),
        # Extras persisted as Section properties via a follow-up Cypher
        "section_type": (
            str(md["section_type"]) if md.get("section_type") is not None else None
        ),
        "breadcrumb_path": _build_breadcrumb_string(md.get("headers")),
        "chunk_id": md.get("chunk_id"),
    }


def _load_chunks_from_pkl(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load a HierarchicalChunkingManager pickle and convert every Document to the dict shape consumed by ``graph_loader``."""
    with path.open("rb") as f:
        documents = pickle.load(f)
    if not isinstance(documents, list):
        raise ValueError(
            f"Chunk pickle must contain a list of Documents: {path}"
        )
    return [_document_to_chunk_dict(doc) for doc in documents]

def _set_section_extras(tx, rows: List[Dict[str, Any]]) -> None:
    """Persist ``section_type``, ``breadcrumb_path``, and ``chunk_id`` on already-created Section nodes. Properties that are None are left untouched."""
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (s:Section {uid: row.uid})
        SET s.section_type     = coalesce(row.section_type, s.section_type),
            s.breadcrumb_path  = coalesce(row.breadcrumb_path, s.breadcrumb_path),
            s.chunk_id         = coalesce(row.chunk_id, s.chunk_id)
        """,
        rows=rows,
    )


def _write_chunks_to_graph(
    driver: Driver,
    doc_id: str,
    chunks: List[Dict[str, Any]],
    *,
    batch_size: int = 200,
    replace_existing_document: bool = True,
) -> None:
    """Replicates the orchestration of ``graph_loader.build_graph_from_chunks`` but works from an in-memory chunk list (so we can source from .pkl). Reuses all of graph_loader's low-level helpers."""
    if not chunks:
        logger.warning("Empty chunks list for %s; nothing to write.", doc_id)
        return

    doc_ids = {c["doc_id"] for c in chunks}
    if len(doc_ids) != 1:
        raise ValueError(
            f"Chunks for {doc_id} contain multiple doc_id values: {sorted(doc_ids)}"
        )
    actual = next(iter(doc_ids))
    if actual != doc_id:
        raise ValueError(
            f"doc_id mismatch: expected {doc_id}, got {actual}"
        )

    normalized_sections = [normalize_section_record(c) for c in chunks]
    section_uids = [s["uid"] for s in normalized_sections]

    parent_child_pairs: List[Dict[str, str]] = []
    next_pairs: List[Dict[str, str]] = []
    for i, s in enumerate(normalized_sections):
        parent_id = s.get("parent_section_id")
        if parent_id:
            parent_child_pairs.append({
                "parent_uid": make_section_uid(doc_id, parent_id),
                "child_uid": s["uid"],
            })
        if i > 0:
            next_pairs.append({
                "prev_uid": normalized_sections[i - 1]["uid"],
                "next_uid": s["uid"],
            })

    # Extras carried per uid for the follow-up SET pass.
    extras_rows = [
        {
            "uid": normalized_sections[i]["uid"],
            "section_type": chunks[i].get("section_type"),
            "breadcrumb_path": chunks[i].get("breadcrumb_path") or "",
            "chunk_id": chunks[i].get("chunk_id"),
        }
        for i in range(len(chunks))
    ]

    with driver.session() as session:
        session.execute_write(setup_schema)
        session.execute_write(create_document, doc_id)

        if replace_existing_document:
            session.execute_write(delete_existing_document_sections, doc_id)
            session.execute_write(delete_orphan_concepts)
        else:
            if session.execute_read(document_sections_exist, doc_id):
                raise ValueError(
                    f"Document {doc_id} already has sections in the graph. "
                    "Use replace_existing_document=True to reload it."
                )

        for batch in chunked(normalized_sections, batch_size):
            session.execute_write(create_sections_batch, batch)

        for uid_batch in chunked(
            [{"uid": uid} for uid in section_uids], batch_size,
        ):
            session.execute_write(
                link_document_sections_batch,
                doc_id,
                [row["uid"] for row in uid_batch],
            )

        for batch in chunked(parent_child_pairs, batch_size):
            session.execute_write(link_parent_child_batch, batch)

        for batch in chunked(next_pairs, batch_size):
            session.execute_write(link_next_batch, batch)

        for batch in chunked(extras_rows, batch_size):
            session.execute_write(_set_section_extras, batch)

    logger.info(
        "Structural graph built for %s | sections=%d | parent_child=%d | next=%d",
        doc_id, len(normalized_sections), len(parent_child_pairs), len(next_pairs),
    )

def _process_document_structural(
    driver: Driver,
    paths: KGPaths,
    doc_id: str,
    *,
    batch_size: int = 200,
    replace_existing_document: bool = True,
) -> None:
    chunks_file = paths.chunks_file(doc_id)
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Missing chunks pickle for {doc_id}: {chunks_file}"
        )
    chunks = _load_chunks_from_pkl(chunks_file)
    _write_chunks_to_graph(
        driver=driver,
        doc_id=doc_id,
        chunks=chunks,
        batch_size=batch_size,
        replace_existing_document=replace_existing_document,
    )


def _process_document_prose_entities(
    driver: Driver,
    paths: KGPaths,
    doc_id: str,
    *,
    use_acronym_validation: bool = True,
    **add_entities_kwargs: Any,
) -> Dict[str, int]:
    acronym_dir = paths.acronyms_folder() if use_acronym_validation else None
    return add_entities_from_sections(
        driver=driver,
        doc_id=doc_id,
        acronym_dir=acronym_dir,
        use_acronym_validation=use_acronym_validation,
        entity_review_output_dir=paths.entity_review_dir(),
        **add_entities_kwargs,
    )


def _process_document_recommendations(
    driver: Driver,
    paths: KGPaths,
    doc_id: str,
    *,
    replace_existing: bool = True,
) -> Dict[str, int]:
    rec_file = paths.recommendations_file(doc_id)
    if not rec_file.exists():
        logger.warning(
            "No recommendations JSON for %s at %s -- skipping.",
            doc_id, rec_file,
        )
        return {"recommendations": 0, "accepted_concepts": 0, "rejected_concepts": 0}
    return add_recommendations_from_extractions(
        driver=driver,
        extractions_path=rec_file,
        replace_existing=replace_existing,
    )


def _process_document_section_links(
    driver: Driver,
    paths: KGPaths,
    doc_id: str,
) -> Optional[Dict[str, int]]:
    """Apply SectionLinkingManager output to the graph."""
    link_file = paths.section_linking_file(doc_id)
    if link_file is None:
        return None

    import json
    try:
        records = json.loads(link_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load section links for %s: %s", doc_id, exc)
        return None

    if not records:
        return {"links": 0, "edges_created": 0, "empty_links": 0}

    payload: List[Dict[str, Any]] = []
    empty_count = 0
    for r in records:
        if not r.get("chunk_section_ids"):
            # match_strategy == 'empty', nothing to attach.
            empty_count += 1
            continue
        rec_uid = f"{doc_id}::{r['recommendation_id']}"
        section_uids = [
            f"{doc_id}::{sid}" for sid in r["chunk_section_ids"] if sid
        ]
        if not section_uids:
            empty_count += 1
            continue
        payload.append({
            "recommendation_uid": rec_uid,
            "section_uids": section_uids,
            "match_strategy": r.get("match_strategy"),
            "target_section_id": r.get("target_section_id"),
            "target_section_title": r.get("target_section_title"),
            "linking_version": r.get("linking_version"),
            "linked_at": r.get("linked_at"),
        })

    if not payload:
        return {"links": 0, "edges_created": 0, "empty_links": empty_count}

    with driver.session() as session:
        # Step A: drop any existing CONTAINS_RECOMMENDATION edges to the
        # recommendations we are about to relink. This wipes both the
        # heuristic edges from add_recommendations and stale links from
        # an earlier run.
        session.run(
            """
            UNWIND $payload AS link
            MATCH (r:Recommendation {uid: link.recommendation_uid})
            OPTIONAL MATCH (:Section)-[old:CONTAINS_RECOMMENDATION]->(r)
            DELETE old
            """,
            payload=payload,
        )

        # Step B: create the authoritative edges. Sections referenced by the linker but absent from the graph
        # (e.g. chunks skipped as empty during structural load) simply do not match and are silently ignored.
        result = session.run(
            """
            UNWIND $payload AS link
            MATCH (r:Recommendation {uid: link.recommendation_uid})
            UNWIND link.section_uids AS section_uid
            MATCH (s:Section {uid: section_uid})
            MERGE (s)-[e:CONTAINS_RECOMMENDATION]->(r)
            SET e.match_strategy       = link.match_strategy,
                e.target_section_id    = link.target_section_id,
                e.target_section_title = link.target_section_title,
                e.linking_version      = link.linking_version,
                e.linked_at            = link.linked_at
            RETURN count(e) AS edges_created
            """,
            payload=payload,
        )
        record = result.single()
        edges_created = int(record["edges_created"]) if record else 0

    return {
        "links": len(payload),
        "edges_created": edges_created,
        "empty_links": empty_count,
    }


def process_document(
    driver: Driver,
    paths: KGPaths,
    doc_id: str,
    *,
    skip_prose_entities: bool = False,
    skip_recommendations: bool = False,
    skip_section_links: bool = False,
    replace_existing_document: bool = True,
    add_entities_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"doc_id": doc_id}

    logger.info("[%s] structural graph", doc_id)
    _process_document_structural(
        driver, paths, doc_id,
        replace_existing_document=replace_existing_document,
    )
    result["structural"] = "ok"

    if not skip_prose_entities:
        logger.info("[%s] prose entity extraction", doc_id)
        result["prose_entities"] = _process_document_prose_entities(
            driver, paths, doc_id, **(add_entities_kwargs or {})
        )
        print(result["prose_entities"])

    if not skip_recommendations:
        logger.info("[%s] recommendation writer", doc_id)
        result["recommendations"] = _process_document_recommendations(
            driver, paths, doc_id,
        )

    if not skip_section_links:
        logger.info("[%s] section linking (optional)", doc_id)
        link_stats = _process_document_section_links(driver, paths, doc_id)
        if link_stats is not None:
            result["section_links"] = link_stats

    return result


def _run_disambiguation(driver: Driver) -> Dict[str, int]:
    logger.info("[global] concept disambiguation")
    return disambiguate_concepts(driver=driver, delete_orphans=True)


def _run_umls_normalization(
    driver: Driver,
    paths: KGPaths,
    *,
    mode: str = "hybrid",
    use_acronyms: bool = True,
    scispacy_kwargs: Optional[Dict[str, Any]] = None,
    api_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run UMLS normalization on Concept nodes.

    mode:
      - "scispacy": local scispaCy + KB only.
      - "api": UMLS REST API only.
      - "hybrid" (default): scispaCy first, then API on whatever didn't
        get a confident scispaCy match. Each Concept ends up with
        ``umls_source`` set to "scispacy" or "api".
      - "fuzzy": degraded mode (fuzzy duplicate detection only).
    """
    out: Dict[str, Any] = {"mode": mode}
    acronym_dir = paths.acronyms_folder() if use_acronyms else None
    review_dir = paths.umls_review_dir()
    scispacy_kwargs = dict(scispacy_kwargs or {})
    api_kwargs = dict(api_kwargs or {})

    if mode == "scispacy":
        logger.info("[global] UMLS normalization (scispacy only)")
        out["scispacy"] = normalize_concepts_with_umls(
            driver=driver, backend=SCISPACY_BACKEND,
            use_acronyms=use_acronyms, acronym_dir=acronym_dir,
            review_output_dir=review_dir, **scispacy_kwargs,
        )
        return out

    if mode == "api":
        logger.info("[global] UMLS normalization (api only)")
        out["api"] = normalize_concepts_with_umls(
            driver=driver, backend=UMLS_API_BACKEND,
            use_acronyms=use_acronyms, acronym_dir=acronym_dir,
            review_output_dir=review_dir, **api_kwargs,
        )
        return out

    if mode == "fuzzy":
        logger.info("[global] UMLS normalization (fuzzy only)")
        out["fuzzy"] = normalize_concepts_with_umls(
            driver=driver, backend=FUZZY_ONLY_BACKEND,
            use_acronyms=use_acronyms, acronym_dir=acronym_dir,
            review_output_dir=review_dir,
        )
        return out

    if mode != "hybrid":
        raise ValueError(
            f"Unknown UMLS mode {mode!r}; expected one of "
            "'scispacy', 'api', 'hybrid', 'fuzzy'."
        )

    logger.info("[global] UMLS normalization (hybrid, pass 1/2: scispacy)")
    out["scispacy"] = normalize_concepts_with_umls(
        driver=driver, backend=SCISPACY_BACKEND,
        use_acronyms=use_acronyms, acronym_dir=acronym_dir,
        review_output_dir=review_dir, **scispacy_kwargs,
    )
    logger.info("[global] UMLS normalization (hybrid, pass 2/2: api)")
    out["api"] = normalize_concepts_with_umls(
        driver=driver, backend=UMLS_API_BACKEND,
        use_acronyms=use_acronyms, acronym_dir=acronym_dir,
        review_output_dir=review_dir, **api_kwargs,
    )
    return out


def _run_embeddings(
    driver: Driver,
    doc_ids: Iterable[str],
    *,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_dimensions: Optional[int] = None,
    batch_size: int = 8,
    **embedding_kwargs: Any,
) -> List[Dict[str, int]]:
    stats_all: List[Dict[str, int]] = []
    for doc_id in doc_ids:
        logger.info("[global] embeddings for doc=%s", doc_id)
        stats_all.append(add_embeddings_to_sections(
            driver=driver,
            doc_id=doc_id,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            batch_size=batch_size,
            **embedding_kwargs,
        ))
    return stats_all


def _run_sanity_checks(
    driver: Driver,
    *,
    mode: str = "full",
) -> Optional[Dict[str, Any]]:
    try:
        from knowledge_graph.sanity_checks import run_sanity_checks
    except ImportError:
        logger.warning("sanity_checks module not importable; skipping.")
        return None
    logger.info("[global] sanity checks (mode=%s)", mode)
    return run_sanity_checks(driver=driver, mode=mode)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_kg(
    driver: Driver,
    paths: KGPaths,
    doc_ids: Iterable[str],
    *,
    # per-document toggles
    replace_existing_document: bool = True,
    skip_prose_entities: bool = False,
    skip_recommendations: bool = False,
    skip_section_links: bool = False,
    add_entities_kwargs: Optional[Dict[str, Any]] = None,
    # global toggles
    skip_disambiguation: bool = False,
    skip_umls_normalization: bool = False,
    skip_embeddings: bool = False,
    skip_sanity_checks: bool = False,
    # UMLS config
    umls_mode: str = "hybrid",
    umls_use_acronyms: bool = True,
    umls_scispacy_kwargs: Optional[Dict[str, Any]] = None,
    umls_api_kwargs: Optional[Dict[str, Any]] = None,
    # embedding config
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_dimensions: Optional[int] = None,
    embedding_batch_size: int = 8,
    embedding_kwargs: Optional[Dict[str, Any]] = None,
    # sanity
    sanity_check_mode: str = "full",
) -> Dict[str, Any]:
    """Build (or rebuild) the knowledge graph from preprocessed artifacts.

    For each ``doc_id``, runs the per-document writers in order (structural -> prose entities -> recommendations -> section links).
    After all documents have been ingested, runs the global passes (disambiguation -> UMLS normalization -> embeddings -> sanity).
    """
    doc_ids = list(doc_ids)
    summary: Dict[str, Any] = {"per_document": [], "global": {}}

    for doc_id in doc_ids:
        per_doc = process_document(
            driver=driver,
            paths=paths,
            doc_id=doc_id,
            replace_existing_document=replace_existing_document,
            skip_prose_entities=skip_prose_entities,
            skip_recommendations=skip_recommendations,
            skip_section_links=skip_section_links,
            add_entities_kwargs=add_entities_kwargs,
        )
        summary["per_document"].append(per_doc)

    if not skip_disambiguation:
        summary["global"]["disambiguation"] = _run_disambiguation(driver)

    if not skip_umls_normalization:
        summary["global"]["umls"] = _run_umls_normalization(
            driver, paths,
            mode=umls_mode,
            use_acronyms=umls_use_acronyms,
            scispacy_kwargs=umls_scispacy_kwargs,
            api_kwargs=umls_api_kwargs,
        )

    if not skip_embeddings:
        summary["global"]["embeddings"] = _run_embeddings(
            driver=driver,
            doc_ids=doc_ids,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            batch_size=embedding_batch_size,
            **(embedding_kwargs or {}),
        )

    if not skip_sanity_checks:
        summary["global"]["sanity_checks"] = _run_sanity_checks(
            driver, mode=sanity_check_mode,
        )

    return summary