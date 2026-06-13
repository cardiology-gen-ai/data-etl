import json
import logging
import pathlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neo4j import Driver

from knowledge_graph.entity_schema import ALLOWED_TYPES, BLOCKLIST_NAMES, normalize_concept
from knowledge_graph.recommendation_extraction_manager import ExtractionEntry, ExtractionsCatalog
from knowledge_graph.umls_semantics import expected_semgroups_for_role

from knowledge_graph.validate_entities import validate_concepts_against_source


logger = logging.getLogger(__name__)


def setup_recommendation_schema(tx) -> None:
    """Indexes and constraints. Concept-side constraints are owned by ``add_entities.setup_entity_schema`` -- we only add Recommendation indexes here."""
    tx.run(
        """
        CREATE CONSTRAINT recommendation_uid IF NOT EXISTS
        FOR (r:Recommendation)
        REQUIRE r.uid IS UNIQUE
        """
    )
    tx.run(
        """
        CREATE INDEX recommendation_doc_id IF NOT EXISTS
        FOR (r:Recommendation)
        ON (r.doc_id)
        """
    )
    tx.run(
        """
        CREATE INDEX recommendation_modality IF NOT EXISTS
        FOR (r:Recommendation)
        ON (r.modality)
        """
    )
    tx.run(
        """
        CREATE INDEX recommendation_class IF NOT EXISTS
        FOR (r:Recommendation)
        ON (r.class)
        """
    )


def _recommendation_uid(entry: ExtractionEntry) -> str:
    """Globally unique uid for a Recommendation node."""
    return f"{entry.doc_id}::{entry.recommendation_id}"


def _section_uid(doc_id: str, container_id: Optional[str]) -> Optional[str]:
    """Heuristic mapping from (doc_id, container_id) to a Section.uid."""
    if not container_id:
        return None
    return f"{doc_id}::{container_id}"


def _iter_entity_dicts(entry: ExtractionEntry) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    """Yield (role, index, term_dict) for each EntityTerm in a successful extraction. term_dict has keys: entity_text, type, qualifiers (dict)."""
    if not entry.ok or not entry.extraction:
        return
    pop = entry.extraction.get("population") or {}
    act = entry.extraction.get("action") or {}

    for i, t in enumerate(pop.get("requires") or []):
        if t.get("entity_text"):
            yield "requires", i, t
    for i, t in enumerate(pop.get("excludes") or []):
        if t.get("entity_text"):
            yield "excludes", i, t

    intv = act.get("intervention")
    if intv and intv.get("entity_text"):
        yield "intervention", 0, intv

    for i, t in enumerate(act.get("alternatives") or []):
        if t:
            yield "alternative", i, t

    purp = act.get("purpose")
    if purp:
        yield "purpose", 0, purp


_ROLE_TO_RELTYPE = {
    "requires": "HAS_INDICATION",
    "excludes": "HAS_CONTRAINDICATION",
    "intervention": "RECOMMENDS_ACTION",
    "alternative": "RECOMMENDS_ACTION",
    "purpose": "HAS_PURPOSE",
}


def _normalize_term_to_concept(
    role: str,
    index: int,
    term: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Apply entity_schema normalization to one term. Returns a dict
    suitable for both Concept MERGE and role-edge write, or None if the
    term is rejected by the local schema (blocklisted, bad type, etc).
    """
    if isinstance(term, str):  # TODO: check (e.g. from alternatives)
        return None
    raw_name = term.get("entity_text")
    raw_type = term.get("type")

    normalized = normalize_concept({"name": raw_name, "type": raw_type})
    if normalized is None:
        return None

    qualifiers = term.get("qualifiers") or {}
    # Flatten qualifiers into edge props -- Neo4j doesn't store nested maps.
    flat_q = {
        f"qualifier_{k}": v
        for k, v in qualifiers.items()
        if v not in (None, "", False)
        # keep `negated=True` though
    }
    if qualifiers.get("negated"):
        flat_q["qualifier_negated"] = True

    return {
        "name": normalized["name"],
        "type": normalized["type"],
        "raw_name": str(raw_name) if raw_name is not None else None,
        "raw_type": str(raw_type) if raw_type is not None else None,
        "role": role,
        "role_index": index,
        "expected_semgroups": expected_semgroups_for_role(role),
        **flat_q,
    }


def _apply_tolerant_validation(
    concepts: List[Dict[str, Any]],
    source_text: str,
    acronyms: Optional[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run upstream validate_concepts_against_source if available.

    Returns (accepted_with_evidence, rejected). The accepted list carries
    validation evidence (support_method, matched_text, ...) merged into
    the original concept dicts.
    """
    if validate_concepts_against_source is None or not concepts:
        return concepts, []

    inputs = [
        {"name": c["name"], "type": c["type"], "raw_name": c.get("raw_name"),
         "raw_type": c.get("raw_type")}
        for c in concepts
    ]
    result = validate_concepts_against_source(
        concepts=inputs,
        source_text=source_text,
        allowed_types=ALLOWED_TYPES,
        blocklist_names=BLOCKLIST_NAMES,
        acronyms=acronyms,
    )

    by_key = {(a.get("name"), a.get("type")): a for a in result.get("accepted", [])}
    accepted_out: List[Dict[str, Any]] = []
    rejected_out: List[Dict[str, Any]] = list(result.get("rejected", []))

    for c in concepts:
        key = (c["name"], c["type"])
        match = by_key.get(key)
        if match is None:
            continue
        merged = {**c, **{k: v for k, v in match.items() if k not in c}}
        # validate_entities can rewrite the name (acronym expansion) -- honour it.
        merged["name"] = match.get("name", c["name"])
        merged["type"] = match.get("type", c["type"])
        accepted_out.append(merged)

    return accepted_out, rejected_out


def _clear_existing_recommendation(tx, uid: str) -> None:
    """Detach-delete any prior Recommendation node with this uid, so
    reruns don't accumulate stale edges/qualifiers."""
    tx.run(
        """
        MATCH (r:Recommendation {uid: $uid})
        DETACH DELETE r
        """,
        uid=uid,
    )


def _write_recommendation_node(tx, entry: ExtractionEntry, uid: str) -> None:
    extraction = entry.extraction or {}
    tx.run(
        """
        MERGE (r:Recommendation {uid: $uid})
        SET r.recommendation_id = $rec_id,
            r.doc_id = $doc_id,
            r.table_id = $table_id,
            r.row_index = $row_index,
            r.table_caption = $table_caption,
            r.section_path = $section_path,
            r.container_id = $container_id,
            r.source_text = $source_text,
            r.group_header = $group_header,
            r.effective_source = $effective_source,
            r.class = $class_,
            r.level = $level,
            r.modality = $modality,
            r.logical_operator = $logical_operator,
            r.extraction_notes = $extraction_notes,
            r.prompt_version = $prompt_version,
            r.model = $model,
            r.extracted_at = $extracted_at,
            r.validation_flags = $validation_flags,
            r.updated_at = datetime()
        """,
        uid=uid,
        rec_id=entry.recommendation_id,
        doc_id=entry.doc_id,
        table_id=entry.table_id,
        row_index=entry.row_index,
        table_caption=entry.table_caption,
        section_path=entry.section_path,
        container_id=entry.container_id,
        source_text=entry.source_text,
        group_header=entry.group_header,
        effective_source=entry.effective_source,
        class_=entry.catalog_class,
        level=entry.catalog_level,
        modality=extraction.get("modality"),
        logical_operator=(extraction.get("population") or {}).get("logical_operator"),
        extraction_notes=extraction.get("extraction_notes"),
        prompt_version=entry.prompt_version,
        model=entry.model,
        extracted_at=entry.extracted_at,
        validation_flags=list(entry.validation_flags.keys()) if entry.validation_flags else [],
    )


def _attach_section(tx, uid: str, section_uid: Optional[str]) -> None:
    if not section_uid:
        return
    tx.run(
        """
        MATCH (r:Recommendation {uid: $uid})
        MATCH (s:Section {uid: $section_uid})
        MERGE (s)-[:CONTAINS_RECOMMENDATION]->(r)
        """,
        uid=uid,
        section_uid=section_uid,
    )


def _write_role_edges(
    tx,
    rec_uid: str,
    section_uid: Optional[str],
    concepts: List[Dict[str, Any]],
) -> None:
    """Write Concept node, Recommendation->Concept role edge, and (when
    section_uid is known) a co-located Section->Concept :MENTIONS edge."""
    if not concepts:
        return

    # Group by role-type because Cypher rel types can't be parameterized.
    by_reltype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in concepts:
        rel = _ROLE_TO_RELTYPE.get(c["role"], "MENTIONS")
        by_reltype[rel].append(c)

    for reltype, group in by_reltype.items():
        tx.run(
            f"""
            MATCH (r:Recommendation {{uid: $uid}})
            UNWIND $concepts AS concept

            MERGE (c:Concept {{name: concept.name}})
            ON CREATE SET
                c.observed_types = [concept.type],
                c.type_resolution_status = 'pending',
                c.needs_type_review = false,
                c.created_at = datetime()
            WITH r, concept, c, coalesce(c.observed_types, []) AS old_types
            SET c.observed_types =
                CASE WHEN concept.type IN old_types
                     THEN old_types
                     ELSE old_types + concept.type END,
                c.updated_at = datetime()

            MERGE (r)-[e:{reltype} {{
                role: concept.role,
                role_index: concept.role_index
            }}]->(c)
            SET e.expected_semgroups = concept.expected_semgroups,
                e.raw_name = concept.raw_name,
                e.raw_type = concept.raw_type,
                e.qualifier_severity = concept.qualifier_severity,
                e.qualifier_duration = concept.qualifier_duration,
                e.qualifier_confirmation = concept.qualifier_confirmation,
                e.qualifier_min_count = concept.qualifier_min_count,
                e.qualifier_threshold = concept.qualifier_threshold,
                e.qualifier_dose = concept.qualifier_dose,
                e.qualifier_route = concept.qualifier_route,
                e.qualifier_negated = concept.qualifier_negated,
                e.validation_reason = concept.validation_reason,
                e.support_method = concept.support_method,
                e.matched_text = concept.matched_text,
                e.matched_pattern = concept.matched_pattern,
                e.acronym_short = concept.acronym_short,
                e.acronym_definition = concept.acronym_definition,
                e.updated_at = datetime()
            """,
            uid=rec_uid,
            concepts=group,
        )

    if section_uid:
        tx.run(
            """
            MATCH (s:Section {uid: $section_uid})
            UNWIND $concepts AS concept
            MATCH (c:Concept {name: concept.name})
            MERGE (s)-[m:MENTIONS]->(c)
            ON CREATE SET
                m.observed_types = [concept.type],
                m.created_at = datetime(),
                m.source = 'recommendation',
                m.raw_name = concept.raw_name,
                m.raw_type = concept.raw_type
            ON MATCH SET
                m.observed_types =
                    CASE WHEN concept.type IN coalesce(m.observed_types, [])
                         THEN m.observed_types
                         ELSE coalesce(m.observed_types, []) + concept.type END,
                m.updated_at = datetime()
            """,
            section_uid=section_uid,
            concepts=concepts,
        )


def _process_entry(
    tx,
    entry: ExtractionEntry,
    replace_existing: bool,
) -> Dict[str, int]:
    """Write a single ExtractionEntry to the graph. Returns counters."""
    uid = _recommendation_uid(entry)

    if replace_existing:
        _clear_existing_recommendation(tx, uid)

    if not entry.ok:
        # Persist the failed/empty recommendation node for audit, no edges.
        _write_recommendation_node(tx, entry, uid)
        return {"written": 1, "accepted_concepts": 0, "rejected_concepts": 0}

    # Collect + normalize EntityTerms.
    normalized: List[Dict[str, Any]] = []
    for role, idx, term in _iter_entity_dicts(entry):
        nc = _normalize_term_to_concept(role, idx, term)
        if nc is not None:
            normalized.append(nc)

    # Run tolerant validation against the effective source. Accepted concepts carry validation evidence (support_method, matched_text, ...).
    accepted, rejected = _apply_tolerant_validation(
        concepts=normalized,
        source_text=entry.effective_source,
        acronyms=entry.acronyms_snapshot,
    )

    _write_recommendation_node(tx, entry, uid)
    section_uid = _section_uid(entry.doc_id, entry.container_id)
    _attach_section(tx, uid, section_uid)
    _write_role_edges(tx, uid, section_uid, accepted)

    return {
        "written": 1,
        "accepted_concepts": len(accepted),
        "rejected_concepts": len(rejected),
    }


def load_extractions(path: pathlib.Path) -> ExtractionsCatalog:
    """Load a JSON catalog produced by ExtractionsManager.save()."""
    if not path.exists():
        raise FileNotFoundError(path)
    cat = ExtractionsCatalog()
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Extractions file is not a JSON array: {path}")
    for item in data:
        try:
            cat.catalog.append(ExtractionEntry.from_json(item))
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping malformed extraction entry: %s", exc)
    return cat


def add_recommendations_from_extractions(
    driver: Driver,
    extractions_path: pathlib.Path,
    replace_existing: bool = True,
) -> Dict[str, int]:
    """Write all extractions from one document into the graph."""
    catalog = load_extractions(extractions_path)
    if not catalog.catalog:
        logger.info("Empty extractions catalog at %s", extractions_path)
        return {"recommendations": 0, "accepted_concepts": 0, "rejected_concepts": 0}

    stats = {"recommendations": 0, "accepted_concepts": 0, "rejected_concepts": 0}

    with driver.session() as session:
        session.execute_write(setup_recommendation_schema)

        for entry in catalog.catalog:
            entry_stats = session.execute_write(
                _process_entry, entry, replace_existing
            )
            stats["recommendations"] += entry_stats["written"]
            stats["accepted_concepts"] += entry_stats["accepted_concepts"]
            stats["rejected_concepts"] += entry_stats["rejected_concepts"]

    logger.info(
        "Recommendations written | doc=%s | recommendations=%d | "
        "accepted_concepts=%d | rejected_concepts=%d",
        catalog.catalog[0].doc_id if catalog.catalog else "?",
        stats["recommendations"],
        stats["accepted_concepts"],
        stats["rejected_concepts"],
    )
    return stats