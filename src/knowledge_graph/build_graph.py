"""
build_graph.py

Flexible graph pipeline for the knowledge graph workflow.

Main responsibilities:
- optionally run preprocessing and chunk preparation from PDFs
- optionally extract and cache per-document acronym dictionaries during preprocessing
- optionally load graph structure into Neo4j from cached chunk files
- optionally run entity extraction and embeddings on existing graph data
- optionally use cached document-level acronyms during entity validation
- optionally run global concept disambiguation
- optionally run sanity checks, using the phase-aware sanity_mode provided
  by the pipeline config (for example: structure, entities, embeddings, full)

"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from managers.table_of_contents_manager import GuidelineTOCExtractor
from managers.markdown_conversion_manager import MarkdownConverter
from managers.markdown_manager import MarkdownManager
from managers.hierarchical_chunking_manager import build_hierarchical_chunks
from managers.acronym_extractor import load_or_extract_acronyms

from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver
from knowledge_graph.graph_loader import build_graph_from_chunks
from knowledge_graph.add_entities import add_entities_from_sections
from knowledge_graph.entity_disambiguation import disambiguate_concepts
from knowledge_graph.add_embeddings import add_embeddings_to_sections
from knowledge_graph.sanity_checks import run_sanity_checks
from knowledge_graph.llm_utils import clear_chat_model_cache


logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def optional_path(value: Any) -> Optional[Path]:
    """
    Convert optional config path values to Path objects.

    Accepts:
    - None -> None
    - str -> Path(str)
    - Path -> Path
    """
    if value is None:
        return None
    return Path(value)


def get_acronym_dir(config) -> Path:
    """
    Return the directory used for per-document acronym JSON caches.

    If config.acronym_dir is provided, use that directly.
    Otherwise, default to a sibling directory of chunk_dir named "acronyms".
    """
    acronym_dir = getattr(config, "acronym_dir", None)

    if acronym_dir is not None:
        return Path(acronym_dir)

    return Path(config.chunk_dir).parent / "acronyms"


def get_entity_acronym_dir(config) -> Path:
    """
    Return the acronym directory used during entity validation.

    Priority:
    1. config.entity_acronym_dir, if provided
    2. config.acronym_dir, if provided
    3. sibling directory of chunk_dir named "acronyms"

    This allows acronym extraction and entity validation to share the same cache
    by default, while still allowing an override for entity-only runs.
    """
    entity_acronym_dir = getattr(config, "entity_acronym_dir", None)

    if entity_acronym_dir is not None:
        return Path(entity_acronym_dir)

    return get_acronym_dir(config)


def get_cached_acronym_path(config, doc_id: str) -> Path:
    return get_acronym_dir(config) / f"{doc_id}_acronyms.json"


def should_run_acronym_extraction(config) -> bool:
    """
    Acronym extraction is part of preprocessing by default.

    Set config.run_acronym_extraction = False only if you explicitly want to
    skip this cache generation step.
    """
    return bool(getattr(config, "run_acronym_extraction", True))


def should_use_acronym_validation(config) -> bool:
    """
    Whether entity validation should load cached acronym dictionaries.

    Default: True.

    This does not extract acronyms by itself. It only tells add_entities.py to
    look for already-cached acronym JSON files and use them as validation support.
    """
    return bool(getattr(config, "entity_use_acronym_validation", True))


def should_clear_chat_cache_before_embeddings(config) -> bool:
    """
    Whether to clear the cached chat model before the embeddings phase.

    Default: True.

    This matters when entity extraction and embeddings run in the same Python
    process. Entity extraction loads the chat model; embeddings load the embedding
    model. On GPU-constrained nodes, keeping both cached can cause CUDA OOM.
    """
    return bool(getattr(config, "clear_chat_cache_before_embeddings", True))


def ensure_pipeline_dirs(config) -> None:
    """
    Create all required output/cache directories for the graph pipeline.
    """
    ensure_dir(Path(config.toc_dir))
    ensure_dir(Path(config.markdown_dir))
    ensure_dir(Path(config.image_dir))
    ensure_dir(Path(config.anchor_dir))
    ensure_dir(Path(config.chunk_dir))

    if should_run_acronym_extraction(config):
        ensure_dir(get_acronym_dir(config))

    if (
        getattr(config, "run_entity_extraction", False)
        and should_use_acronym_validation(config)
    ):
        ensure_dir(get_entity_acronym_dir(config))


def requires_neo4j(config) -> bool:
    """
    Neo4j is needed whenever we touch graph/enrichment/sanity stages.
    """
    return any([
        getattr(config, "run_graph_loader", False),
        getattr(config, "run_entity_extraction", False),
        getattr(config, "run_embeddings", False),
        getattr(config, "run_entity_disambiguation", False),
        getattr(config, "run_sanity_checks", False),
    ])


def requires_preprocessing(config) -> bool:
    """
    Preprocessing is required only when explicitly requested.
    Graph loading can also run directly from cached chunk files.
    """
    return bool(getattr(config, "run_preprocessing", False))


def needs_document_level_processing(config) -> bool:
    """
    Whether the pipeline needs to iterate document-by-document for graph/enrichment work.
    """
    return any([
        getattr(config, "run_graph_loader", False),
        getattr(config, "run_entity_extraction", False),
        getattr(config, "run_embeddings", False),
    ])


def validate_preprocessing_paths(config) -> None:
    """
    Ensure that the graph pipeline paths are correct and consistent with the
    MarkdownConverter configuration.
    """
    preprocessing_config = getattr(config, "preprocessing_config", None)
    if preprocessing_config is None:
        raise ValueError(
            "PipelineConfig must provide `preprocessing_config` "
            "for MarkdownConverter initialization."
        )

    converter_input_dir = Path(preprocessing_config.input_folder.folder)
    converter_output_dir = Path(preprocessing_config.output_folder.folder)

    if Path(config.pdf_dir) != converter_input_dir:
        raise ValueError(
            f"config.pdf_dir ({config.pdf_dir}) does not match "
            f"preprocessing_config.input_folder.folder ({converter_input_dir})"
        )

    if Path(config.markdown_dir) != converter_output_dir:
        raise ValueError(
            f"config.markdown_dir ({config.markdown_dir}) does not match "
            f"preprocessing_config.output_folder.folder ({converter_output_dir})"
        )


def get_markdown_converter(config) -> MarkdownConverter:
    """
    Build the MarkdownConverter from the project config.
    """
    preprocessing_config = getattr(config, "preprocessing_config", None)
    if preprocessing_config is None:
        raise ValueError(
            "PipelineConfig must provide `preprocessing_config` "
            "for MarkdownConverter initialization."
        )

    return MarkdownConverter(config=preprocessing_config)


def load_or_extract_toc(config, pdf_path: Path, doc_id: str) -> Dict[str, Any]:
    """
    Load cached TOC if present, otherwise extract it from the PDF.
    """
    toc_path = Path(config.toc_dir) / f"{doc_id}_toc.json"

    if toc_path.exists() and not getattr(config, "force_toc", False):
        logger.info("TOC exists, loading cached version for %s", doc_id)
        return json.loads(toc_path.read_text(encoding="utf-8"))

    logger.info("Extracting TOC for %s", doc_id)
    extractor = GuidelineTOCExtractor(
        pdf_path=str(pdf_path),
        doc_id=doc_id,
    )
    extractor.save(str(toc_path))

    return json.loads(toc_path.read_text(encoding="utf-8"))


def load_or_extract_document_acronyms(
    config,
    pdf_path: Path,
    doc_id: str,
) -> Dict[str, Any]:
    """
    Load or extract the per-document acronym cache.

    This function does not require the already-loaded TOC object.
    It passes the expected TOC cache path to the acronym extractor; if that TOC
    file exists, the extractor can use it, otherwise it safely falls back to PDF
    heuristics only.
    """
    acronym_payload = load_or_extract_acronyms(
        pdf_path=pdf_path,
        doc_id=doc_id,
        toc_path=Path(config.toc_dir) / f"{doc_id}_toc.json",
        acronym_dir=get_acronym_dir(config),
        force=getattr(config, "force_acronyms", False),
        write_output=True,
        sample_size=getattr(config, "acronym_sample_size", 0),
        print_all=getattr(config, "acronym_print_all", False),
    )

    if acronym_payload is None:
        logger.warning(
            "Acronym extractor returned None for %s; using empty failed payload",
            doc_id,
        )
        acronym_payload = {
            "doc_id": doc_id,
            "status": "failed",
            "source": "extractor_returned_none",
            "n_acronyms": 0,
            "n_suspicious": 0,
            "suspicious": [],
            "acronyms": {},
        }

    logger.info(
        "Acronym cache for %s | status=%s | n_acronyms=%s | n_suspicious=%s",
        doc_id,
        acronym_payload.get("status"),
        acronym_payload.get("n_acronyms"),
        acronym_payload.get("n_suspicious"),
    )

    return acronym_payload


def load_or_convert_markdown(
    config,
    md_converter: MarkdownConverter,
    pdf_path: Path,
    doc_id: str,
) -> str:
    """
    Load cached Markdown if present, otherwise convert the PDF.
    """
    md_path = Path(config.markdown_dir) / f"{doc_id}.md"

    if md_path.exists() and not getattr(config, "force_markdown", False):
        logger.info("Markdown exists, loading cached version for %s", doc_id)
    else:
        logger.info("Converting PDF to Markdown for %s", doc_id)
        success, meta = md_converter(pdf_path.name)
        del meta

        if not success:
            raise RuntimeError(f"Markdown conversion failed for document {doc_id}")

    if not md_path.exists():
        raise FileNotFoundError(f"Expected Markdown file not found: {md_path}")

    return md_path.read_text(encoding="utf-8")


def load_or_compute_anchors(
    config,
    pdf_path: Path,
    markdown_text: str,
    doc_id: str,
):
    """
    Load cached page anchors if present, otherwise compute them.
    """
    anchor_path = Path(config.anchor_dir) / f"{doc_id}_page_anchors.json"

    markdown_manager = MarkdownManager(
        filepath=pdf_path,
        text=markdown_text,
    )

    if getattr(config, "force_anchors", False) and anchor_path.exists():
        logger.info("Forcing page anchor recomputation for %s", doc_id)
        anchor_path.unlink()

    logger.info("Loading or computing page anchors for %s", doc_id)
    anchors = markdown_manager.get_page_anchors(cache_path=anchor_path)

    return markdown_manager, anchors


def resolve_toc_tree(toc: Dict[str, Any], doc_id: str):
    """
    Resolve the TOC tree structure from the extracted TOC JSON.
    """
    toc_tree = toc.get("toc_tree")
    if toc_tree is not None:
        return toc_tree

    flat_toc = toc.get("flat_toc")
    if not flat_toc:
        raise ValueError(
            f"TOC for document {doc_id} does not contain `toc_tree` or `flat_toc`"
        )

    if not flat_toc[0].get("children"):
        raise ValueError(
            f"TOC fallback failed for document {doc_id}: "
            "first flat_toc node has no children"
        )

    return flat_toc[0]["children"]


def get_cached_chunk_path(config, doc_id: str) -> Path:
    return Path(config.chunk_dir) / f"{doc_id}_hier_chunks.json"


def list_cached_chunk_paths(config) -> List[Path]:
    chunk_dir = Path(config.chunk_dir)
    if not chunk_dir.exists():
        return []
    return sorted(chunk_dir.glob("*_hier_chunks.json"))


def chunk_path_to_doc_id(chunk_path: Path) -> str:
    suffix = "_hier_chunks.json"
    name = chunk_path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected chunk file name: {chunk_path}")
    return name[:-len(suffix)]


def get_graph_document_ids(driver) -> List[str]:
    """
    Return document ids currently present in Neo4j.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:Document)
            RETURN d.doc_id AS doc_id
            ORDER BY d.doc_id
            """
        )
        return [record["doc_id"] for record in result]


def load_or_build_chunks(
    config,
    toc: Dict[str, Any],
    markdown_manager: MarkdownManager,
    anchors,
    doc_id: str,
) -> Path:
    """
    Load cached hierarchical chunks if present, otherwise build them.
    """
    chunk_path = get_cached_chunk_path(config, doc_id)

    if chunk_path.exists() and not getattr(config, "force_chunks", False):
        logger.info("Chunks exist, loading cached version for %s", doc_id)
        return chunk_path

    logger.info("Building hierarchical chunks for %s", doc_id)

    toc_tree = resolve_toc_tree(toc, doc_id)

    chunks = build_hierarchical_chunks(
        toc_tree=toc_tree,
        markdown_manager=markdown_manager,
        anchors=anchors,
        doc_id=doc_id,
    )

    chunk_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return chunk_path


def preprocess_single_document(
    config,
    md_converter: MarkdownConverter,
    pdf_path: Path,
) -> Dict[str, Any]:
    """
    Run preprocessing/chunk preparation for one PDF.

    Acronym extraction is independent from chunk creation, but it should run
    after the TOC cache has been created/loaded, allowing TOC metadata usage.
    """
    doc_id = pdf_path.stem
    logger.info("=== Preprocessing %s ===", doc_id)

    chunk_path = get_cached_chunk_path(config, doc_id)

    toc: Optional[Dict[str, Any]] = None
    acronym_payload: Optional[Dict[str, Any]] = None

    if should_run_acronym_extraction(config):
        toc = load_or_extract_toc(config, pdf_path, doc_id)

        acronym_payload = load_or_extract_document_acronyms(
            config=config,
            pdf_path=pdf_path,
            doc_id=doc_id,
        )

    if chunk_path.exists() and not getattr(config, "force_chunks", False):
        logger.info("Chunk cache already available for %s", doc_id)
        return {
            "doc_id": doc_id,
            "chunk_path": str(chunk_path),
            "acronym_path": (
                str(get_cached_acronym_path(config, doc_id))
                if should_run_acronym_extraction(config)
                else None
            ),
            "acronym_status": (
                acronym_payload.get("status")
                if acronym_payload is not None
                else None
            ),
            "n_acronyms": (
                acronym_payload.get("n_acronyms")
                if acronym_payload is not None
                else None
            ),
            "n_suspicious_acronyms": (
                acronym_payload.get("n_suspicious")
                if acronym_payload is not None
                else None
            ),
            "source": "preprocessing_cache",
        }

    if toc is None:
        toc = load_or_extract_toc(config, pdf_path, doc_id)

    markdown_text = load_or_convert_markdown(config, md_converter, pdf_path, doc_id)

    markdown_manager, anchors = load_or_compute_anchors(
        config=config,
        pdf_path=pdf_path,
        markdown_text=markdown_text,
        doc_id=doc_id,
    )

    chunk_path = load_or_build_chunks(
        config=config,
        toc=toc,
        markdown_manager=markdown_manager,
        anchors=anchors,
        doc_id=doc_id,
    )

    return {
        "doc_id": doc_id,
        "chunk_path": str(chunk_path),
        "acronym_path": (
            str(get_cached_acronym_path(config, doc_id))
            if should_run_acronym_extraction(config)
            else None
        ),
        "acronym_status": (
            acronym_payload.get("status")
            if acronym_payload is not None
            else None
        ),
        "n_acronyms": (
            acronym_payload.get("n_acronyms")
            if acronym_payload is not None
            else None
        ),
        "n_suspicious_acronyms": (
            acronym_payload.get("n_suspicious")
            if acronym_payload is not None
            else None
        ),
        "source": "preprocessing_built",
    }


def process_document_graph_loading(
    driver,
    config,
    doc_id: str,
    chunk_path: Optional[Path],
) -> Optional[str]:
    """
    Load graph structure for one document.
    """
    if chunk_path is None:
        raise ValueError(
            f"Graph loading requested for document {doc_id}, but chunk_path is missing."
        )

    logger.info("Loading graph structure into Neo4j for document %s", doc_id)

    return build_graph_from_chunks(
        driver=driver,
        chunk_file=chunk_path,
        batch_size=getattr(config, "graph_loader_batch_size", 200),
        replace_existing_document=getattr(
            config,
            "graph_loader_replace_existing_document",
            True,
        ),
    )


def process_document_entity_extraction(
    driver,
    config,
    doc_id: str,
) -> Dict[str, int]:
    """
    Run entity extraction for one document.
    """
    use_acronym_validation = should_use_acronym_validation(config)
    entity_acronym_dir = (
        get_entity_acronym_dir(config)
        if use_acronym_validation
        else None
    )

    logger.info(
        "Extracting entities for document %s | acronym_validation=%s | acronym_dir=%s",
        doc_id,
        use_acronym_validation,
        entity_acronym_dir,
    )

    return add_entities_from_sections(
        driver=driver,
        doc_id=doc_id,
        use_section_text=getattr(config, "entity_use_section_text", False),
        max_sections=getattr(config, "entity_max_sections", None),
        max_sections_per_batch=getattr(config, "entity_max_sections_per_batch", 2),
        max_batch_chars=getattr(config, "entity_max_batch_chars", 12000),
        emergency_max_single_chars=getattr(
            config,
            "entity_emergency_max_single_chars",
            12000,
        ),
        skip_processed=getattr(config, "entity_skip_processed", True),
        replace_section_mentions=getattr(
            config,
            "entity_replace_section_mentions",
            True,
        ),
        export_entity_review=getattr(config, "entity_export_review", True),
        entity_review_output_dir=optional_path(
            getattr(config, "entity_review_output_dir", None)
        ),
        clear_previous_entity_review=getattr(
            config,
            "entity_clear_previous_review",
            True,
        ),
        include_source_preview_in_review=getattr(
            config,
            "entity_include_source_preview_in_review",
            False,
        ),
        acronym_dir=entity_acronym_dir,
        use_acronym_validation=use_acronym_validation,
    )


def process_document_embeddings(
    driver,
    config,
    doc_id: str,
) -> Dict[str, int]:
    """
    Compute embeddings for one document.
    """
    logger.info("Computing embeddings for document %s", doc_id)

    return add_embeddings_to_sections(
        driver=driver,
        doc_id=doc_id,
        max_sections=getattr(config, "embedding_max_sections", None),
        batch_size=getattr(config, "embedding_batch_size", 8),
        force_reembed=getattr(config, "embedding_force_reembed", False),
        include_title=getattr(config, "embedding_include_title", True),
        include_body=getattr(config, "embedding_include_body", True),
        max_chars_per_section=getattr(
            config,
            "embedding_max_chars_per_section",
            8000,
        ),
        allow_title_only=getattr(config, "embedding_allow_title_only", False),
    )


def process_document_graph_and_enrichment(
    driver,
    config,
    doc_id: str,
    chunk_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Backward-compatible single-document helper.

    The main pipeline below no longer uses this for full runs. It now runs phases
    globally across all documents. This helper is kept in case another script
    imports it directly.
    """
    graph_result = None
    entity_stats = None
    embedding_stats = None

    if getattr(config, "run_graph_loader", False):
        graph_result = process_document_graph_loading(
            driver=driver,
            config=config,
            doc_id=doc_id,
            chunk_path=chunk_path,
        )

    if getattr(config, "run_entity_extraction", False):
        entity_stats = process_document_entity_extraction(
            driver=driver,
            config=config,
            doc_id=doc_id,
        )

    if (
        getattr(config, "run_entity_extraction", False)
        and getattr(config, "run_embeddings", False)
        and should_clear_chat_cache_before_embeddings(config)
    ):
        logger.info(
            "Clearing chat model cache before embeddings for document %s",
            doc_id,
        )
        clear_chat_model_cache()

    if getattr(config, "run_embeddings", False):
        embedding_stats = process_document_embeddings(
            driver=driver,
            config=config,
            doc_id=doc_id,
        )

    return {
        "doc_id": doc_id,
        "chunk_path": str(chunk_path) if chunk_path is not None else None,
        "graph_result": graph_result,
        "entity_stats": entity_stats,
        "embedding_stats": embedding_stats,
    }


def build_document_work_items(
    config,
    driver,
    preprocessing_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Decide which documents should be processed for graph/enrichment work.

    Modes:
    - if preprocessing ran in this execution, use those results
    - else if graph loading is requested, use cached chunk files
    - else if only enrichment is requested, use documents already present in Neo4j
    """
    if preprocessing_results:
        logger.info(
            "Using %d preprocessing results as document source for graph/enrichment",
            len(preprocessing_results),
        )
        return [
            {
                "doc_id": row["doc_id"],
                "chunk_path": Path(row["chunk_path"]),
                "source": row.get("source", "preprocessing"),
            }
            for row in preprocessing_results
        ]

    if getattr(config, "run_graph_loader", False):
        chunk_paths = list_cached_chunk_paths(config)
        logger.info(
            "Using %d cached chunk files as document source for graph loading",
            len(chunk_paths),
        )
        return [
            {
                "doc_id": chunk_path_to_doc_id(chunk_path),
                "chunk_path": chunk_path,
                "source": "chunk_cache",
            }
            for chunk_path in chunk_paths
        ]

    doc_ids = get_graph_document_ids(driver)
    logger.info(
        "Using %d existing Neo4j documents as source for enrichment-only run",
        len(doc_ids),
    )
    return [
        {
            "doc_id": doc_id,
            "chunk_path": None,
            "source": "neo4j",
        }
        for doc_id in doc_ids
    ]


def initialize_document_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize a per-document result object that can be updated phase by phase.
    """
    chunk_path = item.get("chunk_path")

    return {
        "doc_id": item["doc_id"],
        "chunk_path": str(chunk_path) if chunk_path is not None else None,
        "source": item.get("source"),
        "graph_result": None,
        "entity_stats": None,
        "embedding_stats": None,
    }


def record_stage_error(
    result: Dict[str, Any],
    stage: str,
    error: Exception,
) -> None:
    """
    Attach structured error information to a document result.
    """
    result.setdefault("failed_stages", []).append(stage)
    result.setdefault("errors", []).append(
        {
            "stage": stage,
            "error": str(error),
        }
    )

    # Backward-friendly top-level fields for quick inspection.
    result["stage"] = stage
    result["error"] = str(error)


def record_stage_skip(
    result: Dict[str, Any],
    stage: str,
    reason: str,
) -> None:
    """
    Attach structured skip information to a document result.
    """
    result.setdefault("skipped_stages", []).append(
        {
            "stage": stage,
            "reason": reason,
        }
    )


def run_graph_pipeline(config) -> Dict[str, Any]:
    """
    Flexible graph pipeline.

    Supported usage patterns:
    - preprocessing only
    - graph loading only from cached chunks
    - entities only from existing Neo4j graph
    - embeddings only from existing Neo4j graph
    - full pipeline

    Execution order for combined/full runs:
    1. preprocess all documents
    2. graph-load all documents
    3. extract entities for all documents
    4. clear chat model cache before embeddings, if needed
    5. compute embeddings for all documents
    6. run global concept disambiguation
    7. run sanity checks
    """
    ensure_pipeline_dirs(config)

    pdf_dir = Path(config.pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    logger.info("Found %d PDF files in %s", len(pdf_files), pdf_dir)

    need_preprocessing = requires_preprocessing(config)
    need_neo4j = requires_neo4j(config)
    need_doc_level_processing = needs_document_level_processing(config)

    if need_preprocessing:
        validate_preprocessing_paths(config)

    md_converter = get_markdown_converter(config) if need_preprocessing else None
    driver = get_neo4j_driver(verify=True) if need_neo4j else None

    preprocessing_results: List[Dict[str, Any]] = []
    document_results: List[Dict[str, Any]] = []
    disambiguation_stats = None
    sanity_summary = None

    try:
        if need_preprocessing:
            if not pdf_files:
                logger.warning(
                    "Preprocessing requested but no PDF files were found in %s",
                    pdf_dir,
                )

            for pdf_path in pdf_files:
                try:
                    prep_result = preprocess_single_document(
                        config=config,
                        md_converter=md_converter,
                        pdf_path=pdf_path,
                    )
                    preprocessing_results.append(prep_result)

                    if not need_doc_level_processing:
                        document_results.append(prep_result)

                except Exception as e:
                    logger.exception(
                        "Failed preprocessing document %s: %s",
                        pdf_path.stem,
                        e,
                    )
                    error_result = {
                        "doc_id": pdf_path.stem,
                        "error": str(e),
                        "stage": "preprocessing",
                    }
                    preprocessing_results.append(error_result)
                    document_results.append(error_result)

        if need_neo4j and need_doc_level_processing:
            work_items = build_document_work_items(
                config=config,
                driver=driver,
                preprocessing_results=[
                    row for row in preprocessing_results if "chunk_path" in row
                ],
            )

            if not work_items:
                logger.warning(
                    "No document work items found for graph/enrichment processing"
                )

            document_results_by_doc: Dict[str, Dict[str, Any]] = {
                item["doc_id"]: initialize_document_result(item)
                for item in work_items
            }

            graph_failed_doc_ids: Set[str] = set()

            if getattr(config, "run_graph_loader", False):
                logger.info("=== Graph loading phase: all documents ===")

                for item in work_items:
                    doc_id = item["doc_id"]
                    result = document_results_by_doc[doc_id]

                    try:
                        graph_result = process_document_graph_loading(
                            driver=driver,
                            config=config,
                            doc_id=doc_id,
                            chunk_path=item["chunk_path"],
                        )

                        result["graph_result"] = graph_result

                        if graph_result is None:
                            raise RuntimeError(
                                "Graph loader returned None; downstream stages "
                                "will be skipped for this document."
                            )

                    except Exception as e:
                        logger.exception(
                            "Failed graph loading document %s: %s",
                            doc_id,
                            e,
                        )
                        record_stage_error(result, "graph_loader", e)
                        graph_failed_doc_ids.add(doc_id)

            if getattr(config, "run_entity_extraction", False):
                logger.info("=== Entity extraction phase: all documents ===")

                for item in work_items:
                    doc_id = item["doc_id"]
                    result = document_results_by_doc[doc_id]

                    if doc_id in graph_failed_doc_ids:
                        record_stage_skip(
                            result,
                            "entity_extraction",
                            "skipped_because_graph_loader_failed",
                        )
                        logger.warning(
                            "Skipping entity extraction for %s because graph loading failed",
                            doc_id,
                        )
                        continue

                    try:
                        result["entity_stats"] = process_document_entity_extraction(
                            driver=driver,
                            config=config,
                            doc_id=doc_id,
                        )

                    except Exception as e:
                        logger.exception(
                            "Failed entity extraction for document %s: %s",
                            doc_id,
                            e,
                        )
                        record_stage_error(result, "entity_extraction", e)

            if (
                getattr(config, "run_entity_extraction", False)
                and getattr(config, "run_embeddings", False)
                and should_clear_chat_cache_before_embeddings(config)
            ):
                logger.info(
                    "Clearing chat model cache before embeddings phase "
                    "because entity extraction and embeddings are both enabled"
                )
                clear_chat_model_cache()

            if getattr(config, "run_embeddings", False):
                logger.info("=== Embedding phase: all documents ===")

                for item in work_items:
                    doc_id = item["doc_id"]
                    result = document_results_by_doc[doc_id]

                    if doc_id in graph_failed_doc_ids:
                        record_stage_skip(
                            result,
                            "embeddings",
                            "skipped_because_graph_loader_failed",
                        )
                        logger.warning(
                            "Skipping embeddings for %s because graph loading failed",
                            doc_id,
                        )
                        continue

                    try:
                        result["embedding_stats"] = process_document_embeddings(
                            driver=driver,
                            config=config,
                            doc_id=doc_id,
                        )

                    except Exception as e:
                        logger.exception(
                            "Failed embeddings for document %s: %s",
                            doc_id,
                            e,
                        )
                        record_stage_error(result, "embeddings", e)

            document_results.extend(document_results_by_doc.values())

        if need_neo4j and getattr(config, "run_entity_disambiguation", False):
            logger.info("Running global concept disambiguation")
            disambiguation_stats = disambiguate_concepts(
                driver,
                delete_orphans=getattr(
                    config,
                    "disambiguation_delete_orphans",
                    True,
                ),
            )

        if need_neo4j and getattr(config, "run_sanity_checks", False):
            sanity_mode = getattr(config, "sanity_mode", "full")
            logger.info(
                "Running global graph sanity checks | mode=%s",
                sanity_mode,
            )
            sanity_summary = run_sanity_checks(
                driver=driver,
                mode=sanity_mode,
                sample_limit=getattr(config, "sanity_sample_limit", 10),
                log_samples=getattr(config, "sanity_log_samples", True),
            )

    finally:
        if driver is not None:
            close_driver(driver)

    summary = {
        "documents_processed": len(document_results),
        "document_results": document_results,
        "disambiguation_stats": disambiguation_stats,
        "sanity_summary": sanity_summary,
    }

    logger.info("Graph pipeline completed")
    return summary