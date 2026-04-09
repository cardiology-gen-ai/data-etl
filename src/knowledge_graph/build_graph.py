import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from managers.table_of_contents_manager import GuidelineTOCExtractor
from managers.markdown_conversion_manager import MarkdownConverter
from managers.markdown_manager import MarkdownManager
from managers.hierarchical_chunking_manager import build_hierarchical_chunks

from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver
from knowledge_graph.graph_loader import build_graph_from_chunks
from knowledge_graph.add_entities import add_entities_from_sections
from knowledge_graph.entity_disambiguation import disambiguate_concepts
from knowledge_graph.add_embeddings import add_embeddings_to_sections
from knowledge_graph.sanity_checks import run_sanity_checks


logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_pipeline_dirs(config) -> None:
    """
    Create all required output/cache directories for the graph pipeline.
    """
    ensure_dir(Path(config.toc_dir))
    ensure_dir(Path(config.markdown_dir))
    ensure_dir(Path(config.image_dir))
    ensure_dir(Path(config.anchor_dir))
    ensure_dir(Path(config.chunk_dir))


def requires_neo4j(config) -> bool:
    return any([
        getattr(config, "run_graph_loader", False),
        getattr(config, "run_entity_extraction", False),
        getattr(config, "run_embeddings", False),
        getattr(config, "run_entity_disambiguation", False),
        getattr(config, "run_sanity_checks", False),
    ])


def requires_preprocessing(config) -> bool:
    """
    Preprocessing is required when:
    - the user explicitly wants preprocessing, or
    - graph loading is requested and chunk files may need to be built.
    """
    return any([
        getattr(config, "run_preprocessing", False),
        getattr(config, "run_graph_loader", False),
    ])


def validate_preprocessing_paths(config) -> None:
    """
    Ensure that the graph pipeline paths are correct and consistent with the MarkdownConverter configuration.
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
        logger.info("TOC exists, loading cached version")
        return json.loads(toc_path.read_text(encoding="utf-8"))

    logger.info("Extracting TOC")
    extractor = GuidelineTOCExtractor(
        pdf_path=str(pdf_path),
        doc_id=doc_id,
    )
    extractor.save(str(toc_path))

    return json.loads(toc_path.read_text(encoding="utf-8"))


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
        logger.info("Markdown exists, loading cached version")
    else:
        logger.info("Converting PDF to Markdown")
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
        logger.info("Forcing page anchor recomputation")
        anchor_path.unlink()

    logger.info("Loading or computing page anchors")
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
        logger.info("Chunks exist, loading cached version")
        return chunk_path

    logger.info("Building hierarchical chunks")

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
    Run only preprocessing/chunk preparation for one PDF.
    """
    doc_id = pdf_path.stem
    logger.info("=== Preprocessing %s ===", doc_id)

    chunk_path = get_cached_chunk_path(config, doc_id)

    if chunk_path.exists() and not getattr(config, "force_chunks", False):
        logger.info("Chunk cache already available for %s", doc_id)
        return {
            "doc_id": doc_id,
            "chunk_path": str(chunk_path),
        }

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
    }


def process_document_graph_and_enrichment(
    driver,
    config,
    doc_id: str,
    chunk_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run graph loading and/or enrichment for one document.
    """
    graph_result = None
    entity_stats = None
    embedding_stats = None

    if getattr(config, "run_graph_loader", False):
        if chunk_path is None:
            raise ValueError(
                f"Graph loading requested for document {doc_id}, but chunk_path is missing."
            )

        logger.info("Loading graph structure into Neo4j for document %s", doc_id)
        graph_result = build_graph_from_chunks(
            driver=driver,
            chunk_file=chunk_path,
            batch_size=getattr(config, "graph_loader_batch_size", 200),
            replace_existing_document=getattr(
                config,
                "graph_loader_replace_existing_document",
                True,
            ),
        )

    if getattr(config, "run_entity_extraction", False):
        logger.info("Extracting entities for document %s", doc_id)
        entity_stats = add_entities_from_sections(
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
        )

    if getattr(config, "run_embeddings", False):
        logger.info("Computing embeddings for document %s", doc_id)
        embedding_stats = add_embeddings_to_sections(
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
        )

    return {
        "doc_id": doc_id,
        "chunk_path": str(chunk_path) if chunk_path is not None else None,
        "graph_result": graph_result,
        "entity_stats": entity_stats,
        "embedding_stats": embedding_stats,
    }


def run_graph_pipeline(config) -> Dict[str, Any]:
    """
    Flexible graph pipeline.

    Supported usage patterns:
    - preprocessing only
    - graph loading only
    - entities only
    - embeddings only
    - full pipeline
    """
    ensure_pipeline_dirs(config)

    pdf_dir = Path(config.pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    logger.info("Found %d PDF files in %s", len(pdf_files), pdf_dir)

    need_preprocessing = requires_preprocessing(config)
    need_neo4j = requires_neo4j(config)

    if need_preprocessing:
        validate_preprocessing_paths(config)

    md_converter = get_markdown_converter(config) if need_preprocessing else None
    driver = get_neo4j_driver(verify=True) if need_neo4j else None

    document_results: List[Dict[str, Any]] = []
    disambiguation_stats = None
    sanity_summary = None

    try:
        if need_preprocessing:
            for pdf_path in pdf_files:
                try:
                    prep_result = preprocess_single_document(
                        config=config,
                        md_converter=md_converter,
                        pdf_path=pdf_path,
                    )

                    if need_neo4j and any([
                        getattr(config, "run_graph_loader", False),
                        getattr(config, "run_entity_extraction", False),
                        getattr(config, "run_embeddings", False),
                    ]):
                        result = process_document_graph_and_enrichment(
                            driver=driver,
                            config=config,
                            doc_id=prep_result["doc_id"],
                            chunk_path=Path(prep_result["chunk_path"]),
                        )
                    else:
                        result = prep_result

                    document_results.append(result)

                except Exception as e:
                    logger.exception("Failed processing document %s: %s", pdf_path.stem, e)
                    document_results.append(
                        {
                            "doc_id": pdf_path.stem,
                            "error": str(e),
                        }
                    )

        else:
            # No preprocessing requested: use document ids from available PDFs only
            # to scope document-level enrichment steps.
            for pdf_path in pdf_files:
                try:
                    result = process_document_graph_and_enrichment(
                        driver=driver,
                        config=config,
                        doc_id=pdf_path.stem,
                        chunk_path=None,
                    )
                    document_results.append(result)

                except Exception as e:
                    logger.exception("Failed processing document %s: %s", pdf_path.stem, e)
                    document_results.append(
                        {
                            "doc_id": pdf_path.stem,
                            "error": str(e),
                        }
                    )

        if need_neo4j and getattr(config, "run_entity_disambiguation", False):
            logger.info("Running global concept disambiguation")
            disambiguation_stats = disambiguate_concepts(driver)

        if need_neo4j and getattr(config, "run_sanity_checks", False):
            logger.info("Running global graph sanity checks")
            sanity_summary = run_sanity_checks(
                driver=driver,
                sample_limit=getattr(config, "sanity_sample_limit", 10),
                log_samples=getattr(config, "sanity_log_samples", True),
            )

    finally:
        close_driver(driver)

    summary = {
        "documents_processed": len(document_results),
        "document_results": document_results,
        "disambiguation_stats": disambiguation_stats,
        "sanity_summary": sanity_summary,
    }

    logger.info("Graph pipeline completed")
    return summary