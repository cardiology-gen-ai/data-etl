import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from dotenv import load_dotenv

from config.manager import PreprocessingConfig
from knowledge_graph.build_graph import run_graph_pipeline
from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver


logger = logging.getLogger(__name__)


@dataclass
class GraphPipelineConfig:
    """
    Minimal config object expected by build_graph.py.

    Important:
    - pdf_dir must match preprocessing_config.input_folder.folder
    - markdown_dir must match preprocessing_config.output_folder.folder
    """
    pdf_dir: Path
    toc_dir: Path
    markdown_dir: Path
    image_dir: Path
    anchor_dir: Path
    chunk_dir: Path
    preprocessing_config: Any

    # Preprocessing stage toggle
    run_preprocessing: bool = False

    # Cache / recomputation flags
    force_toc: bool = False
    force_markdown: bool = False
    force_anchors: bool = False
    force_chunks: bool = False

    # Pipeline stage toggles
    run_graph_loader: bool = True
    run_entity_extraction: bool = True
    run_embeddings: bool = True
    run_entity_disambiguation: bool = True
    run_sanity_checks: bool = True

    # Graph loader
    graph_loader_batch_size: int = 200
    graph_loader_replace_existing_document: bool = True

    # Entity extraction
    entity_use_section_text: bool = True
    entity_max_sections: Optional[int] = None
    entity_max_sections_per_batch: int = 2
    entity_max_batch_chars: int = 12000
    entity_emergency_max_single_chars: int = 12000
    entity_skip_processed: bool = True

    # Embeddings
    embedding_max_sections: Optional[int] = None
    embedding_batch_size: int = 8
    embedding_force_reembed: bool = False
    embedding_include_title: bool = True
    embedding_include_body: bool = True
    embedding_max_chars_per_section: int = 8000

    # Sanity checks
    sanity_sample_limit: int = 10
    sanity_log_samples: bool = True


def _get_optional_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _resolve_config_path_from_env() -> Path:
    """
    Resolve CONFIG_PATH from the environment.

    Handles occasional malformed values like:
        CONFIG_PATH=CONFIG_PATH=/some/path/config.json
    by stripping the repeated prefix if present.
    """
    raw = _get_optional_env("CONFIG_PATH")
    if not raw:
        raise RuntimeError("Missing required environment variable: CONFIG_PATH")

    if raw.startswith("CONFIG_PATH="):
        raw = raw[len("CONFIG_PATH="):].strip()

    path = Path(raw).expanduser().resolve()
    return path


def clear_graph_data() -> None:
    """
    Delete all nodes and relationships from Neo4j, but keep constraints/indexes.
    """
    logger.warning("Clearing all existing Neo4j nodes and relationships")

    driver = get_neo4j_driver(verify=True)
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j graph data cleared")
    finally:
        close_driver(driver)


def align_preprocessing_paths(
    preprocessing_config: Any,
    pdf_dir: Path,
) -> Any:
    """
    Align preprocessing_config paths to absolute paths so they are consistent
    with the graph pipeline config and with build_graph.py validation.
    """
    pdf_dir = pdf_dir.resolve()
    markdown_dir = (pdf_dir.parent / "mddocs").resolve()

    preprocessing_config.input_folder.folder = pdf_dir
    preprocessing_config.output_folder.folder = markdown_dir

    preprocessing_config.input_folder.parent_folder = str(pdf_dir.parent)
    preprocessing_config.input_folder.child_folder = pdf_dir.name

    preprocessing_config.output_folder.parent_folder = str(markdown_dir.parent)
    preprocessing_config.output_folder.child_folder = markdown_dir.name

    return preprocessing_config


def make_graph_pipeline_config(
    preprocessing_config: Any,
    pdf_dir: Path,
    work_root: Path,
    run_preprocessing: bool = False,
    run_graph_loader: bool = True,
    run_entity_extraction: bool = False,
    run_embeddings: bool = False,
    run_entity_disambiguation: bool = False,
    run_sanity_checks: bool = True,
    graph_loader_replace_existing_document: bool = True,
    entity_use_section_text: bool = True,
) -> GraphPipelineConfig:
    """
    Build the graph pipeline config.
    """
    pdf_dir = pdf_dir.resolve()
    work_root = work_root.resolve()

    return GraphPipelineConfig(
        pdf_dir=pdf_dir,
        toc_dir=(work_root / "toc").resolve(),
        markdown_dir=Path(preprocessing_config.output_folder.folder).resolve(),
        image_dir=(work_root / "images").resolve(),
        anchor_dir=(work_root / "anchors").resolve(),
        chunk_dir=(work_root / "chunks").resolve(),
        preprocessing_config=preprocessing_config,

        run_preprocessing=run_preprocessing,

        force_toc=False,
        force_markdown=False,
        force_anchors=False,
        force_chunks=False,

        run_graph_loader=run_graph_loader,
        run_entity_extraction=run_entity_extraction,
        run_embeddings=run_embeddings,
        run_entity_disambiguation=run_entity_disambiguation,
        run_sanity_checks=run_sanity_checks,

        graph_loader_batch_size=200,
        graph_loader_replace_existing_document=graph_loader_replace_existing_document,

        entity_use_section_text=entity_use_section_text,
        entity_max_sections=None,
        entity_max_sections_per_batch=2,
        entity_max_batch_chars=12000,
        entity_emergency_max_single_chars=12000,
        entity_skip_processed=True,

        embedding_max_sections=None,
        embedding_batch_size=8,
        embedding_force_reembed=False,
        embedding_include_title=True,
        embedding_include_body=True,
        embedding_max_chars_per_section=8000,

        sanity_sample_limit=10,
        sanity_log_samples=True,
    )


def inject_kg_runtime_env(
    kg_chat_model: Optional[str] = None,
    kg_embedding_model: Optional[str] = None,
    kg_local_files_only: bool = True,
    kg_chat_max_new_tokens: int = 512,
) -> None:
    """
    Inject runtime environment variables for knowledge_graph.llm_utils.

    This remains a pragmatic bridge until model/runtime settings are moved
    into a cleaner explicit config object.
    """
    if kg_chat_model:
        os.environ["KG_CHAT_MODEL"] = kg_chat_model

    if kg_embedding_model:
        os.environ["KG_EMBEDDING_MODEL"] = kg_embedding_model

    os.environ["KG_LOCAL_FILES_ONLY"] = "true" if kg_local_files_only else "false"
    os.environ["KG_CHAT_MAX_NEW_TOKENS"] = str(kg_chat_max_new_tokens)

    logger.info(
        "Injected KG runtime env | chat=%s | embedding=%s | local_files_only=%s | max_new_tokens=%s",
        kg_chat_model,
        kg_embedding_model,
        kg_local_files_only,
        kg_chat_max_new_tokens,
    )


def main(
    pdf_dir: Path,
    work_root: Path,
    clear_neo4j_before_run: bool = False,
    run_preprocessing: bool = False,
    run_graph_loader: bool = True,
    run_entity_extraction: bool = False,
    run_embeddings: bool = False,
    run_entity_disambiguation: bool = False,
    run_sanity_checks: bool = True,
    graph_loader_replace_existing_document: bool = True,
    entity_use_section_text: bool = True,
    kg_chat_model: Optional[str] = None,
    kg_embedding_model: Optional[str] = None,
    kg_local_files_only: bool = True,
    kg_chat_max_new_tokens: int = 512,
):
    """
    Run the KG pipeline.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    loaded = load_dotenv(env_path)

    logger.info("Loading .env from: %s", env_path)
    logger.info(".env loaded: %s", loaded)

    required_env = [
        "CONFIG_PATH",
    ]
    missing = [k for k in required_env if not _get_optional_env(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {missing}. "
            f"Expected .env at {env_path}"
        )

    inject_kg_runtime_env(
        kg_chat_model=kg_chat_model,
        kg_embedding_model=kg_embedding_model,
        kg_local_files_only=kg_local_files_only,
        kg_chat_max_new_tokens=kg_chat_max_new_tokens,
    )

    config_path = _resolve_config_path_from_env()
    logger.info("Using config path: %s", config_path)

    app_id = _get_optional_env("KG_APP_ID") or "cardiology_protocols"

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    if app_id not in raw_config:
        raise KeyError(f"App id '{app_id}' not found in config file {config_path}")

    app_config = raw_config[app_id]
    preprocessing_dict = app_config["preprocessing"]

    preprocessing_config = PreprocessingConfig.from_config(
        preprocessing_dict,
        embeddings=None,
    )

    preprocessing_config = align_preprocessing_paths(
        preprocessing_config=preprocessing_config,
        pdf_dir=pdf_dir,
    )

    config = make_graph_pipeline_config(
        preprocessing_config=preprocessing_config,
        pdf_dir=pdf_dir,
        work_root=work_root,
        run_preprocessing=run_preprocessing,
        run_graph_loader=run_graph_loader,
        run_entity_extraction=run_entity_extraction,
        run_embeddings=run_embeddings,
        run_entity_disambiguation=run_entity_disambiguation,
        run_sanity_checks=run_sanity_checks,
        graph_loader_replace_existing_document=graph_loader_replace_existing_document,
        entity_use_section_text=entity_use_section_text,
    )

    if clear_neo4j_before_run:
        clear_graph_data()

    summary = run_graph_pipeline(config)

    logger.info("Documents processed: %d", summary["documents_processed"])

    if summary.get("disambiguation_stats") is not None:
        logger.info("Disambiguation stats: %s", summary["disambiguation_stats"])

    if summary.get("sanity_summary") is not None:
        logger.info(
            "Sanity summary | issue_checks=%d | error_checks=%d | warning_checks=%d | info_checks=%d",
            summary["sanity_summary"]["checks_with_issues"],
            summary["sanity_summary"]["error_checks_with_issues"],
            summary["sanity_summary"]["warning_checks_with_issues"],
            summary["sanity_summary"]["info_checks_with_issues"],
        )

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    pdf_dir = project_root / "test_data" / "pdfdocs"
    work_root = project_root / "test_data" / "graph_cache_test"

    PIPELINE_PHASE = os.getenv(
        "KG_PIPELINE_PHASE",
        "preprocess",
    )  # preprocess, graph, entities, embeddings, full

    # Runtime model settings
    KG_CHAT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
    KG_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

    if PIPELINE_PHASE == "preprocess":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            clear_neo4j_before_run=False,
            run_preprocessing=True,
            run_graph_loader=False,
            run_entity_extraction=False,
            run_embeddings=False,
            run_entity_disambiguation=False,
            run_sanity_checks=False,
            entity_use_section_text=True,
            kg_chat_model=KG_CHAT_MODEL,
            kg_embedding_model=KG_EMBEDDING_MODEL,
            kg_local_files_only=True,
            kg_chat_max_new_tokens=512,
        )

    elif PIPELINE_PHASE == "graph":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            clear_neo4j_before_run=True,
            run_preprocessing=False,
            run_graph_loader=True,
            run_entity_extraction=False,
            run_embeddings=False,
            run_entity_disambiguation=False,
            run_sanity_checks=True,
            graph_loader_replace_existing_document=True,
            entity_use_section_text=True,
            kg_chat_model=KG_CHAT_MODEL,
            kg_embedding_model=KG_EMBEDDING_MODEL,
            kg_local_files_only=True,
            kg_chat_max_new_tokens=512,
        )

    elif PIPELINE_PHASE == "entities":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            clear_neo4j_before_run=False,
            run_preprocessing=False,
            run_graph_loader=False,
            run_entity_extraction=True,
            run_embeddings=False,
            run_entity_disambiguation=True,
            run_sanity_checks=True,
            entity_use_section_text=True,
            kg_chat_model=KG_CHAT_MODEL,
            kg_embedding_model=KG_EMBEDDING_MODEL,
            kg_local_files_only=True,
            kg_chat_max_new_tokens=512,
        )

    elif PIPELINE_PHASE == "embeddings":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            clear_neo4j_before_run=False,
            run_preprocessing=False,
            run_graph_loader=False,
            run_entity_extraction=False,
            run_embeddings=True,
            run_entity_disambiguation=False,
            run_sanity_checks=True,
            entity_use_section_text=True,
            kg_chat_model=KG_CHAT_MODEL,
            kg_embedding_model=KG_EMBEDDING_MODEL,
            kg_local_files_only=True,
            kg_chat_max_new_tokens=512,
        )

    elif PIPELINE_PHASE == "full":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            clear_neo4j_before_run=True,
            run_preprocessing=True,
            run_graph_loader=True,
            run_entity_extraction=True,
            run_embeddings=True,
            run_entity_disambiguation=True,
            run_sanity_checks=True,
            graph_loader_replace_existing_document=True,
            entity_use_section_text=True,
            kg_chat_model=KG_CHAT_MODEL,
            kg_embedding_model=KG_EMBEDDING_MODEL,
            kg_local_files_only=True,
            kg_chat_max_new_tokens=512,
        )

    else:
        raise ValueError(
            f"Unsupported PIPELINE_PHASE='{PIPELINE_PHASE}'. "
            "Use one of: 'preprocess', 'graph', 'entities', 'embeddings', 'full'."
        )