import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from dotenv import load_dotenv

from config.manager import PreprocessingConfig
from knowledge_graph.build_graph import run_graph_pipeline
from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver


logger = logging.getLogger(__name__)


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_RUN_LOG_HANDLER_NAME = "kg_run_file"
_CURRENT_RUN_LOG_CONTEXT: Optional[tuple[Path, str, Path]] = None


@dataclass
class GraphPipelineConfig:
    """
    Minimal config object expected by build_graph.py.

    Important:
    - pdf_dir must match preprocessing_config.input_folder.folder
    - markdown_dir must match preprocessing_config.output_folder.folder
    - acronym_dir stores one cached acronym dictionary per document
    """

    pdf_dir: Path
    toc_dir: Path
    markdown_dir: Path
    image_dir: Path
    anchor_dir: Path
    chunk_dir: Path
    acronym_dir: Path
    preprocessing_config: Any

    # Preprocessing stage toggle
    run_preprocessing: bool = False

    # Cache / recomputation flags
    force_toc: bool = False
    force_markdown: bool = False
    force_anchors: bool = False
    force_chunks: bool = False
    force_acronyms: bool = False

    # Acronym extraction
    # This is used only during preprocessing/full runs.
    # It creates cached JSON files and does not write anything to Neo4j.
    run_acronym_extraction: bool = True
    acronym_sample_size: int = 0
    acronym_print_all: bool = False

    # Pipeline stage toggles
    run_graph_loader: bool = True
    run_entity_extraction: bool = True
    run_embeddings: bool = True
    run_entity_disambiguation: bool = True
    run_entity_normalization: bool = False
    run_sanity_checks: bool = True

    # Graph loader
    graph_loader_batch_size: int = 200
    graph_loader_min_text_chars_to_embed: int = 20
    graph_loader_replace_existing_document: bool = True

    # Entity extraction
    entity_use_section_text: bool = True
    entity_max_sections: Optional[int] = None
    entity_max_sections_per_batch: int = 1
    entity_max_batch_chars: int = 30000
    entity_emergency_max_single_chars: int = 12000
    entity_skip_processed: bool = True
    entity_replace_section_mentions: bool = True

    # Entity acronym validation
    # This is used during entity extraction. It consumes cached acronym JSON files.
    entity_use_acronym_validation: bool = True
    entity_acronym_dir: Optional[Path] = None

    # Entity review exports
    entity_export_review: bool = True
    entity_review_output_dir: Optional[Path] = None
    entity_clear_previous_review: bool = True
    entity_include_source_preview_in_review: bool = False

    # Embeddings
    embedding_max_sections: Optional[int] = None
    embedding_batch_size: int = 8
    embedding_force_reembed: bool = False
    embedding_include_title: bool = True
    embedding_include_body: bool = True
    embedding_max_chars_per_section: int = 8000
    embedding_allow_title_only: bool = False

    # Runtime / memory behavior
    clear_chat_cache_before_embeddings: bool = True

    # Entity disambiguation
    disambiguation_delete_orphans: bool = True

    # Entity UMLS normalization
    entity_normalization_doc_id: Optional[str] = None
    entity_normalization_backend: str = "umls_api"
    entity_normalization_model_name: str = "en_core_sci_sm"
    entity_normalization_linker_name: str = "umls"
    entity_normalization_threshold: float = 0.85
    entity_normalization_max_candidates: int = 3
    entity_normalization_use_acronyms: bool = True
    entity_normalization_acronym_dir: Optional[Path] = None
    entity_normalization_force: bool = False
    entity_normalization_dry_run: bool = False
    entity_normalization_export_review: bool = True
    entity_normalization_review_output_dir: Optional[Path] = None
    entity_normalization_fuzzy_threshold: int = 90
    entity_normalization_api_cache_dir: Optional[Path] = None
    entity_normalization_api_timeout: float = 30.0
    entity_normalization_api_rate_limit_per_second: float = 5.0
    entity_normalization_local_files_only: bool = False
    entity_normalization_min_available_memory_gb: float = 8.0

    # Sanity checks
    sanity_mode: Optional[str] = "full"
    sanity_sample_limit: int = 10
    sanity_log_samples: bool = True
    quality_max_chunk_chars: int = 50000


def _get_optional_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _get_env_bool(name: str, default: bool) -> bool:
    value = _get_optional_env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_int(name: str, default: int) -> int:
    value = _get_optional_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as e:
        raise RuntimeError(
            f"Environment variable {name} must be an integer, got: {value!r}"
        ) from e


def _get_env_optional_int(name: str) -> Optional[int]:
    value = _get_optional_env(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as e:
        raise RuntimeError(
            f"Environment variable {name} must be an integer, got: {value!r}"
        ) from e


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"Configuration value {name} must be an integer, got: {value!r}"
        ) from e


def _coerce_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"Configuration value {name} must be a number, got: {value!r}"
        ) from e


def _get_config_value(config: dict, *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _get_env_or_config_str(
    env_name: str,
    config_value: Any,
    default: Optional[str] = None,
) -> Optional[str]:
    env_value = _get_optional_env(env_name)
    if env_value is not None:
        return env_value
    if config_value is None:
        return default
    return str(config_value)


def _get_env_or_config_bool(
    env_name: str,
    config_value: Any,
    default: bool,
) -> bool:
    env_value = _get_optional_env(env_name)
    if env_value is not None:
        return _coerce_bool(env_value, env_name)
    if config_value is None:
        return default
    return _coerce_bool(config_value, env_name)


def _get_env_or_config_int(
    env_name: str,
    config_value: Any,
    default: int,
) -> int:
    env_value = _get_optional_env(env_name)
    if env_value is not None:
        return _coerce_int(env_value, env_name)
    if config_value is None:
        return default
    return _coerce_int(config_value, env_name)


def _get_env_or_config_float(
    env_name: str,
    config_value: Any,
    default: float,
) -> float:
    env_value = _get_optional_env(env_name)
    if env_value is not None:
        return _coerce_float(env_value, env_name)
    if config_value is None:
        return default
    return _coerce_float(config_value, env_name)


def _get_env_or_config_optional_int(
    env_name: str,
    config_value: Any,
) -> Optional[int]:
    env_value = _get_optional_env(env_name)
    if env_value is not None:
        return _coerce_int(env_value, env_name)
    if config_value in (None, ""):
        return None
    return _coerce_int(config_value, env_name)


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

    return Path(raw).expanduser().resolve()


def load_app_config_from_env() -> tuple[dict, Path, str]:
    """
    Load the configured app section from CONFIG_PATH.
    """
    config_path = _resolve_config_path_from_env()
    app_id = _get_optional_env("KG_APP_ID") or "cardiology_protocols"

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    if app_id not in raw_config:
        raise KeyError(f"App id '{app_id}' not found in config file {config_path}")

    return raw_config[app_id], config_path, app_id


def _resolve_project_path(
    value: Any,
    default: Path,
    project_root: Path,
) -> Path:
    if value in (None, ""):
        return default.resolve()

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def _resolve_project_path_from_env(
    name: str,
    default: Path,
    project_root: Path,
) -> Path:
    """
    Resolve a path-like env var relative to the project root.
    """
    return _resolve_project_path(_get_optional_env(name), default, project_root)


def _resolve_sanity_mode_from_phase(phase: str) -> Optional[str]:
    """
    Map pipeline phase to sanity-check mode.

    Mapping:
    - preprocess -> None
    - graph -> structure
    - entities -> entities
    - embeddings -> embeddings
    - normalization -> entities
    - full -> full
    """
    phase = phase.strip().lower()

    mapping = {
        "preprocess": None,
        "graph": "structure",
        "entities": "entities",
        "embeddings": "embeddings",
        "normalization": "entities",
        "full": "full",
    }

    if phase not in mapping:
        raise ValueError(
            f"Unsupported PIPELINE_PHASE='{phase}'. "
            "Use one of: 'preprocess', 'graph', 'entities', 'embeddings', "
            "'normalization', 'full'."
        )

    return mapping[phase]


def _find_run_log_handler() -> Optional[logging.FileHandler]:
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "name", "") == _RUN_LOG_HANDLER_NAME
        ):
            return handler

    return None


def configure_run_logging(
    work_root: Path,
    phase: str,
    project_root: Path,
    log_to_file: Optional[bool] = None,
    configured_log_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Add per-run file logging under ``<work_root>/logs`` unless overridden.

    This preserves existing console logging and avoids duplicate file handlers
    when the module is imported or ``main()`` is called multiple times in the
    same Python process.
    """
    enabled = (
        log_to_file
        if log_to_file is not None
        else _get_env_bool("KG_LOG_TO_FILE", True)
    )

    global _CURRENT_RUN_LOG_CONTEXT

    if not enabled:
        existing_handler = _find_run_log_handler()
        if existing_handler is not None:
            logging.getLogger().removeHandler(existing_handler)
            existing_handler.close()
        _CURRENT_RUN_LOG_CONTEXT = None
        return None

    configured_dir_value = _get_optional_env("KG_LOG_DIR")
    if configured_dir_value is not None:
        log_dir = _resolve_project_path(
            configured_dir_value,
            work_root / "logs",
            project_root,
        )
    elif configured_log_dir is not None:
        log_dir = Path(configured_log_dir).expanduser().resolve()
    else:
        log_dir = (work_root / "logs").resolve()

    safe_phase = (phase or "run").strip().lower() or "run"
    run_context = (work_root.resolve(), safe_phase, log_dir.resolve())

    existing_handler = _find_run_log_handler()
    if existing_handler is not None:
        if _CURRENT_RUN_LOG_CONTEXT == run_context:
            return Path(existing_handler.baseFilename).resolve()
        logging.getLogger().removeHandler(existing_handler)
        existing_handler.close()

    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{safe_phase}_{timestamp}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.set_name(_RUN_LOG_HANDLER_NAME)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    _CURRENT_RUN_LOG_CONTEXT = run_context

    logger.info("Writing run log to: %s", log_path)
    return log_path


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
    run_acronym_extraction: bool = True,
    force_toc: bool = False,
    force_markdown: bool = False,
    force_anchors: bool = False,
    force_chunks: bool = False,
    force_acronyms: bool = False,
    acronym_sample_size: int = 0,
    acronym_print_all: bool = False,
    run_graph_loader: bool = True,
    run_entity_extraction: bool = False,
    run_embeddings: bool = False,
    run_entity_disambiguation: bool = False,
    run_sanity_checks: bool = True,
    graph_loader_batch_size: int = 200,
    graph_loader_min_text_chars_to_embed: int = 20,
    graph_loader_replace_existing_document: bool = True,
    entity_use_section_text: bool = True,
    entity_max_sections: Optional[int] = None,
    entity_max_sections_per_batch: int = 1,
    entity_max_batch_chars: int = 30000,
    entity_emergency_max_single_chars: int = 12000,
    entity_skip_processed: bool = True,
    entity_replace_section_mentions: bool = True,
    entity_use_acronym_validation: bool = True,
    entity_acronym_dir: Optional[Path] = None,
    entity_export_review: bool = True,
    entity_review_output_dir: Optional[Path] = None,
    entity_clear_previous_review: bool = True,
    entity_include_source_preview_in_review: bool = False,
    embedding_max_sections: Optional[int] = None,
    embedding_batch_size: int = 8,
    embedding_force_reembed: bool = False,
    embedding_include_title: bool = True,
    embedding_include_body: bool = True,
    embedding_max_chars_per_section: int = 8000,
    embedding_allow_title_only: bool = False,
    clear_chat_cache_before_embeddings: bool = True,
    disambiguation_delete_orphans: bool = True,
    run_entity_normalization: bool = False,
    entity_normalization_doc_id: Optional[str] = None,
    entity_normalization_backend: str = "umls_api",
    entity_normalization_model_name: str = "en_core_sci_sm",
    entity_normalization_linker_name: str = "umls",
    entity_normalization_threshold: float = 0.85,
    entity_normalization_max_candidates: int = 3,
    entity_normalization_use_acronyms: bool = True,
    entity_normalization_acronym_dir: Optional[Path] = None,
    entity_normalization_force: bool = False,
    entity_normalization_dry_run: bool = False,
    entity_normalization_export_review: bool = True,
    entity_normalization_review_output_dir: Optional[Path] = None,
    entity_normalization_fuzzy_threshold: int = 90,
    entity_normalization_api_cache_dir: Optional[Path] = None,
    entity_normalization_api_timeout: float = 30.0,
    entity_normalization_api_rate_limit_per_second: float = 5.0,
    entity_normalization_local_files_only: bool = False,
    entity_normalization_min_available_memory_gb: float = 8.0,
    sanity_mode: Optional[str] = "full",
    quality_max_chunk_chars: int = 50000,
) -> GraphPipelineConfig:
    """
    Build the graph pipeline config.
    """
    pdf_dir = pdf_dir.resolve()
    work_root = work_root.resolve()

    resolved_entity_acronym_dir = (
        entity_acronym_dir.resolve()
        if entity_acronym_dir is not None
        else None
    )

    resolved_entity_review_output_dir = (
        entity_review_output_dir.resolve()
        if entity_review_output_dir is not None
        else None
    )

    resolved_entity_normalization_acronym_dir = (
        entity_normalization_acronym_dir.resolve()
        if entity_normalization_acronym_dir is not None
        else None
    )

    resolved_entity_normalization_review_output_dir = (
        entity_normalization_review_output_dir.resolve()
        if entity_normalization_review_output_dir is not None
        else None
    )
    resolved_entity_normalization_api_cache_dir = (
        entity_normalization_api_cache_dir.resolve()
        if entity_normalization_api_cache_dir is not None
        else None
    )

    return GraphPipelineConfig(
        pdf_dir=pdf_dir,
        toc_dir=(work_root / "toc").resolve(),
        markdown_dir=Path(preprocessing_config.output_folder.folder).resolve(),
        image_dir=(work_root / "images").resolve(),
        anchor_dir=(work_root / "anchors").resolve(),
        chunk_dir=(work_root / "chunks").resolve(),
        acronym_dir=(work_root / "acronyms").resolve(),
        preprocessing_config=preprocessing_config,

        run_preprocessing=run_preprocessing,

        force_toc=force_toc,
        force_markdown=force_markdown,
        force_anchors=force_anchors,
        force_chunks=force_chunks,
        force_acronyms=force_acronyms,

        run_acronym_extraction=run_acronym_extraction,
        acronym_sample_size=acronym_sample_size,
        acronym_print_all=acronym_print_all,

        run_graph_loader=run_graph_loader,
        run_entity_extraction=run_entity_extraction,
        run_embeddings=run_embeddings,
        run_entity_disambiguation=run_entity_disambiguation,
        run_entity_normalization=run_entity_normalization,
        run_sanity_checks=run_sanity_checks,

        graph_loader_batch_size=graph_loader_batch_size,
        graph_loader_min_text_chars_to_embed=graph_loader_min_text_chars_to_embed,
        graph_loader_replace_existing_document=graph_loader_replace_existing_document,

        entity_use_section_text=entity_use_section_text,
        entity_max_sections=entity_max_sections,
        entity_max_sections_per_batch=entity_max_sections_per_batch,
        entity_max_batch_chars=entity_max_batch_chars,
        entity_emergency_max_single_chars=entity_emergency_max_single_chars,
        entity_skip_processed=entity_skip_processed,
        entity_replace_section_mentions=entity_replace_section_mentions,

        entity_use_acronym_validation=entity_use_acronym_validation,
        entity_acronym_dir=resolved_entity_acronym_dir,

        entity_export_review=entity_export_review,
        entity_review_output_dir=resolved_entity_review_output_dir,
        entity_clear_previous_review=entity_clear_previous_review,
        entity_include_source_preview_in_review=entity_include_source_preview_in_review,

        embedding_max_sections=embedding_max_sections,
        embedding_batch_size=embedding_batch_size,
        embedding_force_reembed=embedding_force_reembed,
        embedding_include_title=embedding_include_title,
        embedding_include_body=embedding_include_body,
        embedding_max_chars_per_section=embedding_max_chars_per_section,
        embedding_allow_title_only=embedding_allow_title_only,

        clear_chat_cache_before_embeddings=clear_chat_cache_before_embeddings,

        disambiguation_delete_orphans=disambiguation_delete_orphans,

        entity_normalization_doc_id=entity_normalization_doc_id,
        entity_normalization_backend=entity_normalization_backend,
        entity_normalization_model_name=entity_normalization_model_name,
        entity_normalization_linker_name=entity_normalization_linker_name,
        entity_normalization_threshold=entity_normalization_threshold,
        entity_normalization_max_candidates=entity_normalization_max_candidates,
        entity_normalization_use_acronyms=entity_normalization_use_acronyms,
        entity_normalization_acronym_dir=resolved_entity_normalization_acronym_dir,
        entity_normalization_force=entity_normalization_force,
        entity_normalization_dry_run=entity_normalization_dry_run,
        entity_normalization_export_review=entity_normalization_export_review,
        entity_normalization_review_output_dir=(
            resolved_entity_normalization_review_output_dir
        ),
        entity_normalization_fuzzy_threshold=entity_normalization_fuzzy_threshold,
        entity_normalization_api_cache_dir=resolved_entity_normalization_api_cache_dir,
        entity_normalization_api_timeout=entity_normalization_api_timeout,
        entity_normalization_api_rate_limit_per_second=(
            entity_normalization_api_rate_limit_per_second
        ),
        entity_normalization_local_files_only=entity_normalization_local_files_only,
        entity_normalization_min_available_memory_gb=(
            entity_normalization_min_available_memory_gb
        ),

        sanity_mode=sanity_mode,
        sanity_sample_limit=10,
        sanity_log_samples=True,
        quality_max_chunk_chars=quality_max_chunk_chars,
    )


def inject_kg_runtime_env(
    kg_chat_provider: Optional[str] = None,
    kg_embedding_provider: Optional[str] = None,
    kg_chat_model: Optional[str] = None,
    kg_chat_model_path: Optional[str] = None,
    kg_embedding_model: Optional[str] = None,
    kg_embedding_model_path: Optional[str] = None,
    kg_local_files_only: Optional[bool] = None,
    kg_chat_max_new_tokens: Optional[int] = None,
) -> None:
    """
    Inject runtime environment variables for knowledge_graph.llm_utils.

    This remains a pragmatic bridge until model/runtime settings are moved
    into a cleaner explicit config object.
    """
    if kg_chat_provider:
        os.environ["KG_CHAT_PROVIDER"] = kg_chat_provider

    if kg_embedding_provider:
        os.environ["KG_EMBEDDING_PROVIDER"] = kg_embedding_provider

    if kg_chat_model:
        os.environ["KG_CHAT_MODEL"] = kg_chat_model

    if kg_chat_model_path is not None:
        os.environ["KG_CHAT_MODEL_PATH"] = kg_chat_model_path

    if kg_embedding_model:
        os.environ["KG_EMBEDDING_MODEL"] = kg_embedding_model

    if kg_embedding_model_path is not None:
        os.environ["KG_EMBEDDING_MODEL_PATH"] = kg_embedding_model_path

    if kg_local_files_only is not None:
        os.environ["KG_LOCAL_FILES_ONLY"] = "true" if kg_local_files_only else "false"

    if kg_chat_max_new_tokens is not None:
        os.environ["KG_CHAT_MAX_NEW_TOKENS"] = str(kg_chat_max_new_tokens)

    logger.info(
        "Injected KG runtime env | chat_provider=%s | embedding_provider=%s | chat=%s | embedding=%s | local_files_only=%s | max_new_tokens=%s",
        kg_chat_provider,
        kg_embedding_provider,
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
    run_acronym_extraction: bool = True,
    force_toc: bool = False,
    force_markdown: bool = False,
    force_anchors: bool = False,
    force_chunks: bool = False,
    force_acronyms: bool = False,
    acronym_sample_size: int = 0,
    acronym_print_all: bool = False,
    run_graph_loader: bool = True,
    run_entity_extraction: bool = False,
    run_embeddings: bool = False,
    run_entity_disambiguation: bool = False,
    run_sanity_checks: bool = True,
    graph_loader_batch_size: int = 200,
    graph_loader_min_text_chars_to_embed: int = 20,
    graph_loader_replace_existing_document: bool = True,
    entity_use_section_text: bool = True,
    entity_max_sections: Optional[int] = None,
    entity_max_sections_per_batch: int = 1,
    entity_max_batch_chars: int = 30000,
    entity_emergency_max_single_chars: int = 12000,
    entity_skip_processed: bool = True,
    entity_replace_section_mentions: bool = True,
    entity_use_acronym_validation: bool = True,
    entity_acronym_dir: Optional[Path] = None,
    entity_export_review: bool = True,
    entity_review_output_dir: Optional[Path] = None,
    entity_clear_previous_review: bool = True,
    entity_include_source_preview_in_review: bool = False,
    embedding_max_sections: Optional[int] = None,
    embedding_batch_size: int = 8,
    embedding_force_reembed: bool = False,
    embedding_include_title: bool = True,
    embedding_include_body: bool = True,
    embedding_max_chars_per_section: int = 8000,
    embedding_allow_title_only: bool = False,
    clear_chat_cache_before_embeddings: bool = True,
    disambiguation_delete_orphans: bool = True,
    run_entity_normalization: bool = False,
    entity_normalization_doc_id: Optional[str] = None,
    entity_normalization_backend: str = "umls_api",
    entity_normalization_model_name: str = "en_core_sci_sm",
    entity_normalization_linker_name: str = "umls",
    entity_normalization_threshold: float = 0.85,
    entity_normalization_max_candidates: int = 3,
    entity_normalization_use_acronyms: bool = True,
    entity_normalization_acronym_dir: Optional[Path] = None,
    entity_normalization_force: bool = False,
    entity_normalization_dry_run: bool = False,
    entity_normalization_export_review: bool = True,
    entity_normalization_review_output_dir: Optional[Path] = None,
    entity_normalization_fuzzy_threshold: int = 90,
    entity_normalization_api_cache_dir: Optional[Path] = None,
    entity_normalization_api_timeout: float = 30.0,
    entity_normalization_api_rate_limit_per_second: float = 5.0,
    entity_normalization_local_files_only: bool = False,
    entity_normalization_min_available_memory_gb: float = 8.0,
    sanity_mode: Optional[str] = "full",
    quality_max_chunk_chars: int = 50000,
    kg_chat_provider: Optional[str] = None,
    kg_embedding_provider: Optional[str] = None,
    kg_chat_model: Optional[str] = None,
    kg_chat_model_path: Optional[str] = None,
    kg_embedding_model: Optional[str] = None,
    kg_embedding_model_path: Optional[str] = None,
    kg_local_files_only: Optional[bool] = None,
    kg_chat_max_new_tokens: Optional[int] = None,
    log_to_file: Optional[bool] = None,
    log_dir: Optional[Path] = None,
    pipeline_phase: Optional[str] = None,
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

    if run_sanity_checks and sanity_mode is None:
        raise ValueError("sanity_mode must be set when run_sanity_checks=True")

    configure_run_logging(
        work_root=work_root.resolve(),
        phase=(
            (pipeline_phase or _get_optional_env("KG_PIPELINE_PHASE") or "manual")
            .strip()
            .lower()
        ),
        project_root=Path(__file__).resolve().parent.parent,
        log_to_file=log_to_file,
        configured_log_dir=log_dir,
    )

    inject_kg_runtime_env(
        kg_chat_provider=kg_chat_provider,
        kg_embedding_provider=kg_embedding_provider,
        kg_chat_model=kg_chat_model,
        kg_chat_model_path=kg_chat_model_path,
        kg_embedding_model=kg_embedding_model,
        kg_embedding_model_path=kg_embedding_model_path,
        kg_local_files_only=kg_local_files_only,
        kg_chat_max_new_tokens=kg_chat_max_new_tokens,
    )

    app_config, config_path, app_id = load_app_config_from_env()
    logger.info("Using config path: %s", config_path)
    logger.info("Using app id: %s", app_id)
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
        run_acronym_extraction=run_acronym_extraction,
        force_toc=force_toc,
        force_markdown=force_markdown,
        force_anchors=force_anchors,
        force_chunks=force_chunks,
        force_acronyms=force_acronyms,
        acronym_sample_size=acronym_sample_size,
        acronym_print_all=acronym_print_all,
        run_graph_loader=run_graph_loader,
        run_entity_extraction=run_entity_extraction,
        run_embeddings=run_embeddings,
        run_entity_disambiguation=run_entity_disambiguation,
        run_entity_normalization=run_entity_normalization,
        run_sanity_checks=run_sanity_checks,
        graph_loader_batch_size=graph_loader_batch_size,
        graph_loader_min_text_chars_to_embed=graph_loader_min_text_chars_to_embed,
        graph_loader_replace_existing_document=graph_loader_replace_existing_document,
        entity_use_section_text=entity_use_section_text,
        entity_max_sections=entity_max_sections,
        entity_max_sections_per_batch=entity_max_sections_per_batch,
        entity_max_batch_chars=entity_max_batch_chars,
        entity_emergency_max_single_chars=entity_emergency_max_single_chars,
        entity_skip_processed=entity_skip_processed,
        entity_replace_section_mentions=entity_replace_section_mentions,
        entity_use_acronym_validation=entity_use_acronym_validation,
        entity_acronym_dir=entity_acronym_dir,
        entity_export_review=entity_export_review,
        entity_review_output_dir=entity_review_output_dir,
        entity_clear_previous_review=entity_clear_previous_review,
        entity_include_source_preview_in_review=entity_include_source_preview_in_review,
        embedding_max_sections=embedding_max_sections,
        embedding_batch_size=embedding_batch_size,
        embedding_force_reembed=embedding_force_reembed,
        embedding_include_title=embedding_include_title,
        embedding_include_body=embedding_include_body,
        embedding_max_chars_per_section=embedding_max_chars_per_section,
        embedding_allow_title_only=embedding_allow_title_only,
        clear_chat_cache_before_embeddings=clear_chat_cache_before_embeddings,
        disambiguation_delete_orphans=disambiguation_delete_orphans,
        entity_normalization_doc_id=entity_normalization_doc_id,
        entity_normalization_backend=entity_normalization_backend,
        entity_normalization_model_name=entity_normalization_model_name,
        entity_normalization_linker_name=entity_normalization_linker_name,
        entity_normalization_threshold=entity_normalization_threshold,
        entity_normalization_max_candidates=entity_normalization_max_candidates,
        entity_normalization_use_acronyms=entity_normalization_use_acronyms,
        entity_normalization_acronym_dir=entity_normalization_acronym_dir,
        entity_normalization_force=entity_normalization_force,
        entity_normalization_dry_run=entity_normalization_dry_run,
        entity_normalization_export_review=entity_normalization_export_review,
        entity_normalization_review_output_dir=entity_normalization_review_output_dir,
        entity_normalization_fuzzy_threshold=entity_normalization_fuzzy_threshold,
        entity_normalization_api_cache_dir=entity_normalization_api_cache_dir,
        entity_normalization_api_timeout=entity_normalization_api_timeout,
        entity_normalization_api_rate_limit_per_second=(
            entity_normalization_api_rate_limit_per_second
        ),
        entity_normalization_local_files_only=entity_normalization_local_files_only,
        entity_normalization_min_available_memory_gb=(
            entity_normalization_min_available_memory_gb
        ),
        sanity_mode=sanity_mode,
        quality_max_chunk_chars=quality_max_chunk_chars,
    )

    if clear_neo4j_before_run:
        clear_graph_data()

    summary = run_graph_pipeline(config)

    logger.info("Documents processed: %d", summary["documents_processed"])

    if summary.get("disambiguation_stats") is not None:
        logger.info("Disambiguation stats: %s", summary["disambiguation_stats"])

    if summary.get("normalization_stats") is not None:
        logger.info("UMLS normalization stats: %s", summary["normalization_stats"])

    if summary.get("sanity_summary") is not None:
        logger.info(
            "Sanity summary | mode=%s | issue_checks=%d | error_checks=%d | warning_checks=%d | info_checks=%d",
            summary["sanity_summary"].get("mode"),
            summary["sanity_summary"]["checks_with_issues"],
            summary["sanity_summary"]["error_checks_with_issues"],
            summary["sanity_summary"]["warning_checks_with_issues"],
            summary["sanity_summary"]["info_checks_with_issues"],
        )

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )

    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    loaded = load_dotenv(env_path)

    app_config, config_path, app_id = load_app_config_from_env()

    kg_config = app_config.get("knowledge_graph", {})
    kg_logging_config = kg_config.get("logging", {})

    pdf_dir = _resolve_project_path(
        _get_optional_env("KG_PDF_DIR")
        or _get_config_value(kg_config, "pipeline", "pdf_dir"),
        project_root / "test_data" / "pdfdocs",
        project_root,
    )
    work_root = _resolve_project_path(
        _get_optional_env("KG_WORK_ROOT")
        or _get_config_value(kg_config, "pipeline", "work_root"),
        project_root / "test_data" / "graph_cache_test",
        project_root,
    )

    PIPELINE_PHASE = (
        _get_optional_env("KG_PIPELINE_PHASE")
        or _get_config_value(
            kg_config,
            "pipeline",
            "phase",
            default="preprocess",
        )
    ).strip().lower()  # preprocess, graph, entities, embeddings, normalization, full

    SANITY_MODE = _resolve_sanity_mode_from_phase(PIPELINE_PHASE)
    KG_LOG_TO_FILE = _get_env_or_config_bool(
        "KG_LOG_TO_FILE",
        _get_config_value(kg_logging_config, "enabled"),
        True,
    )
    kg_log_dir_value = (
        _get_optional_env("KG_LOG_DIR")
        or _get_config_value(kg_logging_config, "dir")
    )
    KG_LOG_DIR = (
        _resolve_project_path(
            kg_log_dir_value,
            work_root / "logs",
            project_root,
        )
        if kg_log_dir_value not in (None, "")
        else None
    )

    configure_run_logging(
        work_root=work_root,
        phase=PIPELINE_PHASE,
        project_root=project_root,
        log_to_file=KG_LOG_TO_FILE,
        configured_log_dir=KG_LOG_DIR,
    )

    logger.info("Loading .env from: %s", env_path)
    logger.info(".env loaded: %s", loaded)
    logger.info("Using config path: %s", config_path)
    logger.info("Using app id: %s", app_id)

    # Runtime model settings come from config.json by default. .env can still
    # override them for local/emergency runs without changing tracked config.
    KG_CHAT_PROVIDER = (
        _get_optional_env("KG_CHAT_PROVIDER")
        or _get_optional_env("KG_MODEL_PROVIDER")
        or _get_config_value(kg_config, "providers", "chat_provider")
        or "local_hf"
    )
    KG_EMBEDDING_PROVIDER = (
        _get_optional_env("KG_EMBEDDING_PROVIDER")
        or _get_optional_env("KG_MODEL_PROVIDER")
        or _get_config_value(kg_config, "providers", "embedding_provider")
        or "local_hf"
    )
    KG_CHAT_MODEL = _get_env_or_config_str(
        "KG_CHAT_MODEL",
        _get_config_value(kg_config, "models", "chat_model"),
    ) or (
        "gpt-4.1-mini"
        if KG_CHAT_PROVIDER.strip().lower().replace("-", "_") == "openai"
        else "Qwen/Qwen2.5-14B-Instruct"
    )
    KG_CHAT_MODEL_PATH = _get_env_or_config_str(
        "KG_CHAT_MODEL_PATH",
        _get_config_value(kg_config, "models", "chat_model_path"),
        "",
    )
    KG_EMBEDDING_MODEL = _get_env_or_config_str(
        "KG_EMBEDDING_MODEL",
        _get_config_value(kg_config, "models", "embedding_model"),
    ) or (
        "text-embedding-3-small"
        if KG_EMBEDDING_PROVIDER.strip().lower().replace("-", "_") == "openai"
        else "Qwen/Qwen3-Embedding-8B"
    )
    KG_EMBEDDING_MODEL_PATH = _get_env_or_config_str(
        "KG_EMBEDDING_MODEL_PATH",
        _get_config_value(kg_config, "models", "embedding_model_path"),
        "",
    )
    KG_LOCAL_FILES_ONLY = _get_env_or_config_bool(
        "KG_LOCAL_FILES_ONLY",
        _get_config_value(kg_config, "providers", "local_files_only"),
        default=(
            KG_CHAT_PROVIDER.strip().lower().replace("-", "_") != "openai"
            or KG_EMBEDDING_PROVIDER.strip().lower().replace("-", "_") != "openai"
        ),
    )
    KG_CHAT_MAX_NEW_TOKENS = _get_env_or_config_int(
        "KG_CHAT_MAX_NEW_TOKENS",
        _get_config_value(kg_config, "models", "chat_max_new_tokens"),
        2048,
    )

    runtime_kwargs = {
        "kg_chat_provider": KG_CHAT_PROVIDER,
        "kg_embedding_provider": KG_EMBEDDING_PROVIDER,
        "kg_chat_model": KG_CHAT_MODEL,
        "kg_chat_model_path": KG_CHAT_MODEL_PATH,
        "kg_embedding_model": KG_EMBEDDING_MODEL,
        "kg_embedding_model_path": KG_EMBEDDING_MODEL_PATH,
        "kg_local_files_only": KG_LOCAL_FILES_ONLY,
        "kg_chat_max_new_tokens": KG_CHAT_MAX_NEW_TOKENS,
    }

    limit_kwargs = {
        "graph_loader_batch_size": _get_env_or_config_int(
            "KG_GRAPH_LOADER_BATCH_SIZE",
            _get_config_value(kg_config, "graph_loader", "batch_size"),
            200,
        ),
        "graph_loader_min_text_chars_to_embed": _get_env_or_config_int(
            "MIN_TEXT_CHARS_TO_EMBED",
            _get_config_value(kg_config, "graph_loader", "min_text_chars_to_embed"),
            20,
        ),
        "entity_max_sections": _get_env_or_config_optional_int(
            "KG_ENTITY_MAX_SECTIONS",
            _get_config_value(kg_config, "entities", "max_sections"),
        ),
        "entity_max_sections_per_batch": _get_env_or_config_int(
            "KG_ENTITY_MAX_SECTIONS_PER_BATCH",
            _get_config_value(kg_config, "entities", "max_sections_per_batch"),
            1,
        ),
        "entity_max_batch_chars": _get_env_or_config_int(
            "KG_ENTITY_MAX_BATCH_CHARS",
            _get_config_value(kg_config, "entities", "max_batch_chars"),
            30000,
        ),
        "entity_emergency_max_single_chars": _get_env_or_config_int(
            "KG_ENTITY_EMERGENCY_MAX_SINGLE_CHARS",
            _get_config_value(kg_config, "entities", "emergency_max_single_chars"),
            12000,
        ),
        "entity_skip_processed": _get_env_or_config_bool(
            "KG_ENTITY_SKIP_PROCESSED",
            _get_config_value(kg_config, "entities", "skip_processed"),
            True,
        ),
        "embedding_max_sections": _get_env_or_config_optional_int(
            "KG_EMBEDDING_MAX_SECTIONS",
            _get_config_value(kg_config, "section_embeddings", "max_sections"),
        ),
        "embedding_batch_size": _get_env_or_config_int(
            "KG_EMBEDDING_BATCH_SIZE",
            _get_config_value(kg_config, "section_embeddings", "batch_size"),
            8,
        ),
        "embedding_force_reembed": _get_env_or_config_bool(
            "KG_EMBEDDING_FORCE_REEMBED",
            _get_config_value(kg_config, "section_embeddings", "force_reembed"),
            False,
        ),
        "embedding_include_title": _get_env_or_config_bool(
            "KG_EMBEDDING_INCLUDE_TITLE",
            _get_config_value(kg_config, "section_embeddings", "include_title"),
            True,
        ),
        "embedding_include_body": _get_env_or_config_bool(
            "KG_EMBEDDING_INCLUDE_BODY",
            _get_config_value(kg_config, "section_embeddings", "include_body"),
            True,
        ),
        "embedding_max_chars_per_section": _get_env_or_config_int(
            "KG_EMBEDDING_MAX_CHARS_PER_SECTION",
            _get_config_value(
                kg_config,
                "section_embeddings",
                "max_chars_per_section",
            ),
            8000,
        ),
        "quality_max_chunk_chars": _get_env_or_config_int(
            "KG_QUALITY_MAX_CHUNK_CHARS",
            _get_config_value(kg_config, "quality", "max_chunk_chars"),
            50000,
        ),
    }
    cache_kwargs = {
        "force_toc": _get_env_or_config_bool(
            "KG_FORCE_TOC",
            _get_config_value(kg_config, "pipeline", "force_toc"),
            False,
        ),
        "force_markdown": _get_env_or_config_bool(
            "KG_FORCE_MARKDOWN",
            _get_config_value(kg_config, "pipeline", "force_markdown"),
            False,
        ),
        "force_anchors": _get_env_or_config_bool(
            "KG_FORCE_ANCHORS",
            _get_config_value(kg_config, "pipeline", "force_anchors"),
            False,
        ),
        "force_chunks": _get_env_or_config_bool(
            "KG_FORCE_CHUNKS",
            _get_config_value(kg_config, "pipeline", "force_chunks"),
            False,
        ),
    }

    normalization_acronym_dir_value = (
        _get_optional_env("KG_ENTITY_NORMALIZATION_ACRONYM_DIR")
        or _get_config_value(kg_config, "entity_normalization", "acronym_dir")
    )
    normalization_review_dir_value = (
        _get_optional_env("KG_ENTITY_NORMALIZATION_REVIEW_OUTPUT_DIR")
        or _get_config_value(kg_config, "entity_normalization", "review_output_dir")
    )
    normalization_api_cache_dir_value = (
        _get_optional_env("KG_ENTITY_NORMALIZATION_API_CACHE_DIR")
        or _get_config_value(kg_config, "entity_normalization", "api_cache_dir")
    )

    normalization_kwargs = {
        "entity_normalization_doc_id": _get_env_or_config_str(
            "KG_ENTITY_NORMALIZATION_DOC_ID",
            _get_config_value(kg_config, "entity_normalization", "doc_id"),
        ),
        "entity_normalization_backend": _get_env_or_config_str(
            "KG_ENTITY_NORMALIZATION_BACKEND",
            _get_config_value(kg_config, "entity_normalization", "backend"),
            "umls_api",
        ) or "umls_api",
        "entity_normalization_model_name": _get_env_or_config_str(
            "KG_ENTITY_NORMALIZATION_MODEL_NAME",
            _get_config_value(kg_config, "entity_normalization", "model_name"),
            "en_core_sci_sm",
        ) or "en_core_sci_sm",
        "entity_normalization_linker_name": _get_env_or_config_str(
            "KG_ENTITY_NORMALIZATION_LINKER_NAME",
            _get_config_value(kg_config, "entity_normalization", "linker_name"),
            "umls",
        ) or "umls",
        "entity_normalization_threshold": _get_env_or_config_float(
            "KG_ENTITY_NORMALIZATION_THRESHOLD",
            _get_config_value(kg_config, "entity_normalization", "threshold"),
            0.85,
        ),
        "entity_normalization_max_candidates": _get_env_or_config_int(
            "KG_ENTITY_NORMALIZATION_MAX_CANDIDATES",
            _get_config_value(kg_config, "entity_normalization", "max_candidates"),
            3,
        ),
        "entity_normalization_use_acronyms": _get_env_or_config_bool(
            "KG_ENTITY_NORMALIZATION_USE_ACRONYMS",
            _get_config_value(kg_config, "entity_normalization", "use_acronyms"),
            True,
        ),
        "entity_normalization_acronym_dir": (
            _resolve_project_path(
                normalization_acronym_dir_value,
                work_root / "acronyms",
                project_root,
            )
            if normalization_acronym_dir_value not in (None, "")
            else None
        ),
        "entity_normalization_force": _get_env_or_config_bool(
            "KG_ENTITY_NORMALIZATION_FORCE",
            _get_config_value(kg_config, "entity_normalization", "force"),
            False,
        ),
        "entity_normalization_dry_run": _get_env_or_config_bool(
            "KG_ENTITY_NORMALIZATION_DRY_RUN",
            _get_config_value(kg_config, "entity_normalization", "dry_run"),
            False,
        ),
        "entity_normalization_export_review": _get_env_or_config_bool(
            "KG_ENTITY_NORMALIZATION_EXPORT_REVIEW",
            _get_config_value(kg_config, "entity_normalization", "export_review"),
            True,
        ),
        "entity_normalization_review_output_dir": (
            _resolve_project_path(
                normalization_review_dir_value,
                work_root / "entity_review",
                project_root,
            )
            if normalization_review_dir_value not in (None, "")
            else None
        ),
        "entity_normalization_fuzzy_threshold": _get_env_or_config_int(
            "KG_ENTITY_NORMALIZATION_FUZZY_THRESHOLD",
            _get_config_value(kg_config, "entity_normalization", "fuzzy_threshold"),
            90,
        ),
        "entity_normalization_api_cache_dir": (
            _resolve_project_path(
                normalization_api_cache_dir_value,
                work_root / "umls_api_cache",
                project_root,
            )
            if normalization_api_cache_dir_value not in (None, "")
            else None
        ),
        "entity_normalization_api_timeout": _get_env_or_config_float(
            "KG_ENTITY_NORMALIZATION_API_TIMEOUT",
            _get_config_value(kg_config, "entity_normalization", "api_timeout"),
            30.0,
        ),
        "entity_normalization_api_rate_limit_per_second": _get_env_or_config_float(
            "KG_ENTITY_NORMALIZATION_API_RATE_LIMIT_PER_SECOND",
            _get_config_value(
                kg_config,
                "entity_normalization",
                "api_rate_limit_per_second",
            ),
            5.0,
        ),
        "entity_normalization_local_files_only": _get_env_or_config_bool(
            "KG_ENTITY_NORMALIZATION_LOCAL_FILES_ONLY",
            _get_config_value(kg_config, "entity_normalization", "local_files_only"),
            False,
        ),
        "entity_normalization_min_available_memory_gb": _get_env_or_config_float(
            "KG_ENTITY_NORMALIZATION_MIN_AVAILABLE_MEMORY_GB",
            _get_config_value(
                kg_config,
                "entity_normalization",
                "min_available_memory_gb",
            ),
            8.0,
        ),
    }

    RUN_ENTITY_NORMALIZATION = _get_env_or_config_bool(
        "KG_RUN_ENTITY_NORMALIZATION",
        _get_config_value(kg_config, "entity_normalization", "enabled"),
        False,
    )

    if PIPELINE_PHASE == "preprocess":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            pipeline_phase=PIPELINE_PHASE,
            clear_neo4j_before_run=False,
            run_preprocessing=True,
            run_acronym_extraction=True,
            force_acronyms=False,
            acronym_sample_size=0,
            acronym_print_all=False,
            run_graph_loader=False,
            run_entity_extraction=False,
            run_embeddings=False,
            run_entity_disambiguation=False,
            run_entity_normalization=False,
            run_sanity_checks=False,
            entity_use_section_text=True,
            entity_replace_section_mentions=True,
            entity_use_acronym_validation=True,
            entity_acronym_dir=None,
            entity_export_review=True,
            entity_review_output_dir=None,
            entity_clear_previous_review=True,
            entity_include_source_preview_in_review=False,
            embedding_allow_title_only=False,
            clear_chat_cache_before_embeddings=True,
            disambiguation_delete_orphans=True,
            sanity_mode=SANITY_MODE,
            log_to_file=KG_LOG_TO_FILE,
            log_dir=KG_LOG_DIR,
            **runtime_kwargs,
            **limit_kwargs,
            **cache_kwargs,
            **normalization_kwargs,
        )

    elif PIPELINE_PHASE == "graph":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            pipeline_phase=PIPELINE_PHASE,
            clear_neo4j_before_run=True,
            run_preprocessing=False,
            run_acronym_extraction=False,
            run_graph_loader=True,
            run_entity_extraction=False,
            run_embeddings=False,
            run_entity_disambiguation=False,
            run_entity_normalization=False,
            run_sanity_checks=True,
            graph_loader_replace_existing_document=True,
            entity_use_section_text=True,
            entity_replace_section_mentions=True,
            entity_use_acronym_validation=True,
            entity_acronym_dir=None,
            entity_export_review=True,
            entity_review_output_dir=None,
            entity_clear_previous_review=True,
            entity_include_source_preview_in_review=False,
            embedding_allow_title_only=False,
            clear_chat_cache_before_embeddings=True,
            disambiguation_delete_orphans=True,
            sanity_mode=SANITY_MODE,
            log_to_file=KG_LOG_TO_FILE,
            log_dir=KG_LOG_DIR,
            **runtime_kwargs,
            **limit_kwargs,
            **cache_kwargs,
            **normalization_kwargs,
        )

    elif PIPELINE_PHASE == "entities":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            pipeline_phase=PIPELINE_PHASE,
            clear_neo4j_before_run=False,
            run_preprocessing=False,
            run_acronym_extraction=False,
            run_graph_loader=False,
            run_entity_extraction=True,
            run_embeddings=False,
            run_entity_disambiguation=True,
            run_entity_normalization=RUN_ENTITY_NORMALIZATION,
            run_sanity_checks=True,
            entity_use_section_text=True,
            entity_replace_section_mentions=True,
            entity_use_acronym_validation=True,
            entity_acronym_dir=None,
            entity_export_review=True,
            entity_review_output_dir=None,
            entity_clear_previous_review=True,
            entity_include_source_preview_in_review=False,
            embedding_allow_title_only=False,
            clear_chat_cache_before_embeddings=True,
            disambiguation_delete_orphans=True,
            sanity_mode=SANITY_MODE,
            log_to_file=KG_LOG_TO_FILE,
            log_dir=KG_LOG_DIR,
            **runtime_kwargs,
            **limit_kwargs,
            **cache_kwargs,
            **normalization_kwargs,
        )

    elif PIPELINE_PHASE == "embeddings":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            pipeline_phase=PIPELINE_PHASE,
            clear_neo4j_before_run=False,
            run_preprocessing=False,
            run_acronym_extraction=False,
            run_graph_loader=False,
            run_entity_extraction=False,
            run_embeddings=True,
            run_entity_disambiguation=False,
            run_entity_normalization=False,
            run_sanity_checks=True,
            entity_use_section_text=True,
            entity_replace_section_mentions=True,
            entity_use_acronym_validation=True,
            entity_acronym_dir=None,
            entity_export_review=True,
            entity_review_output_dir=None,
            entity_clear_previous_review=True,
            entity_include_source_preview_in_review=False,
            embedding_allow_title_only=False,
            clear_chat_cache_before_embeddings=True,
            disambiguation_delete_orphans=True,
            sanity_mode=SANITY_MODE,
            log_to_file=KG_LOG_TO_FILE,
            log_dir=KG_LOG_DIR,
            **runtime_kwargs,
            **limit_kwargs,
            **cache_kwargs,
            **normalization_kwargs,
        )

    elif PIPELINE_PHASE == "normalization":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            pipeline_phase=PIPELINE_PHASE,
            clear_neo4j_before_run=False,
            run_preprocessing=False,
            run_acronym_extraction=False,
            run_graph_loader=False,
            run_entity_extraction=False,
            run_embeddings=False,
            run_entity_disambiguation=False,
            run_entity_normalization=True,
            run_sanity_checks=True,
            entity_use_section_text=True,
            entity_replace_section_mentions=True,
            entity_use_acronym_validation=True,
            entity_acronym_dir=None,
            entity_export_review=True,
            entity_review_output_dir=None,
            entity_clear_previous_review=True,
            entity_include_source_preview_in_review=False,
            embedding_allow_title_only=False,
            clear_chat_cache_before_embeddings=True,
            disambiguation_delete_orphans=True,
            sanity_mode=SANITY_MODE,
            log_to_file=KG_LOG_TO_FILE,
            log_dir=KG_LOG_DIR,
            **runtime_kwargs,
            **limit_kwargs,
            **cache_kwargs,
            **normalization_kwargs,
        )

    elif PIPELINE_PHASE == "full":
        main(
            pdf_dir=pdf_dir,
            work_root=work_root,
            pipeline_phase=PIPELINE_PHASE,
            clear_neo4j_before_run=True,
            run_preprocessing=True,
            run_acronym_extraction=True,
            force_acronyms=False,
            acronym_sample_size=0,
            acronym_print_all=False,
            run_graph_loader=True,
            run_entity_extraction=True,
            run_embeddings=True,
            run_entity_disambiguation=True,
            run_entity_normalization=RUN_ENTITY_NORMALIZATION,
            run_sanity_checks=True,
            graph_loader_replace_existing_document=True,
            entity_use_section_text=True,
            entity_replace_section_mentions=True,
            entity_use_acronym_validation=True,
            entity_acronym_dir=None,
            entity_export_review=True,
            entity_review_output_dir=None,
            entity_clear_previous_review=True,
            entity_include_source_preview_in_review=False,
            embedding_allow_title_only=False,
            clear_chat_cache_before_embeddings=True,
            disambiguation_delete_orphans=True,
            sanity_mode=SANITY_MODE,
            log_to_file=KG_LOG_TO_FILE,
            log_dir=KG_LOG_DIR,
            **runtime_kwargs,
            **limit_kwargs,
            **cache_kwargs,
            **normalization_kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported PIPELINE_PHASE='{PIPELINE_PHASE}'. "
            "Use one of: 'preprocess', 'graph', 'entities', 'embeddings', "
            "'normalization', 'full'."
        )
