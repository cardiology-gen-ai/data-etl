import os
import pathlib
from enum import Enum
from typing import Tuple, Dict, Any, Optional, List, Literal

from langchain.embeddings import Embeddings
from pydantic import BaseModel, Field

from managers.chunking.chunking_manager import TextSplitterConfig, TextSplitterName
from cardiology_gen_ai import IndexingConfig
from cardiology_gen_ai.config.manager import ConfigManager


class ImageManagerConfig(BaseModel):
    """Configure image extraction and placement for Markdown conversion."""
    dpi: int = 200 #: int, default ``200`` : Rendering DPI used when rasterize PDF regions into images.
    tol: float = 40.0 #: float, default ``40.0`` : Tolerance (in PDF points) used to merge touching/nearby image rectangles.
    pad: float = 16.0 #: float, default ``16.0`` : Padding (in points) added around detected image boxes before rendering.
    caption_keywords: Tuple[str] = ("Figure",) #: Tuple[str], default ``("Figure",)`` : Keywords that indicate lines likely to be **captions** (case-sensitive unless handled upstream).

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "ImageManagerConfig":
        """Create an :class:`~src.config.manager.ImageManagerConfig` from a plain dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Mapping with keys (``dpi``, ``tol``, ``pad``, ``caption_keywords``).

        Returns
        -------
        :class:`~src.config.manager.ImageManagerConfig`
            Config instance with ``caption_keywords`` converted to a tuple.
        """
        caption_keywords = tuple(config_dict["caption_keywords"])
        other_config_dict = {k: v for k,v in config_dict.items() if k != "caption_keywords"}
        return cls(caption_keywords=caption_keywords, **other_config_dict)


class FileStorageConfig(BaseModel):
    """Storage configuration for input/output folders.

    .. rubric:: Notes

    - After initialization, ``folder`` is set to ``Path(parent_folder) / child_folder``
    """
    parent_folder: str #: str : Base directory used to resolve child folders.
    child_folder: str #: str : Relative child directory (e.g., ``"pdf"`` or ``"markdown"``).
    folder: pathlib.Path = None #: :class:`pathlib.Path`, optional : Resolved path combining ``parent_folder / child_folder``; set automatically in :meth:`model_post_init`.
    allowed_extensions: Optional[List[str]] = None #:  list[str] | None, optional : Optional whitelist of allowed file extensions (e.g., ``[".pdf"]`` or ``[".md"]``).

    def model_post_init(self,  __context: Any) -> None:
        """Resolve the concrete :class:`pathlib.Path` for ``folder`` after init."""
        self.folder = pathlib.Path(os.getenv("DATA_ROOT")) / self.parent_folder / self.child_folder


class ChunkingStrategy(Enum):
    flat = "flat"
    hierarchical = "hierarchical"


class ChunkingManagerConfig(BaseModel):
    """Configure the text splitting pipeline used in preprocessing.

    Parameters
    ----------
    splitter : list[:class:`~src.managers.chunking_manager.TextSplitterConfig`]
        Ordered list of splitter configurations. It is typically built via :meth:`~src.config.manager.ChunkingManagerConfig.from_config`.
    """
    strategy: ChunkingStrategy
    splitter: List[TextSplitterConfig] = None

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any], embeddings: Optional[Embeddings] = None) -> "ChunkingManagerConfig":
        """Construct a :class:`ChunkingManagerConfig` from a plain dictionary.

        .. rubric:: Logic

        - If ``markdown_first`` is ``True`` or ``splitter == "markdown"``, a :class:`~src.managers.chunking_manager.TextSplitterConfig` with ``name=``:class:`~src.managers.chunking_manager.TextSplitterName`.markdown_splitter is prepended (``header_levels`` defaults to 2).
        - If ``splitter`` is provided and is not ``"markdown"`` (or no markdown was added yet), append a second splitter using that strategy and the given hyper‑parameters. ``embeddings`` is passed so token‑aware splitters can compute lengths properly.


        Parameters
        ----------
        config_dict : Dict[str, Any]
            Mapping controlling splitter composition and hyper-params.
        embeddings : :langchain:`HuggingFaceEmbeddings <huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html>`
            Embedding backend used by token-aware splitters.

        Returns
        -------
        :class:`~src.config.manager.ChunkingManagerConfig`
            Config with a populated ``splitter`` list.
        """
        chunking_strategy = ChunkingStrategy(config_dict["strategy"])
        splitter_list = []
        if chunking_strategy == ChunkingStrategy.flat:
            assert not (config_dict.get("markdown_first", False) and config_dict.get("toc_first", False))
            markdown_first = config_dict.get("markdown_first", False)
            splitter = config_dict.get("splitter", None)
            if markdown_first is True or splitter == "markdown":
                splitter_list.append(TextSplitterConfig(
                    name=TextSplitterName.markdown_splitter,
                    header_levels=config_dict.get("header_levels", 2),
                ))
            elif config_dict.get("toc_first", False):
                splitter_list.append(TextSplitterConfig(
                    name=TextSplitterName.hierarchical_splitter, header_levels=config_dict.get("header_levels", 2))
                )
            if splitter is not None and (len(splitter_list) == 0 or splitter != "markdown"):
                    other_config_dict = {k: v for k, v in config_dict.items() if k != "splitter"}
                    splitter_list.append(TextSplitterConfig(
                        name=TextSplitterName(splitter), embeddings=embeddings, **other_config_dict,
                    ))
        else:
            splitter_list.append(TextSplitterConfig(
                name=TextSplitterName.hierarchical_splitter, header_levels=config_dict.get("header_levels", 2))
            )
        return cls(strategy=chunking_strategy, splitter=splitter_list)


class ParsingConfig(BaseModel):
    backend: Literal["docling", "mineru"] = "docling"

    # MinerU execution mode
    mineru_runtime: Literal["local", "artifacts"] = "local"
    mineru_artifacts_root: Optional[str] = None

    # MinerU-only knobs
    mineru_force_ocr: bool = False
    mineru_language: str = "en"
    mineru_backend: Literal[
        "pipeline",
        "vlm-transformers",
        "vlm-sglang-engine",
        "vlm-sglang-client",
        "hybrid-auto-engine",
    ] = "pipeline"
    mineru_formula_enable: bool = True
    mineru_table_enable: bool = True
    mineru_server_url: Optional[str] = None

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "ParsingConfig":
        """Build a :class:`ParsingConfig` from a plain dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Mapping with ``backend`` and optionally any of the ``mineru_*`` keys.

        Returns
        -------
        :class:`~src.config.manager.ParsingConfig`
            Validated parsing configuration.
        """
        return cls(**config_dict)


class PreprocessingConfig(BaseModel):
    """Top‑level configuration for the preprocessing phase."""
    image_manager: ImageManagerConfig = ImageManagerConfig() #:  :class:`ImageManagerConfig`, default constructed : Parameters for image extraction and placement.
    input_folder: FileStorageConfig #: :class:`FileStorageConfig` : Where input PDFs are read from.
    output_folder: FileStorageConfig #: :class:`FileStorageConfig` : Where Markdown and extracted images are saved.
    cache_folder: FileStorageConfig  # TODO: this is for docling cache, check consistency
    anchors_folder: Optional[FileStorageConfig] = None
    tocs_folder: Optional[FileStorageConfig] = None
    chunks_folder: Optional[FileStorageConfig] = None
    attributor: Literal["lexical", "semantic"] = "semantic"
    chunking_manager: ChunkingManagerConfig #: :class:`ChunkingManagerConfig` : How to split text into chunks.
    parsing: ParsingConfig = Field(default_factory=ParsingConfig) #: :class:`ParsingConfig`, default constructed : PDF parsing backend selection (Docling vs MinerU) and backend-specific options.

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any], embeddings: Optional[Embeddings] = None) -> "PreprocessingConfig":
        """Build a :class:`PreprocessingConfig` from a nested mapping.

        .. rubric:: Expected schema

        ``config_dict`` should contain:

        - ``storage``: ``{"parent_folder", "input_folder", "output_folder", "allowed_extensions"}``
        - ``images``: options for :class:`ImageManagerConfig`
        - ``chunking``: options for :class:`ChunkingManagerConfig`
        - ``parsing`` (optional): options for :class:`ParsingConfig`. If omitted,
          the default ``ParsingConfig`` (Docling backend) is used.

        The resulting instance has ``input_folder.folder`` and ``output_folder.folder``
        resolved by :class:`FileStorageConfig`.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Preprocessing subsection of the app configuration.
        embeddings : :langchain:`HuggingFaceEmbeddings <huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html>`
            Embedding backend forwarded to token-aware splitters.

        Returns
        -------
        :class:`~src.config.manager.PreprocessingConfig`
            Validated preprocessing configuration.

        """
        preprocessing_storage_dict = config_dict["storage"]
        input_folder = FileStorageConfig(
            parent_folder=preprocessing_storage_dict["parent_folder"],
            child_folder=preprocessing_storage_dict["input_folder"],
            allowed_extensions=preprocessing_storage_dict["allowed_extensions"]
        )
        output_folder = FileStorageConfig(
            parent_folder=preprocessing_storage_dict["parent_folder"],
            child_folder=preprocessing_storage_dict["output_folder"],
            allowed_extensions=["md"]
        )
        cache_folder, anchors_folder, tocs_folder, chunks_folder = [None for _ in range(4)]
        if preprocessing_storage_dict.get("cache_folder") is not None:
            cache_folder = FileStorageConfig(
                parent_folder=output_folder.parent_folder + "/" + output_folder.child_folder,
                child_folder=preprocessing_storage_dict["cache_folder"],
                allowed_extensions=["json"]
            )
        if preprocessing_storage_dict.get("anchors_folder") is not None:
            anchors_folder = FileStorageConfig(
                parent_folder=output_folder.parent_folder + "/" + output_folder.child_folder,
                child_folder=preprocessing_storage_dict["anchors_folder"],
                allowed_extensions=["json"]
            )
        if preprocessing_storage_dict.get("tocs_folder") is not None:
            tocs_folder = FileStorageConfig(
                parent_folder=output_folder.parent_folder + "/" + output_folder.child_folder,
                child_folder=preprocessing_storage_dict["tocs_folder"],
                allowed_extensions=["json"]
            )
            if preprocessing_storage_dict.get("chunks_folder") is not None:
                chunks_folder = FileStorageConfig(
                    parent_folder=tocs_folder.parent_folder + "/" + tocs_folder.child_folder,
                    child_folder=preprocessing_storage_dict["chunks_folder"],
                    allowed_extensions=["pkl"]
                )
        image_manager_dict = config_dict["images"]
        image_manager = ImageManagerConfig.from_config(image_manager_dict)
        chunking_manager_dict = config_dict["chunking"]
        chunking_manager = ChunkingManagerConfig.from_config(chunking_manager_dict, embeddings=embeddings)
        parsing_dict = config_dict.get("parsing")
        parsing = ParsingConfig.from_config(parsing_dict) if parsing_dict else ParsingConfig()
        if config_dict["parsing"].get("backend") == "mineru":
            scratch_folder = FileStorageConfig(
                parent_folder=preprocessing_storage_dict["parent_folder"],
                child_folder=config_dict["parsing"].get("mineru_artifacts_root"),
                allowed_extensions=["md", "json", "jpg"]
            )
            parsing.mineru_artifacts_root = scratch_folder.folder.as_posix()
        other_keys_dict = {k: v for k, v in config_dict.items() if k not in
                           ["storage", "allowed_extensions", "parent_folder", "input_folder", "output_folder",
                            "cache_folder", "anchors_folder", "tocs_folder", "chunks_folder",
                            "images", "chunking", "parsing"]}
        return cls(input_folder=input_folder, output_folder=output_folder, image_manager=image_manager,
                   chunking_manager=chunking_manager, anchors_folder=anchors_folder, tocs_folder=tocs_folder,
                   cache_folder=cache_folder, chunks_folder=chunks_folder, parsing=parsing, **other_keys_dict)


class ETLConfig(BaseModel):
    """Aggregate configuration for the full ETL stack (i.e. preprocessing and indexing)."""
    preprocessing: PreprocessingConfig #: :class:`~src.config.manager.PreprocessingConfig` : Preprocessing phase configuration.
    indexing: IndexingConfig #: :class:`cardiology_gen_ai.models.IndexingConfig` : Indexing backend and persistence settings.
    # embeddings: Optional[EmbeddingConfig] = None #: :class:`cardiology_gen_ai.models.EmbeddingConfig` : Embedding model (callable + dimensionality); also passed to preprocessing.

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "ETLConfig":
        """Construct an :class:`ETLConfig` from a nested mapping.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Mapping with ``embeddings``, ``preprocessing``, and ``indexing`` sections.

        Returns
        -------
        :class:`~src.config.manager.ETLConfig`
            Aggregated configuration with nested sections validated.
        """
        # embedding_config = None
        # if config_dict.get("embeddings"):
        #    embedding_dict = config_dict["embeddings"]
        #    embedding_config = EmbeddingConfig.from_config(embedding_dict)
        preprocessing_dict = config_dict["preprocessing"]
        # preprocessing_config_embeddings = embedding_config.model if embedding_config else None
        preprocessing_config = PreprocessingConfig.from_config(preprocessing_dict)
        indexing_dict = config_dict["indexing"]
        indexing_config = IndexingConfig.from_config(indexing_dict)
        other_config_dict = \
            {k: v for k, v in config_dict.items() if k not in ["preprocessing", "indexing", "embeddings"]}
        return cls(preprocessing=preprocessing_config, indexing=indexing_config, **other_config_dict)


class ETLConfigManager(ConfigManager):
    """Helper that loads :class:`~src.config.manager.ETLConfig` using app‑level configuration paths.

    Parameters
    ----------
    config_path : str | None, optional
        Filesystem path to the current ETL configuration file. Defaults to the ``CONFIG_PATH`` environment variable.
    app_id : str, default ``"cardiology_protocols"``
        Application identifier used by the base :class:`~cardiology_gen_ai.config.manager.ConfigManager`.
    """
    config: ETLConfig #: :class:`~src.config.manager.ETLConfig` : Parsed configuration ready to be consumed by the application.
    def __init__(self,
                 config_path: str = os.getenv("CONFIG_PATH"),
                 app_id: str = "cardiology_protocols"):
        super().__init__(config_path, app_id)
        self.config = ETLConfig.from_config(self._config)