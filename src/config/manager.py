import os
import pathlib
from typing import Tuple, Dict, Any, Optional, List

from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel

from src.managers.chunking_manager import TextSplitterConfig, TextSplitterName
from cardiology_gen_ai import IndexingConfig, EmbeddingConfig
from cardiology_gen_ai.config.manager import ConfigManager


class ImageManagerConfig(BaseModel):
    dpi: int = 200
    tol: float = 40.0
    pad: float = 16.0
    caption_keywords: Tuple[str] = ("Figure",)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "ImageManagerConfig":
        caption_keywords = tuple(config_dict["caption_keywords"])
        other_config_dict = {k: v for k,v in config_dict.items() if k != "caption_keywords"}
        return cls(caption_keywords=caption_keywords, **other_config_dict)


class FileStorageConfig(BaseModel):
    parent_folder: str
    child_folder: str
    folder: pathlib.Path = None
    allowed_extensions: Optional[List[str]] = None

    def model_post_init(self,  __context: Any) -> None:
        self.folder = pathlib.Path(self.parent_folder) / self.child_folder


class ChunkingManagerConfig(BaseModel):
    splitter: List[TextSplitterConfig]

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any], embeddings: HuggingFaceEmbeddings) -> "ChunkingManagerConfig":
        markdown_first = config_dict.get("markdown_first", False)
        splitter_list = []
        splitter = config_dict.get("splitter", None)
        if markdown_first is True or splitter == "markdown":
            splitter_list.append(TextSplitterConfig(
                name=TextSplitterName.markdown_splitter,
                header_levels=config_dict.get("header_levels", 2),
            ))
        if splitter is not None and (len(splitter_list) == 0 or splitter != "markdown"):
                other_config_dict = {k: v for k, v in config_dict.items() if k != "splitter"}
                splitter_list.append(TextSplitterConfig(
                    name=TextSplitterName(splitter), embeddings=embeddings, **other_config_dict,
                ))
        return cls(splitter=splitter_list)


class PreprocessingConfig(BaseModel):
    image_manager: ImageManagerConfig = ImageManagerConfig()
    input_folder: FileStorageConfig
    output_folder: FileStorageConfig
    chunking_manager: ChunkingManagerConfig

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any], embeddings: HuggingFaceEmbeddings) -> "PreprocessingConfig":
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
        image_manager_dict = config_dict["images"]
        image_manager = ImageManagerConfig.from_config(image_manager_dict)
        chunking_manager_dict = config_dict["chunking"]
        chunking_manager = ChunkingManagerConfig.from_config(chunking_manager_dict, embeddings=embeddings)
        return cls(input_folder=input_folder, output_folder=output_folder, image_manager=image_manager,
                   chunking_manager=chunking_manager, **config_dict)


class ETLConfig(BaseModel):
    preprocessing: PreprocessingConfig
    indexing: IndexingConfig
    embeddings: EmbeddingConfig

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "ETLConfig":
        embedding_dict = config_dict["embeddings"]
        embedding_config = EmbeddingConfig.from_config(embedding_dict)
        preprocessing_dict = config_dict["preprocessing"]
        preprocessing_config = (
            PreprocessingConfig.from_config(preprocessing_dict, embeddings=embedding_config.model))
        indexing_dict = config_dict["indexing"]
        indexing_config = IndexingConfig.from_config(indexing_dict)
        other_config_dict = \
            {k: v for k, v in config_dict.items() if k not in ["preprocessing", "indexing", "embeddings"]}
        return cls(preprocessing=preprocessing_config, indexing=indexing_config, embeddings=embedding_config,
                   **other_config_dict)


class ETLConfigManager(ConfigManager):
    def __init__(self,
                 config_path: str = os.getenv("CONFIG_PATH"),
                 app_config_path: str = os.getenv("APP_CONFIG_PATH"),
                 app_id: str = "cardiology_protocols"):
        super().__init__(config_path, app_config_path, app_id)
        self.config = ETLConfig.from_config(self._app_config)
