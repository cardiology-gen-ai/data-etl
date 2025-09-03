import os
import pathlib
import re
import json
from enum import Enum
from typing import Tuple, Dict, Any, Optional, List

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from transformers import AutoConfig

from src.managers.chunking_manager import TextSplitterConfig, TextSplitterName


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


class EmbeddingConfig(BaseModel):
    model_name: str
    model: HuggingFaceEmbeddings = None
    dim: int

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "EmbeddingConfig":
        kwargs = config_dict["kwargs"]
        if not isinstance(kwargs, dict):
            kwargs = dict()
        encode_kwargs = {k: v for k, v in kwargs.items() if k in ["normalize_embeddings", "prompt_name"]}
        model_kwargs = {k: v for k, v in kwargs.items() if k in ["device"]}
        if model_kwargs.get("device", None) is None:
            model_kwargs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = config_dict["deployment"]
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        dim = AutoConfig.from_pretrained(model_name).hidden_size
        return cls(model_name=model_name, model=model, dim=dim)


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


class IndexTypeNames(Enum):
    qdrant = "qdrant"
    faiss = "faiss"


class DistanceTypeNames(Enum):
    cosine = "cosine"
    euclidean = "euclidean"


class RetrievalTypeNames(Enum):
    dense = "dense"
    sparse = "sparse"
    hybrid = "hybrid"


class IndexingConfig(BaseModel):
    name: str
    description: str
    folder: pathlib.Path = pathlib.Path("index")
    type: IndexTypeNames
    distance: DistanceTypeNames
    retrieval_mode: RetrievalTypeNames = RetrievalTypeNames.dense

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "IndexingConfig":
        index_type = IndexTypeNames(config_dict["type"])
        distance = DistanceTypeNames(config_dict["distance"])
        retrieval = config_dict.get("retrieval_mode", RetrievalTypeNames.dense)
        retrieval_mode = RetrievalTypeNames(retrieval) if isinstance(retrieval, str) is not None else retrieval
        other_config_dict = {k: v for k, v in config_dict.items() if k not in ["type", "distance", "retrieval_mode"]}
        return cls(type=index_type, distance=distance, retrieval_mode=retrieval_mode, **other_config_dict)


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


class ETLConfigManager:
    def __init__(self, config_path: str = os.getenv("CONFIG_PATH"), app_id: str = "cardiology_protocols"):
        self._config_path = config_path
        self._app_id = app_id
        self._config = self._load_config()
        self._app_config = self._get_app_config()
        self.config = ETLConfig.from_config(self._app_config)

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self._config_path, "r") as config_file:
                raw_json = config_file.read()

                def replace_env_var(match):
                    var_name = match.group(1)
                    return os.environ.get(var_name, f"<MISSING:{var_name}>")

                interpolated_json = re.sub(r"\$\{(\w+)\}", replace_env_var, raw_json)
                return json.loads(interpolated_json)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self._config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in configuration file at {self._config_path}")

    def _get_app_config(self) -> Dict[str, Any]:
        app_config = self._config.get(self._app_id)
        if not app_config:
            raise ValueError(f"No configuration found for application: {self._app_id}")
        return app_config
