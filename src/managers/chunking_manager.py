import pathlib
from enum import Enum
from typing import Optional, Any, List

from langchain_core.documents import Document
from pydantic import BaseModel

from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings  # TODO: if needed, replace with custom model

from src.utils.singleton import Singleton


class TextSplitterName(Enum):
    markdown_splitter = "markdown"
    recursive_splitter = "recursive"
    semantic_splitter = "semantic"
    sentence_splitter = "sentence"


class TextSplitterConfig(BaseModel):
    name: TextSplitterName
    splitter: TextSplitter | SemanticChunker | MarkdownHeaderTextSplitter = None
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embeddings: Optional[Embeddings] = None  # TODO: make an initializer [possibly called also by config manager]
    header_levels: int = 2
    sentence_transformer_name: Optional[str] = None

    def model_post_init(self,  __context: Any) -> None:
        if self.name == TextSplitterName.markdown_splitter:
            headers_to_split_on = [("".join(["#"]*level), "Header " + str(level))
                                   for level in range(1, self.header_levels+1)]
            self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        elif self.name == TextSplitterName.recursive_splitter:
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        elif self.name == TextSplitterName.semantic_splitter:
            self.splitter = SemanticChunker(embeddings=self.embeddings, min_chunk_size=self.chunk_size)
        elif self.name == TextSplitterName.sentence_splitter:
            self.splitter = SentenceTransformersTokenTextSplitter(
                model_name=self.sentence_transformer_name,chunk_overlap=self.chunk_overlap, tokenizer=self.chunk_size)
        else:
            ValueError(f"{self.name.value} is not a valid splitter, valid splitters are "
                       f"{[n.value for n in TextSplitterName]}")

    class Config:
        arbitrary_types_allowed = True


class ChunkingManager(metaclass=Singleton):
    def __init__(self, splitter_list: List[TextSplitterConfig]):
        self.splitter_list = splitter_list

    def __call__(self, filepath: pathlib.Path) -> List[Document]:
        return self.split_text(filepath)

    def split_text(self, filepath: pathlib.Path) -> List[Document]:
        file_text = filepath.read_text(encoding="utf-8")
        doc_chunks = []
        for splitter_idx, splitter in enumerate(self.splitter_list):
            if splitter_idx == 0:
                doc_chunks = splitter.splitter.split_text(file_text)
            else:
                assert not isinstance(splitter.splitter, MarkdownHeaderTextSplitter)  # does not implement split_docs
                doc_chunks = splitter.splitter.split_documents(doc_chunks)
        for chunk_idx, chunk in enumerate(doc_chunks):
            chunk_headers = {}
            if isinstance(self.splitter_list[0].splitter, MarkdownHeaderTextSplitter):
                headers_metadata_keys = \
                    ["Header " + str(level) for level in range(1, self.splitter_list[0].header_levels+1)]
                chunk_headers = {k: v for k, v in chunk.metadata.items() if k in headers_metadata_keys}
            chunk_metadata_dict = {"filename": str(filepath), "chunk_idx": chunk_idx, "headers": chunk_headers,
                                   "n_tokens": 0}
            chunk.metadata = chunk_metadata_dict
        return doc_chunks