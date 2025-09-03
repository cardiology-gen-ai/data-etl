import pathlib
from enum import Enum
from typing import Any, List, Dict, Optional

from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from transformers import AutoTokenizer

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
    embeddings: Optional[HuggingFaceEmbeddings] = None
    header_levels: int = 2

    def model_post_init(self,  __context: Any) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.embeddings.model_name) if self.embeddings is not None else None
        chunk_size = self.chunk_size if tokenizer is None else min(self.chunk_size, self.embeddings._client.max_seq_length - 10)
        chunk_overlap = self.chunk_overlap if tokenizer is None else self.chunk_overlap * chunk_size / self.chunk_size
        if self.name == TextSplitterName.markdown_splitter:
            headers_to_split_on = [("".join(["#"]*level), "Header " + str(level))
                                   for level in range(1, self.header_levels+1)]
            self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        elif self.name == TextSplitterName.recursive_splitter:
            self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                 chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer)
        elif self.name == TextSplitterName.semantic_splitter:
            self.splitter = SemanticChunker(embeddings=self.embeddings, min_chunk_size=int(chunk_size/3))
        elif self.name == TextSplitterName.sentence_splitter:
            self.splitter = SentenceTransformersTokenTextSplitter.from_huggingface_tokenizer(
               tokenizer=tokenizer, chunk_overlap=chunk_overlap, tokens_per_chunk=chunk_size)

    class Config:
        arbitrary_types_allowed = True


class ChunkMetadata(BaseModel):
    filename: str
    chunk_idx: int
    headers: Dict[str, str]
    n_tokens: int


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
            n_tokens = 0
            first_splitter_with_embeddings = \
                [splitter for splitter in self.splitter_list if splitter.embeddings is not None]
            if len(first_splitter_with_embeddings) > 0:
                n_tokens = first_splitter_with_embeddings[0].splitter._length_function(chunk.page_content)
            chunk_metadata = ChunkMetadata(
                filename=str(filepath),
                chunk_idx=chunk_idx,
                headers=chunk_headers,
                n_tokens=n_tokens
            )
            chunk.metadata = chunk_metadata.model_dump()
        return doc_chunks