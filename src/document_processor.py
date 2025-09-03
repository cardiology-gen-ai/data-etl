import pathlib
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.managers.markdown_conversion_manager import MarkdownConverter, DocumentMetadata
from src.managers.chunking_manager import ChunkingManager
from src.managers.index_manager import IndexManager


SUPPORTED_EXTENSION = ["pdf", "xps", "epub", "mobi", "fb2", "cbz", "svg", "txt"]


class DocumentProcessor:
    def __init__(self, filename: str, markdown_converter: MarkdownConverter, chunking_manager: ChunkingManager,
                 index_manager: IndexManager, filepath: Optional[pathlib.Path] = None,
                 md_filepath: Optional[pathlib.Path] = None):
        self.filename = filename
        self.file_extension = None
        self.filepath = filepath
        self.md_filepath = md_filepath
        self.markdown_converter = markdown_converter
        self.chunking_manager = chunking_manager
        self.index_manager = index_manager

    def detect_file_extension(self) -> bool:
        self.file_extension =  self.filename.split(".")[-1]
        return self.file_extension in SUPPORTED_EXTENSION

    def convert_document(self) -> Tuple[bool, DocumentMetadata]:
        conversion_status, doc_metadata = self.markdown_converter(self.filename)
        self.filepath = pathlib.Path(doc_metadata.filepath) if self.filepath is None else self.filepath
        self.md_filepath = pathlib.Path(doc_metadata.md_filepath) if self.md_filepath is None else self.md_filepath
        if self.file_extension is None:
            _ = self.detect_file_extension()
        doc_metadata.file_extension = self.file_extension
        return conversion_status, doc_metadata

    def chunk_document(self) -> List[Document]:
        assert self.md_filepath is not None
        return self.chunking_manager(self.md_filepath)

    def add_document_to_vectorstore(self) -> int:
        document_chunks = self.chunk_document()
        self.index_manager.add_document(document_chunks)
        return len(document_chunks)

    def delete_document_from_vectorstore(self) -> int:
        assert self.md_filepath is not None
        return self.index_manager.delete_document(self.md_filepath)

    def process_document(self) -> DocumentMetadata:
        successful_conversion, doc_metadata = self.convert_document()
        if successful_conversion:
            n_doc_chunks = self.add_document_to_vectorstore()
            doc_metadata.n_chunks = n_doc_chunks
        return doc_metadata
