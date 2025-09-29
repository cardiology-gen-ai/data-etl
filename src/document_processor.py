import pathlib
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.managers.markdown_conversion_manager import MarkdownConverter, DocumentMetadata
from src.managers.chunking_manager import ChunkingManager
from src.managers.index_manager import IndexManager


SUPPORTED_EXTENSION = ["pdf", "xps", "epub", "mobi", "fb2", "cbz", "svg", "txt"]


class DocumentProcessor:
    """Coordinate Markdown conversion, chunking, and indexing for a document.

    Parameters
    ----------
    filename : str
        Name of the input file to process. Used together with the converter’s configuration to resolve full paths.
    markdown_converter : :class:`~src.managers.markdown_conversion_manager.MarkdownConverter`
        Component that performs PDF→Markdown conversion and image placement; returns a :class:`~src.managers.markdown_conversion_manager.DocumentMetadata`.
    chunking_manager : :class:`~src.managers.chunking_manager.ChunkingManager`
        Component that turns a Markdown file into a list of :langchain_core:`Document <documents/langchain_core.documents.base.Document.html>` chunks with metadata.
    index_manager : :class:`~src.managers.index_manager.IndexManager`
        Manager that adds/removes chunks to/from the vector store
    filepath : pathlib.Path, optional
        Explicit path to the input file. If omitted, it is inferred by the converter.
    md_filepath : pathlib.Path, optional
        Explicit path to the Markdown output. If omitted, it is filled after conversion.
    """
    filename: str #: str : Filename.
    file_extension: Optional[str] #: str :  File extension detected by :meth:`~src.document_processor.DocumentProcessor.detect_file_extension` (without the dot).
    filepath: Optional[pathlib.Path] #: pathlib.Path : Resolved input file path.
    md_filepath: Optional[pathlib.Path] #: pathlib.Path : Resolved Markdown path.
    markdown_converter: MarkdownConverter #: MarkdownConverter : Markdown converter.
    chunking_manager: ChunkingManager #: ChunkingManager : Chunking strategy.
    index_manager: IndexManager #: IndexManager : Indexing strategy.

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
        """Detect and validate the file extension against ``SUPPORTED_EXTENSION``.

        Returns
        -------
        bool
            ``True`` if the extension is supported, ``False`` otherwise.
        """
        self.file_extension =  self.filename.split(".")[-1]
        return self.file_extension in SUPPORTED_EXTENSION

    def convert_document(self) -> Tuple[bool, DocumentMetadata]:
        """Convert the input document to Markdown and update local paths.

        Invokes the :class:`~src.managers.markdown_conversion_manager.MarkdownConverter` (callable) and, on success, fills :py:attr:`~src.document_processor.DocumentProcessor.filepath`, :py:attr:`~src.document_processor.DocumentProcessor.md_filepath`, and
        :py:attr:`~src.document_processor.DocumentProcessor.file_extension` (if it was not previously detected).

        Returns
        -------
        Tuple[bool, :class:`~src.managers.markdown_conversion_manager.DocumentMetadata`]
            ``(success, metadata)``. On success, ``metadata.file_extension`` is set to the
            detected extension.
        """
        conversion_status, doc_metadata = self.markdown_converter(self.filename)
        self.filepath = pathlib.Path(doc_metadata.filepath) if self.filepath is None else self.filepath
        self.md_filepath = pathlib.Path(doc_metadata.md_filepath) if self.md_filepath is None else self.md_filepath
        if self.file_extension is None:
            _ = self.detect_file_extension()
        doc_metadata.file_extension = self.file_extension
        return conversion_status, doc_metadata

    def chunk_document(self) -> List[Document]:
        """Split the produced Markdown file into :langchain_core:`Document <documents/langchain_core.documents.base.Document.html>`` chunks.

        Returns
        -------
        List[:langchain_core:`Document <documents/langchain_core.documents.base.Document.html>`]
            List of chunked documents produced by :class:`~src.managers.chunking_manager.ChunkingManager`

        Raises
        ------
        AssertionError
            If :py:attr:`~src.document_processor.DocumentProcessor.md_filepath` is not set.
        """
        assert self.md_filepath is not None
        return self.chunking_manager(self.md_filepath)

    def add_document_to_vectorstore(self) -> int:
        """Chunk the document and add all chunks to the vector store.

        Returns
        -------
        int
            Number of chunks added.
        """
        document_chunks = self.chunk_document()
        self.index_manager.add_document(document_chunks)
        return len(document_chunks)

    def delete_document_from_vectorstore(self) -> int:
        """Remove all chunks for this document from the vector store.

        Returns
        -------
        int
            Number of removed chunks/documents as reported by the backend.

        Raises
        ------
        AssertionError
            If :py:attr:`~src.document_processor.DocumentProcessor.md_filepath` is not set.
        """
        assert self.md_filepath is not None
        return self.index_manager.delete_document(self.md_filepath)

    def process_document(self) -> DocumentMetadata:
        """Run the full pipeline: convert → chunk → index.

        On successful conversion, the chunks are added to the vector store and
        ``n_chunks`` is populated in the returned :class:`~src.managers.markdown_conversion_manager.DocumentMetadata`.

        Returns
        -------
        DocumentMetadata
            Metadata about the processed document; ``n_chunks`` is set on success.
        """
        successful_conversion, doc_metadata = self.convert_document()
        if successful_conversion:
            n_doc_chunks = self.add_document_to_vectorstore()
            doc_metadata.n_chunks = n_doc_chunks
        return doc_metadata
