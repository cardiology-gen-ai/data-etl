import pathlib
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config.manager import PreprocessingConfig
from managers.chunking.chunking_manager import ChunkingManager, TextSplitterConfig, TextSplitterName, ChunkMetadata
from managers.chunking.hierarchical_chunking_manager import HierarchicalChunkingManager


class FlatChunkingManager(ChunkingManager):
    """Coordinates a chain of one or more splitters and emits annotated documents.

    The first splitter receives a raw string (file text). Subsequent splitters must implement ``split_documents`` and will refine the previously produced list of documents.

    .. rubric:: Call signature

    ``__call__(filepath)`` is an alias of :py:meth:`split_text`.

    Parameters
    ----------
    splitter_list : list[:class:`~src.managers.chunking_manager.TextSplitterConfig`]
        Ordered splitter pipeline. The first may be a Markdown splitter; the others must
        implement ``split_documents``.
    """
    splitter_list: List[TextSplitterConfig] #: list[:class:`~src.managers.chunking_manager.TextSplitterConfig`] : Ordered list of text splitters.
    def __init__(self, config: PreprocessingConfig, app_id: str):
        self.config = config
        self.app_id = app_id
        self.splitter_list = self.config.chunking_manager.splitter

    def __call__(self, filepath: pathlib.Path) -> List[Document]:
        """Alias for :py:meth:`split_text` for ergonomic use.

        Parameters
        ----------
        filepath : pathlib.Path
            Path to the text file to split.

        Returns
        -------
        List[:langchain_core:`Document <documents/langchain_core.documents.base.Document.html>`]
            List of documents produced by the configured pipeline.
        """
        return self.split_text(filepath)

    def split_text(self, filepath: pathlib.Path) -> List[Document]:
        """Split a text file and attach :class:`ChunkMetadata` to each chunk.

        .. rubric:: Processing steps

        1. Read the file content.
        2. Apply the first splitter via ``split_text``.
        3. Iteratively refine the result with the remaining splitters via ``split_documents``.
        4. If the first splitter is Markdown, capture header metadata (e.g., ``Header 1``, ``Header 2``).
        5. If any splitter has embeddings, compute an estimated token length for each chunk.
        6. Replace ``chunk.metadata`` with a serialized :class:`ChunkMetadata`.

        Parameters
        ----------
        filepath : pathlib.Path
            Path to the text file to split.

        Returns
        -------
        List[:langchain_core:`Document <documents/langchain_core.documents.base.Document.html>`]
            A list of :langchain_core:`Document <documents/langchain_core.documents.base.Document.html>` items with enriched metadata.

        Raises
        ------
        AssertionError
            If a non-first splitter is a :class:`~langchain_text_splitters.markdown.MarkdownHeaderTextSplitter` (it lacks ``split_documents``).
        """
        file_text = filepath.read_text(encoding="utf-8")
        doc_chunks = []
        for splitter_idx, splitter in enumerate(self.splitter_list):
            if splitter_idx == 0:
                if splitter.name == TextSplitterName.hierarchical_splitter:
                    hierarchical_chunker = HierarchicalChunkingManager(
                        config=self.config, header_levels=splitter.header_levels, app_id=self.app_id
                    )
                    doc_chunks = hierarchical_chunker(filepath)
                else:
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
            if self.splitter_list[0].name == TextSplitterName.hierarchical_splitter:
                chunk_headers = chunk.metadata["headers"]
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
