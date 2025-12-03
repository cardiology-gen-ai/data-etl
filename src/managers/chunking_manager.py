import pathlib
from enum import Enum
from typing import Any, List, Dict, Optional

from pydantic import BaseModel
from langchain.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from transformers import AutoTokenizer

from cardiology_gen_ai.utils.singleton import Singleton


class TextSplitterName(Enum):
    """Supported text splitting strategies."""
    markdown_splitter = "markdown" #: Splits Markdown by headers using :class:`~langchain_text_splitters.markdown.MarkdownHeaderTextSplitter`.
    recursive_splitter = "recursive" #: Uses :class:`~langchain_text_splitters.character.RecursiveCharacterTextSplitter` (tokenizer-aware when embeddings are provided).
    semantic_splitter = "semantic" #: Uses :class:`~langchain_experimental.text_splitter.SemanticChunker` (embedding-based semantic boundaries).
    sentence_splitter = "sentence" #: Uses :class:`~langchain_text_splitters.sentence_transformers.SentenceTransformersTokenTextSplitter` (token count per chunk).


class TextSplitterConfig(BaseModel):
    """Configuration wrapper that builds the concrete text splitter on init.

    This model normalizes ``chunk_size`` and ``chunk_overlap`` based on the optional
    Hugging Face tokenizer derived from ``embeddings``. If an embedding model is provided,
    the effective chunk size is clamped to the tokenizer's max sequence length minus a small
    margin, and the overlap is rescaled proportionally.

    .. rubric:: Notes

    - If the ``markdown_splitter`` is present, it is automatically placed first in a pipeline, hence ``split_text`` is used.
    """
    name: TextSplitterName #: :class:`~src.managers.chunking_manager.TextSplitterName` : Strategy to build.
    splitter: TextSplitter | SemanticChunker | MarkdownHeaderTextSplitter = None #: object, optional : A prebuilt splitter. If omitted (which is the default behavior), it is created in :py:meth:`model_post_init` matching ``name``.
    chunk_size: int = 1000 #: int, default ``1000`` : Preferred chunk token/character budget. When a tokenizer is available it is interpreted as tokens; otherwise it is characters.
    chunk_overlap: int = 150 #: int, default ``150`` : Overlap between adjacent chunks (same unit as ``chunk_size``).
    embeddings: Optional[Embeddings] = None #: :langchain:`HuggingFaceEmbeddings <huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html>`, optional : Embeddings object. If provided, its underlying tokenizer and model metadata drive chunk sizing.
    header_levels: int = 2 #: int, default ``2`` : For Markdown splitting, number of header levels (e.g. ``#`` to ``###...``) to split on.

    def model_post_init(self,  __context: Any) -> None:
        """Finalize the splitter instance and normalize chunking hyper‑parameters.

        .. rubric:: Logic

        - If ``embeddings`` is provided, derive a Hugging Face tokenizer via :meth:`~transformers.AutoTokenizer.from_pretrained` using the embeddings' model name.
        - Clamp ``chunk_size`` to ``embeddings._client.max_seq_length``, keeping a small safety margin.
        - Rescale ``chunk_overlap`` proportionally to the adjusted ``chunk_size``.
        - Instantiate the concrete splitter implementation according to :class:`TextSplitterName`:

            * ``markdown_splitter`` → :class:`~langchain_text_splitters.markdown.MarkdownHeaderTextSplitter` with ``header_levels``.
            * ``recursive_splitter`` → :meth:`~langchain_text_splitters.character.RecursiveCharacterTextSplitter.from_huggingface_tokenizer`.
            * ``semantic_splitter`` → :class:`~langchain_experimental.text_splitter.SemanticChunker`.
            * ``sentence_splitter`` → :meth:`~langchain_text_splitters.sentence_transformers.SentenceTransformersTokenTextSplitter.from_huggingface_tokenizer`.

        .. rubric:: Notes

        Some splitter implementations expect integer values for overlaps; if you supply floats,
        internal casting may occur. Consider using integer values explicitly if your version requires it.
        """
        if self.name == TextSplitterName.markdown_splitter:
            headers_to_split_on = [("".join(["#"]*level), "Header " + str(level))
                                   for level in range(1, self.header_levels+1)]
            self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        elif self.name == TextSplitterName.recursive_splitter:
            self.splitter = RecursiveCharacterTextSplitter(
                 chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len, is_separator_regex=False
            )
        elif self.name == TextSplitterName.semantic_splitter:
            self.splitter = SemanticChunker(embeddings=self.embeddings, min_chunk_size=int(self.chunk_size/3))
        elif self.name == TextSplitterName.sentence_splitter:
            self.splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=self.chunk_overlap, tokens_per_chunk=self.chunk_size
            )

    class Config:
        arbitrary_types_allowed = True


class ChunkMetadata(BaseModel):
    """Per‑chunk metadata attached to each produced :langchain_core:`Document <documents/langchain_core.documents.base.Document.html>`."""
    filename: str #: str : Path of the source file.
    chunk_idx: int #: int : Zero-based progressive index within the document.
    headers: Dict[str, str] #: Dict[str, str] : Markdown headers captured when the first splitter is a Markdown splitter.
    n_tokens: int #: int : Estimated token count for the chunk computed via the first splitter (if any) that has embeddings.


class ChunkingManager(metaclass=Singleton):
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
    def __init__(self, splitter_list: List[TextSplitterConfig]):
        self.splitter_list = splitter_list

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
