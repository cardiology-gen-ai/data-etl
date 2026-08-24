import json
import pathlib
from enum import Enum
from typing import Any, List, Dict, Optional, Sequence

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



class PrebuiltChunkSource(str, Enum):
    """Supported prebuilt retrieval artifacts.

    These modes bypass the legacy Markdown splitters and only adapt already
    prepared JSON records to LangChain ``Document`` objects.
    """

    fixed_chunks = "fixed_chunks"
    hierarchical_section_view = "hierarchical_section_view"


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

    @staticmethod
    def _normalize_prebuilt_records(
        payload: Any,
        source_type: PrebuiltChunkSource,
    ) -> List[Dict[str, Any]]:
        """Extract a record list from one supported prebuilt JSON payload."""
        if source_type == PrebuiltChunkSource.fixed_chunks:
            if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
                raise TypeError(
                    "Fixed-chunk JSON must be an object containing a 'chunks' list"
                )
            records = payload["chunks"]
        elif isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
            records = payload["chunks"]
        else:
            raise TypeError(
                "Hierarchical Section-view JSON must be a list or an object "
                "containing a 'chunks' list"
            )

        if not all(isinstance(record, dict) for record in records):
            raise TypeError("Every prebuilt chunk record must be a JSON object")
        return [dict(record) for record in records]

    @staticmethod
    def _normalize_string_list(values: Any, fallback: Any = None) -> List[str]:
        """Return non-empty string values while preserving their input order."""
        if values is None:
            values = [] if fallback is None else [fallback]
        elif isinstance(values, (str, int, float)):
            values = [values]
        elif not isinstance(values, Sequence):
            raise TypeError(f"Expected a sequence of identifiers, got {type(values)!r}")

        normalized: List[str] = []
        seen: set[str] = set()
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)
        return normalized

    @staticmethod
    def _embedding_text(section_title: str, text: str) -> str:
        """Build the common title/body representation used for vectorization."""
        return f"Title: {section_title}\n\nBody:\n{text}"

    def load_prebuilt_chunks(
        self,
        filepath: pathlib.Path | str,
        source_type: PrebuiltChunkSource | str,
    ) -> List[Document]:
        """Load fixed chunks or a hierarchical Section view without re-splitting.

        The existing :meth:`split_text` path is intentionally untouched. This
        method is an explicit opt-in adapter for MinerU-derived retrieval
        artifacts. It preserves section provenance and emits the same
        ``Title: ... / Body: ...`` page-content template for both corpora.

        Parameters
        ----------
        filepath:
            JSON artifact to load.
        source_type:
            ``fixed_chunks`` or ``hierarchical_section_view``.

        Returns
        -------
        list[Document]
            Retrieval documents in source order. Structural Section-view
            records are omitted.
        """
        path = pathlib.Path(filepath)
        normalized_source_type = PrebuiltChunkSource(source_type)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = self._normalize_prebuilt_records(payload, normalized_source_type)

        documents: List[Document] = []
        seen_record_ids: set[str] = set()
        observed_doc_ids: set[str] = set()

        for source_index, record in enumerate(records):
            if normalized_source_type == PrebuiltChunkSource.hierarchical_section_view:
                role = str(record.get("section_view_role") or "").strip()
                if role == "structural":
                    continue
                if role != "retrieval":
                    raise ValueError(
                        "Hierarchical Section-view record at index "
                        f"{source_index} has invalid section_view_role={role!r}"
                    )

            # Defensive checks. The fixed corpus and Section-view builders
            # should already enforce these invariants, but the vector path must
            # never index excluded, structural, or empty content.
            if bool(record.get("excluded")):
                continue
            if record.get("embed") is False:
                continue
            if bool(record.get("is_empty")):
                continue
            text = str(record.get("text") or "").strip()
            if not text:
                continue

            if normalized_source_type == PrebuiltChunkSource.fixed_chunks:
                record_id = str(
                    record.get("fixed_chunk_id") or record.get("chunk_id") or ""
                ).strip()
                retrieval_strategy = str(
                    record.get("strategy") or "fixed_within_section"
                )
            else:
                record_id = str(record.get("retrieval_unit_id") or "").strip()
                retrieval_strategy = str(
                    record.get("retrieval_strategy") or "hierarchical_sections"
                )

            if not record_id:
                raise ValueError(
                    f"Prebuilt record at index {source_index} has no stable identifier"
                )
            if record_id in seen_record_ids:
                raise ValueError(f"Duplicate prebuilt record identifier: {record_id}")
            seen_record_ids.add(record_id)

            doc_id = str(record.get("doc_id") or "").strip()
            if not doc_id:
                raise ValueError(f"Prebuilt record {record_id} has no doc_id")
            observed_doc_ids.add(doc_id)

            section_id = str(
                record.get("section_id")
                or record.get("source_section_id")
                or ""
            ).strip()
            section_title = str(record.get("section_title") or "").strip()
            display_title = section_title or section_id or record_id

            source_section_ids = self._normalize_string_list(
                record.get("source_section_ids"),
                fallback=record.get("source_section_id") or section_id,
            )
            source_chunk_ids = self._normalize_string_list(
                record.get("source_chunk_ids"),
                fallback=record.get("source_chunk_id") or record.get("chunk_id"),
            )
            if not source_section_ids:
                raise ValueError(f"Prebuilt record {record_id} has no source sections")
            if not source_chunk_ids:
                raise ValueError(f"Prebuilt record {record_id} has no source chunks")

            source_key = f"{doc_id}::{normalized_source_type.value}"
            retrieval_unit_key = f"{source_key}::{record_id}"
            metadata: Dict[str, Any] = {
                # Keep filename for provenance and legacy compatibility. New
                # prebuilt replacement uses the location-independent source_key.
                "filename": str(path),
                "source_key": source_key,
                # Globally unique across documents and retrieval representations.
                "retrieval_unit_key": retrieval_unit_key,
                "chunk_idx": len(documents),
                "headers": {"Section": display_title},
                "n_tokens": 0,
                # Common retrieval metadata.
                "prebuilt_source_type": normalized_source_type.value,
                "record_id": record_id,
                "doc_id": doc_id,
                "retrieval_strategy": retrieval_strategy,
                "embedding_text_template": "title_body_v1",
                "section_id": section_id or None,
                "section_title": section_title or None,
                "section_level": record.get("section_level"),
                "parent_section_id": record.get("parent_section_id"),
                "page_start": record.get("page_start"),
                "page_end": record.get("page_end"),
                "source_section_ids": source_section_ids,
                "source_chunk_ids": source_chunk_ids,
            }

            if normalized_source_type == PrebuiltChunkSource.fixed_chunks:
                metadata.update(
                    {
                        "fixed_chunk_id": record_id,
                        "fixed_part_index": record.get("fixed_part_index"),
                        "fixed_part_count": record.get("fixed_part_count"),
                        "chunk_size": record.get("chunk_size"),
                        "chunk_overlap": record.get("chunk_overlap"),
                        "contains_table": bool(record.get("contains_table")),
                        "oversized_atomic_table": bool(
                            record.get("oversized_atomic_table")
                        ),
                    }
                )
            else:
                metadata.update(
                    {
                        "retrieval_unit_id": record_id,
                        "section_view_role": "retrieval",
                        "aggregation_max_level": record.get(
                            "aggregation_max_level"
                        ),
                        "is_aggregated": bool(record.get("is_aggregated")),
                        "represented_section_ids": self._normalize_string_list(
                            record.get("represented_section_ids"),
                            fallback=section_id,
                        ),
                    }
                )

            metadata = {
                key: value for key, value in metadata.items() if value is not None
            }
            documents.append(
                Document(
                    page_content=self._embedding_text(display_title, text),
                    metadata=metadata,
                )
            )

        if not documents:
            raise ValueError(f"No retrievable records found in {path}")
        if len(observed_doc_ids) != 1:
            raise ValueError(
                "A prebuilt artifact must contain exactly one doc_id; found "
                f"{sorted(observed_doc_ids)}"
            )
        return documents

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
