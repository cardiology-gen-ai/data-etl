import json
import logging
import os
import pathlib
import math
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import NAMESPACE_URL, uuid5

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from cardiology_gen_ai import (
    IndexTypeNames,
    DistanceTypeNames,
    IndexingConfig,
    EmbeddingConfig,
    Vectorstore,
    QdrantVectorstore,
    FaissVectorstore,
)
from cardiology_gen_ai.utils.logger import get_logger
from cardiology_gen_ai.utils.singleton import Singleton


class L2NormalizedEmbeddings(Embeddings):
    """Normalize document and query embeddings before FAISS cosine search."""

    def __init__(self, base: Embeddings):
        self.base = base

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise ValueError("Cannot L2-normalize a zero embedding vector")
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._normalize(vector) for vector in self.base.embed_documents(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self._normalize(self.base.embed_query(text))


class EditableVectorstore(Vectorstore, ABC):
    """Abstract editable vector store."""

    @abstractmethod
    def create_vectorstore(self, **kwargs) -> QdrantVectorStore | FAISS:
        pass

    @abstractmethod
    def _delete_vectorstore(self) -> None:
        pass

    def delete_vectorstore(self) -> None:
        if self.vectorstore_exists():
            self._delete_vectorstore()

    @abstractmethod
    def delete_from_vectorstore(
        self,
        source_value: str,
        metadata_key: str = "filename",
    ) -> int:
        pass

    @abstractmethod
    def add_to_vectorstore(
        self,
        doc: Document | List[Document],
        ids: Optional[List[str]] = None,
    ) -> None:
        pass

    def _add_to_vectorstore(
        self,
        doc: Document | List[Document],
        ids: Optional[List[str]] = None,
    ) -> None:
        documents = [doc] if isinstance(doc, Document) else list(doc)
        kwargs = {"ids": ids} if ids is not None else {}
        self.vectorstore.add_documents(documents=documents, **kwargs)


class EditableQdrantVectorstore(EditableVectorstore, QdrantVectorstore):
    """Editable Qdrant backend with dense + sparse hybrid configuration."""

    def create_vectorstore(self, embeddings_model: EmbeddingConfig) -> QdrantVectorStore:
        if not self.vectorstore_exists():
            distance = (
                Distance.COSINE
                if self.config.distance == DistanceTypeNames.cosine
                else Distance.EUCLID
            )
            self.client.create_collection(
                collection_name=self.config.name,
                vectors_config=VectorParams(
                    size=embeddings_model.dim,
                    distance=distance,
                ),
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
        qdrant_vectorstore = QdrantVectorStore.construct_instance(
            collection_name=self.config.name,
            embedding=embeddings_model.model,
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            vector_name="dense",
            sparse_vector_name="sparse",
            content_payload_key="page_content",
            metadata_payload_key="metadata",
            force_recreate=True,
        )
        self.vectorstore = qdrant_vectorstore
        return qdrant_vectorstore

    def _delete_vectorstore(self) -> None:
        self.client.delete_collection(collection_name=self.config.name)

    def delete_from_vectorstore(
        self,
        source_value: str,
        metadata_key: str = "filename",
    ) -> int:
        current_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=f"metadata.{metadata_key}",
                    match=models.MatchValue(value=str(source_value)),
                )
            ]
        )
        del_vectorstore_points, _ = self.client.scroll(
            collection_name=self.config.name,
            limit=1000000,
            with_payload=["page_content", "metadata"],
            with_vectors=True,
            scroll_filter=current_filter,
        )

        # Migration path for indices built before source_key was introduced.
        if (
            not del_vectorstore_points
            and metadata_key == "source_key"
            and "::" in source_value
        ):
            doc_id, source_type = source_value.rsplit("::", 1)
            legacy_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.doc_id",
                        match=models.MatchValue(value=doc_id),
                    ),
                    models.FieldCondition(
                        key="metadata.prebuilt_source_type",
                        match=models.MatchValue(value=source_type),
                    ),
                ]
            )
            del_vectorstore_points, _ = self.client.scroll(
                collection_name=self.config.name,
                limit=1000000,
                with_payload=["page_content", "metadata"],
                with_vectors=True,
                scroll_filter=legacy_filter,
            )

        for point in del_vectorstore_points:
            self.vectorstore.delete(ids=[point.id])
        return len(del_vectorstore_points)

    def add_to_vectorstore(
        self,
        doc: Document | List[Document],
        ids: Optional[List[str]] = None,
    ) -> None:
        return EditableVectorstore._add_to_vectorstore(self, doc, ids=ids)


class EditableFaissVectorstore(EditableVectorstore, FaissVectorstore):
    """Editable FAISS backend with on-disk persistence."""

    def _embedding_function(self, embeddings_model: EmbeddingConfig) -> Embeddings:
        if self.config.distance == DistanceTypeNames.cosine:
            return L2NormalizedEmbeddings(embeddings_model.model)
        return embeddings_model.model

    def _distance_strategy(self) -> DistanceStrategy:
        if self.config.distance == DistanceTypeNames.cosine:
            return DistanceStrategy.MAX_INNER_PRODUCT
        return DistanceStrategy.EUCLIDEAN_DISTANCE

    def create_vectorstore(self, embeddings_model: EmbeddingConfig, **kwargs) -> FAISS:
        import faiss

        self._ensure_folder()
        faiss_index = (
            faiss.IndexFlatIP(embeddings_model.dim)
            if self.config.distance == DistanceTypeNames.cosine
            else faiss.IndexFlatL2(embeddings_model.dim)
        )
        faiss_vectorstore = FAISS(
            embedding_function=self._embedding_function(embeddings_model),
            index=faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            normalize_L2=False,
            distance_strategy=self._distance_strategy(),
        )
        self.vectorstore = faiss_vectorstore
        self.vectorstore.save_local(
            folder_path=self.config.folder.as_posix(),
            index_name=self.config.name,
        )
        return faiss_vectorstore

    def load_vectorstore(
        self,
        embeddings_model: EmbeddingConfig,
        **kwargs,
    ) -> FAISS:
        self._ensure_folder()
        faiss_vectorstore = FAISS.load_local(
            folder_path=self.config.folder.as_posix(),
            index_name=self.config.name,
            embeddings=self._embedding_function(embeddings_model),
            allow_dangerous_deserialization=True,
            normalize_L2=False,
            distance_strategy=self._distance_strategy(),
        )
        self.vectorstore = faiss_vectorstore
        return faiss_vectorstore

    def _delete_vectorstore(self) -> None:
        os.remove(
            (pathlib.Path(self.config.folder) / f"{self.config.name}.faiss").as_posix()
        )
        os.remove(
            (pathlib.Path(self.config.folder) / f"{self.config.name}.pkl").as_posix()
        )

    def _ensure_folder(self) -> None:
        self.config.folder.mkdir(parents=True, exist_ok=True)

    def delete_from_vectorstore(
        self,
        source_value: str,
        metadata_key: str = "filename",
    ) -> int:
        del_documents = [
            docstore_id
            for docstore_id, doc in self.vectorstore.docstore._dict.items()
            if str(doc.metadata.get(metadata_key) or "") == str(source_value)
        ]

        # Migration path for indices built before source_key was introduced.
        if (
            not del_documents
            and metadata_key == "source_key"
            and "::" in source_value
        ):
            doc_id, source_type = source_value.rsplit("::", 1)
            del_documents = [
                docstore_id
                for docstore_id, doc in self.vectorstore.docstore._dict.items()
                if str(doc.metadata.get("doc_id") or "") == doc_id
                and str(doc.metadata.get("prebuilt_source_type") or "")
                == source_type
            ]

        if del_documents:
            self.vectorstore.delete(ids=del_documents)
            self.vectorstore.save_local(
                folder_path=str(self.config.folder),
                index_name=self.config.name,
            )
        return len(del_documents)

    def add_to_vectorstore(
        self,
        doc: Document | List[Document],
        ids: Optional[List[str]] = None,
    ) -> None:
        documents = [doc] if isinstance(doc, Document) else list(doc)
        kwargs = {"ids": ids} if ids is not None else {}
        self.vectorstore.add_documents(documents=documents, **kwargs)
        self.vectorstore.save_local(
            folder_path=self.config.folder.as_posix(),
            index_name=self.config.name,
        )


class IndexManager(metaclass=Singleton):
    """High-level manager for vector index lifecycle operations."""

    logger: logging.Logger
    config: IndexingConfig
    embeddings: EmbeddingConfig
    vectorstore: EditableVectorstore

    def __init__(self, config: IndexingConfig, embeddings: EmbeddingConfig):
        self.logger = get_logger("Indexing based on LangChain VectorStores")
        self.config = config
        self.embeddings = embeddings
        self.config.folder.mkdir(parents=True, exist_ok=True)
        self._save_config()
        self.vectorstore = (
            EditableQdrantVectorstore(config=self.config)
            if IndexTypeNames(self.config.type) == IndexTypeNames.qdrant
            else EditableFaissVectorstore(config=self.config)
        )

    def _save_config(self, filename: str = "config.json") -> None:
        self.config.folder.mkdir(parents=True, exist_ok=True)
        config_file = pathlib.Path(self.config.folder) / filename
        saved = False
        if config_file.is_file():
            with config_file.open("r", encoding="utf-8") as handle:
                existing_config_json = json.load(handle)
            existing_config_embedding = existing_config_json["embeddings"]
            existing_config_indexing = existing_config_json["indexing"]
            existing_config_indexing["type"] = (
                existing_config_indexing["type"]
                if isinstance(existing_config_indexing["type"], list)
                else [existing_config_indexing["type"]]
            )
            if (
                self.config.name == existing_config_indexing["name"]
                and self.config.distance.value == existing_config_indexing["distance"]
                and self.embeddings.model_name
                == existing_config_embedding["deployment"]
            ):
                existing_config_indexing["type"].append(self.config.type.value)
                existing_config_indexing["type"] = list(
                    set(existing_config_indexing["type"])
                )
                with config_file.open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "indexing": existing_config_indexing,
                            "embeddings": existing_config_embedding,
                        },
                        handle,
                        indent=2,
                    )
                saved = True
        if not saved:
            with config_file.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "indexing": self.config.to_config(),
                        "embeddings": self.embeddings.to_config(),
                    },
                    handle,
                    indent=2,
                )

    @staticmethod
    def _normalize_documents(
        doc: Document | List[Document],
    ) -> List[Document]:
        documents = [doc] if isinstance(doc, Document) else list(doc)
        if not documents:
            raise ValueError("Cannot index an empty document list")
        if not all(isinstance(document, Document) for document in documents):
            raise TypeError("All indexed items must be LangChain Document objects")
        for document in documents:
            filename = str(document.metadata.get("filename") or "").strip()
            if not filename:
                raise ValueError("Every indexed document must define metadata['filename']")
        return documents

    @staticmethod
    def _stable_vector_ids(documents: List[Document]) -> Optional[List[str]]:
        """Return deterministic backend-safe IDs for prebuilt records.

        Legacy Markdown chunks have no ``record_id`` and retain LangChain-generated
        IDs. Prebuilt corpora must define ``record_id`` for every document. UUID5
        keeps the ID deterministic while remaining valid for both FAISS and Qdrant.
        """
        record_ids = [
            str(document.metadata.get("record_id") or "").strip()
            for document in documents
        ]
        has_record_ids = [bool(record_id) for record_id in record_ids]
        if any(has_record_ids) and not all(has_record_ids):
            raise ValueError(
                "A document batch cannot mix records with and without metadata['record_id']"
            )
        if not any(has_record_ids):
            return None
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Duplicate metadata['record_id'] values in indexing batch")
        return [
            str(uuid5(NAMESPACE_URL, f"cardiology-rag:{record_id}"))
            for record_id in record_ids
        ]

    def create_index(self) -> None:
        try:
            self.vectorstore.create_vectorstore(embeddings_model=self.embeddings)
            assert self.vectorstore.vectorstore is not None
            self.logger.info(f"Index {self.config.name} created successfully.")
        except Exception as exc:
            self.logger.error(f"Error creating index {self.config.name}: {exc}")
            raise

    def get_n_documents_in_vectorstore(self) -> int:
        return self.vectorstore.get_n_documents_in_vectorstore()

    def delete_index(self) -> None:
        try:
            self.vectorstore.delete_vectorstore()
            self.logger.info(f"Index {self.config.name} deleted successfully.")
        except Exception as exc:
            self.logger.error(f"Error deleting index {self.config.name}: {exc}")
            raise

    def load_index(self) -> None:
        try:
            self.vectorstore.load_vectorstore(
                embeddings_model=self.embeddings,
                retrieval_mode=self.config.retrieval_mode.value,
            )
            self.logger.info(f"Index {self.config.name} loaded successfully.")
        except Exception as exc:
            self.logger.error(f"Error loading {self.config.name} index: {exc}")
            raise

    @staticmethod
    def _replacement_selector(documents: List[Document]) -> tuple[str, List[str]]:
        """Choose a stable replacement key for a homogeneous document batch.

        Prebuilt records use ``source_key`` (``doc_id`` + artifact type), so an
        index can be moved or rebuilt from a differently located source file.
        Legacy records fall back to the historical ``filename`` key.
        """
        source_keys = [
            str(document.metadata.get("source_key") or "").strip()
            for document in documents
        ]
        has_source_keys = [bool(value) for value in source_keys]
        if any(has_source_keys) and not all(has_source_keys):
            raise ValueError(
                "A document batch cannot mix records with and without "
                "metadata['source_key']"
            )
        if all(has_source_keys):
            return "source_key", sorted(set(source_keys))
        return "filename", sorted(
            {str(document.metadata["filename"]) for document in documents}
        )

    def delete_document(
        self,
        source_value: str | pathlib.Path,
        metadata_key: str = "filename",
    ) -> int:
        try:
            normalized_value = str(source_value)
            n_doc_deleted = self.vectorstore.delete_from_vectorstore(
                normalized_value,
                metadata_key=metadata_key,
            )
            if n_doc_deleted > 0:
                self.logger.info(
                    "Document source deleted successfully | key=%s | value=%s "
                    "| chunks=%d",
                    metadata_key,
                    normalized_value,
                    n_doc_deleted,
                )
            return n_doc_deleted
        except Exception as exc:
            self.logger.error(
                "Error deleting document source | key=%s | value=%s | error=%s",
                metadata_key,
                source_value,
                exc,
            )
            raise

    def add_document(self, doc: Document | List[Document]) -> None:
        """Replace previous chunks for each source, then add the new batch.

        Prebuilt records are replaced by a stable ``metadata['source_key']``
        independent of the local filesystem path. Legacy Markdown chunks keep
        the historical filename-based replacement. Deterministic vector-store
        IDs continue to derive from ``metadata['record_id']``.
        """
        documents = self._normalize_documents(doc)
        stable_ids = self._stable_vector_ids(documents)
        metadata_key, source_values = self._replacement_selector(documents)

        doc_present = sum(
            self.delete_document(value, metadata_key=metadata_key)
            for value in source_values
        )
        if doc_present > 0:
            self.logger.info("Overwriting document(s)")

        try:
            self.vectorstore.add_to_vectorstore(documents, ids=stable_ids)
            self.logger.info(
                "Document(s) added successfully | sources=%d | chunks=%d "
                "| replacement_key=%s",
                len(source_values),
                len(documents),
                metadata_key,
            )
        except Exception as exc:
            self.logger.error(f"Error adding document(s): {exc}")
            raise
