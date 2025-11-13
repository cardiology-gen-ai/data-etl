import configparser
import json
import logging
import os
import pathlib
from typing import List
from abc import ABC, abstractmethod

from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from qdrant_client import models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_community.vectorstores import FAISS

from cardiology_gen_ai import (IndexTypeNames, DistanceTypeNames, IndexingConfig, EmbeddingConfig,
                               Vectorstore, QdrantVectorstore, FaissVectorstore)
from cardiology_gen_ai.utils.logger import get_logger
from cardiology_gen_ai.utils.singleton import Singleton


class EditableVectorstore(Vectorstore, ABC):
    """Abstract editable vector store.

    Subclasses must implement creation/deletion of the underlying index and the
    ability to add or remove documents by filename.

    .. rubric:: Notes

    ``self.vectorstore`` is assumed to be set by :meth:`create_vectorstore` and
    to implement LangChain's ``add_documents``/``delete`` APIs.
    """

    @abstractmethod
    def create_vectorstore(self, **kwargs) -> QdrantVectorStore | FAISS:
        """Create or connect to the underlying vector store.

        Returns
        -------
        :class:`~langchain_qdrant.qdrant.QdrantVectorStore` | :class:`~langchain_community.vectorstores.faiss.FAISS`
            The constructed vector store instance.
        """
        pass

    @abstractmethod
    def _delete_vectorstore(self) -> None:
        """Delete the underlying index.

        Subclasses should implement the destructive operation only; callers
        should use :meth:`delete_vectorstore` to perform safe deletion.
        """
        pass

    def delete_vectorstore(self) -> None:
        """Safely delete the vector store if it exists."""
        if self.vectorstore_exists():
            self._delete_vectorstore()
        return

    @abstractmethod
    def delete_from_vectorstore(self, filename: pathlib.Path) -> int:
        """Remove all chunks/documents associated with ``filename``.

        Parameters
        ----------
        filename : :class:`pathlib.Path`
            Path to source file whose chunks should be removed.

        Returns
        -------
        int
            Number of removed items.
        """
        pass

    @abstractmethod
    def add_to_vectorstore(self, doc: Document | List[Document]) -> None:
        """Add one or more :langchain_core:`Document <documents/langchain_core.documents.base.Document.html>` items.
        """
        pass

    def _add_to_vectorstore(self, doc: Document | List[Document]) -> None:
        """Default implementation delegating to ``self.vectorstore.add_documents``."""
        if isinstance(doc, Document):
            self.vectorstore.add_documents(documents=[doc])
        else:
            self.vectorstore.add_documents(documents=doc)


class EditableQdrantVectorstore(EditableVectorstore, QdrantVectorstore):
    """Editable Qdrant backend with dense + sparse hybrid configuration.

    The collection is created on demand with the desired distance metric and
    vector sizes inferred from the provided embedding configuration.
    """

    def create_vectorstore(self, embeddings_model: EmbeddingConfig) -> QdrantVectorStore:
        """Create/connect a Qdrant collection and construct a :class:`~langchain_qdrant.qdrant.QdrantVectorStore`.

        Parameters
        ----------
        embeddings_model : :class:`cardiology_gen_ai.models.EmbeddingConfig`
            Embedding model config providing ``dim`` and ``model``.

        Returns
        -------
        :py:class:`~langchain_qdrant.qdrant.QdrantVectorStore`
            Constructed :py:class:`~langchain_qdrant.qdrant.QdrantVectorStore` bound to
            ``self.config.name``.
        """
        if not self.vectorstore_exists():
            distance = Distance.COSINE if self.config.distance == DistanceTypeNames.cosine else Distance.EUCLID
            self.client.create_collection(
                collection_name=self.config.name,
                vectors_config=VectorParams(size=embeddings_model.dim, distance=distance),
                sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
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
        """Drop the Qdrant collection configured in ``self.config.name``."""
        self.client.delete_collection(collection_name=self.config.name)

    def delete_from_vectorstore(self, filename: pathlib.Path) -> int:
        """Delete all points matching ``metadata.filename == str(filename)``.

        Parameters
        ----------
        filename : :class:`pathlib.Path`
            Source filename used to filter points.

        Returns
        -------
        int
            Number of deleted points.
        """
        current_filter = models.Filter(must=[models.FieldCondition(
             key="metadata.filename", match=models.MatchValue(value=str(filename)))]
        )
        del_vectorstore_points, _ = self.client.scroll(
            collection_name=self.config.name,
            limit=1000000,
            with_payload=["page_content", "metadata"],
            with_vectors=True,
            scroll_filter=current_filter,
        )
        if len(del_vectorstore_points) > 0:
            for p in del_vectorstore_points:
                self.vectorstore.delete(ids=[p.id])
        return len(del_vectorstore_points)

    def add_to_vectorstore(self, doc: Document | List[Document]) -> None:
        """Add documents to Qdrant (dense + sparse) using the base helper."""
        return Vectorstore._add_to_vectorstore(self, doc)


class EditableFaissVectorstore(EditableVectorstore, FaissVectorstore):
    """Editable :class:`~langchain_community.vectorstores.faiss.FAISS` backend with on-disk persistence.

    The index type depends on the configured distance metric: :faiss:`IndexFlatIP <struct/structfaiss_1_1IndexFlatIP.html>` for
    cosine similarity (with L2 normalization), or :faiss:`IndexFlatL2 <struct/structfaiss_1_1IndexFlatL2.html>` for Euclidean distance.
    """

    def create_vectorstore(self, embeddings_model: EmbeddingConfig, **kwargs) -> FAISS:
        """Instantiate a :class:`~langchain_community.vectorstores.faiss.FAISS` vector store and persist it.

        Parameters
        ----------
        embeddings_model : :class:`cardiology_gen_ai.models.EmbeddingConfig`
            Embedding model config providing ``dim`` and ``model``.

        Returns
        -------
        :py:class:`~langchain_community.vectorstores.faiss.FAISS`
            Constructed FAISS vector store.
        """
        import faiss
        faiss_index = faiss.IndexFlatIP(embeddings_model.dim) if self.config.distance == DistanceTypeNames.cosine \
            else faiss.IndexFlatL2(embeddings_model.dim)
        normalize = True if self.config.distance == DistanceTypeNames.cosine else False
        faiss_vectorstore = FAISS(
            embedding_function=embeddings_model.model,
            index=faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            normalize_L2=normalize,
        )
        self.vectorstore = faiss_vectorstore
        self.vectorstore.save_local(folder_path=self.config.folder.as_posix(), index_name=self.config.name)
        return faiss_vectorstore

    def _delete_vectorstore(self) -> None:
        """Remove the persisted :class:`~langchain_community.vectorstores.faiss.FAISS` ``.faiss`` and ``.pkl`` files."""
        os.remove((pathlib.Path(self.config.folder) / (self.config.name + ".faiss")).as_posix())
        os.remove((pathlib.Path(self.config.folder) / (self.config.name + ".pkl")).as_posix())

    def _ensure_folder(self) -> None:
        """Ensure the :class:`~langchain_community.vectorstores.faiss.FAISS` persistence folder exists."""
        self.config.folder.mkdir(parents=True, exist_ok=True)

    def delete_from_vectorstore(self, filename: pathlib.Path) -> int:
        """Delete all stored docs whose metadata ``filename`` matches the input.

        Parameters
        ----------
        filename : :class:`pathlib.Path`
            Source filename used to select stored documents.

        Returns
        -------
        int
            Number of deleted documents.
        """
        del_documents = [doc_id for doc_id, doc in self.vectorstore.docstore._dict.items()
                         if doc.metadata["filename"] == str(filename)]
        if len(del_documents):
            self.vectorstore.delete(ids=del_documents)
            self.vectorstore.save_local(folder_path=str(self.config.folder), index_name=self.config.name)
        return len(del_documents)

    def add_to_vectorstore(self, doc: Document | List[Document]) -> None:
        """Add documents and persist the :class:`~langchain_community.vectorstores.faiss.FAISS` index immediately."""
        # Vectorstore.add_to_vectorstore(self, doc)
        docs = doc if isinstance(doc, list) else [doc]
        self.vectorstore.add_documents(docs)
        self.vectorstore.save_local(folder_path=self.config.folder.as_posix(), index_name=self.config.name)


class IndexManager(metaclass=Singleton):
    """High-level manager for vector index lifecycle operations.

    Parameters
    ----------
    config : :class:`cardiology_gen_ai.models.IndexingConfig`
        Backend and persistence configuration.
    embeddings : :class:`cardiology_gen_ai.models.EmbeddingConfig`
        Embedding model configuration (callable + dimensionality).
    """
    logger: logging.Logger  #: :class:`~logging.Logger` : Named logger ("Indexing based on LangChain VectorStores").
    config: IndexingConfig  #: :class:`cardiology_gen_ai.models.IndexingConfig` : Backend and persistence configuration used by this manager.
    embeddings: EmbeddingConfig  #: :class:`cardiology_gen_ai.models.EmbeddingConfig` : Embedding model (callable and vector dimensionality).
    vectorstore: EditableVectorstore  #: :class:`~src.managers.index_manager.EditableVectorstore` : Concrete editable vector store selected from ``config.type``, either :class:`~src.managers.index_manager.EditableQdrantVectorstore` (wraps :py:class:`~langchain_qdrant.qdrant.QdrantVectorStore`) or :class:`~src.managers.index_manager.EditableFaissVectorstore` (wraps :py:class:`~langchain_community.vectorstores.faiss.FAISS`).

    def __init__(self, config: IndexingConfig, embeddings: EmbeddingConfig):
        self.logger = get_logger("Indexing based on LangChain VectorStores")
        self.config = config
        self.embeddings = embeddings
        self._save_config()
        self.vectorstore: EditableVectorstore = (
            EditableQdrantVectorstore(config=self.config)) if IndexTypeNames(self.config.type) == IndexTypeNames.qdrant \
            else EditableFaissVectorstore(config=self.config)

    def _save_config(self, filename="config.json") -> None:
        """Save configuration to disk."""
        config_file = pathlib.Path(self.config.folder) / filename
        saved = False
        if config_file.is_file():
            with open(str(config_file), "r") as f:
                existing_config_json = json.load(f)
            existing_config_embedding = existing_config_json["embeddings"]
            existing_config_indexing = existing_config_json["indexing"]
            existing_config_indexing["type"] = existing_config_indexing["type"] \
                if isinstance(existing_config_indexing["type"], list) else [existing_config_indexing["type"]]
            if (self.config.name == existing_config_indexing["name"] and
                    self.config.distance.value == existing_config_indexing["distance"] and
                    self.embeddings.model_name == existing_config_embedding["deployment"]):
                existing_config_indexing["type"].append(self.config.type.value)
                existing_config_indexing["type"] = list(set(existing_config_indexing["type"]))
                with open(str(config_file), "w") as f:
                    json.dump({"indexing": existing_config_indexing, "embeddings": existing_config_embedding},
                              f, indent=2)
                saved = True
        if not saved:
            print(f"Saving {str(config_file)}.")
            with open(str(config_file), "w") as f:
                json.dump(
                    {"indexing": self.config.to_config(), "embeddings": self.embeddings.to_config()}, f, indent=2)

    def create_index(self) -> None:
        """Create the underlying index/collection using the configured backend.
        Logs a success message or raises if creation fails.
        """
        try:
            self.vectorstore.create_vectorstore(embeddings_model=self.embeddings)
            assert self.vectorstore.vectorstore is not None
            self.logger.info(f"Index {self.config.name} created successfully.")
        except Exception as e:
            self.logger.info(f"Error creating index {self.config.name}: {str(e)}")
            raise

    def get_n_documents_in_vectorstore(self) -> int:
        """Return the number of stored documents/chunks, as reported by the backend."""
        return self.vectorstore.get_n_documents_in_vectorstore()

    def delete_index(self) -> None:
        """Delete the underlying index/collection and log the operation."""
        try:
            self.vectorstore.delete_vectorstore()
            self.logger.info(f"Index {self.config.name} deleted successfully.")
        except Exception as e:
            self.logger.info(f"Error deleting index {self.config.name}: {str(e)}")
            raise

    def load_index(self) -> None:
        """Load or connect to an existing index for retrieval.
        Uses backend-specific ``load_vectorstore`` with the configured retrieval mode.
        """
        try:
            self.vectorstore.load_vectorstore(embeddings_model=self.embeddings,
                                              retrieval_mode=self.config.retrieval_mode.value)
            self.logger.info(f"Index {self.config.name} loaded successfully.")
        except Exception as e:
            self.logger.info(f"Error loading {self.config.name} index: {str(e)}")
            raise

    def delete_document(self, filename: pathlib.Path) -> int:
        """Remove any chunks/documents associated with ``filename`` from the index.

        Parameters
        ----------
        filename : :class:`pathlib.Path`
            Path to the file whose chunks should be removed.

        Returns
        -------
        int
            Number of removed items.
        """
        try:
            n_doc_deleted = self.vectorstore.delete_from_vectorstore(filename)
            if n_doc_deleted > 0:
                self.logger.info(f"Document {filename} deleted successfully ({n_doc_deleted} chunks removed)")
            return n_doc_deleted
        except Exception as e:
            self.logger.info(f"Error deleting document {filename}: {str(e)}")
            raise

    def add_document(self, doc: Document | List[Document]) -> None:
        """Add a document or list of documents; overwrite any existing entries.

        If entries already exist for the same ``metadata['filename']``, they are
        deleted first to avoid duplicates.

        Parameters
        ----------
        doc : :langchain_core:`Document <documents/langchain_core.documents.base.Document.html>` | List[:langchain_core:`Document <documents/langchain_core.documents.base.Document.html>`]
            Document or list of documents to add.
        """
        filename = doc.metadata["filename"] if isinstance(doc, Document) else [d.metadata["filename"] for d in doc]
        doc_present = self.delete_document(filename) if isinstance(doc, Document) \
            else sum(self.delete_document(f) for f in filename)
        if doc_present > 0:
            self.logger.info(f"Overwriting document(s)")
        try:
            self.vectorstore.add_to_vectorstore(doc)
            self.logger.info("Document(s) added successfully")
        except Exception as e:
            self.logger.info(f"Error adding document(s): {str(e)}")
            raise
