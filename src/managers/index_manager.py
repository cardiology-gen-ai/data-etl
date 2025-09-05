import os
import pathlib
from typing import List
from abc import ABC, abstractmethod

from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.vectorstores import FAISS

from cardiology_gen_ai import IndexTypeNames, DistanceTypeNames, IndexingConfig, \
    EmbeddingConfig
from src.utils.logger import get_logger
from src.utils.singleton import Singleton


class Vectorstore(BaseModel, ABC):
    config: IndexingConfig
    vectorstore: QdrantVectorStore | FAISS = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def vectorstore_exists(self) -> bool:
        pass

    @abstractmethod
    def create_vectorstore(self, **kwargs) -> QdrantVectorStore | FAISS:
        pass

    @abstractmethod
    def load_vectorstore(self, **kwargs) -> QdrantVectorStore | FAISS:
        pass

    @abstractmethod
    def get_n_documents_in_vectorstore(self) -> int:
        pass

    @abstractmethod
    def _delete_vectorstore(self) -> None:
        pass

    def delete_vectorstore(self) -> None:
        if self.vectorstore_exists():
            self._delete_vectorstore()
        return

    @abstractmethod
    def delete_from_vectorstore(self, filename: pathlib.Path) -> int:
        pass

    @abstractmethod
    def add_to_vectorstore(self, doc: Document | List[Document]) -> None:
        pass

    def _add_to_vectorstore(self, doc: Document | List[Document]) -> None:
        if isinstance(doc, Document):
            self.vectorstore.add_documents(documents=[doc])
        else:
            self.vectorstore.add_documents(documents=doc)


class QdrantVectorstore(Vectorstore):
    url: str = os.getenv("QDRANT_URL")  # TODO: maybe should be changed
    client: QdrantClient = QdrantClient(url)
    vectorstore: QdrantVectorStore = None

    def vectorstore_exists(self) -> bool:
        return any(collection.name == self.config.name for collection in self.client.get_collections().collections)

    def create_vectorstore(self, embeddings_model: EmbeddingConfig) -> QdrantVectorStore:
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

    def get_n_documents_in_vectorstore(self) -> int:
        return self.vectorstore.client.count(self.vectorstore.collection_name, exact=True).count

    def _delete_vectorstore(self) -> None:
        self.client.delete_collection(collection_name=self.config.name)

    def load_vectorstore(self, embeddings_model: EmbeddingConfig, retrieval_mode: str) -> QdrantVectorStore:
        retrieval_mode_dict = \
            {"dense": RetrievalMode.DENSE, "sparse": RetrievalMode.SPARSE, "hybrid": RetrievalMode.HYBRID}
        retrieval_mode = retrieval_mode_dict.get(self.config.retrieval_mode.value)
        qdrant_vectorstore = QdrantVectorStore.from_existing_collection(
            url=self.url,
            collection_name=self.config.name,
            embedding=embeddings_model.model,
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            vector_name="dense",
            sparse_vector_name="sparse",
            content_payload_key="page_content",
            metadata_payload_key="metadata",
            retrieval_mode=retrieval_mode,
        )
        self.vectorstore = qdrant_vectorstore
        return qdrant_vectorstore

    def delete_from_vectorstore(self, filename: pathlib.Path) -> int:
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
        return Vectorstore._add_to_vectorstore(self, doc)

    # search
    # metadata_search_filter = models.Filter(must=[models.FieldCondition(
    #     key="metadata.filename", match=models.MatchValue(value="data/mddocs/Acute_Pulmonary_Embolism_2019.md"))]
    # )
    # retriever = qdrant_vectorstore.as_retriever(
    #     search_type="similarity",  # similarity, mmr, similarity_score_threshold
    #     search_kwargs={
    #         "k": 3,  # amount of documents to return
    #         # "fetch_k": 25,  # amount of documents to pass to the mmr algorithm
    #         # "filter": metadata_search_filter
    #         # "score_threshold": 0.7  # only for search_type=similarity_score_threshold
    #         # "hybrid_fusion": FusionQuery(fusion=Fusion("rrf"))
    #     }
    # )


class FaissVectorstore(Vectorstore):
    vectorstore: FAISS = None

    def vectorstore_exists(self) -> bool:
        vectorstore_embedding_path = self.config.folder / (self.config.name + ".faiss")
        vectorstore_doc_path = self.config.folder / (self.config.name + ".pkl")
        return vectorstore_embedding_path.is_file() and vectorstore_doc_path.is_file()

    def create_vectorstore(self, embeddings_model: EmbeddingConfig, **kwargs) -> FAISS:
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

    def get_n_documents_in_vectorstore(self) -> int:
        return int(self.vectorstore.index.ntotal)

    def _delete_vectorstore(self) -> None:
        os.remove((pathlib.Path(self.config.folder) / (self.config.name + ".faiss")).as_posix())
        os.remove((pathlib.Path(self.config.folder) / (self.config.name + ".pkl")).as_posix())

    def _ensure_folder(self):
        self.config.folder.mkdir(parents=True, exist_ok=True)

    def load_vectorstore(self, embeddings_model: EmbeddingConfig, **kwargs) -> FAISS:
        faiss_vectorstore = FAISS.load_local(
            folder_path=self.config.folder.as_posix(),
            index_name=self.config.name,
            embeddings=embeddings_model.model,
            allow_dangerous_deserialization=True,
        )
        self.vectorstore = faiss_vectorstore
        return faiss_vectorstore

    def delete_from_vectorstore(self, filename: pathlib.Path) -> int:
        del_documents = [doc_id for doc_id, doc in self.vectorstore.docstore._dict.items()
                         if doc.metadata["filename"] == str(filename)]
        if len(del_documents):
            self.vectorstore.delete(ids=del_documents)
            self.vectorstore.save_local(folder_path=str(self.config.folder), index_name=self.config.name)
        return len(del_documents)

    def add_to_vectorstore(self, doc: Document | List[Document]) -> None:
        # Vectorstore.add_to_vectorstore(self, doc)
        docs = doc if isinstance(doc, list) else [doc]
        self.vectorstore.add_documents(docs)
        self.vectorstore.save_local(folder_path=self.config.folder.as_posix(), index_name=self.config.name)

    # search
    # retriever = faiss_vectorstore.as_retriever(
    #     search_type="similarity",  # similarity, mmr, similarity_score_threshold
    #     search_kwargs={
    #         "k": 3,  # amount of documents to return
    #         # "fetch_k": 25,  # amount of documents to pass to the mmr algorithm
    #         # "filter": metadata_search_filter
    #         "score_threshold": 0.7  # only for search_type=similarity_score_threshold
    #     }
    # )
    # search_result = retriever.invoke("ciao")
    # embeddings = retriever.vectorstore.embeddings  # TODO: this might be useful in the agentic pipeline


class IndexManager(metaclass=Singleton):
    def __init__(self, config: IndexingConfig, embeddings: EmbeddingConfig):
        self.logger = get_logger("Indexing based on LangChain VectorStores")
        self.config = config
        self.embeddings = embeddings
        self.vectorstore = (
            QdrantVectorstore(config=self.config)) if IndexTypeNames(self.config.type) == IndexTypeNames.qdrant \
            else FaissVectorstore(config=self.config)

    def create_index(self):
        try:
            self.vectorstore.create_vectorstore(embeddings_model=self.embeddings)
            assert self.vectorstore.vectorstore is not None
            self.logger.info(f"Index {self.config.name} created successfully.")
        except Exception as e:
            self.logger.info(f"Error creating index {self.config.name}: {str(e)}")
            raise

    def get_n_documents_in_vectorstore(self):
        return self.vectorstore.get_n_documents_in_vectorstore()

    def delete_index(self):
        try:
            self.vectorstore.delete_vectorstore()
            self.logger.info(f"Index {self.config.name} deleted successfully.")
        except Exception as e:
            self.logger.info(f"Error deleting index {self.config.name}: {str(e)}")
            raise

    def load_index(self):
        try:
            self.vectorstore.load_vectorstore(embeddings_model=self.embeddings,
                                              retrieval_mode=self.config.retrieval_mode.value)
            self.logger.info(f"Index {self.config.name} loaded successfully.")
        except Exception as e:
            self.logger.info(f"Error loading {self.config.name} index: {str(e)}")
            raise

    def delete_document(self, filename: pathlib.Path) -> int:
        try:
            n_doc_deleted = self.vectorstore.delete_from_vectorstore(filename)
            if n_doc_deleted > 0:
                self.logger.info(f"Document {filename} deleted successfully ({n_doc_deleted} chunks removed)")
            return n_doc_deleted
        except Exception as e:
            self.logger.info(f"Error deleting document {filename}: {str(e)}")
            raise

    def add_document(self, doc: Document | List[Document]):
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
