Index Manager
==============

Editable vector store backends (:langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>` / :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>`) with an indexing manager.

This module provides an abstract :class:`~src.managers.index_manager.EditableVectorstore` interface that adds
"edit" capabilities (create, delete index; add/remove documents) on top of a
:langchain:`LangChain VectorStore <core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore>`. Two concrete implementations are included:

- :class:`~src.managers.index_manager.EditableQdrantVectorstore`, backed by :langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>` (dense + sparse hybrid),
- :class:`~src.managers.index_manager.EditableFaissVectorstore`, backed by :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>`.


An :class:`~src.managers.index_manager.IndexManager` orchestrates lifecycle operations (create/load/delete
index, add/remove documents) using configuration objects and an embedding model.

.. rubric:: Notes

- ``EmbeddingConfig`` is expected to expose ``model`` (callable embedding function) and ``dim`` (int, embedding dimensionality).
- ``IndexingConfig`` should provide ``name``, ``type`` (e.g., ``qdrant`` or ``faiss``), ``folder`` for FAISS persistence, and optionally ``retrieval_mode``/``distance``.
- The Qdrant backend creates a sparse BM25 head via ``FastEmbedSparse('Qdrant/bm25')`` and sets payload keys to ``page_content`` (content) and ``metadata`` (metadata).

.. automodule:: src.managers.index_manager
   :members:
   :exclude-members:
   :member-order: bysource