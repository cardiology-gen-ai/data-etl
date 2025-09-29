Chunking Manager
================

This module defines a small abstraction over multiple text splitters (e.g.,
Markdown, recursive character/token, semantic, and sentence/token splitters),
plus a manager that runs them as a pipeline and annotates each produced
:langchain:`Document <core/documents/langchain_core.documents.base.Document.html>` with rich metadata.


.. rubric:: Notes

- The first splitter in the pipeline may be a Markdown header splitter (i.e., if a markdown splitter is required, it is placed as first splitter); subsequent splitters must implement ``split_documents``.
- When an embedding model is provided, chunk sizing is token‑aware via the Hugging Face tokenizer inferred from the embeddings object.


.. automodule:: src.managers.chunking_manager
   :members:
   :exclude-members:
   :member-order: bysource
