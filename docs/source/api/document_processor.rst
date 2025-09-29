Document Processor
==================

High-level document processing pipeline: PDF → Markdown → chunks → index.

This module exposes :class:`~src.document_processor.DocumentProcessor`, a thin orchestrator that:

1. converts an input file to Markdown using :class:`~src.managers.markdown_conversion_manager.MarkdownConverter`,
2. splits the Markdown into chunks via :class:`~src.managers.chunking_manager.ChunkingManager`,
3. adds or removes those chunks from a vectorstore index managed by :class:`~src.managers.index_manager.IndexManager`.

.. mermaid::
    :caption: ETL pipeline for a single file.

     sequenceDiagram
     participant DP as DocumentProcessor
     participant MC as MarkdownConverter
     participant IM as ImageManager
     participant MM as MarkdownManager
     participant CM as ChunkingManager
     participant IDX as IndexManager
     participant VS as EditableVectorstore

     DP->>MC: __call__(filename)
     MC->>IM: load_images_catalog()/extract_visual_images()
     IM-->>MC: ImagesCatalog
     MC->>MM: find_page_anchors_in_markdown()
     MM-->>MC: anchors
     MC-->>DP: DocumentMetadata (md_filepath, ...)
     DP->>CM: __call__(md_filepath)
     CM-->>DP: List[Document] chunks
     DP->>IDX: add_document(chunks)
     IDX->>VS: add_to_vectorstore(chunks)
     VS-->>IDX: ok

.. autoclass:: src.document_processor.DocumentProcessor
   :members:
   :member-order: bysource
   :show-inheritance:

See also
--------

- :mod:`src.managers.markdown_conversion_manager`
- :mod:`src.managers.chunking_manager`
- :class:`src.managers.index_manager.IndexManager`