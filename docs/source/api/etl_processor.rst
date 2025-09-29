ETL Processor
=============

End-to-end ETL orchestrator (File → Markdown → chunks → vector index).

This module defines :class:`~src.etl_processor.ETLProcessor`, a high-level coordinator that wires
configuration loading, index initialization, document conversion/chunking, and catalog/metadata updates into a
single ETL pipeline built on :langchain:`LangChain <index.html>` and :pymupdf:`PyMuPDF <index.html>`.

.. rubric:: Notes

- The processor defers configuration to :class:`~src.config.manager.ETLConfigManager`, then initializes a concrete vector store via :class:`~src.managers.index_manager.IndexManager`.
- Per-file processing is handled by :class:`~src.document_processor.DocumentProcessor`.
- Documents metadata are saved to ``<output_folder>/documents_metadata.json``.

.. mermaid::
   :caption: ETL pipeline for a batch of files stored in a folder.

   flowchart LR
     subgraph ETL["ETLProcessor"]
       perform["perform_etl()"];
       process["process_file(filename)"];
       saveMeta["_save_docs_metadata()"];
     end

     subgraph Preproc["Preprocessing"]
       mc["MarkdownConverter"];
       im["ImageManager"];
       mm["MarkdownManager"];
       cm["ChunkingManager"];
     end

     subgraph Index["Indexing"]
       idx["IndexManager"];
       evs["EditableVectorstore"];
       qdr["Qdrant collection\n(dense+sparse)"];
       fai["FAISS index\n(.faiss/.pkl)"];
     end

     subgraph Storage["Storage"]
       inF[("(input_folder)")];
       outF[("(output_folder)")];
       md["&lt;doc&gt;.md"];
       img[/"&lt;doc&gt;_images/"/];
       meta["(documents_metadata.json)"];
     end

     perform --> inF --> process;
     process --> mc;
     mc --> im --> img;
     mc --> mm --> md;
     mc --> cm --> docs["List&lt;Document&gt; chunks"];
     docs["List&lt;Document&gt; chunks"] --> idx --> evs;
     evs --> qdr;
     evs --> fai;
     perform --> saveMeta --> meta;
     md -.-> outF;
     img -.-> outF;


.. autoclass:: src.etl_processor.ETLProcessor
   :members:
   :member-order: bysource
   :show-inheritance:

See also
--------

- :mod:`src.config.manager`
- :mod:`src.managers.index_manager`
- :class:`src.document_processor.DocumentProcessor`