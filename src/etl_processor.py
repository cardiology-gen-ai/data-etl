import itertools
import json
import logging
import os
import pathlib
from typing import Optional, Tuple, List

from src.config.manager import ETLConfigManager, ETLConfig
from src.managers.markdown_conversion_manager import MarkdownConverter, DocumentMetadata
from src.document_processor import DocumentProcessor
from src.managers.chunking_manager import ChunkingManager
from src.managers.index_manager import IndexManager

from cardiology_gen_ai.utils.singleton import Singleton
from cardiology_gen_ai.utils.logger import get_logger


class ETLProcessor(metaclass=Singleton):
    """Coordinate the full ETL pipeline for a given application ID.

    Parameters
    ----------
    app_id : str
        Logical application identifier used by :class:`~src.config.manager.ETLConfigManager`
        to load the appropriate configuration set.
    """
    logger: logging.Logger #: logging.Logger : Named logger ("ETL Processor based on LangChain and PyMuPDF").
    app_id: str #: str : Application identifier.
    config: ETLConfig #: ETLConfig : Loaded and validated configuration.
    index_manager: IndexManager #: :class:`~src.managers.index_manager.IndexManager` : Backend-agnostic manager for vector indexes (Qdrant/FAISS).
    markdown_converter: MarkdownConverter #: :class:`~src.managers.markdown_conversion_manager.MarkdownConverter` : Component that converts PDFs to Markdown and places images.
    chunking_manager: ChunkingManager #: :class:`~src.managers.chunking_manager.ChunkingManager` : Component that splits Markdown into :langchain_core:`Document <documents/langchain_core.documents.base.Document.html>` chunks.

    def __init__(self, app_id: str):
        self.logger = get_logger("ETL Processor based on LangChain and PyMuPDF")
        self.app_id = app_id
        self.config = ETLConfigManager(app_id=app_id).config
        self.index_manager = IndexManager(config=self.config.indexing, embeddings=self.config.embeddings)
        self._initialize_index()
        self.markdown_converter = MarkdownConverter(config=self.config.preprocessing)
        self.chunking_manager = ChunkingManager(self.config.preprocessing.chunking_manager.splitter)

    def _initialize_index(self):
        """Create or load the target index, depending on its current state.

        .. rubric:: Behavior

        - If the vectorstore already exists, load it with the configured retrieval mode.
        - Otherwise, create a new vectorstore based on ``self.config.indexing``.
        """
        self.logger.info(f"Initializing {self.index_manager.config.name} index")
        try:
            if self.index_manager.vectorstore.vectorstore_exists():
                self.logger.info(f"Index {self.index_manager.config.name} already exists, loading it.")
                self.index_manager.vectorstore.load_vectorstore(
                    embeddings_model=self.config.embeddings, retrieval_mode=self.config.indexing.retrieval_mode)
                self.logger.info(f"Index {self.index_manager.config.name} loaded successfully.")
            else:
                self.logger.info(f"Index {self.index_manager.config.name} does not exist, creating it.")
                self.index_manager.vectorstore.create_vectorstore(self.config.embeddings)
                self.logger.info(f"Index {self.index_manager.config.name} created successfully.")
        except Exception as e:
            self.logger.info(f"Error initializing {self.index_manager.config.name} index: {str(e)}")
            raise

    def process_file(self, filename: str, filepath: Optional[pathlib.Path] = None,
                 md_filepath: Optional[pathlib.Path] = None) -> Tuple[bool, DocumentMetadata | None]:
        """Run the ETL pipeline on a single file.

        .. rubric:: Steps

        1. Resolve ``filepath`` if not provided (from ``config.preprocessing.input_folder.folder``).
        2. Instantiate a :class:`~src.document_processor.DocumentProcessor` and validate the file extension against the allowed set.
        3. Convert → chunk → index via :meth:`~src.document_processor.DocumentProcessor.process_document`.
        4. Return a success flag and the resulting :class:`~src.managers.markdown_conversion_manager.DocumentMetadata` (if any).

        Parameters
        ----------
        filename : str
            Input file name to process.
        filepath : pathlib.Path, optional
            Optional explicit path to the input file.
        md_filepath : pathlib.Path, optional
            Optional explicit path for the Markdown file.

        Returns
        -------
        Tuple[bool, :class:`~src.managers.markdown_conversion_manager.DocumentMetadata` | None]
        """
        self.logger.info(f"Processing file: {filename} for ETL.")
        if filepath is None:
            filepath = self.config.preprocessing.input_folder.folder / filename
        try:
            document_processor = DocumentProcessor(
                filename = filename,
                markdown_converter = self.markdown_converter,
                chunking_manager = self.chunking_manager,
                index_manager = self.index_manager,
                filepath = filepath,
                md_filepath = md_filepath
            )
            supported_extensions = document_processor.detect_file_extension()
            allowed_extensions = self.config.preprocessing.input_folder.allowed_extensions
            if not supported_extensions or document_processor.file_extension not in allowed_extensions:
                self.logger.info(f"File extension {document_processor.file_extension} not processable.")
                return False, None
            doc_metadata: DocumentMetadata = document_processor.process_document()
            return True, doc_metadata
        except Exception as e:
            self.logger.info(f"Error processing {filename}: {e}")
            return False, None

    def _save_docs_metadata(self, docs_metadata_list: List[DocumentMetadata]) -> None:
        """Persist a list of :class:`~src.managers.markdown_conversion_manager.DocumentMetadata` to ``documents_metadata.json``.

        The file is written under ``config.preprocessing.output_folder.folder``.

        Parameters
        ----------
        docs_metadata_list : list[:class:`~src.managers.markdown_conversion_manager.DocumentMetadata`]
            List of metadata objects to serialize
        """
        docs_metadata_dict_list = \
            [doc_metadata.model_dump(mode="json", exclude_none=True) for doc_metadata in docs_metadata_list]
        with open(self.config.preprocessing.output_folder.folder / "documents_metadata.json", "w",
                  encoding="utf-8") as f:
            json.dump(docs_metadata_dict_list, f, ensure_ascii=False, indent=2)

    def update_documents_metadata(self, doc_metadata: DocumentMetadata, create_if_missing: bool = True) -> None:
        """Upsert a single document's metadata into ``documents_metadata.json``.

        If the file exists, the entry for ``doc_metadata.filename`` is replaced; if
        absent and ``create_if_missing`` is ``True``, a new file containing the single entry is created.

        Parameters
        ----------
        doc_metadata : :class:`~src.managers.markdown_conversion_manager.DocumentMetadata`
            Metadata to upsert.
        create_if_missing : bool
            Create the metadata file when missing.
        """
        docs_metadata_path = self.config.preprocessing.output_folder.folder / "documents_metadata.json"
        if docs_metadata_path.exists():
            docs_metadata_dict_list = json.loads(docs_metadata_path.read_text(encoding="utf-8"))
            docs_metadata_list = \
                [DocumentMetadata(**doc_metadata_dict) for doc_metadata_dict in docs_metadata_dict_list]
            other_docs_metadata = [prev_doc_metadata for prev_doc_metadata in docs_metadata_list
                                   if prev_doc_metadata.filename != doc_metadata.filename]
            updated_docs_metadata_list = other_docs_metadata + [doc_metadata]
            self._save_docs_metadata(updated_docs_metadata_list)
        elif create_if_missing:
            updated_docs_metadata_list = [doc_metadata]
            self._save_docs_metadata(updated_docs_metadata_list)

    def perform_etl(self) -> None:
        """Process all allowed files in the configured input folder.

        The method scans ``input_folder.folder`` for files whose extensions match
        ``allowed_extensions``, runs :meth:`process_file` for each, logs aggregate
        statistics, persists the collected metadata, and logs the Markdown output
        directory on success.

        Raises
        ------
        Exception
            Re-raised for fatal pipeline errors after logging.
        """
        self.logger.info("Starting ETL process.")
        self.logger.info(f"Directory containing input files: {self.config.preprocessing.input_folder.folder}.")
        input_files = list(itertools.chain.from_iterable(
            [[f for f in os.listdir(self.config.preprocessing.input_folder.folder.as_posix())
              if f.lower().endswith(allowed_extension)]
             for allowed_extension in self.config.preprocessing.input_folder.allowed_extensions]))
        conversion_status_list, doc_metadata_list = [], []
        try:
            for filename in input_files:
                conversion_status, doc_metadata = self.process_file(filename)
                conversion_status_list.append(conversion_status)
                doc_metadata_list.append(doc_metadata)
            self.logger.info(f"Successfully processed: {sum(conversion_status_list)} PDF(s)")
            self.logger.info(f"Parsing failed on {len(conversion_status_list) - sum(conversion_status_list)} PDF(s)")
            self._save_docs_metadata(doc_metadata_list)
            self.logger.info(f"Directory containing Markdown files: {self.config.preprocessing.output_folder.folder}")
        except Exception as e:
            self.logger.error(f"Error performing ETL: {e}")
            raise
