import itertools
import json
import os
import pathlib
from typing import Optional, Tuple, List

from src.config.manager import ETLConfigManager
from src.managers.markdown_conversion_manager import MarkdownConverter, DocumentMetadata
from src.document_processor import DocumentProcessor
from src.managers.chunking_manager import ChunkingManager
from src.managers.index_manager import IndexManager

from cardiology_gen_ai.utils.singleton import Singleton
from cardiology_gen_ai.utils.logger import get_logger


class ETLProcessor(metaclass=Singleton):
    def __init__(self, app_id: str):
        self.logger = get_logger("ETL Processor based on LangChain and PyMuPDF")
        self.app_id = app_id
        self.config = ETLConfigManager(app_id=app_id).config
        self.index_manager = IndexManager(config=self.config.indexing, embeddings=self.config.embeddings)
        self._initialize_index()
        self.markdown_converter = MarkdownConverter(config=self.config.preprocessing)
        self.chunking_manager = ChunkingManager(self.config.preprocessing.chunking_manager.splitter)

    def _initialize_index(self):
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

    def _save_docs_metadata(self, docs_metadata_list: List[DocumentMetadata]):
        docs_metadata_dict_list = \
            [doc_metadata.model_dump(mode="json", exclude_none=True) for doc_metadata in docs_metadata_list]
        with open(self.config.preprocessing.output_folder.folder / "documents_metadata.json", "w",
                  encoding="utf-8") as f:
            json.dump(docs_metadata_dict_list, f, ensure_ascii=False, indent=2)

    def update_documents_metadata(self, doc_metadata: DocumentMetadata, create_if_missing: bool = True):
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

    def perform_etl(self):
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
