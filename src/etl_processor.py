import itertools
import json
import logging
import os
import pathlib
from typing import Optional, Tuple, List

from src.config.manager import ETLConfigManager, ETLConfig
from src.managers.markdown_conversion_manager import MarkdownConverter, DocumentMetadata
from src.document_processor import DocumentProcessor
from src.managers.chunking_manager import (
    ChunkingManager,
    PrebuiltChunkSource,
)
from src.managers.index_manager import IndexManager
from cardiology_gen_ai.utils.singleton import Singleton
from cardiology_gen_ai.utils.logger import get_logger


class ETLProcessor(metaclass=Singleton):
    """Coordinate the full ETL pipeline for a given application ID."""

    logger: logging.Logger
    app_id: str
    config: ETLConfig
    index_manager: IndexManager
    markdown_converter: MarkdownConverter
    chunking_manager: ChunkingManager

    def __init__(self, app_id: str):
        self.logger = get_logger("ETL Processor based on LangChain and PyMuPDF")
        self.app_id = app_id
        self.config = ETLConfigManager(app_id=app_id).config
        self.index_manager = IndexManager(
            config=self.config.indexing,
            embeddings=self.config.embeddings,
        )
        self._initialize_index()
        self.markdown_converter = MarkdownConverter(config=self.config.preprocessing)
        self.chunking_manager = ChunkingManager(
            self.config.preprocessing.chunking_manager.splitter
        )

    def _initialize_index(self):
        self.logger.info(f"Initializing {self.index_manager.config.name} index")
        try:
            if self.index_manager.vectorstore.vectorstore_exists():
                self.logger.info(
                    f"Index {self.index_manager.config.name} already exists, loading it."
                )
                self.index_manager.vectorstore.load_vectorstore(
                    embeddings_model=self.config.embeddings,
                    retrieval_mode=self.config.indexing.retrieval_mode,
                )
                self.logger.info(
                    f"Index {self.index_manager.config.name} loaded successfully."
                )
            else:
                self.logger.info(
                    f"Index {self.index_manager.config.name} does not exist, creating it."
                )
                self.index_manager.vectorstore.create_vectorstore(
                    self.config.embeddings
                )
                self.logger.info(
                    f"Index {self.index_manager.config.name} created successfully."
                )
        except Exception as exc:
            self.logger.info(
                f"Error initializing {self.index_manager.config.name} index: {exc}"
            )
            raise

    def process_file(
        self,
        filename: str,
        filepath: Optional[pathlib.Path] = None,
        md_filepath: Optional[pathlib.Path] = None,
        force_md_conv: bool = True,
        existing_metadata_path: str | None = None,
    ) -> Tuple[bool, DocumentMetadata | None]:
        self.logger.info(f"Processing file: {filename} for ETL.")
        if filepath is None:
            filepath = self.config.preprocessing.input_folder.folder / filename
        try:
            document_processor = DocumentProcessor(
                filename=filename,
                markdown_converter=self.markdown_converter,
                chunking_manager=self.chunking_manager,
                index_manager=self.index_manager,
                filepath=filepath,
                md_filepath=md_filepath,
            )
            supported_extensions = document_processor.detect_file_extension()
            allowed_extensions = (
                self.config.preprocessing.input_folder.allowed_extensions
            )
            if (
                not supported_extensions
                or document_processor.file_extension not in allowed_extensions
            ):
                self.logger.info(
                    f"File extension {document_processor.file_extension} not processable."
                )
                return False, None
            doc_metadata = document_processor.process_document(
                force_md_conv=force_md_conv,
                existing_metadata_path=existing_metadata_path,
            )
            return True, doc_metadata
        except Exception as exc:
            self.logger.info(f"Error processing {filename}: {exc}")
            return False, None

    def process_prebuilt_chunks(
        self,
        source_path: pathlib.Path | str,
        source_type: PrebuiltChunkSource | str,
    ) -> int:
        """Load and index one already-built retrieval artifact.

        This is an explicit alternative entry point to ``process_file``. It
        bypasses PDF conversion, Markdown generation and legacy splitting, and
        therefore cannot alter the upstream MinerU/KG pipeline.
        """
        path = pathlib.Path(source_path)
        if path.suffix.lower() != ".json":
            raise ValueError(f"Prebuilt chunk source must be JSON: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)

        normalized_source_type = PrebuiltChunkSource(source_type)
        documents = self.chunking_manager.load_prebuilt_chunks(
            path,
            normalized_source_type,
        )
        self.index_manager.add_document(documents)

        max_chars = max(len(document.page_content) for document in documents)
        self.logger.info(
            "Prebuilt artifact indexed | source=%s | type=%s | chunks=%d | "
            "max_embedding_chars=%d",
            path,
            normalized_source_type.value,
            len(documents),
            max_chars,
        )
        return len(documents)

    def _save_docs_metadata(
        self,
        docs_metadata_list: List[DocumentMetadata],
    ) -> None:
        embedding_name = self.config.embeddings.model_name.replace("/", "_")
        metadata_filename = f"documents_metadata_{embedding_name}.json"
        output_dir = self.config.preprocessing.output_folder.folder
        output_path = output_dir / metadata_filename
        docs_metadata_dict_list = [
            doc_metadata.model_dump(mode="json", exclude_none=True)
            for doc_metadata in docs_metadata_list
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                docs_metadata_dict_list,
                handle,
                ensure_ascii=False,
                indent=2,
            )
        self.logger.info(f"Saved metadata to: {output_path}")

    def update_documents_metadata(
        self,
        doc_metadata: DocumentMetadata,
        create_if_missing: bool = True,
    ) -> None:
        embedding_name = self.config.embeddings.model_name.replace("/", "_")
        metadata_filename = f"documents_metadata{embedding_name}.json"
        docs_metadata_path = (
            self.config.preprocessing.output_folder.folder / metadata_filename
        )
        if docs_metadata_path.exists():
            docs_metadata_dict_list = json.loads(
                docs_metadata_path.read_text(encoding="utf-8")
            )
            docs_metadata_list = [
                DocumentMetadata(**doc_metadata_dict)
                for doc_metadata_dict in docs_metadata_dict_list
            ]
            other_docs_metadata = [
                prev_doc_metadata
                for prev_doc_metadata in docs_metadata_list
                if prev_doc_metadata.filename != doc_metadata.filename
            ]
            updated_docs_metadata_list = other_docs_metadata + [doc_metadata]
            self._save_docs_metadata(updated_docs_metadata_list)
        elif create_if_missing:
            self._save_docs_metadata([doc_metadata])

    def perform_etl(
        self,
        force_md_conv: bool = True,
        existing_metadata_path: str | None = None,
    ) -> None:
        self.logger.info("Starting ETL process.")
        self.logger.info(
            "Directory containing input files: "
            f"{self.config.preprocessing.input_folder.folder}."
        )
        input_files = list(
            itertools.chain.from_iterable(
                [
                    [
                        filename
                        for filename in os.listdir(
                            self.config.preprocessing.input_folder.folder.as_posix()
                        )
                        if filename.lower().endswith(allowed_extension)
                    ]
                    for allowed_extension in (
                        self.config.preprocessing.input_folder.allowed_extensions
                    )
                ]
            )
        )
        conversion_status_list, doc_metadata_list = [], []
        try:
            for filename in input_files:
                conversion_status, doc_metadata = self.process_file(
                    filename,
                    force_md_conv=force_md_conv,
                    existing_metadata_path=existing_metadata_path,
                )
                self.logger.info(
                    f"File {filename} processed with status: {conversion_status}."
                )
                conversion_status_list.append(conversion_status)
                doc_metadata_list.append(doc_metadata)
            self.logger.info(
                f"Successfully processed: {sum(conversion_status_list)} PDF(s)"
            )
            self.logger.info(
                "Parsing failed on "
                f"{len(conversion_status_list) - sum(conversion_status_list)} PDF(s)"
            )
            self._save_docs_metadata(doc_metadata_list)
            self.logger.info(
                "Directory containing Markdown files: "
                f"{self.config.preprocessing.output_folder.folder}"
            )
        except Exception as exc:
            self.logger.error(f"Error performing ETL: {exc}")
            raise
