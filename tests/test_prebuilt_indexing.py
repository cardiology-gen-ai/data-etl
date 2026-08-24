import pathlib
import tempfile
import unittest
from uuid import UUID

from langchain_core.documents import Document

from src.managers.chunking_manager import PrebuiltChunkSource
from src.managers.index_manager import IndexManager
from src.etl_processor import ETLProcessor


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _Vectorstore:
    def __init__(self):
        self.deleted = []
        self.add_calls = []

    def delete_from_vectorstore(self, source_value, metadata_key="filename"):
        self.deleted.append((metadata_key, str(source_value)))
        return 0

    def add_to_vectorstore(self, documents, ids=None):
        self.add_calls.append((list(documents), None if ids is None else list(ids)))


class _ChunkingManager:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def load_prebuilt_chunks(self, path, source_type):
        self.calls.append((pathlib.Path(path), source_type))
        return list(self.documents)


class _IndexManager:
    def __init__(self):
        self.calls = []

    def add_document(self, documents):
        self.calls.append(list(documents))


class IndexManagerPrebuiltTests(unittest.TestCase):
    def _manager(self):
        manager = object.__new__(IndexManager)
        manager.logger = _Logger()
        manager.vectorstore = _Vectorstore()
        return manager

    @staticmethod
    def _doc(filename, record_id=None, source_key=None):
        metadata = {"filename": filename}
        if record_id is not None:
            metadata["record_id"] = record_id
        if source_key is not None:
            metadata["source_key"] = source_key
        return Document(page_content="Title: T\n\nBody:\nB", metadata=metadata)

    def test_prebuilt_ids_are_deterministic_backend_safe_uuids(self):
        manager = self._manager()
        docs = [
            self._doc("artifact.json", "record-a", "Doc::fixed_chunks"),
            self._doc("artifact.json", "record-b", "Doc::fixed_chunks"),
        ]

        manager.add_document(docs)
        _, first_ids = manager.vectorstore.add_calls[-1]
        manager.add_document(docs)
        _, second_ids = manager.vectorstore.add_calls[-1]

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), 2)
        self.assertEqual(len(set(first_ids)), 2)
        for value in first_ids:
            UUID(value)

    def test_each_stable_source_is_deleted_only_once(self):
        manager = self._manager()
        docs = [
            self._doc("old/a.json", "a", "DocA::fixed_chunks"),
            self._doc("old/a.json", "b", "DocA::fixed_chunks"),
            self._doc("old/b.json", "c", "DocB::fixed_chunks"),
        ]

        manager.add_document(docs)

        self.assertEqual(
            manager.vectorstore.deleted,
            [
                ("source_key", "DocA::fixed_chunks"),
                ("source_key", "DocB::fixed_chunks"),
            ],
        )
        self.assertEqual(len(manager.vectorstore.add_calls), 1)

    def test_prebuilt_replacement_is_independent_of_filename(self):
        manager = self._manager()
        first = [
            self._doc("machine-a/artifact.json", "a", "Doc::fixed_chunks")
        ]
        moved = [
            self._doc("machine-b/artifact.json", "a", "Doc::fixed_chunks")
        ]

        manager.add_document(first)
        manager.add_document(moved)

        self.assertEqual(
            manager.vectorstore.deleted,
            [
                ("source_key", "Doc::fixed_chunks"),
                ("source_key", "Doc::fixed_chunks"),
            ],
        )

    def test_legacy_documents_keep_filename_replacement_and_generated_ids(self):
        manager = self._manager()
        docs = [self._doc("legacy.md"), self._doc("legacy.md")]

        manager.add_document(docs)

        _, ids = manager.vectorstore.add_calls[-1]
        self.assertIsNone(ids)
        self.assertEqual(manager.vectorstore.deleted, [("filename", "legacy.md")])

    def test_duplicate_record_ids_are_rejected_before_indexing(self):
        manager = self._manager()
        docs = [
            self._doc("artifact.json", "duplicate", "Doc::fixed_chunks"),
            self._doc("artifact.json", "duplicate", "Doc::fixed_chunks"),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            manager.add_document(docs)

        self.assertEqual(manager.vectorstore.deleted, [])
        self.assertEqual(manager.vectorstore.add_calls, [])

    def test_mixed_legacy_and_prebuilt_ids_are_rejected(self):
        manager = self._manager()
        docs = [
            self._doc("artifact.json", "record-a", "Doc::fixed_chunks"),
            self._doc("artifact.json"),
        ]

        with self.assertRaisesRegex(ValueError, "cannot mix"):
            manager.add_document(docs)

    def test_mixed_source_key_presence_is_rejected(self):
        manager = self._manager()
        docs = [
            self._doc("artifact.json", "record-a", "Doc::fixed_chunks"),
            self._doc("artifact.json", "record-b"),
        ]

        with self.assertRaisesRegex(ValueError, "source_key"):
            manager.add_document(docs)

    def test_empty_batch_is_rejected(self):
        manager = self._manager()
        with self.assertRaisesRegex(ValueError, "empty"):
            manager.add_document([])

    def test_missing_filename_is_rejected(self):
        manager = self._manager()
        doc = Document(
            page_content="body",
            metadata={"record_id": "a", "source_key": "Doc::fixed_chunks"},
        )
        with self.assertRaisesRegex(ValueError, "filename"):
            manager.add_document([doc])


class ETLProcessorPrebuiltTests(unittest.TestCase):
    def test_prebuilt_path_bypasses_conversion_and_indexes_loaded_documents(self):
        documents = [
            Document(
                page_content="Title: T\n\nBody:\nB",
                metadata={
                    "filename": "artifact.json",
                    "record_id": "a",
                    "source_key": "Doc::fixed_chunks",
                },
            )
        ]
        processor = object.__new__(ETLProcessor)
        processor.logger = _Logger()
        processor.chunking_manager = _ChunkingManager(documents)
        processor.index_manager = _IndexManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "artifact.json"
            path.write_text("{}", encoding="utf-8")
            count = processor.process_prebuilt_chunks(
                path,
                PrebuiltChunkSource.fixed_chunks,
            )

        self.assertEqual(count, 1)
        self.assertEqual(
            processor.chunking_manager.calls[0][1],
            PrebuiltChunkSource.fixed_chunks,
        )
        self.assertEqual(processor.index_manager.calls, [documents])

    def test_non_json_prebuilt_source_is_rejected(self):
        processor = object.__new__(ETLProcessor)
        processor.logger = _Logger()
        processor.chunking_manager = _ChunkingManager([])
        processor.index_manager = _IndexManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "artifact.txt"
            path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be JSON"):
                processor.process_prebuilt_chunks(
                    path,
                    PrebuiltChunkSource.fixed_chunks,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
