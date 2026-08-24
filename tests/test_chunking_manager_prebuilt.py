import json
import pathlib
import tempfile
import unittest

from langchain_core.documents import Document

from managers.chunking_manager import ChunkingManager, PrebuiltChunkSource


def make_manager(splitter_list=None):
    """Bypass the project Singleton so each unit test is isolated."""
    manager = object.__new__(ChunkingManager)
    manager.splitter_list = list(splitter_list or [])
    return manager


class _LegacySplitter:
    def split_text(self, text):
        return [Document(page_content=text, metadata={})]


class _LegacyConfig:
    splitter = _LegacySplitter()
    embeddings = None
    header_levels = 0


class ChunkingManagerPrebuiltTests(unittest.TestCase):
    def _write_json(self, folder, filename, payload):
        path = pathlib.Path(folder) / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_legacy_split_text_behavior_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "legacy.md"
            path.write_text("legacy text", encoding="utf-8")
            manager = make_manager([_LegacyConfig()])

            documents = manager(path)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, "legacy text")
            self.assertEqual(
                documents[0].metadata,
                {
                    "filename": str(path),
                    "chunk_idx": 0,
                    "headers": {},
                    "n_tokens": 0,
                },
            )

    def test_fixed_chunks_are_loaded_with_common_embedding_text(self):
        payload = {
            "version": "fixed_within_section_v2",
            "strategy": "fixed_within_section",
            "chunks": [
                {
                    "fixed_chunk_id": "Doc::fixed::6.8.2::0000",
                    "chunk_id": "Doc::fixed::6.8.2::0000",
                    "doc_id": "Doc",
                    "strategy": "fixed_within_section",
                    "text": "Genetic testing body.",
                    "source_section_id": "6.8.2",
                    "source_section_ids": ["6.8.2"],
                    "source_chunk_id": "Doc:6.8.2:0",
                    "source_chunk_ids": ["Doc:6.8.2:0"],
                    "section_title": "Genetic testing",
                    "section_level": 3,
                    "parent_section_id": "6.8",
                    "page_start": 27,
                    "page_end": 32,
                    "fixed_part_index": 0,
                    "fixed_part_count": 1,
                    "chunk_size": 2000,
                    "chunk_overlap": 300,
                    "contains_table": False,
                    "oversized_atomic_table": False,
                    "excluded": False,
                    "embed": True,
                },
                {
                    "fixed_chunk_id": "Doc::fixed::excluded",
                    "doc_id": "Doc",
                    "text": "must not be indexed",
                    "source_section_ids": ["x"],
                    "source_chunk_ids": ["x"],
                    "excluded": True,
                    "embed": True,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "fixed.json", payload)
            documents = make_manager().load_prebuilt_chunks(
                path, PrebuiltChunkSource.fixed_chunks
            )

        self.assertEqual(len(documents), 1)
        document = documents[0]
        self.assertEqual(
            document.page_content,
            "Title: Genetic testing\n\nBody:\nGenetic testing body.",
        )
        self.assertEqual(document.metadata["filename"], str(path))
        self.assertEqual(document.metadata["source_key"], "Doc::fixed_chunks")
        self.assertEqual(document.metadata["record_id"], "Doc::fixed::6.8.2::0000")
        self.assertEqual(document.metadata["source_section_ids"], ["6.8.2"])
        self.assertEqual(document.metadata["source_chunk_ids"], ["Doc:6.8.2:0"])
        self.assertEqual(document.metadata["prebuilt_source_type"], "fixed_chunks")
        self.assertEqual(document.metadata["embedding_text_template"], "title_body_v1")

    def test_hierarchical_loader_excludes_structural_records(self):
        payload = [
            {
                "chunk_id": "Doc:6.8:0",
                "doc_id": "Doc",
                "section_id": "6.8",
                "section_title": "Genetics",
                "section_level": 2,
                "text": "",
                "is_empty": True,
                "excluded": False,
                "embed": False,
                "retrieval_unit_id": None,
                "retrieval_strategy": "max_level_4",
                "section_view_role": "structural",
                "source_section_ids": [],
                "source_chunk_ids": [],
            },
            {
                "chunk_id": "Doc:6.8.2:0",
                "doc_id": "Doc",
                "section_id": "6.8.2",
                "section_title": "Genetic testing",
                "section_level": 3,
                "parent_section_id": "6.8",
                "text": "Hierarchical retrieval body.",
                "is_empty": False,
                "excluded": False,
                "embed": True,
                "retrieval_unit_id": "Doc:6.8.2:0::retrieval::max_level_4",
                "retrieval_strategy": "max_level_4",
                "aggregation_max_level": 4,
                "is_aggregated": False,
                "section_view_role": "retrieval",
                "source_section_ids": ["6.8.2"],
                "source_chunk_ids": ["Doc:6.8.2:0"],
                "represented_section_ids": ["6.8.2"],
                "page_start": 27,
                "page_end": 32,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "section_view.json", payload)
            documents = make_manager().load_prebuilt_chunks(
                path, "hierarchical_section_view"
            )

        self.assertEqual(len(documents), 1)
        document = documents[0]
        self.assertEqual(
            document.metadata["retrieval_unit_id"],
            "Doc:6.8.2:0::retrieval::max_level_4",
        )
        self.assertEqual(document.metadata["section_view_role"], "retrieval")
        self.assertEqual(
            document.metadata["source_key"],
            "Doc::hierarchical_section_view",
        )
        self.assertEqual(document.metadata["aggregation_max_level"], 4)
        self.assertEqual(document.metadata["source_section_ids"], ["6.8.2"])

    def test_duplicate_stable_ids_are_rejected(self):
        record = {
            "fixed_chunk_id": "duplicate",
            "doc_id": "Doc",
            "text": "body",
            "source_section_ids": ["1"],
            "source_chunk_ids": ["Doc:1:0"],
            "excluded": False,
            "embed": True,
        }
        payload = {"chunks": [record, dict(record)]}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "fixed.json", payload)
            with self.assertRaisesRegex(ValueError, "Duplicate prebuilt record"):
                make_manager().load_prebuilt_chunks(path, "fixed_chunks")

    def test_mixed_document_ids_are_rejected(self):
        payload = {
            "chunks": [
                {
                    "fixed_chunk_id": "a",
                    "doc_id": "DocA",
                    "text": "body a",
                    "source_section_ids": ["1"],
                    "source_chunk_ids": ["DocA:1:0"],
                    "excluded": False,
                    "embed": True,
                },
                {
                    "fixed_chunk_id": "b",
                    "doc_id": "DocB",
                    "text": "body b",
                    "source_section_ids": ["2"],
                    "source_chunk_ids": ["DocB:2:0"],
                    "excluded": False,
                    "embed": True,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "fixed.json", payload)
            with self.assertRaisesRegex(ValueError, "exactly one doc_id"):
                make_manager().load_prebuilt_chunks(path, "fixed_chunks")

    def test_unknown_section_view_role_is_rejected(self):
        payload = [
            {
                "chunk_id": "Doc:1:0",
                "doc_id": "Doc",
                "section_id": "1",
                "section_title": "Section",
                "text": "body",
                "section_view_role": "unexpected",
                "source_section_ids": ["1"],
                "source_chunk_ids": ["Doc:1:0"],
                "excluded": False,
                "embed": True,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "section_view.json", payload)
            with self.assertRaisesRegex(ValueError, "invalid section_view_role"):
                make_manager().load_prebuilt_chunks(
                    path, "hierarchical_section_view"
                )

    def test_empty_retrieval_artifact_is_rejected(self):
        payload = [
            {
                "chunk_id": "Doc:1:0",
                "doc_id": "Doc",
                "section_id": "1",
                "section_title": "Section",
                "text": "",
                "section_view_role": "structural",
                "excluded": False,
                "embed": False,
                "is_empty": True,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, "section_view.json", payload)
            with self.assertRaisesRegex(ValueError, "No retrievable records"):
                make_manager().load_prebuilt_chunks(
                    path, "hierarchical_section_view"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
