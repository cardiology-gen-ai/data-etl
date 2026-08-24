from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import main as main_module


class MainManifestMultiSourceTests(unittest.TestCase):
    def test_manifest_records_multiple_sources_and_total_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            source_a = root / "cardiomyopathies.json"
            source_a.write_text(
                json.dumps(
                    {
                        "version": "fixed_within_section_v2",
                        "doc_id": "Cardiomyopathies_2023",
                        "chunks": [],
                    }
                ),
                encoding="utf-8",
            )

            source_b = root / "cardio_oncology.json"
            source_b.write_text(
                json.dumps(
                    {
                        "version": "fixed_within_section_v2",
                        "doc_id": "Cardio_Oncology_2022",
                        "chunks": [],
                    }
                ),
                encoding="utf-8",
            )

            config_path = root / "config_rag.json"
            config_path.write_text(
                '{"app": {}}',
                encoding="utf-8",
            )

            index_folder = root / "index"
            index_folder.mkdir()

            processor = MagicMock()
            processor.app_id = "prototype"
            processor.config.indexing.name = "prototype_index"
            processor.config.indexing.folder = index_folder
            processor.config.indexing.type.value = "faiss"
            processor.config.indexing.distance.value = "cosine"

            processor.config.embeddings.model_name = (
                "text-embedding-3-small"
            )
            processor.config.embeddings.dim = 1536

            processor.index_manager.get_n_documents_in_vectorstore.return_value = (
                30
            )

            sources = [
                {
                    "source_path": source_a,
                    "source_type": "fixed_chunks",
                    "source_key": (
                        "Cardiomyopathies_2023::fixed_chunks"
                    ),
                    "artifact": {
                        "doc_id": "Cardiomyopathies_2023",
                        "strategy": "fixed_within_section",
                    },
                    "indexed_chunk_count": 12,
                    "expected_chunk_count": 12,
                },
                {
                    "source_path": source_b,
                    "source_type": "fixed_chunks",
                    "source_key": (
                        "Cardio_Oncology_2022::fixed_chunks"
                    ),
                    "artifact": {
                        "doc_id": "Cardio_Oncology_2022",
                        "strategy": "fixed_within_section",
                    },
                    "indexed_chunk_count": 18,
                    "expected_chunk_count": 18,
                },
            ]

            with (
                patch.object(
                    main_module,
                    "_package_versions",
                    return_value={"langchain": "test-version"},
                ),
                patch.object(
                    main_module,
                    "_git_provenance",
                    return_value={
                        "commit": "abc123",
                        "dirty": False,
                    },
                ),
                patch.object(
                    main_module,
                    "_index_output_files",
                    return_value=[],
                ),
            ):
                output = main_module._write_manifest(
                    processor=processor,
                    config_path=config_path,
                    mode="configured",
                    sources=sources,
                )

            payload = json.loads(
                output.read_text(encoding="utf-8")
            )

            self.assertEqual(
                payload["schema_version"],
                "rag_index_build_v3",
            )
            self.assertEqual(
                payload["mode"],
                "configured",
            )

            self.assertEqual(
                len(payload["sources"]),
                2,
            )

            # The singular compatibility alias is intentionally reserved
            # for prebuilt mode with exactly one source.
            self.assertNotIn("source", payload)

            self.assertEqual(
                payload["indexed_chunk_count"],
                30,
            )
            self.assertEqual(
                payload["stored_vector_count"],
                30,
            )

            by_key = {
                source["source_key"]: source
                for source in payload["sources"]
            }

            cm = by_key[
                "Cardiomyopathies_2023::fixed_chunks"
            ]
            oncology = by_key[
                "Cardio_Oncology_2022::fixed_chunks"
            ]

            self.assertEqual(
                cm["indexed_chunk_count"],
                12,
            )
            self.assertEqual(
                cm["expected_chunk_count"],
                12,
            )
            self.assertEqual(
                Path(cm["path"]).name,
                "cardiomyopathies.json",
            )
            self.assertEqual(
                cm["artifact"]["doc_id"],
                "Cardiomyopathies_2023",
            )

            self.assertEqual(
                oncology["indexed_chunk_count"],
                18,
            )
            self.assertEqual(
                oncology["expected_chunk_count"],
                18,
            )
            self.assertEqual(
                Path(oncology["path"]).name,
                "cardio_oncology.json",
            )
            self.assertEqual(
                oncology["artifact"]["doc_id"],
                "Cardio_Oncology_2022",
            )

            # Every source and the config must carry reproducibility hashes.
            for source in payload["sources"]:
                self.assertEqual(
                    len(source["sha256"]),
                    64,
                )

            self.assertEqual(
                len(payload["config"]["sha256"]),
                64,
            )

            self.assertEqual(
                payload["git"],
                {
                    "commit": "abc123",
                    "dirty": False,
                },
            )
            self.assertEqual(
                payload["runtime"]["packages"],
                {
                    "langchain": "test-version",
                },
            )

            self.assertEqual(
                payload["embeddings"]["model"],
                "text-embedding-3-small",
            )
            self.assertEqual(
                payload["embeddings"]["dimensions"],
                1536,
            )


if __name__ == "__main__":
    unittest.main()
