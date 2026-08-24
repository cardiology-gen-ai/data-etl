import json
import pathlib
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src import main as main_module


class ConfiguredMainTests(unittest.TestCase):
    def test_run_config_defaults_to_legacy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "config.json"
            path.write_text(json.dumps({"app": {}}), encoding="utf-8")
            self.assertEqual(
                main_module._load_run_config(path, "app"),
                {"mode": "legacy"},
            )

    def test_relative_paths_are_resolved_from_config_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            config_dir = root / "config-dir"
            launch_dir = root / "elsewhere"
            config_dir.mkdir()
            launch_dir.mkdir()
            config_path = config_dir / "config_rag.json"
            config_path.write_text("{}", encoding="utf-8")

            with patch("pathlib.Path.cwd", return_value=launch_dir):
                resolved = main_module._resolve_from_config(
                    config_path,
                    "mineru_test/fixed.json",
                )

            self.assertEqual(
                resolved,
                (config_dir / "mineru_test/fixed.json").resolve(),
            )

    def test_unified_rag_config_contains_fixed_and_hierarchical_runs(self):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        config_path = repo_root / "config_rag.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))

        fixed = payload["cardiology_rag_fixed_prototype"]
        hierarchical = payload["cardiology_rag_hierarchical_prototype"]

        self.assertEqual(fixed["run"]["source_type"], "fixed_chunks")
        self.assertEqual(fixed["run"]["expected_chunk_count"], 293)
        self.assertEqual(
            hierarchical["run"]["source_type"],
            "hierarchical_section_view",
        )
        self.assertEqual(hierarchical["run"]["expected_chunk_count"], 166)
        self.assertTrue(
            fixed["indexing"]["folder"].startswith("prototype/vectorstores/")
        )
        self.assertTrue(
            hierarchical["indexing"]["folder"].startswith(
                "prototype/vectorstores/"
            )
        )

    def test_manifest_contains_reproducibility_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "version": "fixed_within_section_v2",
                        "strategy": "fixed_within_section",
                        "doc_id": "Doc",
                        "chunk_size": 2000,
                        "chunk_overlap": 300,
                        "source_record_count": 202,
                        "eligible_source_record_count": 170,
                        "output_chunk_count": 293,
                        "chunks": [],
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config_rag.json"
            config_path.write_text('{"app": {}}', encoding="utf-8")

            index_folder = root / "index"
            index_folder.mkdir()
            (index_folder / "prototype_index.faiss").write_bytes(b"faiss")
            (index_folder / "prototype_index.pkl").write_bytes(b"docstore")
            (index_folder / "config.json").write_text("{}", encoding="utf-8")

            processor = MagicMock()
            processor.app_id = "prototype"
            processor.config.indexing.name = "prototype_index"
            processor.config.indexing.folder = index_folder
            processor.config.indexing.type.value = "faiss"
            processor.config.indexing.distance.value = "cosine"
            processor.config.embeddings.model_name = "text-embedding-3-small"
            processor.config.embeddings.dim = 1536
            processor.index_manager.get_n_documents_in_vectorstore.return_value = 293

            with (
                patch.object(
                    main_module,
                    "_package_versions",
                    return_value={"langchain": "1.2.1"},
                ),
                patch.object(
                    main_module,
                    "_git_provenance",
                    return_value={"commit": "abc123", "dirty": False},
                ),
            ):
                output = main_module._write_manifest(
                    processor=processor,
                    config_path=config_path,
                    mode="prebuilt",
                    sources=[
                        {
                            "source_path": source,
                            "source_type": "fixed_chunks",
                            "source_key": "Doc::fixed_chunks",
                            "artifact": {
                                "chunk_size": 2000,
                                "chunk_overlap": 300,
                            },
                            "indexed_chunk_count": 293,
                            "expected_chunk_count": 293,
                        }
                    ],
                )

            payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(payload["schema_version"], "rag_index_build_v3")
            self.assertEqual(payload["mode"], "prebuilt")
            self.assertEqual(len(payload["sources"]), 1)
            self.assertEqual(payload["sources"][0], payload["source"])
            self.assertEqual(payload["indexed_chunk_count"], 293)
            self.assertEqual(payload["stored_vector_count"], 293)
            self.assertEqual(payload["source"]["type"], "fixed_chunks")
            self.assertEqual(payload["source"]["source_key"], "Doc::fixed_chunks")
            self.assertEqual(payload["source"]["artifact"]["chunk_size"], 2000)
            self.assertEqual(payload["source"]["artifact"]["chunk_overlap"], 300)
            self.assertEqual(len(payload["source"]["sha256"]), 64)
            self.assertEqual(len(payload["config"]["sha256"]), 64)
            self.assertEqual(payload["git"]["commit"], "abc123")
            self.assertEqual(payload["runtime"]["packages"]["langchain"], "1.2.1")
            self.assertEqual(payload["index"]["distance"], "cosine")
            self.assertIn("faiss", payload["index"]["output_files"])
            self.assertEqual(
                payload["index"]["output_files"]["faiss"]["size_bytes"],
                5,
            )
            self.assertEqual(payload["embeddings"]["dimensions"], 1536)


if __name__ == "__main__":
    unittest.main(verbosity=2)
