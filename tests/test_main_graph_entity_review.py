import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main_graph


class MainGraphEntityReviewSettingsTests(unittest.TestCase):
    def test_default_review_output_dir_is_under_work_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            work_root = project_root / "first_prototype"

            with patch.dict(os.environ, {}, clear=True):
                settings = main_graph.resolve_entity_review_settings(
                    kg_config={"entities": {}},
                    work_root=work_root,
                    project_root=project_root,
                )

        self.assertTrue(settings["entity_export_review"])
        self.assertEqual(
            settings["entity_review_output_dir"],
            (work_root / "entity_review").resolve(),
        )
        self.assertTrue(settings["entity_clear_previous_review"])
        self.assertFalse(settings["entity_include_source_preview_in_review"])

    def test_env_review_output_dir_overrides_config_and_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            work_root = project_root / "first_prototype"
            config_dir = project_root / "config_review"
            env_dir = project_root / "env_review"

            with patch.dict(
                os.environ,
                {"KG_ENTITY_REVIEW_OUTPUT_DIR": str(env_dir)},
                clear=True,
            ):
                settings = main_graph.resolve_entity_review_settings(
                    kg_config={
                        "entities": {
                            "review_output_dir": str(config_dir),
                        }
                    },
                    work_root=work_root,
                    project_root=project_root,
                )

        self.assertEqual(settings["entity_review_output_dir"], env_dir.resolve())

    def test_config_review_flags_are_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            work_root = project_root / "first_prototype"

            with patch.dict(os.environ, {}, clear=True):
                settings = main_graph.resolve_entity_review_settings(
                    kg_config={
                        "entities": {
                            "export_review": False,
                            "clear_previous_review": False,
                            "include_source_preview_in_review": True,
                        }
                    },
                    work_root=work_root,
                    project_root=project_root,
                )

        self.assertFalse(settings["entity_export_review"])
        self.assertFalse(settings["entity_clear_previous_review"])
        self.assertTrue(settings["entity_include_source_preview_in_review"])

    def test_normalization_exact_threshold_resolves_from_config_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            work_root = project_root / "first_prototype"

            with patch.dict(os.environ, {}, clear=True):
                settings = main_graph._resolve_normalization_kwargs(
                    {
                        "entity_normalization": {
                            "threshold": 0.85,
                            "exact_threshold": 0.76,
                            "create_same_as_edges": True,
                            "create_fuzzy_candidate_edges": True,
                        }
                    },
                    work_root=work_root,
                    project_root=project_root,
                )
            self.assertEqual(settings["entity_normalization_exact_threshold"], 0.76)
            self.assertTrue(settings["entity_normalization_create_same_as_edges"])
            self.assertTrue(
                settings["entity_normalization_create_fuzzy_candidate_edges"]
            )

            with patch.dict(
                os.environ,
                {
                    "KG_ENTITY_NORMALIZATION_EXACT_THRESHOLD": "0.77",
                    "KG_ENTITY_NORMALIZATION_CREATE_SAME_AS_EDGES": "false",
                    "KG_ENTITY_NORMALIZATION_CREATE_FUZZY_CANDIDATE_EDGES": "false",
                },
                clear=True,
            ):
                settings = main_graph._resolve_normalization_kwargs(
                    {
                        "entity_normalization": {
                            "threshold": 0.85,
                            "exact_threshold": 0.76,
                        }
                    },
                    work_root=work_root,
                    project_root=project_root,
                )
            self.assertEqual(settings["entity_normalization_exact_threshold"], 0.77)
            self.assertFalse(settings["entity_normalization_create_same_as_edges"])
            self.assertFalse(
                settings["entity_normalization_create_fuzzy_candidate_edges"]
            )


if __name__ == "__main__":
    unittest.main()
