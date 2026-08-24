from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from managers.clean_chunk_source_manager import (
    CLEAN_SOURCE_KIND,
    force_text_cleaning,
    resolve_section_view_chunk_source,
    text_cleaning_enabled,
)
from managers.text_cleaning_manager import VERSION


class CleanChunkSourceManagerTests(unittest.TestCase):
    def test_clean_chunks_are_selected_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chunks = root / "chunks"
            chunks.mkdir()
            source = chunks / "Doc_hier_chunks.json"
            source.write_text(
                json.dumps([
                    {
                        "chunk_id": "Doc:1:0",
                        "doc_id": "Doc",
                        "section_id": "1",
                        "parent_section_id": None,
                        "section_title": "Introduction",
                        "section_level": 1,
                        "text": "Downloaded from https://example.org by guest on 1 January 2026\nClinical text.",
                        "is_empty": False,
                        "excluded": False,
                        "embed": True,
                        "quality_flags": [],
                    }
                ]),
                encoding="utf-8",
            )
            config = SimpleNamespace(
                chunk_dir=chunks,
                clean_chunk_dir=root / "clean_chunks",
                text_cleaning_audit_dir=root / "audit",
                run_text_cleaning=True,
                force_text_cleaning=False,
            )

            resolved = resolve_section_view_chunk_source(config, source)
            self.assertEqual(resolved.source_kind, CLEAN_SOURCE_KIND)
            self.assertEqual(resolved.text_cleaning_version, VERSION)
            self.assertNotEqual(resolved.source_path, resolved.canonical_path)
            self.assertTrue(resolved.source_path.exists())
            self.assertTrue(resolved.text_cleaning_audit_path.exists())
            cleaned = json.loads(resolved.source_path.read_text(encoding="utf-8"))
            self.assertEqual(cleaned[0]["text"], "Clinical text.")

            reused = resolve_section_view_chunk_source(config, source)
            self.assertEqual(reused.text_cleaning_cache_status, "reused")
            self.assertEqual(reused.source_sha256, resolved.source_sha256)

    def test_disabled_cleaning_uses_canonical_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "Doc_hier_chunks.json"
            source.write_text("[]", encoding="utf-8")
            config = SimpleNamespace(
                chunk_dir=source.parent,
                run_text_cleaning=False,
            )
            resolved = resolve_section_view_chunk_source(config, source)
            self.assertEqual(resolved.source_path, source.resolve())
            self.assertFalse(resolved.text_cleaning_enabled)
            self.assertIsNone(resolved.text_cleaning_version)


    def test_explicit_cleaning_config_wins_over_environment(self) -> None:
        config = SimpleNamespace(
            run_text_cleaning=True,
            force_text_cleaning=False,
        )
        with patch.dict(
            os.environ,
            {
                "KG_RUN_TEXT_CLEANING": "false",
                "KG_FORCE_TEXT_CLEANING": "true",
            },
            clear=False,
        ):
            self.assertTrue(text_cleaning_enabled(config))
            self.assertFalse(force_text_cleaning(config))

    def test_environment_is_fallback_when_cleaning_config_is_unspecified(self) -> None:
        config = SimpleNamespace(
            run_text_cleaning=None,
            force_text_cleaning=None,
        )
        with patch.dict(
            os.environ,
            {
                "KG_RUN_TEXT_CLEANING": "false",
                "KG_FORCE_TEXT_CLEANING": "true",
            },
            clear=False,
        ):
            self.assertFalse(text_cleaning_enabled(config))
            self.assertTrue(force_text_cleaning(config))


if __name__ == "__main__":
    unittest.main()
