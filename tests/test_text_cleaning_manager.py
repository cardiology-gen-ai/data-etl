from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from managers.text_cleaning_manager import (
    VERSION,
    clean_text,
    load_or_build_clean_chunks,
    sha_file,
    validate_balanced_tables,
    validate_clean_cache,
)


class TextCleaningManagerTests(unittest.TestCase):
    def test_complete_table_is_preserved_byte_for_byte(self) -> None:
        table = '<table class="x"><tr><td>A  B</td></tr></table>'
        raw = f"Before  text\n{table}\nAfter  text"
        cleaned, stats = clean_text(raw)
        self.assertIn(table, cleaned)
        self.assertEqual(stats["table_blocks_preserved"], 1)

    def test_unbalanced_table_fails_before_cleaning(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unbalanced table HTML"):
            validate_balanced_tables("Text <table><tr><td>x</td></tr>")
        with self.assertRaisesRegex(ValueError, "Unbalanced table HTML"):
            clean_text("Text <table><tr><td>x</td></tr>")

    def test_page_footer_is_removed_only_as_full_line(self) -> None:
        raw = (
            "Clinical prose mentions ESC Guidelines in a sentence.\n"
            "ESC Guidelines 3509\n"
            "3510 ESC Guidelines\n"
            "ESC Guidelines\n"
            "4237\n"
            "4236\n"
            "ESC Guidelines\n"
            "More prose."
        )
        cleaned, stats = clean_text(raw)
        self.assertIn("Clinical prose mentions ESC Guidelines in a sentence.", cleaned)
        self.assertNotIn("ESC Guidelines 3509", cleaned)
        self.assertNotIn("3510 ESC Guidelines", cleaned)
        self.assertNotIn("4237", cleaned)
        self.assertNotIn("4236", cleaned)
        self.assertEqual(stats["publisher_noise_lines_removed"], 6)

    def test_split_download_watermark_is_removed(self) -> None:
        raw = (
            "Useful text.\n"
            "D\n"
            "ow\n"
            "nloaded from\n"
            "https://academic.oup.com/eurheartj/article/44/37/3503/7246608 "
            "by guest on 08 June 2025\n"
            "More useful text."
        )
        cleaned, stats = clean_text(raw)
        self.assertEqual(cleaned, "Useful text.\nMore useful text.")
        self.assertEqual(stats["download_watermark_blocks_removed"], 1)

    def test_multi_reference_superscript_is_removed(self) -> None:
        cleaned, stats = clean_text("Treatment.<sup>12,14–16</sup>")
        self.assertEqual(cleaned, "Treatment.")
        self.assertEqual(stats["numeric_superscripts_removed"], 1)

    def test_unit_exponents_are_converted(self) -> None:
        cleaned, stats = clean_text("Area 4 cm<sup>2</sup> and volume 2 m<sup>3</sup>.")
        self.assertEqual(cleaned, "Area 4 cm² and volume 2 m³.")
        self.assertEqual(stats["exponent_superscripts_converted"], 2)

    def test_single_numeric_reference_after_prose_is_removed(self) -> None:
        cleaned, stats = clean_text(
            "Treatment is recommended.<sup>5</sup> HF<sup>14</sup> therapy."
        )
        self.assertEqual(cleaned, "Treatment is recommended. HF therapy.")
        self.assertEqual(stats["single_reference_superscripts_removed"], 2)

    def test_power_of_ten_is_preserved(self) -> None:
        cleaned, stats = clean_text("Expression 10<sup>5</sup> cells.")
        self.assertEqual(cleaned, "Expression 10^{5} cells.")
        self.assertEqual(stats["scientific_numeric_superscripts_preserved"], 1)

    def test_single_letter_square_is_preserved(self) -> None:
        cleaned, stats = clean_text("Use x<sup>2</sup> in the equation.")
        self.assertEqual(cleaned, "Use x^{2} in the equation.")
        self.assertEqual(stats["scientific_numeric_superscripts_preserved"], 1)

    def test_truncated_reference_superscript_is_removed(self) -> None:
        cleaned, stats = clean_text("Evidence.<sup>719–</sup> Next sentence.")
        self.assertEqual(cleaned, "Evidence. Next sentence.")
        self.assertEqual(stats["truncated_reference_superscripts_removed"], 1)

    def test_rendered_truncated_reference_is_removed_idempotently(self) -> None:
        cleaned, stats = clean_text("Evidence.^{719–} Next sentence.")
        self.assertEqual(cleaned, "Evidence. Next sentence.")
        self.assertEqual(
            stats["rendered_truncated_reference_markers_removed"], 1
        )

    def test_corrupt_blocks_adjacent_to_table_are_removed(self) -> None:
        table = "<table><tr><td>Condition</td></tr></table>"
        raw = (
            "Useful clinical prose.\n\n"
            "trial burden and mana[gement] [in] [cardio]\n\n"
            f"{table}\n\n"
            "g ypy;,;,;, [g];, [ypy];, [yp] ion fracti[on]; "
            "[NDLVC], [non-d]il[ated] l[eft] [ventr]i[cu]l[ar] "
            "[card]i[omyopathy]; [QRS], [Q], [R], [a] "
            "verylonggluedalphabetictokenwithoutseparation\n\n"
            "More useful clinical prose."
        )
        cleaned, stats = clean_text(raw)
        self.assertIn("Useful clinical prose.", cleaned)
        self.assertIn("More useful clinical prose.", cleaned)
        self.assertIn(table, cleaned)
        self.assertNotIn("mana[gement]", cleaned)
        self.assertNotIn("g ypy", cleaned)
        self.assertEqual(
            stats["high_confidence_corruption_blocks_removed"], 2
        )
        self.assertEqual(len(stats["removed_corruption_blocks"]), 2)

    def test_normal_bracketed_prose_is_preserved(self) -> None:
        raw = (
            "Patients [especially older adults] may require monitoring.\n\n"
            "A separate paragraph [with context] remains valid."
        )
        cleaned, stats = clean_text(raw)
        self.assertIn("[especially older adults]", cleaned)
        self.assertIn("[with context]", cleaned)
        self.assertEqual(
            stats["high_confidence_corruption_blocks_removed"], 0
        )

    def test_cache_is_reused_and_source_change_invalidates_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "chunks"
            output_dir = root / "clean_chunks"
            audit_dir = root / "audit"
            input_dir.mkdir()
            source = input_dir / "Doc_hier_chunks.json"
            source_payload = [
                {
                    "chunk_id": "Doc::1",
                    "doc_id": "Doc",
                    "section_id": "1",
                    "section_title": "Title",
                    "section_level": 1,
                    "excluded": False,
                    "embed": True,
                    "is_empty": False,
                    "text": "Text  with  spaces.",
                }
            ]
            source.write_text(json.dumps(source_payload), encoding="utf-8")
            source_hash_before = sha_file(source)

            output_path, first = load_or_build_clean_chunks(
                source, output_dir, audit_dir
            )
            self.assertEqual(first["cache_status"], "rebuilt")
            self.assertEqual(first["cleaning_version"], VERSION)
            self.assertEqual(sha_file(source), source_hash_before)

            reused_path, second = load_or_build_clean_chunks(
                source, output_dir, audit_dir
            )
            self.assertEqual(reused_path, output_path)
            self.assertEqual(second["cache_status"], "reused")

            source_payload[0]["text"] = "Changed source."
            source.write_text(json.dumps(source_payload), encoding="utf-8")
            rebuilt_path, third = load_or_build_clean_chunks(
                source, output_dir, audit_dir
            )
            self.assertEqual(rebuilt_path, output_path)
            self.assertEqual(third["cache_status"], "rebuilt")

            audit_path = audit_dir / "Doc_text_cleaning_audit.json"
            valid, reason, _ = validate_clean_cache(
                source, output_path, audit_path
            )
            self.assertTrue(valid, reason)


if __name__ == "__main__":
    unittest.main()
