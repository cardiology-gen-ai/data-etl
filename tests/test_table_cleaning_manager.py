from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from managers.tables.table_cleaning_manager import (
    VERSION,
    compose_normalized_recommendation,
    detect_text_quality_flags,
    parse_catalog_record,
    process_catalog_file,
    sha256_text,
    split_terminal_annotations,
    split_terminal_citations,
)


def raw_record(
    table_id: str,
    html: str,
    *,
    classification: str = "table_unclassified",
    caption: list[str] | None = None,
    source_index: int = 0,
    link_status: str = "matched_exact",
) -> dict:
    return {
        "version": "table_catalog_v2_2",
        "doc_id": "Doc",
        "table_id": table_id,
        "source_index": source_index,
        "page_idx": source_index,
        "page": source_index + 1,
        "caption": caption or [],
        "footnotes": [],
        "raw_html": html,
        "raw_html_sha256": sha256_text(html),
        "classification": classification,
        "classification_reasons": [],
        "quality_flags": [],
        "link_status": link_status,
        "chunk_id": "Doc:1:0",
        "section_id": "1",
        "section_title": "Section",
        "chunk_table_index": 1,
        "chunk_source_path": "chunks.json",
        "chunk_source_order": 0,
        "excluded": False,
        "embed": True,
    }


class TableCleaningManagerTests(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(VERSION, "table_render_conservative_v1_4")

    def test_structured_guidance_uses_general_structure_and_semantics(self) -> None:
        html = (
            "<table><tr><td>Topic</td><td>Practical guidance</td></tr>"
            "<tr><td>Activity</td><td>Patients should receive individualized advice.</td></tr>"
            "<tr><td>Travel</td><td>Risks and local rules should be discussed.</td></tr>"
            "<tr><td>Work</td><td>Occupational implications should be reviewed.</td></tr>"
            "</table>"
        )
        parsed = parse_catalog_record(
            raw_record(
                "Doc::table::0001",
                html,
                classification="recommendation_candidate",
                caption=["Practical guidance and considerations"],
            )
        )
        self.assertEqual(parsed["classification"], "structured_guidance_table")
        self.assertEqual(parsed["recommendation_rows"], [])
        self.assertEqual(parsed["recommendation_fragments"], [])
        self.assertNotIn("unresolved_recommendation_rows", parsed["quality_flags"])

    def test_single_prescriptive_row_becomes_ungraded_fragment(self) -> None:
        html = (
            "<table><tr><td>Imaging should be considered in selected patients.</td>"
            "<td><img src='grade.png'/></td></tr></table>"
        )
        parsed = parse_catalog_record(raw_record("Doc::table::0002", html))
        self.assertEqual(parsed["classification"], "recommendation_text_fragment")
        self.assertEqual(parsed["recommendation_rows"], [])
        self.assertEqual(len(parsed["recommendation_fragments"]), 1)
        fragment = parsed["recommendation_fragments"][0]
        self.assertIsNone(fragment["class"])
        self.assertIsNone(fragment["level"])
        self.assertTrue(fragment["active_for_retrieval"])
        self.assertIn("missing_class_level", fragment["quality_flags"])
        self.assertIn("embedded_image_present", fragment["quality_flags"])

    def test_superscript_terminal_citations_are_separated(self) -> None:
        text, refs, marker = split_terminal_citations(
            "Treatment is recommended.^{300-302, 305}"
        )
        self.assertEqual(text, "Treatment is recommended.")
        self.assertEqual(refs, ["300-302", "305"])
        self.assertIsNone(marker)

    def test_plain_terminal_citations_and_marker_are_separated(self) -> None:
        text, refs, marker = split_terminal_citations(
            "Treatment should be considered.c808,811"
        )
        self.assertEqual(text, "Treatment should be considered.")
        self.assertEqual(refs, ["808", "811"])
        self.assertEqual(marker, "c")

    def test_internal_numbers_are_not_removed(self) -> None:
        original = "Use 75–100 mg daily in patients treated in 2021."
        text, refs, marker = split_terminal_citations(original)
        self.assertEqual(text, original)
        self.assertEqual(refs, [])
        self.assertIsNone(marker)

    def test_auxiliary_continuation_is_combined_with_context(self) -> None:
        normalized, dependent, kind = compose_normalized_recommendation(
            "In eligible patients, surgery:",
            "is recommended over medical therapy.12",
        )
        self.assertTrue(dependent)
        self.assertEqual(kind, "auxiliary_verb")
        self.assertEqual(
            normalized,
            "In eligible patients, surgery is recommended over medical therapy.12",
        )

    def test_prepositional_bullet_completes_recommendation_header(self) -> None:
        normalized, dependent, kind = compose_normalized_recommendation(
            "Implantation is recommended:d",
            "• in patients with previous cardiac arrest.",
        )
        self.assertTrue(dependent)
        self.assertEqual(kind, "prepositional_list_item")
        self.assertEqual(
            normalized,
            "Implantation is recommended in patients with previous cardiac arrest.",
        )

    def test_noun_bullet_preserves_colon_after_governing_header(self) -> None:
        normalized, dependent, kind = compose_normalized_recommendation(
            "The following tests are recommended:",
            "• complete blood count;",
        )
        self.assertTrue(dependent)
        self.assertEqual(kind, "recommendation_header_list_item")
        self.assertEqual(
            normalized,
            "The following tests are recommended: complete blood count;",
        )

    def test_unrelated_section_context_is_not_prepended(self) -> None:
        normalized, dependent, kind = compose_normalized_recommendation(
            "Drug therapy",
            "Drug A should be considered.",
        )
        self.assertFalse(dependent)
        self.assertIsNone(kind)
        self.assertEqual(normalized, "Drug A should be considered.")

    def test_recommendation_record_preserves_raw_and_adds_normalized_fields(self) -> None:
        html = (
            "<table><tr><td>Recommendations</td><td>Class</td><td>Level</td></tr>"
            "<tr><td colspan='3'>In eligible patients, procedure:</td></tr>"
            "<tr><td>is recommended.18,20-22</td><td>I</td><td>A</td></tr>"
            "</table>"
        )
        parsed = parse_catalog_record(
            raw_record("Doc::table::0003", html, classification="recommendation_candidate")
        )
        row = parsed["recommendation_rows"][0]
        self.assertEqual(row["raw_recommendation"], "is recommended.18,20-22")
        self.assertEqual(row["recommendation"], "is recommended.")
        self.assertEqual(row["citation_numbers"], ["18", "20-22"])
        self.assertEqual(
            row["normalized_recommendation"],
            "In eligible patients, procedure is recommended.",
        )
        self.assertTrue(row["context_dependent"])

    def test_process_file_writes_separate_fragment_outputs_and_clear_summary(self) -> None:
        recommendation_html = (
            "<table><tr><td>Recommendations</td><td>Class</td><td>Level</td></tr>"
            "<tr><td>Treatment is recommended.12</td><td>I</td><td>A</td></tr>"
            "</table>"
        )
        fragment_html = (
            "<table><tr><td>Monitoring should be considered.</td>"
            "<td><img src='grade.png'/></td></tr></table>"
        )
        records = [
            raw_record(
                "Doc::table::0004",
                recommendation_html,
                classification="recommendation_candidate",
                source_index=0,
            ),
            raw_record(
                "Doc::table::0005",
                fragment_html,
                source_index=1,
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "Doc_tables_raw.jsonl"
            input_path.write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            output_dir = root / "processed"
            summary = process_catalog_file(input_path, output_dir, force=True)

            self.assertEqual(summary["active_recommendation_row_count"], 1)
            self.assertEqual(summary["active_recommendation_fragment_count"], 1)
            self.assertEqual(summary["active_table_fragment_count"], 2)
            self.assertEqual(summary["active_linked_chunk_table_count"], 1)
            self.assertIn("active_structured_guidance_fragment_count", summary)
            self.assertIn("active_structured_guidance_logical_table_count", summary)
            self.assertIn("active_recommendations_with_footnotes_count", summary)
            self.assertIn("active_orphan_list_item_count", summary)
            self.assertIn("active_suspicious_internal_text_glue_count", summary)
            fragment_path = Path(
                summary["output_files"]["active_recommendation_fragments"]
            )
            fragment = json.loads(fragment_path.read_text(encoding="utf-8").strip())
            self.assertIsNone(fragment["class"])
            self.assertIsNone(fragment["level"])


    def test_terminal_footnote_and_references_are_separated(self) -> None:
        text, refs, markers = split_terminal_annotations(
            "Treatment should be considered.e,53,55"
        )
        self.assertEqual(text, "Treatment should be considered.")
        self.assertEqual(refs, ["53", "55"])
        self.assertEqual(markers, ["e"])

    def test_terminal_multiple_footnotes_without_references_are_separated(self) -> None:
        text, refs, markers = split_terminal_annotations(
            "Treatment should be considered.d,e"
        )
        self.assertEqual(text, "Treatment should be considered.")
        self.assertEqual(refs, [])
        self.assertEqual(markers, ["d", "e"])

    def test_terminal_superscript_footnote_is_separated(self) -> None:
        text, refs, markers = split_terminal_annotations(
            "Surveillance may be reduced.^{f}"
        )
        self.assertEqual(text, "Surveillance may be reduced.")
        self.assertEqual(refs, [])
        self.assertEqual(markers, ["f"])

    def test_ambiguous_internal_glue_is_flagged_but_not_rewritten(self) -> None:
        raw = "Screening for risk factorscis recommended."
        flags = detect_text_quality_flags(raw)
        self.assertIn("possible_internal_text_glue", flags)
        self.assertIn("possible_footnote_marker_before_is", flags)
        html = (
            "<table><tr><td>Recommendations</td><td>Class</td><td>Level</td></tr>"
            f"<tr><td>{raw}</td><td>I</td><td>C</td></tr></table>"
        )
        parsed = parse_catalog_record(
            raw_record("Doc::table::0010", html, classification="recommendation_candidate")
        )
        row = parsed["recommendation_rows"][0]
        self.assertEqual(row["raw_recommendation"], raw)
        self.assertEqual(row["recommendation"], raw)
        self.assertIn("possible_internal_text_glue", row["text_quality_flags"])

    def test_squared_unit_glue_is_safely_spaced_in_normalized_text(self) -> None:
        html = (
            "<table><tr><td>Recommendations</td><td>Class</td><td>Level</td></tr>"
            "<tr><td>A dose of 250 mg/m2of drug should be considered.</td>"
            "<td>IIa</td><td>B</td></tr></table>"
        )
        parsed = parse_catalog_record(
            raw_record("Doc::table::0011", html, classification="recommendation_candidate")
        )
        row = parsed["recommendation_rows"][0]
        self.assertEqual(
            row["raw_recommendation"],
            "A dose of 250 mg/m2of drug should be considered.",
        )
        self.assertEqual(
            row["recommendation"],
            "A dose of 250 mg/m2 of drug should be considered.",
        )
        self.assertIn("missing_space_after_squared_unit", row["text_quality_flags"])

    def test_orphan_list_item_uses_section_title_only_as_context_hint(self) -> None:
        html = (
            "<table><tr><td>Recommendations</td><td>Class</td><td>Level</td></tr>"
            "<tr><td>• FFR should be used;</td><td>I</td><td>A</td></tr></table>"
        )
        record = raw_record(
            "Doc::table::0012", html, classification="recommendation_candidate"
        )
        record["section_title"] = "Functional assessment"
        parsed = parse_catalog_record(record)
        row = parsed["recommendation_rows"][0]
        self.assertTrue(row["context_dependent"])
        self.assertEqual(row["context_dependency_kind"], "orphan_list_item")
        self.assertEqual(row["context_hint"], "Functional assessment")
        self.assertEqual(row["context_hint_source"], "section_title")
        self.assertEqual(row["normalized_recommendation"], row["recommendation"])

    def test_rerun_drops_stale_derived_unresolved_flag(self) -> None:
        html = (
            "<table><tr><td>Topic</td><td>General guidance</td></tr>"
            "<tr><td>A</td><td>Detailed advice for the first situation.</td></tr>"
            "<tr><td>B</td><td>Detailed advice for the second situation.</td></tr>"
            "<tr><td>C</td><td>Detailed advice for the third situation.</td></tr>"
            "</table>"
        )
        record = raw_record("Doc::table::0006", html, caption=["General guidance"])
        record["quality_flags"] = ["unresolved_recommendation_rows"]
        parsed = parse_catalog_record(record)
        self.assertNotIn("unresolved_recommendation_rows", parsed["quality_flags"])


if __name__ == "__main__":
    unittest.main()
