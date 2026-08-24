from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from managers.tables.table_catalog_manager import (
    ChunkTableOccurrence,
    SourceTable,
    _link_tables,
    canonicalize_table_html,
    sha256_text as catalogue_sha256,
)
from managers.tables.table_cleaning_manager import (
    group_logical_tables,
    parse_catalog_record,
    process_catalog_file,
)


def source(index: int, html: str, caption: list[str] | None = None) -> SourceTable:
    return SourceTable(
        source_index=index,
        page_idx=index,
        block_index=index,
        bbox=None,
        caption=caption or [],
        footnotes=[],
        raw_html=html,
        image_source=None,
        source_format="test",
        quality_flags=[],
    )


def chunk(order: int, html: str, *, excluded: bool = False) -> ChunkTableOccurrence:
    canonical = canonicalize_table_html(html)
    return ChunkTableOccurrence(
        doc_id="Doc",
        chunk_id=f"Doc:1:{order}",
        section_id="1",
        section_title="Section",
        excluded=excluded,
        embed=not excluded,
        chunk_table_index=order + 1,
        start_offset=0,
        end_offset=len(html),
        raw_html=html,
        exact_sha256=catalogue_sha256(html),
        canonical_sha256=catalogue_sha256(canonical),
        source_path="chunks.json",
        source_order=order,
    )


def raw_record(
    table_id: str,
    html: str,
    *,
    classification: str = "table_unclassified",
    caption: list[str] | None = None,
    link_status: str = "matched_exact",
    source_index: int = 0,
    fragment_group_id: str | None = None,
    fragment_index: int | None = None,
    fragment_count: int | None = None,
) -> dict:
    record = {
        "version": "table_catalog_v2_2",
        "doc_id": "Doc",
        "table_id": table_id,
        "source_index": source_index,
        "page_idx": source_index,
        "page": source_index + 1,
        "caption": caption or [],
        "footnotes": [],
        "raw_html": html,
        "raw_html_sha256": catalogue_sha256(html),
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
    if fragment_group_id:
        record.update(
            {
                "fragment_group_id": fragment_group_id,
                "fragment_index": fragment_index,
                "fragment_count": fragment_count,
                "fragment_match_mode": "exact_row_concatenation",
            }
        )
    return record


class TablePipelineLinkingTests(unittest.TestCase):
    def test_one_to_one_exact_link_is_preserved(self) -> None:
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        links, summary, unmatched = _link_tables([source(0, html)], [chunk(0, html)])
        self.assertEqual(links[0]["link_status"], "matched_exact")
        self.assertEqual(summary["linked_chunk_table_count"], 1)
        self.assertEqual(summary["fragment_sequence_count"], 0)
        self.assertEqual(unmatched, [])

    def test_consecutive_fragments_link_to_one_chunk(self) -> None:
        first = "<table><tr><td>A</td><td>1</td></tr></table>"
        second = "<table><tr><td>B</td><td>2</td></tr></table>"
        combined = (
            "<table><tr><td>A</td><td>1</td></tr>"
            "<tr><td>B</td><td>2</td></tr></table>"
        )
        links, summary, unmatched = _link_tables(
            [source(0, first), source(1, second)],
            [chunk(0, combined)],
        )
        self.assertEqual(
            [item["link_status"] for item in links],
            ["matched_fragment_sequence", "matched_fragment_sequence"],
        )
        self.assertEqual(links[0]["fragment_group_id"], links[1]["fragment_group_id"])
        self.assertEqual([links[0]["fragment_index"], links[1]["fragment_index"]], [1, 2])
        self.assertEqual(summary["fragment_sequence_count"], 1)
        self.assertEqual(summary["fragment_linked_source_table_count"], 2)
        self.assertEqual(summary["linked_chunk_table_count"], 1)
        self.assertEqual(unmatched, [])

    def test_fragment_linking_is_not_fuzzy(self) -> None:
        first = "<table><tr><td>A</td><td>1</td></tr></table>"
        second = "<table><tr><td>B</td><td>2</td></tr></table>"
        changed = (
            "<table><tr><td>A</td><td>1</td></tr>"
            "<tr><td>B</td><td>DIFFERENT</td></tr></table>"
        )
        links, summary, unmatched = _link_tables(
            [source(0, first), source(1, second)],
            [chunk(0, changed)],
        )
        self.assertTrue(all(item["link_status"] == "not_linked" for item in links))
        self.assertEqual(summary["fragment_sequence_count"], 0)
        self.assertEqual(len(unmatched), 1)

    def test_wide_sparse_gene_matrix_stays_clinical(self) -> None:
        html = (
            "<table><tr><td>Gene</td><td>HCM</td><td>DCM</td><td>NDLVC</td>"
            "<td>ARVC</td><td>RCM</td><td>Phenotype</td></tr>"
            "<tr><td>MYH7</td><td>●</td><td>●</td><td></td><td></td><td></td>"
            "<td>Myopathy</td></tr></table>"
        )
        parsed = parse_catalog_record(
            raw_record(
                "Doc::table::0001",
                html,
                classification="acronym_or_glossary",
                caption=["Table 10 Overview of genes and phenotypes"],
            )
        )
        self.assertEqual(parsed["classification"], "clinical_table")
        self.assertIn("clinical_matrix_caption", parsed["classification_reasons"])
        self.assertNotIn("unresolved_recommendation_rows", parsed["quality_flags"])

    def test_non_recommendation_table_gets_no_unresolved_flag(self) -> None:
        html = (
            "<table><tr><td></td><td>Definition</td><td>Wording</td></tr>"
            "<tr><td>Class I</td><td>Beneficial treatment</td>"
            "<td>Is recommended</td></tr></table>"
        )
        parsed = parse_catalog_record(
            raw_record(
                "Doc::table::0002",
                html,
                classification="recommendation_candidate",
            )
        )
        self.assertEqual(parsed["classification"], "clinical_table")
        self.assertEqual(parsed["unresolved_patterns"], [])
        self.assertNotIn("unresolved_recommendation_rows", parsed["quality_flags"])

    def test_true_recommendation_fragment_is_kept_as_ungraded_fragment(self) -> None:
        html = (
            "<table><tr><td>Treatment</td>"
            "<td>should be considered.</td></tr></table>"
        )
        parsed = parse_catalog_record(raw_record("Doc::table::0003", html))

        self.assertEqual(
            parsed["classification"],
            "recommendation_text_fragment",
        )
        self.assertNotIn(
            "unresolved_recommendation_rows",
            parsed["quality_flags"],
        )
        self.assertEqual(parsed["unresolved_patterns"], [])
        self.assertEqual(len(parsed["recommendation_fragments"]), 1)
        fragment = parsed["recommendation_fragments"][0]
        self.assertIsNone(fragment["class"])
        self.assertIsNone(fragment["level"])
        self.assertIn("missing_class_level", fragment["quality_flags"])

    def test_group_header_is_inherited_across_fragment_group(self) -> None:
        first_html = (
            "<table><tr><td>Recommendations</td><td>Class</td><td>Level</td></tr>"
            "<tr><td colspan='3'>Drug therapy</td></tr>"
            "<tr><td>Drug A is recommended.</td><td>I</td><td>A</td></tr></table>"
        )
        second_html = (
            "<table><tr><td>Drug B should be considered.</td><td>IIa</td><td>B</td>"
            "</tr></table>"
        )
        first = parse_catalog_record(
            raw_record(
                "Doc::table::0004",
                first_html,
                classification="recommendation_candidate",
                source_index=0,
                link_status="matched_fragment_sequence",
                fragment_group_id="group-1",
                fragment_index=1,
                fragment_count=2,
            )
        )
        second = parse_catalog_record(
            raw_record(
                "Doc::table::0005",
                second_html,
                classification="recommendation_candidate",
                source_index=1,
                link_status="matched_fragment_sequence",
                fragment_group_id="group-1",
                fragment_index=2,
                fragment_count=2,
            )
        )
        logical = group_logical_tables([first, second])
        self.assertEqual(len(logical), 1)
        rows = logical[0]["recommendation_rows"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["group_header"], "Drug therapy")
        self.assertTrue(rows[1]["group_header_inherited"])
        self.assertEqual(logical[0]["fragment_count"], 2)

    def test_process_file_flattens_context_enriched_logical_rows(self) -> None:
        first_html = (
            "<table><tr><td>Recommendations</td><td>Class</td><td>Level</td></tr>"
            "<tr><td colspan='3'>Imaging</td></tr>"
            "<tr><td>Test A is recommended.</td><td>I</td><td>B</td></tr></table>"
        )
        second_html = (
            "<table><tr><td>Test B should be considered.</td><td>IIa</td><td>C</td>"
            "</tr></table>"
        )
        records = [
            raw_record(
                "Doc::table::0006",
                first_html,
                classification="recommendation_candidate",
                source_index=0,
                link_status="matched_fragment_sequence",
                fragment_group_id="group-2",
                fragment_index=1,
                fragment_count=2,
            ),
            raw_record(
                "Doc::table::0007",
                second_html,
                classification="recommendation_candidate",
                source_index=1,
                link_status="matched_fragment_sequence",
                fragment_group_id="group-2",
                fragment_index=2,
                fragment_count=2,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "Doc_tables_raw.jsonl"
            source_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            summary = process_catalog_file(source_path, root / "out", force=True)
            output = [
                json.loads(line)
                for line in (root / "out" / "Doc_recommendations_active.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(summary["active_recommendation_row_count"], 2)
            self.assertEqual(summary["active_linked_chunk_table_count"], 1)
            self.assertEqual(output[1]["group_header"], "Imaging")
            self.assertTrue(output[1]["group_header_inherited"])

    def test_raw_html_hash_is_unchanged(self) -> None:
        html = "<table><tr><td>Value</td><td>42</td></tr></table>"
        parsed = parse_catalog_record(raw_record("Doc::table::0008", html))
        self.assertTrue(parsed["raw_html_integrity_ok"])
        self.assertTrue(parsed["raw_html_unchanged"])
        self.assertEqual(parsed["raw_html"], html)


if __name__ == "__main__":
    unittest.main()
