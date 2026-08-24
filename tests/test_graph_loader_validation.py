from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from knowledge_graph.graph_loader import (
    build_graph_from_chunks,
    validate_section_view_records,
)
from managers.retrieval_unit_manager_section_view import (
    build_retrieval_section_view,
)


def make_chunk(
    section_id: str,
    *,
    level: int,
    parent: str | None,
    text: str = "",
    embed: bool = False,
    excluded: bool = False,
) -> dict:
    return {
        "chunk_id": f"chunk::{section_id}",
        "doc_id": "TestDoc",
        "section_id": section_id,
        "parent_section_id": parent,
        "section_title": f"Section {section_id}",
        "section_level": level,
        "text": text,
        "is_empty": not bool(text),
        "excluded": excluded,
        "embed": embed,
        "quality_flags": [],
    }


def canonical_chunks() -> list[dict]:
    return [
        make_chunk(
            "1",
            level=1,
            parent=None,
        ),
        make_chunk(
            "1.1",
            level=2,
            parent="1",
        ),
        make_chunk(
            "1.1.1",
            level=3,
            parent="1.1",
        ),
        make_chunk(
            "1.1.1.1",
            level=4,
            parent="1.1.1",
            text="Alpha clinical evidence.",
            embed=True,
        ),
        make_chunk(
            "1.1.1.2",
            level=4,
            parent="1.1.1",
            text="Beta clinical evidence.",
            embed=True,
        ),
    ]


def sections_view() -> list[dict]:
    return build_retrieval_section_view(
        canonical_chunks(),
        max_level=None,
    )


def aggregated_view() -> list[dict]:
    return build_retrieval_section_view(
        canonical_chunks(),
        max_level=3,
    )


class FailIfUsedDriver:
    def session(self):
        raise AssertionError(
            "Neo4j session must not be opened before validation succeeds"
        )


class GraphLoaderValidationTests(unittest.TestCase):
    def test_valid_sections_view_returns_expected_summary(self):
        summary = validate_section_view_records(
            sections_view()
        )

        self.assertEqual(summary["doc_id"], "TestDoc")
        self.assertEqual(
            summary["retrieval_strategy"],
            "sections",
        )
        self.assertEqual(
            summary["aggregation_mode"],
            "none",
        )
        self.assertIsNone(
            summary["aggregation_max_level"]
        )
        self.assertEqual(summary["section_count"], 5)
        self.assertEqual(
            summary["retrieval_section_count"],
            2,
        )
        self.assertEqual(
            summary["structural_section_count"],
            3,
        )
        self.assertEqual(
            summary["aggregated_section_count"],
            0,
        )
        self.assertEqual(
            summary["source_section_count"],
            2,
        )

    def test_valid_aggregated_view_returns_expected_summary(self):
        summary = validate_section_view_records(
            aggregated_view()
        )

        self.assertEqual(
            summary["retrieval_strategy"],
            "max_level_3",
        )
        self.assertEqual(
            summary["aggregation_mode"],
            "merge_below_level",
        )
        self.assertEqual(
            summary["aggregation_max_level"],
            3,
        )
        self.assertEqual(summary["section_count"], 3)
        self.assertEqual(
            summary["retrieval_section_count"],
            1,
        )
        self.assertEqual(
            summary["structural_section_count"],
            2,
        )
        self.assertEqual(
            summary["aggregated_section_count"],
            1,
        )
        self.assertEqual(
            summary["source_section_count"],
            2,
        )

    def test_canonical_chunks_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "section_view_role",
        ):
            validate_section_view_records(
                canonical_chunks()
            )

    def test_duplicate_retrieval_unit_id_is_rejected(self):
        view = sections_view()

        retrieval = [
            section
            for section in view
            if section["section_view_role"] == "retrieval"
        ]

        self.assertEqual(len(retrieval), 2)

        retrieval[1]["retrieval_unit_id"] = (
            retrieval[0]["retrieval_unit_id"]
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicate retrieval_unit_id",
        ):
            validate_section_view_records(view)

    def test_mixed_document_ids_are_rejected(self):
        view = sections_view()
        view[-1]["doc_id"] = "OtherDoc"

        with self.assertRaisesRegex(
            ValueError,
            "exactly one doc_id",
        ):
            validate_section_view_records(view)

    def test_structural_section_cannot_have_retrieval_unit_id(self):
        view = sections_view()

        structural = next(
            section
            for section in view
            if section["section_view_role"] == "structural"
        )
        structural["retrieval_unit_id"] = "illegal-unit"

        with self.assertRaisesRegex(
            ValueError,
            "structural Section cannot have a retrieval_unit_id",
        ):
            validate_section_view_records(view)

    def test_missing_parent_is_rejected(self):
        view = sections_view()

        changed = copy.deepcopy(view)
        changed[-1]["parent_section_id"] = "missing-parent"

        with self.assertRaisesRegex(
            ValueError,
            "is not present in the Section view",
        ):
            validate_section_view_records(changed)

    def test_source_count_must_match_provenance(self):
        view = sections_view()

        retrieval = next(
            section
            for section in view
            if section["section_view_role"] == "retrieval"
        )
        retrieval["source_count"] += 1

        with self.assertRaisesRegex(
            ValueError,
            "does not match len\\(source_section_ids\\)",
        ):
            validate_section_view_records(view)

    def test_invalid_view_fails_before_any_neo4j_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid_section_view.json"

            # Deliberately write canonical chunks rather than a Section View.
            path.write_text(
                json.dumps(canonical_chunks()),
                encoding="utf-8",
            )

            driver = FailIfUsedDriver()

            with self.assertRaisesRegex(
                ValueError,
                "Invalid retrieval Section view",
            ):
                build_graph_from_chunks(
                    driver,
                    path,
                )


if __name__ == "__main__":
    unittest.main()
