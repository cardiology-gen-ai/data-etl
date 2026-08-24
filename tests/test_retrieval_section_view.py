from __future__ import annotations

import copy
import unittest

from managers.retrieval_unit_manager_section_view import (
    build_retrieval_section_view,
    validate_retrieval_section_view,
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
    """
    Synthetic hierarchy:

    1                  L1 structural
    └── 1.1            L2 structural
        └── 1.1.1      L3 structural / aggregation owner
            ├── 1.1.1.1 L4 retrieval source
            └── 1.1.1.2 L4 retrieval source
    """
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


class RetrievalSectionViewTests(unittest.TestCase):
    def test_sections_strategy_keeps_sources_separate(self):
        chunks = canonical_chunks()

        view = build_retrieval_section_view(
            chunks,
            max_level=None,
        )

        self.assertEqual(
            [section["section_id"] for section in view],
            [
                "1",
                "1.1",
                "1.1.1",
                "1.1.1.1",
                "1.1.1.2",
            ],
        )

        retrieval = [
            section
            for section in view
            if section["section_view_role"] == "retrieval"
        ]
        structural = [
            section
            for section in view
            if section["section_view_role"] == "structural"
        ]

        self.assertEqual(len(retrieval), 2)
        self.assertEqual(len(structural), 3)

        for section in retrieval:
            self.assertFalse(section["is_aggregated"])
            self.assertEqual(
                section["source_section_ids"],
                [section["section_id"]],
            )
            self.assertTrue(section["embed"])

        report = validate_retrieval_section_view(
            chunks=chunks,
            section_view=view,
            max_level=None,
        )
        self.assertTrue(report["valid"], report["errors"])

    def test_max_level_three_aggregates_l4_sources(self):
        chunks = canonical_chunks()

        view = build_retrieval_section_view(
            chunks,
            max_level=3,
        )

        self.assertEqual(
            [section["section_id"] for section in view],
            ["1", "1.1", "1.1.1"],
        )

        owner = next(
            section
            for section in view
            if section["section_id"] == "1.1.1"
        )

        self.assertEqual(
            owner["section_view_role"],
            "retrieval",
        )
        self.assertTrue(owner["is_aggregated"])
        self.assertTrue(owner["embed"])
        self.assertFalse(owner["is_empty"])

        self.assertEqual(
            owner["source_section_ids"],
            ["1.1.1.1", "1.1.1.2"],
        )
        self.assertEqual(
            owner["absorbed_source_section_ids"],
            ["1.1.1.1", "1.1.1.2"],
        )
        self.assertEqual(
            owner["absorbed_section_ids"],
            ["1.1.1.1", "1.1.1.2"],
        )
        self.assertFalse(owner["root_has_local_text"])

        self.assertIn(
            "Alpha clinical evidence.",
            owner["text"],
        )
        self.assertIn(
            "Beta clinical evidence.",
            owner["text"],
        )

        report = validate_retrieval_section_view(
            chunks=chunks,
            section_view=view,
            max_level=3,
        )
        self.assertTrue(report["valid"], report["errors"])

    def test_structural_ancestors_are_empty_and_not_embeddable(self):
        view = build_retrieval_section_view(
            canonical_chunks(),
            max_level=3,
        )

        structural = [
            section
            for section in view
            if section["section_view_role"] == "structural"
        ]

        self.assertEqual(
            [section["section_id"] for section in structural],
            ["1", "1.1"],
        )

        for section in structural:
            self.assertEqual(section["text"], "")
            self.assertTrue(section["is_empty"])
            self.assertFalse(section["embed"])
            self.assertEqual(section["source_section_ids"], [])
            self.assertIsNone(section["retrieval_unit_id"])
            self.assertIsNone(
                section["content_owner_section_id"]
            )

    def test_build_is_deterministic_and_does_not_mutate_input(self):
        chunks = canonical_chunks()
        original = copy.deepcopy(chunks)

        first = build_retrieval_section_view(
            chunks,
            max_level=3,
        )
        second = build_retrieval_section_view(
            chunks,
            max_level=3,
        )

        self.assertEqual(first, second)
        self.assertEqual(chunks, original)

    def test_missing_required_canonical_field_fails_closed(self):
        chunks = canonical_chunks()
        del chunks[0]["excluded"]

        with self.assertRaisesRegex(
            ValueError,
            "missing fields",
        ):
            build_retrieval_section_view(
                chunks,
                max_level=3,
            )

    def test_duplicate_section_id_fails_closed(self):
        chunks = canonical_chunks()
        chunks[-1]["section_id"] = "1.1.1.1"

        with self.assertRaisesRegex(
            ValueError,
            "duplicate section_id",
        ):
            build_retrieval_section_view(
                chunks,
                max_level=3,
            )


if __name__ == "__main__":
    unittest.main()
