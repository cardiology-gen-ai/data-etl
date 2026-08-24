#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from managers.fixed_chunking_manager import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TABLE_RE,
    build_fixed_corpus,
    split_section_text,
    validate_fixed_corpus,
    write_fixed_corpus,
)


def source_row(
    *,
    chunk_id: str,
    section_id: str,
    text: str,
    embed: bool = True,
    excluded: bool = False,
    is_empty: bool = False,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "Demo_2026",
        "section_id": section_id,
        "printed_section_id": section_id,
        "parent_section_id": None,
        "section_title": f"Section {section_id}",
        "section_level": 1,
        "section_type": "body",
        "page_start": 1,
        "page_end": 2,
        "is_empty": is_empty,
        "excluded": excluded,
        "embed": embed,
        "part_index": 0,
        "part_count": 1,
        "quality_flags": [],
        "boundary_source": "test",
        "text": text,
    }


class SplitSectionTextTests(unittest.TestCase):
    def test_fixed_chunks_respect_limit_and_overlap(self) -> None:
        text = " ".join(f"word{i}" for i in range(120))
        chunks = split_section_text(text, chunk_size=120, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["char_count"] <= 120 for chunk in chunks))
        self.assertTrue(all(chunk["text"].strip() for chunk in chunks))
        self.assertTrue(all(not chunk["contains_table"] for chunk in chunks))

    def test_table_is_atomic_and_not_duplicated(self) -> None:
        table = "<table><tr><td>" + ("clinical value " * 30) + "</td></tr></table>"
        text = f"Paragraph before the table.\n\n{table}\n\nParagraph after the table."
        chunks = split_section_text(text, chunk_size=100, chunk_overlap=20)
        rendered_tables = []
        for chunk in chunks:
            rendered_tables.extend(TABLE_RE.findall(chunk["text"]))
        self.assertEqual(rendered_tables, [table])
        table_chunk = next(chunk for chunk in chunks if chunk["contains_table"])
        self.assertEqual(table_chunk["text"], table)
        self.assertTrue(table_chunk["oversized_atomic_table"])

    def test_invalid_overlap_fails(self) -> None:
        with self.assertRaises(ValueError):
            split_section_text("abc", chunk_size=10, chunk_overlap=10)

    def test_standalone_continued_is_removed(self) -> None:
        table_a = "<table><tr><td>" + ("A " * 80) + "</td></tr></table>"
        table_b = "<table><tr><td>" + ("B " * 80) + "</td></tr></table>"
        text = f"{table_a}\n\nContinued\n\n{table_b}"
        chunks = split_section_text(text, chunk_size=100, chunk_overlap=20)
        self.assertEqual([chunk["text"] for chunk in chunks], [table_a, table_b])

    def test_recommended_defaults(self) -> None:
        self.assertEqual(DEFAULT_CHUNK_SIZE, 2000)
        self.assertEqual(DEFAULT_CHUNK_OVERLAP, 300)


class BuildFixedCorpusTests(unittest.TestCase):
    def _write_input(self, directory: Path, rows: list[dict]) -> Path:
        path = directory / "Demo_2026_hier_chunks_clean.json"
        path.write_text(
            json.dumps({"chunks": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def test_only_active_rows_are_emitted_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = [
                source_row(
                    chunk_id="Demo_2026::1",
                    section_id="1",
                    text="Active text " * 40,
                ),
                source_row(
                    chunk_id="Demo_2026::2",
                    section_id="2",
                    text="Excluded text",
                    excluded=True,
                    embed=False,
                ),
                source_row(
                    chunk_id="Demo_2026::3",
                    section_id="3",
                    text="Structural text",
                    embed=False,
                ),
                source_row(
                    chunk_id="Demo_2026::4",
                    section_id="4",
                    text="",
                    is_empty=True,
                    embed=False,
                ),
            ]
            input_path = self._write_input(directory, rows)
            payload = build_fixed_corpus(
                input_path,
                chunk_size=100,
                chunk_overlap=20,
            )
            self.assertEqual(payload["eligible_source_record_count"], 1)
            self.assertGreater(payload["output_chunk_count"], 1)
            self.assertEqual(payload["skipped_source_records"]["excluded"], 1)
            self.assertEqual(payload["skipped_source_records"]["not_embeddable"], 2)
            for chunk in payload["chunks"]:
                self.assertEqual(chunk["source_section_ids"], ["1"])
                self.assertEqual(chunk["source_chunk_ids"], ["Demo_2026::1"])
                self.assertTrue(chunk["embed"])
                self.assertFalse(chunk["excluded"])
            report = validate_fixed_corpus(payload, source_rows=rows)
            self.assertTrue(report["valid"])

    def test_output_is_deterministic_and_overwrite_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = [
                source_row(
                    chunk_id="Demo_2026::1",
                    section_id="1",
                    text="Deterministic clinical prose. " * 25,
                )
            ]
            input_path = self._write_input(directory, rows)
            output_dir = directory / "fixed"
            output_path, first = write_fixed_corpus(
                input_path,
                output_dir,
                chunk_size=120,
                chunk_overlap=20,
            )
            first_bytes = output_path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_fixed_corpus(
                    input_path,
                    output_dir,
                    chunk_size=120,
                    chunk_overlap=20,
                )
            output_path_2, second = write_fixed_corpus(
                input_path,
                output_dir,
                chunk_size=120,
                chunk_overlap=20,
                force=True,
            )
            self.assertEqual(output_path, output_path_2)
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, output_path_2.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
