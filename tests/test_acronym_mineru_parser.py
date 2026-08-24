#!/usr/bin/env python3
"""Deterministic regression tests for the structured MinerU acronym parser."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from managers.acronym_extractor import (
    extract_acronym_payload_from_mineru,
    looks_like_short_sequence,
)
from managers.acronym_mineru_parser import (
    extract_mineru_acronym_candidates,
    find_mineru_artifact,
    infer_doc_id,
    normalize_space,
)


def title_block(text: str, index: int) -> dict:
    return {
        "type": "title",
        "index": index,
        "lines": [{"spans": [{"type": "text", "content": text}]}],
    }


def text_block(text: str, index: int) -> dict:
    return {
        "type": "text",
        "index": index,
        "lines": [{"spans": [{"type": "text", "content": text}]}],
    }


def table_block(raw_html: str, index: int) -> dict:
    return {
        "type": "table",
        "index": index,
        "blocks": [
            {
                "type": "table_body",
                "lines": [
                    {
                        "spans": [
                            {
                                "type": "table",
                                "html": raw_html,
                            }
                        ]
                    }
                ],
            }
        ],
    }



def test_mineru_acronym_parser_regression() -> None:
    payload = {
        "pdf_info": [
            {
                "page_idx": 5,
                "preproc_blocks": [
                    title_block("Abbreviations and acronyms", 1),
                    text_block(
                        "ABC\nAlpha beta concept\n"
                        "LONG\nLong definition\ncontinued line\n"
                        "tx\nTreatment",
                        2,
                    ),
                ],
            },
            {
                "page_idx": 6,
                "preproc_blocks": [
                    table_block(
                        "<table>"
                        "<tr><td>C</td><td>Chemotherapy cycle</td>"
                        "<td>XYZ</td><td>Xylophone yield zone</td></tr>"
                        "<tr><td>M</td><td>Months</td>"
                        "<td colspan='2'>1. Preamble</td></tr>"
                        "<tr><td>N</td><td>No</td>"
                        "<td colspan='2'>Narrative body text</td></tr>"
                        "<tr><td>\\uparrow QTc</td>"
                        "<td>Corrected QT interval prolongation</td>"
                        "<td></td><td></td></tr>"
                        "</table>",
                        3,
                    )
                ],
            },
            {
                "page_idx": 7,
                "preproc_blocks": [
                    table_block(
                        "<table><tr><td>Class I</td>"
                        "<td>Is recommended</td></tr></table>",
                        4,
                    )
                ],
            },
        ]
    }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_path = root / "MinerU_Test_Doc__20260101000000.json"
        new_path = root / "MinerU_Test_Doc__20260805000000.json"
        old_path.write_text(json.dumps({"pdf_info": []}), encoding="utf-8")
        new_path.write_text(json.dumps(payload), encoding="utf-8")

        selected, count = find_mineru_artifact(root, "Test_Doc")
        assert selected == new_path.resolve(), (selected, new_path)
        assert count == 2
        assert infer_doc_id(new_path) == "Test_Doc"
        assert normalize_space(r"\uparrow QTc") == "↑QTc"

        result = extract_mineru_acronym_candidates(
            selected,
            doc_id="Test_Doc",
            short_detector=looks_like_short_sequence,
            candidate_file_count=count,
        )
        pairs = {item.short: item.definition for item in result.candidates}

        assert result.heading_found
        assert result.page_start_idx == 5
        assert result.page_end_idx == 6
        assert pairs["ABC"] == "Alpha beta concept"
        assert pairs["LONG"] == "Long definition continued line"
        assert pairs["tx"] == "Treatment"
        assert pairs["C"] == "Chemotherapy cycle"
        assert pairs["M"] == "Months"
        assert pairs["N"] == "No"
        assert pairs["XYZ"] == "Xylophone yield zone"
        assert pairs["↑QTc"] == "Corrected QT interval prolongation"
        assert "Class I" not in pairs
        assert "1. Preamble" not in pairs

        full_payload = extract_acronym_payload_from_mineru(
            doc_id="Test_Doc",
            mineru_dir=root,
            mineru_file=new_path,
        )
        assert full_payload is not None
        assert full_payload["status"] == "success"
        assert full_payload["source"] == "mineru_front_matter"
        assert isinstance(full_payload["acronyms"], dict)
        assert full_payload["n_acronyms"] == len(full_payload["acronyms"])
        assert full_payload["pdf_fallback_used"] is False
        assert full_payload["mineru_conflict_count"] == 0
        assert full_payload["acronyms"]["C"] == "Chemotherapy cycle"
