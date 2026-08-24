from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from managers.tables.table_catalog_manager import (
    build_catalog_for_file,
    canonicalize_table_html,
    extract_mineru_tables,
    infer_doc_id,
    load_table_config,
)


class TableCatalogManagerTests(unittest.TestCase):
    def test_extracts_rich_pdf_info_without_reading_para_blocks(self) -> None:
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        payload = {
            "pdf_info": [
                {
                    "page_idx": 2,
                    "preproc_blocks": [
                        {
                            "type": "table",
                            "index": 7,
                            "bbox": [1, 2, 3, 4],
                            "blocks": [
                                {
                                    "type": "table_caption",
                                    "lines": [
                                        {"spans": [{"content": "Table caption"}]}
                                    ],
                                },
                                {
                                    "type": "table_body",
                                    "lines": [
                                        {
                                            "spans": [
                                                {
                                                    "html": html,
                                                    "image_path": "image.jpg",
                                                }
                                            ]
                                        }
                                    ],
                                },
                                {
                                    "type": "table_footnote",
                                    "lines": [
                                        {"spans": [{"content": "Footnote"}]}
                                    ],
                                },
                            ],
                        }
                    ],
                    # Same table repeated here: it must be ignored.
                    "para_blocks": [
                        {
                            "type": "table",
                            "blocks": [
                                {
                                    "type": "table_body",
                                    "lines": [{"spans": [{"html": html}]}],
                                }
                            ],
                        }
                    ],
                }
            ],
            "_version_name": "3.4.4",
        }

        tables, metadata = extract_mineru_tables(payload)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].page_idx, 2)
        self.assertEqual(tables[0].caption, ["Table caption"])
        self.assertEqual(tables[0].footnotes, ["Footnote"])
        self.assertEqual(tables[0].image_source, "image.jpg")
        self.assertEqual(tables[0].raw_html, html)
        self.assertEqual(metadata["source_format"], "mineru_pdf_info")

    def test_extracts_flat_content_list(self) -> None:
        payload = [
            {"type": "text", "text": "ignored"},
            {
                "type": "table",
                "page_idx": 4,
                "bbox": [1, 2, 3, 4],
                "table_caption": ["Recommendations"],
                "table_footnote": ["a Footnote"],
                "table_body": "<table><tr><td>Recommendation</td></tr></table>",
                "img_path": "images/table.jpg",
            },
        ]
        tables, metadata = extract_mineru_tables(payload)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].page_idx, 4)
        self.assertEqual(metadata["source_format"], "mineru_content_list")

    def test_canonicalization_removes_outer_wrapper_and_tag_gaps(self) -> None:
        source = "<html><body>\n<table>\n<tr><td>A</td></tr>\n</table>\n</body></html>"
        self.assertEqual(
            canonicalize_table_html(source),
            "<table><tr><td>A</td></tr></table>",
        )

    def test_catalog_links_exact_table_to_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_path = root / "Demo_content_list.json"
            output_dir = root / "catalog"
            chunk_dir = root / "chunks"
            chunk_dir.mkdir()

            html = "<table><tr><td>Recommendation</td><td>Class</td><td>Level</td></tr></table>"
            source_path.write_text(
                json.dumps(
                    [
                        {
                            "type": "table",
                            "page_idx": 3,
                            "table_body": html,
                            "table_caption": ["Recommendations for testing"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (chunk_dir / "Demo_hier_chunks_clean.json").write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": "Demo:1:0",
                            "section_id": "1",
                            "section_title": "Testing",
                            "text": f"Before\n{html}\nAfter",
                            "embed": True,
                            "excluded": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            summary = build_catalog_for_file(
                source_path,
                output_dir,
                chunk_dir=chunk_dir,
            )
            self.assertEqual(summary["table_count"], 1)
            self.assertEqual(summary["linked_chunk_table_count"], 1)
            self.assertEqual(summary["link_status_counts"], {"matched_exact": 1})

            record = json.loads(
                (output_dir / "Demo_tables_raw.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(record["chunk_id"], "Demo:1:0")
            self.assertEqual(record["section_id"], "1")
            self.assertEqual(record["classification"], "recommendation_candidate")
            self.assertEqual(record["raw_html"], html)


    def test_loads_paths_from_project_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "knowledge_graph": {
                            "tables": {
                                "content_list_dir": "mineru_test/content_list",
                                "chunk_dir": "mineru_test/clean_chunks",
                                "catalog_dir": "mineru_test/table_catalogs",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            paths = load_table_config(config_path)
            self.assertEqual(paths["input_dir"], root / "mineru_test/content_list")
            self.assertEqual(paths["chunk_dir"], root / "mineru_test/clean_chunks")
            self.assertEqual(paths["output_dir"], root / "mineru_test/table_catalogs")

    def test_infer_doc_id_strips_mineru_prefix_and_timestamp(self) -> None:
        path = Path("MinerU_Cardiac_Pacing_&_CRT_2021__20260804204043.json")
        self.assertEqual(infer_doc_id(path), "Cardiac_Pacing_&_CRT_2021")


if __name__ == "__main__":
    unittest.main()
