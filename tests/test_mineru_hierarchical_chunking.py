import unittest
from types import SimpleNamespace


from managers.hierarchical_chunking_manager import (
    build_hierarchical_chunks,
    is_effectively_empty,
    validate_section_boundaries,
)


def sec(section_id, title, level=1, children=None):
    return {
        "id": section_id,
        "printed_id": section_id,
        "title": title,
        "level": level,
        "page_start": 1,
        "page_end": 1,
        "type": "body",
        "children": children or [],
    }


def chunks_for(markdown, toc_tree):
    manager = SimpleNamespace(text=markdown, filepath=None)
    return build_hierarchical_chunks(
        toc_tree=toc_tree,
        markdown_manager=manager,
        anchors={},
        doc_id="doc",
        min_words=1,
    )


class MinerUHierarchicalChunkingTests(unittest.TestCase):
    def test_markdown_markup_only_text_is_effectively_empty(self):
        self.assertTrue(is_effectively_empty(""))
        self.assertTrue(is_effectively_empty("   "))
        self.assertTrue(is_effectively_empty("##"))
        self.assertTrue(is_effectively_empty("***"))
        self.assertTrue(is_effectively_empty("##\n"))
        self.assertTrue(is_effectively_empty("##\n***"))
        self.assertFalse(is_effectively_empty("Clinical management"))
        self.assertFalse(is_effectively_empty("**Clinical management**"))
        self.assertFalse(is_effectively_empty("# Heading with words"))
        self.assertFalse(is_effectively_empty("`HCM`"))

    def test_markdown_heading_prefix_is_part_of_next_anchor(self):
        toc = [
            sec(
                "7.2",
                "Dilated cardiomyopathy",
                children=[
                    sec(
                        "7.2.1",
                        "Diagnosis",
                        level=2,
                        children=[sec("7.2.1.1", "Index case", level=3)],
                    )
                ],
            )
        ]
        markdown = (
            "## 7.2. Dilated cardiomyopathy\n\n"
            "## 7.2.1. Diagnosis\n\n"
            "## 7.2.1.1. Index case\n\n"
            "Dilated cardiomyopathy is defined by ventricular dilatation "
            "and systolic dysfunction."
        )

        chunks = chunks_for(markdown, toc)
        report = validate_section_boundaries(toc, chunks, markdown, "doc")
        by_id = {chunk["section_id"]: chunk for chunk in chunks}

        self.assertEqual(by_id["7.2"]["text"], "")
        self.assertTrue(by_id["7.2"]["is_empty"])
        self.assertFalse(by_id["7.2"]["embed"])
        self.assertEqual(by_id["7.2.1"]["text"], "")
        self.assertTrue(by_id["7.2.1"]["is_empty"])
        self.assertFalse(by_id["7.2.1"]["embed"])
        self.assertIn("ventricular dilatation", by_id["7.2.1.1"]["text"])
        self.assertFalse(
            any(chunk["text"].rstrip().endswith("##") for chunk in chunks)
        )
        self.assertEqual(report["child_text_also_in_parent"], [])

    def test_parent_direct_text_does_not_include_child_markdown_marker(self):
        toc = [
            sec(
                "6",
                "Diagnostic work-up",
                children=[sec("6.1", "Initial assessment", level=2)],
            )
        ]
        markdown = (
            "## 6. Diagnostic work-up\n\n"
            "Clinical evaluation should follow a systematic diagnostic pathway.\n\n"
            "## 6.1. Initial assessment\n\n"
            "Initial assessment includes history and examination."
        )

        chunks = chunks_for(markdown, toc)
        by_id = {chunk["section_id"]: chunk for chunk in chunks}
        parent = by_id["6"]

        self.assertEqual(
            parent["text"],
            "Clinical evaluation should follow a systematic diagnostic pathway.",
        )
        self.assertFalse(parent["is_empty"])
        self.assertTrue(parent["embed"])
        self.assertFalse(parent["text"].endswith("##"))
        self.assertNotIn("6.1. Initial assessment", parent["text"])

    def test_validator_still_reports_significant_child_duplication(self):
        toc = [sec("1", "Parent", children=[sec("1.1", "Child", level=2)])]
        chunks = [
            {
                "section_id": "1",
                "text": "Clinical management text.",
                "is_empty": False,
                "quality_flags": [],
            },
            {
                "section_id": "1.1",
                "text": "Clinical management text.",
                "is_empty": False,
                "quality_flags": [],
            },
        ]

        report = validate_section_boundaries(toc, chunks, "", "doc")

        self.assertEqual(
            report["child_text_also_in_parent"],
            [{"parent_section_id": "1", "child_section_id": "1.1"}],
        )

    def test_heading_with_same_line_prose_preserves_prose(self):
        toc = [sec("1", "Management"), sec("2", "Next")]
        markdown = (
            "## Table of contents\n"
            "1. Management ..... 10\n\n"
            "1. Management By convention, patients should be reviewed.\n\n"
            "2. Next\nAfter."
        )

        chunks = chunks_for(markdown, toc)
        first = chunks[0]

        self.assertEqual(first["section_id"], "1")
        self.assertIn("By convention, patients should be reviewed.", first["text"])
        self.assertNotIn("1. Management", first["text"])

    def test_fused_headings_do_not_absorb_child_into_parent(self):
        toc = [
            sec(
                "6.4",
                "Cardiac arrhythmias",
                children=[sec("6.4.1", "Atrial fibrillation", level=2)],
            )
        ]
        markdown = (
            "## 6.4. Cardiac arrhythmias 6.4.1. Atrial fibrillation\n\n"
            "AF may occur in patients with cancer."
        )

        chunks = chunks_for(markdown, toc)
        by_id = {chunk["section_id"]: chunk for chunk in chunks}

        self.assertEqual(by_id["6.4"]["text"], "")
        self.assertIn("AF may occur", by_id["6.4.1"]["text"])

    def test_printed_toc_duplicate_is_ignored(self):
        # Use a retrieval-active section. "Preamble" is intentionally excluded
        # by the current section policy, so it is not a valid fixture for
        # testing duplicate printed-TOC anchor rejection.
        toc = [sec("1", "Overview"), sec("2", "Introduction")]
        markdown = (
            "## Table of contents\n"
            "1. Overview ..... 3509\n"
            "2. Introduction ..... 3511\n\n"
            "# 1. Overview\n\n"
            "Guidelines evaluate available evidence.\n\n"
            "## 2. Introduction\nIntro."
        )

        chunks = chunks_for(markdown, toc)

        self.assertFalse(chunks[0]["excluded"])
        self.assertTrue(chunks[0]["embed"])
        self.assertTrue(chunks[0]["text"].startswith("Guidelines evaluate"))

    def test_parent_without_direct_text_is_kept_empty(self):
        toc = [sec("1", "Parent", children=[sec("1.1", "Child", level=2)])]
        markdown = "1. Parent\n1.1. Child\nChild text."

        chunks = chunks_for(markdown, toc)
        by_id = {chunk["section_id"]: chunk for chunk in chunks}

        self.assertTrue(by_id["1"]["is_empty"])
        self.assertEqual(by_id["1"]["text"], "")
        self.assertIn("Child text.", by_id["1.1"]["text"])

    def test_empty_clinical_leaf_is_reported(self):
        toc = [sec("1", "Empty leaf"), sec("2", "Next")]
        markdown = "1. Empty leaf\n2. Next\nNext text."
        chunks = chunks_for(markdown, toc)
        report = validate_section_boundaries(toc, chunks, markdown, "doc")

        self.assertIn("1", report["empty_leaf_sections"])

    def test_false_front_matter_table_can_release_active_section_text(self):
        # Preserve the false-HTML-table regression while avoiding "Preamble",
        # which is intentionally excluded by the current retrieval policy.
        toc = [sec("1", "Overview"), sec("2", "Introduction")]
        markdown = (
            '<table><tr><td>ABC</td><td>Acronym body</td>'
            '<td colspan="2">1. Overview</td></tr>'
            '<tr><td>DEF</td><td>Definition</td>'
            '<td colspan="2" rowspan="2">Guidelines evaluate and summarize evidence.</td>'
            "</tr></table>\n\n"
            "## 2. Introduction\nIntro."
        )

        chunks = chunks_for(markdown, toc)

        self.assertFalse(chunks[0]["excluded"])
        self.assertTrue(chunks[0]["embed"])
        self.assertIn(
            "Guidelines evaluate and summarize evidence.",
            chunks[0]["text"],
        )
        self.assertNotIn("DEF", chunks[0]["text"])

    def test_real_table_stays_atomic_inside_section(self):
        toc = [sec("1", "Tables"), sec("2", "Next")]
        markdown = (
            "1. Tables\n\n"
            "<table><tr><td>6.4. Cardiac arrhythmias</td><td>Value</td></tr></table>\n\n"
            "2. Next\nNext text."
        )

        chunks = chunks_for(markdown, toc)

        self.assertIn("<table>", chunks[0]["text"])
        self.assertIn("</table>", chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
