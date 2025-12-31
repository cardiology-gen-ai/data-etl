"""
Hierarchical Chunking Manager (GraphRAG-ready)

- Uses TOC hierarchy + page anchors + markdown headings
- Produces embedding-safe hierarchical chunks
- Excludes front matter and back matter
- Saves chunks to disk
"""

import json
import pathlib
import logging
import re
from typing import Dict, List, Any

from markdown_manager import MarkdownManager



# Logging

logger = logging.getLogger("hierarchical_chunker")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)



# TOC helpers
def get_leaf_sections(toc_tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return all leaf sections from a hierarchical TOC."""
    leaves: List[Dict[str, Any]] = []

    def visit(node: Dict[str, Any]):
        children = node.get("children", [])
        if not children:
            leaves.append(node)
        else:
            for c in children:
                visit(c)

    for root in toc_tree:
        visit(root)

    return leaves


# Page-based slicing (narrowing only)
def extract_markdown_by_pages(
    text: str,
    anchors: Dict[int, int],
    page_start: int,
    page_end: int,
) -> str:
    """Slice markdown using cached or computed page anchors."""
    if not anchors:
        return text

    pages = sorted(anchors.keys())

    start_page = max(p for p in pages if p <= page_start)
    start_idx = anchors[start_page]

    later_pages = [p for p in pages if p > page_end]
    end_idx = anchors[min(later_pages)] if later_pages else len(text)

    return text[start_idx:end_idx]



# Heading-based extraction
def extract_section_by_header(markdown: str, section_id: str) -> str:
    """
    Extract section content using numbered ESC-style headers.

    Matches:
    **7.1. Diagnosis**
    ### **3.1. Definitions**
    """

    escaped_id = re.escape(section_id)

    header_rx = re.compile(
        rf"(?m)^\s*(?:#+\s*)?\*{{0,2}}{escaped_id}\.?\s+.+$"
    )

    m = header_rx.search(markdown)
    if not m:
        return ""

    start = m.end()

    next_header_rx = re.compile(
        r"(?m)^\s*(?:#+\s*)?\*{0,2}\d+(?:\.\d+)*\.?\s+.+$"
    )

    m_next = next_header_rx.search(markdown, pos=start)
    end = m_next.start() if m_next else len(markdown)

    return markdown[start:end].strip()



# Adaptive splitting (embedding-safe)

def split_if_too_large(
    text: str,
    section_id: str,
    max_chars: int = 4000,
) -> List[str]:
    """
    Split large sections using inline numbered subsections
    only if size exceeds threshold.
    """

    if len(text) <= max_chars:
        return [text]

    sub_rx = re.compile(
        rf"(?m)^\s*(?:#+\s*)?\*{{0,2}}{re.escape(section_id)}\.\d+(?:\.\d+)*\.?\s+.+$"
    )

    matches = list(sub_rx.finditer(text))
    if not matches:
        return [text]

    chunks: List[str] = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    return chunks if chunks else [text]


# Chunk builder
def build_hierarchical_chunks(
    toc_tree: List[Dict[str, Any]],
    markdown_manager: MarkdownManager,
    anchors: Dict[int, int],
    doc_id: str,
) -> List[Dict[str, Any]]:

    logger.info("Extracting leaf sections from TOC")
    leaf_sections = get_leaf_sections(toc_tree)
    logger.info("Found %d leaf sections", len(leaf_sections))

    chunks: List[Dict[str, Any]] = []

    for sec in leaf_sections:

        if sec["type"] in {"front_matter", "back_matter"}:
            continue

        narrowed = extract_markdown_by_pages(
            text=markdown_manager.text,
            anchors=anchors,
            page_start=sec["page_start"],
            page_end=sec["page_end"],
        )

        section_text = extract_section_by_header(
            markdown=narrowed,
            section_id=sec.get("id"),
        )

        if not section_text:
            continue

        split_texts = split_if_too_large(
            text=section_text,
            section_id=sec.get("id"),
        )

        for i, txt in enumerate(split_texts):
            chunks.append({
                "chunk_id": f"{doc_id}:{sec.get('id')}:{i}",
                "doc_id": doc_id,
                "section_id": sec.get("id"),
                "section_title": sec["title"],
                "section_level": sec["level"],
                "section_type": sec["type"],
                "page_start": sec["page_start"],
                "page_end": sec["page_end"],
                "text": txt,
            })

    logger.info("Built %d hierarchical chunks", len(chunks))
    return chunks



# Main entry point
def run_hierarchical_chunking(
    toc_path: pathlib.Path,
    markdown_path: pathlib.Path,
    pdf_path: pathlib.Path,
    anchor_cache_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:

    logger.info("Starting hierarchical chunking")

    toc = json.loads(toc_path.read_text(encoding="utf-8"))
    markdown_text = markdown_path.read_text(encoding="utf-8")

    markdown_manager = MarkdownManager(
        filepath=pdf_path,
        text=markdown_text,
    )

    anchors = markdown_manager.get_page_anchors(
        cache_path=anchor_cache_path
    )

    chunks = build_hierarchical_chunks(
        toc_tree=toc["toc_tree"],
        markdown_manager=markdown_manager,
        anchors=anchors,
        doc_id=toc["doc_id"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Saved chunks to %s", output_path)
    logger.info("Hierarchical chunking completed successfully")

if __name__ == "__main__":
    run_hierarchical_chunking(
        toc_path=pathlib.Path("../../test_data/toc/Cardiomyopathies_2023_toc.json"),
        markdown_path=pathlib.Path("../../test_data/mddocs/Cardiomyopathies_2023.md"),
        pdf_path=pathlib.Path("../../test_data/pdfdocs/Cardiomyopathies_2023.pdf"),
        anchor_cache_path=pathlib.Path(
            "../../test_data/anchors/Cardiomyopathies_2023_page_anchors.json"
        ),
        output_path=pathlib.Path(
            "../../test_data/chunks/Cardiomyopathies_2023_hier_chunks.json"
        ),
    )
