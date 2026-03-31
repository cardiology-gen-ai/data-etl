"""
Hierarchical Chunker 

- Anchor-narrowed start detection (robust)
- TOC-order end boundaries (for leaf sections with no children)
- Parents stop at first child (prevents parent swallowing children)
- Skips TOC dot-leader lines as headers (or else we ingest TOC entries)
- Keeps empty parents, drops empty leaves
- Does NOT split oversized chunks #TODO: implement splitting later if needed, depends on use case
"""

import json
import pathlib
import logging
import re
from typing import Dict, List, Any, Optional

from managers.markdown_manager import MarkdownManager

# Logging
logger = logging.getLogger("hierarchical_chunker")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# TODO: Exclusion rules (check if there are others to add)
EXCLUDED_TITLE_KEYWORDS = [
    "table of contents", "list of figures", "list of tables",
    "references", "bibliography",
    "abbreviations", "acronyms",
    "acknowledgements", "acknowledgments",
    "appendix", "supplementary data",
    "data availability",
    "author information", "disclaimer",
    "index"
]

def is_excluded_section(sec: Dict[str, Any]) -> bool:
    title = (sec.get("title") or "").lower()
    if sec.get("type") in {"front_matter", "back_matter", "toc"}:
        return True
    return any(k in title for k in EXCLUDED_TITLE_KEYWORDS)


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))

# TODO: refine leakage detection (what should be considered maximum leakage length?)
def looks_like_leakage(text: str, min_words: int = 10) -> bool:
    if not text or not text.strip():
        return True
    if word_count(text) < min_words:
        return True
    first = text.strip().split("\n", 1)[0]
    # dot-leader TOC line
    if re.search(r"\.{5,}\s*\d+\s*$", first):
        return True
    return False


def is_toc_entry_line(line: str) -> bool:
    # e.g. "1. Preamble ............ 3509"
    return bool(re.search(r"\.{5,}\s*\d+\s*$", line.strip()))



# Find all sections in TOC tree as flat list with parent references
def collect_sections(
    toc_nodes: List[Dict[str, Any]],
    parent_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in toc_nodes:
        entry = {
            **node,
            "_parent_id": parent_id,
            "_has_children": bool(node.get("children")),
        }
        out.append(entry)
        for child in node.get("children", []):
            out.extend(collect_sections([child], node.get("id")))
    return out




def _normalize_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())

# Compute Jaccard overlap score between line title and section title
def _title_overlap_score(line_title: str, section_title: Optional[str]) -> float:
    if not section_title:
        return 0.0
    lt = set(_normalize_tokens(line_title))
    st = set(_normalize_tokens(section_title))
    if not lt or not st:
        return 0.0
    return len(lt & st) / len(lt | st)


# Body start (skip front matter / TOC)
def find_body_start(markdown: str) -> int:
    """
    Find the first real numbered markdown heading (Section 1),
    e.g. "### **1. Preamble**".
    """
    rx = re.compile(r"(?m)^\s*#{1,6}\s+.*?\b1\.\s+.+$")
    m = rx.search(markdown)
    return m.start() if m else 0


def _compute_search_window(
    anchors: Dict[int, int],
    page_start: int,
    page_end: int,
    markdown_len: int,
    window: int = 1,
) -> tuple[int, int]:
    """
    Returns (start_offset, end_offset) in markdown for a page range.
    Falls back to (0, len(markdown)) if anchors are missing.
    """
    if not anchors:
        return 0, markdown_len

    pages = sorted(anchors.keys())
    if not pages:
        return 0, markdown_len

    start_page = max(min(page_start - window, pages[-1]), pages[0])
    end_page = min(max(page_end + window, pages[0]), pages[-1])

    start_offset = anchors.get(start_page, anchors[pages[0]])
    end_offset = anchors.get(end_page + 1, markdown_len)

    return start_offset, end_offset



# Header locator (returns header start/end offsets)
def locate_header(
    markdown: str,
    section_id: str,
    section_title: Optional[str],
    search_start: int,
    search_end: int,
) -> Optional[tuple[int, int]]:
    """
    Find the best matching header line for section_id inside [search_start, search_end].
    Returns (header_start, header_end) or None.
    Skips TOC dot-leader lines.
    """

    escaped = re.escape(section_id.strip())
    depth = len(section_id.split("."))

    # Important: enforce boundary ex: "2D" doesn't match section "2" (form the abbreviation lists to avoid leakage)
    header_rx = re.compile(
        rf"""
        (?m)^\s*
        (?:\#{1,6}\s*)?
        .*?
        (?P<num>{escaped})
        (?:\.(?=\s)|(?=\s|$))
        \s*(?P<title>.*)?
        $
        """,
        re.VERBOSE,
    )

    matches = list(header_rx.finditer(markdown, search_start, search_end))
    if not matches:
        return None

    best = None
    best_score = -1.0

    for m in matches:
        line = m.group(0)
        if is_toc_entry_line(line):
            continue

        title = (m.group("title") or "").strip()

        # If header line is just "1." peek next non-empty line
        if not title:
            tail = markdown[m.end():search_end]
            for ln in tail.splitlines():
                if ln.strip():
                    title = ln.strip()
                    break

        score = _title_overlap_score(title, section_title)

        if score > best_score:
            best = m
            best_score = score

    if best is None:
        return None

    if depth == 1 and section_title and best_score < 0.15:
        return None

    return best.start(), best.end()



# Content extraction with TOC boundaries

def extract_section_text(
    markdown: str,
    anchors: Dict[int, int],
    sec: Dict[str, Any],
    next_sec: Optional[Dict[str, Any]],
    first_child: Optional[Dict[str, Any]],
    body_start: int,
    window: int = 1,
) -> str:
    """
    Extract section content using:
    - header position found within anchor window (fallback to global guarded)
    - end boundary:
        - first child header if present
        - else next section header (TOC order)
        - else window end / EOF
    """

    md_len = len(markdown)

    # 1) narrow the search region for this section
    win_start, win_end = _compute_search_window(
        anchors=anchors,
        page_start=sec["page_start"],
        page_end=sec["page_end"],
        markdown_len=md_len,
        window=window,
    )

    # never search before body_start in global windows
    win_start = max(win_start, body_start)

    # Find this section header (prefer within window)
    header_pos = locate_header(
        markdown=markdown,
        section_id=sec["id"],
        section_title=sec.get("title"),
        search_start=win_start,
        search_end=win_end,
    )

    # fallback: guarded global search (still skips front matter)
    if header_pos is None:
        header_pos = locate_header(
            markdown=markdown,
            section_id=sec["id"],
            section_title=sec.get("title"),
            search_start=body_start,
            search_end=md_len,
        )

    if header_pos is None:
        return ""

    header_start, header_end = header_pos
    content_start = header_end

    # Determine end boundary
    end_candidates: List[int] = []

    # End at first child header (prevents parent swallowing children)
    if first_child:
        child_header = locate_header(
            markdown=markdown,
            section_id=first_child["id"],
            section_title=first_child.get("title"),
            search_start=content_start,
            search_end=win_end,
        )
        if child_header is None:
            child_header = locate_header(
                markdown=markdown,
                section_id=first_child["id"],
                section_title=first_child.get("title"),
                search_start=content_start,
                search_end=md_len,
            )
        if child_header:
            end_candidates.append(child_header[0])

    # If no child boundary, end at next section in TOC order
    if not end_candidates and next_sec:
        next_header = locate_header(
            markdown=markdown,
            section_id=next_sec["id"],
            section_title=next_sec.get("title"),
            search_start=content_start,
            search_end=md_len,
        )
        if next_header:
            end_candidates.append(next_header[0])

    # Fallback end = window end or EOF
    end_candidates.append(win_end if win_end > content_start else md_len)

    content_end = min(end_candidates)
    return markdown[content_start:content_end].strip()



# Chunk builder (no splitting yet)
def build_hierarchical_chunks(
    toc_tree: List[Dict[str, Any]],
    markdown_manager: MarkdownManager,
    anchors: Dict[int, int],
    doc_id: str,
    min_words: int = 10,
) -> List[Dict[str, Any]]:

    sections = collect_sections(toc_tree)
    logger.info("TOC nodes considered: %d", len(sections))

    # Keep a stable linear order (pre-order) and index by id
    ordered = [s for s in sections if s.get("id")]
    by_id = {s["id"]: s for s in ordered}

    body_start = find_body_start(markdown_manager.text)

    chunks: List[Dict[str, Any]] = []

    for idx, sec in enumerate(ordered):
        sec_id = sec["id"]

        if is_excluded_section(sec):
            continue

        # next section in TOC pre-order (works for leaf sections)
        next_sec = ordered[idx + 1] if idx + 1 < len(ordered) else None

        # first child section (works for parents)
        first_child = None
        children = sec.get("children") or []
        if children:
            # children in flattened list are already entries; but sec here is flat entry
            # so we look up by id from original sec dict if present
            # If 'children' was not kept in this flat entry, we fallback to TOC ordering:
            # in pre-order, the next node is typically the first child.
            if next_sec and next_sec.get("_parent_id") == sec_id:
                first_child = next_sec

        raw_text = extract_section_text(
            markdown=markdown_manager.text,
            anchors=anchors,
            sec=sec,
            next_sec=next_sec,
            first_child=first_child,
            body_start=body_start,
            window=1,
        )

        empty = looks_like_leakage(raw_text, min_words=min_words)

        # Drop empty leaves, keep empty parents
        if empty and not sec["_has_children"]:
            continue

        chunks.append({
            "chunk_id": f"{doc_id}:{sec_id}:0",
            "doc_id": doc_id,
            "section_id": sec_id,
            "parent_section_id": sec["_parent_id"],
            "section_title": sec.get("title"),
            "section_level": sec.get("level"),
            "section_type": sec.get("type"),
            "page_start": sec.get("page_start"),
            "page_end": sec.get("page_end"),
            "text": "" if empty else raw_text,
            "is_empty": empty,
            "embed": not empty,
        })

    logger.info("Built %d chunks", len(chunks))
    return chunks


def main():
    toc_path = pathlib.Path("../../data/toc/Valvular_Heart_Disease_2021_toc.json")
    markdown_path = pathlib.Path("../../data/mddocs/Valvular_Heart_Disease_2021.md")
    pdf_path = pathlib.Path("../../data/pdfdocs/Valvular_Heart_Disease_2021.pdf")
    anchor_cache_path = pathlib.Path(
        "../../data/anchors/Valvular_Heart_Disease_2021_page_anchors.json"
    )
    output_path = pathlib.Path(
        "../../test_data/chunks/Valvular_Heart_Disease_2021_hier_chunks.json"
    )

    logger.info("Starting hierarchical chunking (anchors + TOC boundaries, no split)")

    toc = json.loads(toc_path.read_text(encoding="utf-8"))
    markdown_text = markdown_path.read_text(encoding="utf-8")

    markdown_manager = MarkdownManager(filepath=pdf_path, text=markdown_text)
    anchors = markdown_manager.get_page_anchors(cache_path=anchor_cache_path)

    toc_tree = toc.get("toc_tree") or toc["flat_toc"][0]["children"]

    chunks = build_hierarchical_chunks(
        toc_tree=toc_tree,
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
    logger.info("Done.")


if __name__ == "__main__":
    main()
