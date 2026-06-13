from logging import Logger
from typing import Dict, List, Optional

import fitz
from docling_core.types.doc import DocItemLabel, DoclingDocument, TextItem

from managers.toc_extraction.toc_configs import BaseTocConfig


# ---------------------------------------------------------------------------
# Document title detection (used by TOCExtractionManager + MarkdownConverter)
# ---------------------------------------------------------------------------

def _is_valid_title(title: str, toc_config: BaseTocConfig) -> bool:
    """Return True if ``title`` looks like a real document title."""
    t = title.strip()
    if len(t) < 10:
        return False
    if toc_config.bad_doc_title_patterns is None:
        return True
    return not any(p.search(t) for p in toc_config.bad_doc_title_patterns)


def _extract_document_title_from_doc(
    docling_doc: DoclingDocument,
    pdf_path: str,
    toc_config: BaseTocConfig,
) -> Optional[str]:
    """Detect the document title via 2-path fallback.

    Path 1 — first ``DocItemLabel.TITLE`` item in the DoclingDocument.
    Path 2 — ``title`` field in PDF metadata via fitz (validated).

    Path 3 (unmatched-heading heuristic) is handled separately inside
    ``infer_heading_levels`` after entries are built.

    Parameters
    ----------
    docling_doc : DoclingDocument
        Converted document.
    pdf_path : str
        Path to the source PDF (used for fitz metadata lookup).
    toc_config : BaseTocConfig
        Config providing ``bad_doc_title_patterns`` for validation.

    Returns
    -------
    str or None
        Detected title, or ``None`` if both paths fail.
    """
    # Path 1: Docling TITLE item
    for item, _ in docling_doc.iterate_items():
        if isinstance(item, TextItem) and item.label == DocItemLabel.TITLE:
            if item.text and _is_valid_title(item.text, toc_config):
                return item.text.strip()

    # Path 2: fitz PDF metadata
    try:
        fitz_doc = fitz.open(pdf_path)
        raw_title = (fitz_doc.metadata or {}).get("title", "").strip()
        fitz_doc.close()
        if raw_title and _is_valid_title(raw_title, toc_config):
            return raw_title
    except (FileNotFoundError, RuntimeError, KeyError):
        pass

    return None


# ---------------------------------------------------------------------------
# Section id generation
# ---------------------------------------------------------------------------

def _generate_sequential_id(level: int, level_counters: Dict[int, int]) -> str:
    """Update ``level_counters`` and return the new dotted section id.

    Modifies ``level_counters`` in place.

    Parameters
    ----------
    level : int
        Current heading level (1-based).
    level_counters : Dict[int, int]
        Mutable counter dict shared across calls for one extraction run.

    Returns
    -------
    str
        Dotted id string, e.g. ``"3.1.2"``.
    """
    level_counters[level] = level_counters.get(level, 0) + 1
    # reset deeper levels when we step back up
    for k in [k for k in level_counters if k > level]:
        del level_counters[k]
    return ".".join(str(level_counters[l]) for l in sorted(level_counters) if l <= level)


# ---------------------------------------------------------------------------
# TOC list operations (shared post-processing)
# ---------------------------------------------------------------------------

def _deduplicate_outline(sections: List) -> List:
    """Remove duplicate TOC entries (same level, id, title, page_start)."""
    seen: set = set()
    out = []
    for s in sections:
        key = (s.level, s.id, s.title, s.page_start)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _set_terminal_section(
    sections: List,
    toc_config: BaseTocConfig,
    logger: Logger,
) -> List:
    """Truncate the TOC after the first back-matter boundary section."""
    for i, s in enumerate(sections):
        if (s.title or "").lower().strip().startswith(toc_config.terminal_sections_starts):
            logger.info(f"Truncating TOC after '{s.title}'")
            return sections[:i + 1]
    return sections


def _toc_needs_sorting(sections: List, logger: Logger) -> bool:
    """Return True if the TOC level ordering is structurally broken."""
    if not sections:
        return False
    if sections[0].level != 1:
        logger.warning("TOC sanity: first entry is not level 1")
        return True
    seen_levels = {1}
    for s in sections:
        if s.level == 1:
            continue
        if s.level - 1 not in seen_levels:
            logger.warning(f"TOC sanity: level {s.level} without parent ('{s.title or ''}')")
            return True
        seen_levels.add(s.level)
    return False


def _safe_sort(sections: List, logger: Logger) -> List:
    """Sort sections by page_start as a structural fallback."""
    logger.info("Applying safe fallback TOC sorting by page_start")
    return sorted(sections, key=lambda s: s.page_start)


def normalize_toc_levels(sections: List) -> List:
    """Remove document-title entry and remap levels to fill gaps.

    Step 1 — If the first section is the sole level-1 entry and all
    others are deeper, it is the document title: remove it.

    Step 2 — Remap remaining levels so they start at 1 with no gaps,
    e.g. [2, 4, 5, 8] -> [1, 2, 3, 4].
    """
    if not sections:
        return sections
    if sections[1:] and sections[0].level == 1 and all(s.level > 1 for s in sections[1:]):
        sections = sections[1:]
    if not sections:
        return sections
    unique_levels = sorted(set(s.level for s in sections))
    level_map = {old: new for new, old in enumerate(unique_levels, start=1)}
    for section in sections:
        section.level = level_map[section.level]
    return sections


# ---------------------------------------------------------------------------
# Page-range computation
# ---------------------------------------------------------------------------

def compute_outline_ranges(
    sections: List,
    n_pages: int,
    doc: fitz.Document,
    heading_top_y_threshold: int,
) -> None:
    """Compute ``page_end`` for each section using the next peer's start.

    Uses fitz to detect whether a heading starts near the top of its page
    (suggesting it belongs to the previous page's range).

    Modifies sections in place.
    """
    for i, section in enumerate(sections):
        next_section = next(
            (sections[j] for j in range(i + 1, len(sections)) if sections[j].level <= section.level),
            None,
        )
        if not next_section:
            section.page_end = n_pages
            continue
        end = next_section.page_start - 1
        if next_section.page_start <= n_pages:
            y = _find_heading_top_y(doc, n_pages, next_section)
            if y is not None and y > heading_top_y_threshold:
                end = next_section.page_start
        section.page_end = max(section.page_end, end)


def _find_heading_top_y(
    doc: fitz.Document, n_pages: int, section
) -> Optional[float]:
    """Return the y-coordinate of the section heading on its start page."""
    page_idx = section.page_start - 1
    if not (0 <= page_idx < n_pages):
        return None
    page = doc.load_page(page_idx)
    candidates = (
        [f"{section.id}. {section.title}", f"{section.id} {section.title}"]
        if section.id else []
    ) + [section.title]
    best_y: Optional[float] = None
    for candidate in candidates:
        for r in page.search_for(candidate):
            y = float(r.y0)
            if best_y is None or y < best_y:
                best_y = y
    return best_y


def compute_fallback_ranges(sections: List) -> None:
    """Compute ``page_end`` as ``next_peer.page_start - 1``.

    Used when fitz page-position data is unavailable (Docling / textual paths).
    Modifies sections in place.
    """
    for i, section in enumerate(sections):
        next_section = next(
            (sections[j] for j in range(i + 1, len(sections)) if sections[j].level <= section.level),
            None,
        )
        section.page_end = (
            max(section.page_start, next_section.page_start - 1)
            if next_section else section.page_start
        )


# ---------------------------------------------------------------------------
# Section classification and tree building
# ---------------------------------------------------------------------------

def classify_sections(sections: List, toc_config: BaseTocConfig) -> None:
    """Tag each section as body / front_matter / back_matter in place."""
    # import here to avoid circular imports with TOCSection
    from managers.toc_extraction.table_of_contents_manager import TOCSectionType

    for section in sections:
        t = (section.title or "").lower()
        if toc_config.section_id_re is not None:
            if section.level == 1 and section.id is None and section.page_start == 1:
                section.type = TOCSectionType.front_matter
                continue
        if toc_config.front_matter_section_starts is not None:
            if t.startswith(toc_config.front_matter_section_starts):
                section.type = TOCSectionType.front_matter
                continue
        if t.startswith(toc_config.back_matter_section_starts):
            section.type = TOCSectionType.back_matter


def build_tree(flat: List) -> List:
    """Convert a flat TOC list into a nested tree using level as depth signal."""
    roots: List = []
    stack: List = []
    for section in flat:
        section.children = []
        while stack and stack[-1].level >= section.level:
            stack.pop()
        if stack:
            stack[-1].children.append(section)
        else:
            roots.append(section)
        stack.append(section)
    return roots