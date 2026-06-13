import re
from logging import Logger
from typing import List, Optional

import fitz

from managers.toc_extraction.utils import _deduplicate_outline, _toc_needs_sorting, _safe_sort, _set_terminal_section
from managers.toc_extraction.toc_configs import BaseTocConfig


def extract_toc_from_text(
    doc: fitz.Document,
    n_pages: int,
    toc_config: BaseTocConfig,
    logger: Logger,
) -> List:
    """Parse a human-readable Table of Contents page using regex.

    Parameters
    ----------
    doc : fitz.Document
        Open fitz document.
    n_pages : int
        Total page count.
    toc_config : BaseTocConfig
        Configuration providing ``fallback_toc_sections``, ``toc_line``,
        and related filtering patterns.
    logger : Logger
        Logging utility.

    Returns
    -------
    List[TOCSection]
        Parsed TOC sections.

    Raises
    ------
    ValueError
        If no TOC page is found or no sections can be parsed from it.
    """
    logger.info("Using textual TOC fallback")
    toc_start = _find_toc_page(doc, n_pages, toc_config, logger)
    if toc_start is None:
        raise ValueError("Table of Contents page not found in PDF")

    sections: List = []
    for p in range(toc_start, min(toc_start + 6, n_pages)):
        parsed = _parse_toc_page(doc.load_page(p).get_text("text"), toc_config, logger)
        if not parsed:
            break
        sections.extend(parsed)

    if not sections:
        raise ValueError("No sections parsed from textual TOC")

    sections = _deduplicate_outline(sections)
    if _toc_needs_sorting(sections, logger):
        sections = _safe_sort(sections, logger)
    logger.info(f"Textual TOC: {len(sections)} entries")
    return _set_terminal_section(sections, toc_config, logger)


def _find_toc_page(
    doc: fitz.Document,
    n_pages: int,
    toc_config: BaseTocConfig,
    logger: Logger,
) -> Optional[int]:
    """Return 0-based index of the first TOC page, or None."""
    if toc_config.fallback_toc_sections is None:
        return None
    for i in range(min(10, n_pages)):
        text = doc.load_page(i).get_text("text")
        if any(s in text.lower().replace(" ", "") for s in toc_config.fallback_toc_sections):
            logger.info(f"TOC page found at PDF page {i + 1}")
            return i
    return None


def _parse_toc_page(
    text: str,
    toc_config: BaseTocConfig,
    logger: Logger,
) -> List:
    """Parse one TOC page into TOCSection entries."""
    from managers.toc_extraction.table_of_contents_manager import TOCSection, TOCSectionType

    def is_supplement(title: str) -> bool:
        if toc_config.supplement_patterns is None:
            return False
        return any(re.match(p, title.lower().strip()) for p in toc_config.supplement_patterns)

    # stitch lines that wrap before a page number
    stitched: List[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if toc_config.bad_page_words and (
            any(p in line.lower() for p in toc_config.bad_page_words) or len(line) > 250
        ):
            continue
        buf += " " + line
        if re.search(r"\d+\s*$", line):
            stitched.append(buf.strip())
            buf = ""
    if buf:
        stitched.append(buf.strip())

    entries: List = []
    for logical in stitched:
        assert toc_config.toc_line is not None
        m = toc_config.toc_line.match(logical)
        if not m:
            continue
        section_id, section_title, page_str = m.groups()
        try:
            page = int(page_str)
        except ValueError as exc:
            logger.error(f"Bad page number '{page_str}': {exc}")
            continue
        if not (1 <= page <= 20000):
            continue
        section_title = section_title.strip(". ").strip()
        if not (3 <= len(section_title) <= 140) or section_title.count(" ") > 18:
            continue
        low_title = section_title.lower()
        if toc_config.bad_title_str_pattern and any(
            p in low_title for p in toc_config.bad_title_str_pattern
        ):
            continue
        if toc_config.prose_words and any(w in low_title for w in toc_config.prose_words):
            continue
        if not section_id and is_supplement(section_title):
            continue
        section_level = section_id.count(".") + 1 if section_id else 1
        entries.append(TOCSection(
            id=section_id, title=section_title, level=section_level,
            page_start=page, page_end=page, type=TOCSectionType.body,
        ))
    return entries