from logging import Logger
from typing import Dict, List, Optional, Tuple

import fitz

from managers.toc_extraction.utils import _toc_needs_sorting, _safe_sort, _set_terminal_section, _deduplicate_outline, \
    _generate_sequential_id
from managers.toc_extraction.toc_configs import BaseTocConfig
from utils.text_utils import title_overlap_score


def extract_toc_from_fitz_outline(
    doc: fitz.Document,
    toc_config: BaseTocConfig,
    logger: Logger,
) -> Optional[List]:
    """Extract TOC from the embedded PDF outline.

    Parameters
    ----------
    doc : fitz.Document
        Open fitz document.
    toc_config : BaseTocConfig
        Configuration driving section-id parsing and bad-title detection.
    logger : Logger
        Logging utility.

    Returns
    -------
    List[TOCSection] or None
        Extracted sections, or ``None`` if no outline is embedded.

    Raises
    ------
    RuntimeError
        If the outline exists but contains too many suspicious entries.
    """
    raw = doc.get_toc(simple=False)
    if not raw:
        return None

    file_title = doc.metadata.get("title")
    sections = []
    level_counters: Dict[int, int] = {}
    level_offset = 0

    # import here to avoid circular dependency with table_of_contents_manager
    from managers.toc_extraction.table_of_contents_manager import TOCSection, TOCSectionType

    for level, title, page, *_ in raw:
        if title_overlap_score(file_title, title) >= 0.9:
            level_offset += 1
            continue
        title = (title or "").strip()
        section_id, clean_title = _parse_outline_title(title, toc_config)
        actual_level = level - level_offset

        if section_id is None:
            section_id = _generate_sequential_id(actual_level, level_counters)
        else:
            # keep level_counters consistent even when id comes from title
            _generate_sequential_id(actual_level, level_counters)

        sections.append(TOCSection(
            id=section_id, title=clean_title, level=int(actual_level),
            page_start=int(page), page_end=int(page),
            type=TOCSectionType.body,
        ))

    sections = _deduplicate_outline(sections)

    bad_count = sum(
        1 for s in sections
        if (s.title or "").lower().startswith(toc_config.bad_section_title_starts)
    )
    if bad_count / max(1, len(sections)) > 0.3:
        raise RuntimeError(
            f"Outline rejected: {bad_count}/{len(sections)} suspicious entries"
        )

    if _toc_needs_sorting(sections, logger):
        sections = _safe_sort(sections, logger)
    return _set_terminal_section(sections, toc_config, logger)


def _parse_outline_title(
    title: str, toc_config: BaseTocConfig
) -> Tuple[Optional[str], str]:
    """Extract (section_id, clean_title) from a raw outline title string."""
    if toc_config.section_re is not None:
        m = toc_config.section_re.match(title)
        if m:
            section_id = m.group(1)
            if not (section_id.isdigit() and 1900 <= int(section_id) <= 2100):
                return section_id, title[len(section_id):].strip(" .")
    return None, title