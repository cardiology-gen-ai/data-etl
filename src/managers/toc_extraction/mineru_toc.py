from logging import Logger
from typing import Dict, List

from managers.parsing.parsing_manager import ParsedHeading
from managers.toc_extraction.toc_configs import BaseTocConfig
from managers.toc_extraction.utils import (
    _deduplicate_outline,
    _generate_sequential_id,
    _safe_sort,
    _set_terminal_section,
    _toc_needs_sorting,
)


def extract_toc_from_backend_headings(
    headings: List[ParsedHeading],
    toc_config: BaseTocConfig,
    logger: Logger,
) -> List:
    """Convert ``ParsedHeading`` items into ``TOCSection`` entries.

    Mirrors the behaviour of :func:`extract_toc_from_fitz_outline`:
      - parse an explicit section id from the title when ``section_re`` matches,
      - otherwise generate a dotted sequential id from the heading level,
      - dedupe, sanity-sort and truncate at the back-matter boundary.

    Parameters
    ----------
    headings : List[ParsedHeading]
        Backend-provided heading list (e.g. from MinerU ``content_list``).
    toc_config : BaseTocConfig
        Configuration driving id parsing and bad-title detection.
    logger : Logger
        Logging utility.

    Returns
    -------
    List[TOCSection]
        Extracted sections.  Empty if ``headings`` is empty.

    Raises
    ------
    RuntimeError
        If too many entries look like junk (same threshold as the fitz path).
    """
    if not headings:
        return []

    # import here to avoid circular dependency with table_of_contents_manager
    from managers.toc_extraction.table_of_contents_manager import (
        TOCSection,
        TOCSectionType,
    )

    sections: List = []
    level_counters: Dict[int, int] = {}

    for h in headings:
        title = (h.title or "").strip()
        if not title:
            continue

        section_id, clean_title = _parse_heading_title(title, toc_config)

        if toc_config.section_id_in_title and section_id is None:
            logger.debug(f"Skipping non-numbered backend heading: {title!r}")
            continue

        if section_id is None:
            section_id = _generate_sequential_id(int(h.level), level_counters)
            toc_level = int(h.level)
        else:
            _generate_sequential_id(int(h.level), level_counters)
            toc_level = section_id.count(".") + 1

        sections.append(TOCSection(
            id=section_id,
            title=clean_title,
            level=toc_level,
            page_start=int(h.page),
            page_end=int(h.page),
            type=TOCSectionType.body,
        ))

    sections = _deduplicate_outline(sections)

    bad_count = sum(
        1 for s in sections
        if (s.title or "").lower().startswith(toc_config.bad_section_title_starts)
    )
    if sections and bad_count / max(1, len(sections)) > 0.3:
        raise RuntimeError(
            f"Backend headings rejected: {bad_count}/{len(sections)} "
            f"suspicious entries"
        )

    if _toc_needs_sorting(sections, logger):
        sections = _safe_sort(sections, logger)
    logger.info(f"Backend-headings TOC: {len(sections)} entries")
    return _set_terminal_section(sections, toc_config, logger)


def _parse_heading_title(title: str, toc_config: BaseTocConfig):
    """Extract ``(section_id, clean_title)`` if the title starts with an id."""
    title = (title or "").strip()

    if toc_config.section_re is not None:
        m = toc_config.section_re.match(title)
        if m:
            groupdict = m.groupdict()
            section_id = groupdict.get("id") or m.group(1)
            clean_title = groupdict.get("title") or title[m.end():]

            section_id = section_id.strip().rstrip(".")
            clean_title = clean_title.strip(" .")

            # avoid catching years like "2023" as section ids
            if not (section_id.isdigit() and 1900 <= int(section_id) <= 2100):
                return section_id, clean_title

    return None, title