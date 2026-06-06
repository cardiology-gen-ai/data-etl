"""
Hierarchical Chunker 

- Anchor-narrowed start detection (robust)
- TOC-order end boundaries (for leaf sections with no children)
- Parents stop at first child (prevents parent swallowing children)
- Skips TOC dot-leader lines as headers (or else we ingest TOC entries)
- Preserves one structural record for every non-excluded TOC section
- Parents stop at the first located descendant in TOC order
- Leaf sections stop at the next located section in TOC order
- Splits only oversized sections as a last-resort safety net
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from managers.markdown_manager import MarkdownManager


logger = logging.getLogger("hierarchical_chunker")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_OUT_OF_RANGE_WINDOW_WARNINGS: set[tuple[int, int, int, int]] = set()

EXCLUDED_TITLE_SUBSTRINGS = [
    "table of contents",
    "list of figures",
    "list of tables",
    "references",
    "bibliography",
    "abbreviations",
    "acronyms",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "supplementary data",
    "data availability",
    "author information",
    "disclaimer",
]

# Exact-only exclusion. "Index case" is a real clinical section.
EXCLUDED_TITLE_EXACT = {
    "index",
}


def is_excluded_section(sec: Dict[str, Any]) -> bool:
    title = re.sub(
        r"\s+",
        " ",
        (sec.get("title") or "").strip().lower(),
    )

    if sec.get("type") in {"front_matter", "back_matter", "toc"}:
        return True

    if title in EXCLUDED_TITLE_EXACT:
        return True

    return any(
        keyword in title
        for keyword in EXCLUDED_TITLE_SUBSTRINGS
    )

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def section_match_id(sec: Dict[str, Any]) -> str:
    """Return the section ID that should be matched in the document body.

    A body-reconciled TOC record must be matched with its corrected canonical
    ID. Unresolved duplicate fallbacks still use the printed ID because that is
    the only numbering available in the source TOC/body.
    """
    canonical_id = (sec.get("id") or "").split("__part", 1)[0]

    if sec.get("toc_section_id_corrected_from_body") and canonical_id:
        return canonical_id.split("__dup", 1)[0]

    return (
        sec.get("printed_id")
        or canonical_id
    ).split("__dup", 1)[0].split("__part", 1)[0]


def section_printed_id(sec: Dict[str, Any]) -> str:
    """Preserve the original printed ID in chunk metadata."""
    return (
        sec.get("printed_id")
        or sec.get("toc_original_section_id")
        or sec.get("id")
        or ""
    ).split("__part", 1)[0]

def clean_inline_markup(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text or "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\", " ")
    text = re.sub(r"[*_`#~]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .:-\t")

def is_effectively_empty(text: str) -> bool:
    """Return True only when no usable section text was extracted.

    Short text is not empty: a real guideline section may legitimately contain
    only a sentence or a table pointer. Dot-leader TOC rows are treated as
    non-content, but the section record is still preserved by the caller.
    """
    if not text or not text.strip():
        return True
    first = text.strip().split("\n", 1)[0]
    return bool(re.search(r"(?:\.\s*){5,}\d+\s*$", clean_inline_markup(first)))


def first_non_empty_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def starts_mid_sentence(text: str) -> bool:
    first = clean_inline_markup(first_non_empty_line(text))
    if not first:
        return False
    first = first.lstrip("'\"([{")
    if not first:
        return False
    lowered = first.lower()
    continuation_prefixes = ("and ", "or ", "of ", "that ", "which ")
    return first[0].islower() or lowered.startswith(continuation_prefixes)


def _overlaps_title_suffix(fragment: str, section_title: str) -> bool:
    fragment_tokens = _normalize_tokens(fragment)
    title_tokens = _normalize_tokens(section_title)
    if not fragment_tokens or not title_tokens:
        return False
    if len(fragment_tokens) > 8 or len(fragment) > 100:
        return False
    if len(fragment_tokens) > len(title_tokens):
        return False
    suffix = title_tokens[-len(fragment_tokens):]
    if fragment_tokens == suffix:
        return True
    overlap = len(set(fragment_tokens) & set(suffix)) / max(1, len(set(fragment_tokens)))
    return overlap >= 0.8


def strip_leading_heading_fragment(text: str, section_title: str) -> tuple[str, bool]:
    """Remove only a short repeated tail of the current section title."""
    if not text or not text.strip() or not section_title:
        return text, False

    body = text.lstrip()
    bold = re.match(r"^(?:\*\*(?P<a>[^*\n]{1,100})\*\*|__(?P<b>[^_\n]{1,100})__)", body)
    if bold:
        fragment = bold.group("a") or bold.group("b") or ""
        if _overlaps_title_suffix(fragment, section_title):
            return body[bold.end():].lstrip(), True

    line_end = body.find("\n")
    first_line = body if line_end == -1 else body[:line_end]
    fragment = clean_inline_markup(first_line)
    if (
        fragment
        and word_count(fragment) <= 8
        and len(fragment) <= 100
        and not re.search(r"[.!?]\s*$", first_line.strip())
        and _overlaps_title_suffix(fragment, section_title)
    ):
        return ("" if line_end == -1 else body[line_end + 1:].lstrip()), True
    return text, False


def _canonical_section_id(section_id: str) -> str:
    return (section_id or "").split("__dup", 1)[0].split("__part", 1)[0]


def _is_descendant_id(candidate_id: str, ancestor_id: str) -> bool:
    candidate = _canonical_section_id(candidate_id)
    ancestor = _canonical_section_id(ancestor_id)
    return bool(candidate and ancestor and candidate.startswith(f"{ancestor}."))


_GENERIC_NUMBERED_HEADING_RE = re.compile(
    r"^[ \t]*"
    r"(?P<markdown>#{1,6}[ \t]+)?"
    r"(?:[*_`~ \t]*)?"
    r"(?P<sid>\d+(?:\.\d+)*)"
    r"(?:\.(?=\s|$)|(?=\s|$))"
    r"[ \t*_`~]*"
    r"(?P<title>.*?)"
    r"[ \t*_`~]*$"
)


def _heading_candidate_from_lines(
    lines: List[str],
    line_index: int,
    max_extra_lines: int = 2,
) -> Optional[tuple[str, str, bool]]:
    """Parse a numbered heading candidate, including wrapped headings."""
    raw_line = lines[line_index]
    stripped = raw_line.strip()

    if (
        not stripped
        or "|" in stripped
        or is_toc_entry_line(stripped)
        or is_table_or_summary_line(stripped)
    ):
        return None

    if re.search(
        r"\bESC(?:/EAS|/EACTS)? Guidelines\s+\d{1,6}\b",
        stripped,
        re.IGNORECASE,
    ):
        return None

    match = _GENERIC_NUMBERED_HEADING_RE.match(raw_line)
    if not match:
        return None

    sid = match.group("sid")
    if sid.isdigit() and 1900 <= int(sid) <= 2100:
        return None

    explicit_markdown = bool(match.group("markdown"))
    title_parts: List[str] = []
    inline_title = clean_inline_markup(match.group("title") or "")
    if inline_title:
        title_parts.append(inline_title)

    cursor = line_index + 1
    extra = 0
    while (
        cursor < len(lines)
        and extra < max_extra_lines
        and (not title_parts or len(" ".join(title_parts)) < 120)
    ):
        continuation = lines[cursor].strip()
        if not continuation:
            break
        if (
            "|" in continuation
            or is_toc_entry_line(continuation)
            or is_table_or_summary_line(continuation)
            or _GENERIC_NUMBERED_HEADING_RE.match(lines[cursor])
        ):
            break

        cleaned = clean_inline_markup(continuation)
        if not cleaned or len(cleaned) > 180:
            break

        title_parts.append(cleaned)
        cursor += 1
        extra += 1

    title = clean_inline_markup(" ".join(title_parts))
    if not title:
        return None
    if re.match(r"^(table|figure|recommendation table)\b", title, re.IGNORECASE):
        return None

    return sid, title, explicit_markdown


def embedded_section_heading_ids(
    text: str,
    known_titles: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Find canonical numbered headings embedded inside section text."""
    if not text or not text.strip():
        return []

    lines = text.splitlines()
    found: List[str] = []
    seen: set[str] = set()
    all_known_titles = [
        title
        for titles in (known_titles or {}).values()
        for title in titles
    ]

    for line_index in range(len(lines)):
        parsed = _heading_candidate_from_lines(lines, line_index)
        if parsed is None:
            continue

        sid, title, explicit_markdown = parsed
        if known_titles is not None:
            candidates = known_titles.get(sid)
            if candidates:
                best_overlap = max(
                    (_title_overlap_score(title, candidate) for candidate in candidates),
                    default=0.0,
                )
                threshold = 0.25 if explicit_markdown else 0.35
                if best_overlap < threshold:
                    continue
            else:
                if not explicit_markdown:
                    continue
                best_global_overlap = max(
                    (_title_overlap_score(title, candidate) for candidate in all_known_titles),
                    default=0.0,
                )
                if best_global_overlap < 0.85:
                    continue

        if sid not in seen:
            found.append(sid)
            seen.add(sid)

    return found

def classify_embedded_headings(
    embedded_ids: List[str],
    current_section_id: str,
) -> tuple[List[str], List[str], List[str]]:
    current = _canonical_section_id(current_section_id)
    repeated: List[str] = []
    descendants: List[str] = []
    foreign: List[str] = []
    for embedded_id in embedded_ids:
        embedded = _canonical_section_id(embedded_id)
        if embedded == current:
            repeated.append(embedded_id)
        elif _is_descendant_id(embedded, current):
            descendants.append(embedded_id)
        else:
            foreign.append(embedded_id)
    return repeated, descendants, foreign


def is_toc_entry_line(line: str) -> bool:
    # e.g. "1. Preamble ............ 3509"
    stripped = clean_inline_markup(line)
    return bool(re.search(r"(?:\.\s*){5,}\d+\s*$", stripped))


def is_table_or_summary_line(line: str) -> bool:
    """
    Reject Markdown table rows and dense summary rows as section headers.

    Guideline front matter often contains tables listing many section IDs. Those
    rows can look like valid numbered headings unless we explicitly reject them.
    """
    stripped = line.strip()
    low = stripped.lower()
    return (
        "|" in stripped
        or "<br" in low
        or len(stripped) > 300
        or bool(re.match(r"^recommendations?\s+(for|regarding|on)\b", low))
    )



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
    return re.findall(r"\w+", clean_inline_markup(text).lower())

# Compute Jaccard overlap score between line title and section title
def _title_overlap_score(line_title: str, section_title: Optional[str]) -> float:
    if not section_title:
        return 0.0
    lt = set(_normalize_tokens(line_title))
    st = set(_normalize_tokens(section_title))
    if not lt or not st:
        return 0.0
    return len(lt & st) / len(lt | st)


def _line_end(markdown: str, offset: int, search_end: int) -> int:
    line_end = markdown.find("\n", offset, search_end)
    return search_end if line_end == -1 else line_end


def _heading_context_title(
    markdown: str,
    match_start: int,
    match_end: int,
    search_end: int,
    max_extra_lines: int = 2,
) -> tuple[str, int]:
    """
    Build a title candidate from a heading line plus short continuation lines.
    PDF-to-Markdown often wraps formatted headings across several lines.
    """
    first_end = _line_end(markdown, match_start, search_end)
    lines = [markdown[match_start:first_end]]
    end = first_end

    cursor = first_end + 1
    extra = 0
    while cursor < search_end and extra < max_extra_lines:
        line_end = _line_end(markdown, cursor, search_end)
        line = markdown[cursor:line_end].strip()
        if not line:
            break
        if is_toc_entry_line(line) or is_table_or_summary_line(line):
            break
        if re.match(r"^[ \t]*(?:#{1,6}[ \t]*)?(?:[*_`~ \t]*)?\d+(?:\.\d+)*", line):
            break
        if len(clean_inline_markup(line)) > 160:
            break

        lines.append(line)
        end = line_end
        cursor = line_end + 1
        extra += 1

    return clean_inline_markup(" ".join(lines)), end


# Body start (skip front matter / TOC)
def find_body_start(
    markdown: str,
    ordered_sections: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Locate the first real body heading using canonical TOC order."""
    if ordered_sections:
        for sec in ordered_sections:
            if not sec.get("id") or is_excluded_section(sec):
                continue
            header = locate_boundary_header(markdown, sec, 0, len(markdown))
            if header is not None:
                return header[0]

    rx = re.compile(r"(?m)^\s*#{1,6}\s+.*?\b1\.\s+.+$")
    match = rx.search(markdown)
    return match.start() if match else 0

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

    first_page = pages[0]
    last_page = pages[-1]
    if page_start < first_page or page_end > last_page:
        warning_key = (page_start, page_end, first_page, last_page)
        if warning_key not in _OUT_OF_RANGE_WINDOW_WARNINGS:
            logger.warning(
                "TOC page range %s-%s is outside anchor page range %s-%s; "
                "using global guarded header search for this range",
                page_start,
                page_end,
                first_page,
                last_page,
            )
            _OUT_OF_RANGE_WINDOW_WARNINGS.add(warning_key)
        return 0, markdown_len

    start_page = max(min(page_start - window, pages[-1]), pages[0])
    end_page = min(max(page_end + window, pages[0]), pages[-1])

    start_offset = anchors.get(start_page, anchors[pages[0]])
    end_offset = anchors.get(end_page + 1, markdown_len)

    return start_offset, end_offset



def _same_parent_and_depth(left_id: str, right_id: str) -> bool:
    left = _canonical_section_id(left_id)
    right = _canonical_section_id(right_id)
    if not left or not right or left.count(".") != right.count("."):
        return False
    left_parent = left.rsplit(".", 1)[0] if "." in left else ""
    right_parent = right.rsplit(".", 1)[0] if "." in right else ""
    return left_parent == right_parent


def _ids_compatible_for_title_fallback(
    expected_id: str,
    candidate_id: str,
) -> bool:
    expected = _canonical_section_id(expected_id)
    candidate = _canonical_section_id(candidate_id)
    if not expected or not candidate:
        return False
    if candidate == expected or candidate.startswith(f"{expected}."):
        return True
    return _same_parent_and_depth(expected, candidate)


def _strip_leading_section_id(title: str, section_id: str) -> str:
    return re.sub(
        rf"^\s*{re.escape(section_id)}(?:\.)?\s*",
        "",
        clean_inline_markup(title),
        count=1,
    ).strip()


def _title_prefix_content_start(
    raw_line: str,
    section_id: str,
    section_title: Optional[str],
) -> Optional[int]:
    """Return the offset where same-line body text starts after a heading.

    This handles forms such as::

        6.3.3.4.2. Iloprost. Iloprost is ...
        _8.1.2.1.1. Definition._ Chronic limb ...
        **8.1.2.1.1 Definition** Chronic limb ...

    The function succeeds only when the exact section ID and all normalized
    title tokens occur in order at the beginning of the line.
    """
    title_words = re.findall(r"\w+", section_title or "")
    if not title_words:
        return None

    escaped_id = re.escape(section_id.strip())
    title_pattern = r"\W+".join(re.escape(word) for word in title_words)
    pattern = re.compile(
        rf"^[ \t]*(?:#{{1,6}}[ \t]*)?(?:[*_`~ \t]*)?"
        rf"{escaped_id}(?:\.(?=\s|[*_`~])|(?=\s|[*_`~]))"
        rf"[ \t*_`~.:-]*"
        rf"{title_pattern}"
        rf"(?P<closing_before>[ \t*_`~]*)"
        rf"(?P<punct>[.:])?"
        rf"(?P<closing_after>[ \t*_`~]*)",
        re.IGNORECASE,
    )
    match = pattern.match(raw_line)
    if not match:
        return None

    content_start = match.end()
    if not raw_line[content_start:].strip():
        return None
    return content_start


def locate_header_by_title(
    markdown: str,
    expected_section_id: str,
    section_title: Optional[str],
    search_start: int,
    search_end: int,
    min_score: float = 0.85,
) -> Optional[tuple[int, int, str]]:
    """Recover a numbered body heading by title when exact-ID matching fails."""
    if not section_title:
        return None

    scan_start = markdown.rfind("\n", 0, search_start) + 1
    candidates: List[tuple[float, int, int, str]] = []
    pos = scan_start

    while pos < search_end:
        line_end = markdown.find("\n", pos, search_end)
        if line_end == -1:
            line_end = search_end
        raw_line = markdown[pos:line_end]

        if (
            raw_line.strip()
            and "|" not in raw_line
            and not is_toc_entry_line(raw_line)
            and not is_table_or_summary_line(raw_line)
        ):
            match = _GENERIC_NUMBERED_HEADING_RE.match(raw_line)
            if match:
                candidate_id = match.group("sid")
                if not (
                    candidate_id.isdigit() and 1900 <= int(candidate_id) <= 2100
                ) and _ids_compatible_for_title_fallback(
                    expected_section_id,
                    candidate_id,
                ):
                    absolute_start = pos + match.start()
                    absolute_end = pos + match.end()
                    context_title, context_end = _heading_context_title(
                        markdown=markdown,
                        match_start=absolute_start,
                        match_end=absolute_end,
                        search_end=search_end,
                    )
                    inline_title = clean_inline_markup(match.group("title") or "")
                    context_title = _strip_leading_section_id(context_title, candidate_id)
                    line_score = _title_overlap_score(inline_title, section_title)
                    context_score = _title_overlap_score(context_title, section_title)
                    score = max(line_score, context_score)

                    if score >= min_score:
                        inline_start = _title_prefix_content_start(
                            raw_line,
                            candidate_id,
                            section_title,
                        )
                        if inline_start is not None:
                            header_end = pos + inline_start
                        else:
                            header_end = line_end
                            if (
                                not inline_title
                                or (
                                    context_score > line_score
                                    and context_score >= 0.9
                                )
                            ):
                                header_end = context_end

                        candidates.append(
                            (score, absolute_start, header_end, candidate_id)
                        )

        pos = line_end + 1

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_start, best_end, best_id = candidates[0]
    for other_score, _, _, other_id in candidates[1:]:
        if other_id != best_id and best_score - other_score < 0.05:
            return None
        if best_score - other_score >= 0.05:
            break

    return best_start, best_end, best_id


# Header locator (returns header start/end offsets)
def locate_header(
    markdown: str,
    section_id: str,
    section_title: Optional[str],
    search_start: int,
    search_end: int,
) -> Optional[tuple[int, int]]:
    """Find the best matching block or inline heading for a section ID."""
    escaped = re.escape(section_id.strip())
    header_rx = re.compile(
        rf"^[ \t]*"
        rf"(?:#{{1,6}}[ \t]*)?"
        rf"(?:[*_`~ \t]*)?"
        rf"(?<![\d.])"
        rf"(?P<num>{escaped})"
        rf"(?:"
        rf"\.(?=\s|[A-Za-z*_`<])"
        rf"|"
        rf"(?=[\s*_`<]|$)"
        rf")"
        rf"[ \t]*"
        rf"(?P<title>.*)?$",
        re.IGNORECASE,
    )

    best: Optional[tuple[int, int]] = None
    best_score = -1.0
    found_candidate = False
    scan_start = markdown.rfind("\n", 0, search_start) + 1
    pos = scan_start

    while pos < search_end:
        line_end = markdown.find("\n", pos, search_end)
        if line_end == -1:
            line_end = search_end

        line = markdown[pos:line_end]
        match = header_rx.match(line)
        if not match:
            pos = line_end + 1
            continue

        found_candidate = True
        absolute_start = pos + match.start()
        absolute_end = pos + match.end()
        if absolute_end < search_start:
            pos = line_end + 1
            continue
        if is_toc_entry_line(line) or is_table_or_summary_line(line):
            pos = line_end + 1
            continue

        raw_title = (match.group("title") or "").strip()
        title = raw_title
        if not title:
            tail = markdown[absolute_end:search_end]
            for next_line in tail.splitlines():
                if next_line.strip():
                    title = next_line.strip()
                    break

        context_title, context_end = _heading_context_title(
            markdown=markdown,
            match_start=absolute_start,
            match_end=absolute_end,
            search_end=search_end,
        )
        line_score = _title_overlap_score(title, section_title)
        context_score = _title_overlap_score(context_title, section_title)
        score = max(line_score, context_score)

        inline_start = _title_prefix_content_start(
            line,
            section_id,
            section_title,
        )
        if inline_start is not None:
            header_end_for_content = pos + inline_start
            # Exact ordered title-prefix matching is stronger than a Jaccard
            # score polluted by the same-line body text.
            score = max(score, 1.0)
        else:
            header_end_for_content = line_end
            if not clean_inline_markup(raw_title):
                header_end_for_content = context_end
            elif (
                line_score < 0.75
                and context_score >= 0.85
                and context_score > line_score
            ):
                header_end_for_content = context_end

        if score > best_score:
            best = (absolute_start, header_end_for_content)
            best_score = score

        pos = line_end + 1

    if not found_candidate or best is None:
        return None

    title_tokens = set(_normalize_tokens(section_title or ""))
    min_score = 0.2 if len(title_tokens) > 1 else 0.5
    if section_title and best_score < min_score:
        return None

    return best

def locate_boundary_header(
    markdown: str,
    sec: Dict[str, Any],
    search_start: int,
    search_end: int,
) -> Optional[tuple[int, int]]:
    match_id = section_match_id(sec)
    if not match_id:
        return None
    return locate_header(
        markdown=markdown,
        section_id=match_id,
        section_title=sec.get("title"),
        search_start=search_start,
        search_end=search_end,
    )


def locate_section_header(
    markdown: str,
    anchors: Dict[int, int],
    sec: Dict[str, Any],
    body_start: int,
    window: int = 1,
) -> Dict[str, Any]:
    markdown_len = len(markdown)
    window_start, window_end = _compute_search_window(
        anchors=anchors,
        page_start=sec["page_start"],
        page_end=sec["page_end"],
        markdown_len=markdown_len,
        window=window,
    )
    window_start = max(window_start, body_start)

    flags: List[str] = []
    matched_heading_id: Optional[str] = None
    expected_id = section_match_id(sec)

    header_pos = locate_boundary_header(
        markdown=markdown,
        sec=sec,
        search_start=window_start,
        search_end=window_end,
    )
    if header_pos is not None:
        matched_heading_id = expected_id

    if header_pos is None:
        title_match = locate_header_by_title(
            markdown=markdown,
            expected_section_id=expected_id,
            section_title=sec.get("title"),
            search_start=window_start,
            search_end=window_end,
            min_score=0.85,
        )
        if title_match is not None:
            header_start, header_end, matched_heading_id = title_match
            header_pos = (header_start, header_end)
            flags.append("header_title_fallback")
            if matched_heading_id != expected_id:
                flags.append("header_id_mismatch")

    if header_pos is None:
        header_pos = locate_boundary_header(
            markdown=markdown,
            sec=sec,
            search_start=body_start,
            search_end=markdown_len,
        )
        if header_pos is not None:
            matched_heading_id = expected_id
            flags.append("header_global")

    if header_pos is None:
        title_match = locate_header_by_title(
            markdown=markdown,
            expected_section_id=expected_id,
            section_title=sec.get("title"),
            search_start=body_start,
            search_end=markdown_len,
            min_score=0.9,
        )
        if title_match is not None:
            header_start, header_end, matched_heading_id = title_match
            header_pos = (header_start, header_end)
            flags.extend(["header_global", "header_title_fallback"])
            if matched_heading_id != expected_id:
                flags.append("header_id_mismatch")

    if header_pos is None:
        flags.append("header_not_found")

    return {
        "section_id": sec.get("id"),
        "header_pos": header_pos,
        "matched_heading_id": matched_heading_id,
        "win_start": window_start,
        "win_end": window_end,
        "quality_flags": sorted(set(flags)),
    }



# Content extraction with TOC boundaries

def extract_section_text(
    markdown: str,
    sec: Dict[str, Any],
    section_index: int,
    ordered_sections: List[Dict[str, Any]],
    header_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract direct section text using the canonical TOC order.

    A parent stops at its first located descendant. A leaf stops at the first
    later canonical section whose heading is located after the current heading.
    We never use an arbitrary heading outside the current section's TOC suffix as
    a boundary. If no canonical boundary can be located, the anchor window is a
    conservative fallback and the record is flagged rather than removed.
    """
    md_len = len(markdown)
    flags: List[str] = []
    header_info = header_lookup.get(sec.get("id")) or {}
    flags.extend(header_info.get("quality_flags") or [])
    header_pos = header_info.get("header_pos")
    win_end = int(header_info.get("win_end") or md_len)

    if header_pos is None:
        return {
            "text": "",
            "quality_flags": sorted(set(flags or ["header_not_found"])),
            "header_found": False,
            "boundary_source": "none",
        }

    _, header_end = header_pos
    content_start = header_end
    current_id = sec.get("id") or ""
    later_sections = ordered_sections[section_index + 1:]

    preferred: List[tuple[Dict[str, Any], str]] = []
    remaining: List[tuple[Dict[str, Any], str]] = []

    if sec.get("_has_children"):
        for candidate in later_sections:
            candidate_id = candidate.get("id") or ""
            if _is_descendant_id(candidate_id, current_id):
                source = "first_child" if candidate.get("_parent_id") == current_id else "later_descendant"
                preferred.append((candidate, source))
            else:
                remaining.append((candidate, "next_toc_section"))
                break
    else:
        remaining = [
            (candidate, "next_toc_section" if offset == 0 else "later_toc_section")
            for offset, candidate in enumerate(later_sections)
        ]

    chosen: Optional[tuple[int, str]] = None
    for candidate, source in preferred + remaining:
        boundary = (header_lookup.get(candidate.get("id")) or {}).get("header_pos")
        if boundary and boundary[0] > content_start:
            chosen = (boundary[0], source)
            break

    if chosen is not None:
        content_end, boundary_source = chosen
        if boundary_source == "later_descendant":
            flags.append("first_child_header_not_found")
        elif boundary_source == "later_toc_section":
            flags.append("next_section_header_not_found")
    elif win_end > content_start:
        content_end = win_end
        boundary_source = "window_cap"
        flags.extend(["boundary_uncertain", "boundary_window_cap"])
    else:
        content_end = min(md_len, content_start + 50000)
        boundary_source = "char_cap"
        flags.extend(["boundary_uncertain", "boundary_char_cap"])

    return {
        "text": markdown[content_start:content_end].strip(),
        "quality_flags": sorted(set(flags)),
        "header_found": True,
        "boundary_source": boundary_source,
    }

def split_oversized_text(text: str, max_chunk_chars: int) -> List[str]:
    if len(text) <= max_chunk_chars:
        return [text]

    parts: List[str] = []
    current = ""
    blocks = re.split(r"(\n\s*\n)", text)

    for block in blocks:
        if not block:
            continue
        if len(current) + len(block) <= max_chunk_chars:
            current += block
            continue

        if current.strip():
            parts.append(current.strip())
            current = ""

        while len(block) > max_chunk_chars:
            split_at = block.rfind(" ", 0, max_chunk_chars)
            if split_at < max_chunk_chars // 2:
                split_at = max_chunk_chars
            parts.append(block[:split_at].strip())
            block = block[split_at:].lstrip()

        current = block

    if current.strip():
        parts.append(current.strip())

    return parts or [text[:max_chunk_chars]]



# Chunk builder (no splitting yet)
def build_hierarchical_chunks(
    toc_tree: List[Dict[str, Any]],
    markdown_manager: MarkdownManager,
    anchors: Dict[int, int],
    doc_id: str,
    min_words: int = 10,
    max_chunk_chars: int = 50000,
) -> List[Dict[str, Any]]:
    """Build section-level chunks while preserving the canonical hierarchy."""
    sections = collect_sections(toc_tree)
    logger.info("TOC nodes considered: %d", len(sections))

    ordered = [section for section in sections if section.get("id")]
    body_start = find_body_start(markdown_manager.text, ordered)
    header_lookup = {
        section["id"]: locate_section_header(
            markdown=markdown_manager.text,
            anchors=anchors,
            sec=section,
            body_start=body_start,
            window=1,
        )
        for section in ordered
    }

    known_titles: Dict[str, List[str]] = {}
    for section in ordered:
        match_id = section_match_id(section)
        if match_id and section.get("title"):
            known_titles.setdefault(match_id, []).append(section["title"])

    chunks: List[Dict[str, Any]] = []

    for section_index, sec in enumerate(ordered):
        sec_id = sec["id"]
        if is_excluded_section(sec):
            continue

        extraction = extract_section_text(
            markdown=markdown_manager.text,
            sec=sec,
            section_index=section_index,
            ordered_sections=ordered,
            header_lookup=header_lookup,
        )
        raw_text = extraction["text"]
        quality_flags = list(extraction["quality_flags"])

        raw_text, stripped = strip_leading_heading_fragment(raw_text, sec.get("title") or "")
        if stripped:
            quality_flags.append("stripped_leading_heading_fragment")

        empty = is_effectively_empty(raw_text)
        if empty:
            quality_flags.append("empty_section")
            if not sec.get("_has_children"):
                quality_flags.append("empty_leaf_section")
        else:
            words = word_count(raw_text)
            if words < min_words:
                quality_flags.append("very_short_chunk")
            elif words < 20:
                quality_flags.append("very_short_embedded_chunk")

            if starts_mid_sentence(raw_text):
                quality_flags.append("starts_mid_sentence")

            embedded_ids = embedded_section_heading_ids(raw_text, known_titles=known_titles)
            repeated, descendants, foreign = classify_embedded_headings(
                embedded_ids,
                section_match_id(sec),
            )
            if embedded_ids:
                quality_flags.append("contains_embedded_section_heading")
            if repeated:
                quality_flags.append("contains_repeated_current_heading")
            if descendants:
                quality_flags.append("contains_descendant_section_content")
            if foreign:
                quality_flags.extend([
                    "contains_foreign_section_content",
                    "contains_likely_section_leakage",
                ])

        if raw_text and len(raw_text) > max_chunk_chars:
            quality_flags.append("oversized")
            logger.warning(
                "Section %s:%s exceeds quality threshold | chars=%d | max=%d",
                doc_id,
                sec_id,
                len(raw_text),
                max_chunk_chars,
            )

        text_parts = [""] if empty else split_oversized_text(raw_text, max_chunk_chars)
        if len(text_parts) > 1:
            quality_flags.append("split_oversized")

        for part_idx, part_text in enumerate(text_parts):
            part_section_id = sec_id if part_idx == 0 else f"{sec_id}__part{part_idx + 1}"
            part_flags = sorted(set(quality_flags))
            chunks.append({
                "chunk_id": f"{doc_id}:{part_section_id}:0",
                "doc_id": doc_id,
                "section_id": part_section_id,
                "printed_section_id": section_printed_id(sec),
                "parent_section_id": sec.get("_parent_id") if part_idx == 0 else sec_id,
                "section_title": sec.get("title"),
                "section_level": sec.get("level"),
                "section_type": sec.get("type"),
                "page_start": sec.get("page_start"),
                "page_end": sec.get("page_end"),
                "text": part_text,
                "is_empty": empty,
                # Preserve recall: every non-empty section remains embeddable.
                # Quality flags allow strict downstream policies without data loss.
                "embed": not empty,
                "part_index": part_idx,
                "part_count": len(text_parts),
                "quality_flags": part_flags,
                "boundary_source": extraction["boundary_source"],
            })

    logger.info("Built %d chunks", len(chunks))
    return chunks

