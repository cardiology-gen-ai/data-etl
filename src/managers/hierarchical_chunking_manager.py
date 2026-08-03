"""
Hierarchical chunker.

- Uses anchor-narrowed section-start detection.
- Parents stop at the first located descendant.
- Leaf sections stop at the next located TOC section.
- Preserves one structural record for every TOC section.
- Marks excluded and empty sections as non-embeddable.
- Flags oversized sections without splitting them.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from managers.mineru_markdown_adapter import normalize_heading_for_matching

if TYPE_CHECKING:
    from managers.markdown_manager import MarkdownManager


logger = logging.getLogger("hierarchical_chunker")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_OUT_OF_RANGE_WINDOW_WARNINGS: set[tuple[int, int, int, int]] = set()

EXCLUDED_TITLE_SUBSTRINGS = [
    #front matter
    "table of contents",
    "list of figures",
    "list of tables",
    "abbreviations",
    "acronyms",
    "preamble",
    # Guideline-level summaries that duplicate clinical sections
    "what is new",
    "key messages",
    "what to do and what not to do",
    "'what to do' and 'what not to do'",
    "gaps in evidence",
    "future needs",
    "quality indicators",
    "evidence tables",


    # back matter
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "appendices",
    "supplementary data",
    "data availability",
    "author information",
    "disclaimer",
    "disclaimers"
]

# Exact-only exclusion. "Index case" is a real clinical section.
EXCLUDED_TITLE_EXACT = {
    "index",
}


def is_excluded_section(sec: Dict[str, Any]) -> bool:
    """Return True when a section must be excluded from downstream retrieval."""

    title = (sec.get("title") or "").translate(
        str.maketrans({
            "‘": "'",
            "’": "'",
            "“": '"',
            "”": '"',
            "`": "'",
        })
    )

    title = re.sub(
        r"\s+",
        " ",
        title.strip().lower(),
    )

    if sec.get("type") in {"front_matter", "back_matter", "toc"}:
        return True

    if title in EXCLUDED_TITLE_EXACT:
        return True

    # Robust to quotation marks, capitalization and trailing wording such as
    # "messages from the Guidelines".
    if "what to do" in title and "what not to do" in title:
        return True

    return any(
        keyword in title
        for keyword in EXCLUDED_TITLE_SUBSTRINGS
    )


def is_effectively_excluded_section(sec: Dict[str, Any]) -> bool:
    """Return whether a section is excluded directly or through an ancestor."""
    return bool(sec.get("_excluded_by_policy")) or is_excluded_section(sec)

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
    text = re.sub(r"</(?:td|th|tr|p|div|li|h[1-6])\s*>", " ", text, flags=re.IGNORECASE)
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
    semantic = clean_inline_markup(text)
    if not semantic:
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


def _looks_like_printed_toc_suffix(raw_line: str, prefix_end: int) -> bool:
    """Return True for dot-leader TOC rows after an otherwise valid prefix."""
    suffix = clean_inline_markup(raw_line[prefix_end:])
    return bool(re.match(r"^(?:\.\s*){3,}\d{1,6}\s*$", suffix))


def _header_inside_html_table(markdown: str, offset: int) -> bool:
    before_open = markdown.rfind("<table", 0, offset)
    if before_open == -1:
        return False
    before_close = markdown.rfind("</table>", 0, offset)
    return before_close < before_open


def _html_table_end_after(markdown: str, offset: int) -> int:
    table_end = markdown.find("</table>", offset)
    return -1 if table_end == -1 else table_end + len("</table>")


def _strip_html_to_text(html_fragment: str) -> str:
    text = re.sub(
        r"</(?:td|th|tr|p|div|li|h[1-6])\s*>",
        "\n",
        html_fragment or "",
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _extract_false_table_text(markdown: str, start: int, end: int) -> Optional[str]:
    """Extract narrative cells from a false MinerU HTML table region.

    This is intentionally narrow: it is used only when the matched section
    heading itself is inside a table.  Real tables elsewhere stay atomic.
    """
    table_end = _html_table_end_after(markdown, start)
    if table_end == -1 or table_end > end:
        return None

    table_fragment = markdown[start:table_end]
    rest = markdown[table_end:end]
    cells: List[str] = []
    cell_rx = re.compile(r"<td(?P<attrs>[^>]*)>(?P<body>.*?)</td>", re.IGNORECASE | re.DOTALL)
    for match in cell_rx.finditer(table_fragment):
        attrs = match.group("attrs") or ""
        body = _strip_html_to_text(match.group("body"))
        if not body:
            continue
        structural = bool(re.search(r"\b(?:colspan|rowspan)\s*=", attrs, re.IGNORECASE))
        if structural or len(body) >= 80:
            cells.append(body)

    if not cells:
        return None

    pieces = cells
    if rest.strip():
        pieces.append(rest.strip())
    return "\n\n".join(piece for piece in pieces if piece.strip()).strip()



# Find all sections in TOC tree as flat list with parent references
def collect_sections(
    toc_nodes: List[Dict[str, Any]],
    parent_id: Optional[str] = None,
    ancestor_excluded: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in toc_nodes:
        excluded = ancestor_excluded or is_excluded_section(node)
        entry = {
            **node,
            "_parent_id": parent_id,
            "_has_children": bool(node.get("children")),
            "_excluded_by_policy": excluded,
        }
        out.append(entry)
        out.extend(
            collect_sections(
                node.get("children", []),
                node.get("id"),
                ancestor_excluded=excluded,
            )
        )
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
            if not sec.get("id") or is_effectively_excluded_section(sec):
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


def _title_prefix_match_end(
    raw_line: str,
    section_id: str,
    section_title: Optional[str],
) -> Optional[int]:
    """Return the end offset of a TOC-confirmed heading prefix.

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

    return match.end()


def _title_prefix_content_start(
    raw_line: str,
    section_id: str,
    section_title: Optional[str],
) -> Optional[int]:
    """Return where same-line body text starts after a heading, if present."""
    content_start = _title_prefix_match_end(
        raw_line=raw_line,
        section_id=section_id,
        section_title=section_title,
    )
    if content_start is None or not raw_line[content_start:].strip():
        return None
    return content_start


def _iter_exact_prefix_matches(
    raw_line: str,
    section_id: str,
    section_title: Optional[str],
    allow_embedded: bool = True,
) -> List[tuple[int, int]]:
    """Find exact ``section_id + title`` prefixes in one Markdown block/line."""
    if not section_id or not section_title:
        return []

    if not allow_embedded:
        end = _title_prefix_match_end(raw_line, section_id, section_title)
        return [(0, end)] if end is not None else []

    escaped = re.escape(section_id.strip())
    sid_rx = re.compile(
        rf"(?<![\d.]){escaped}"
        rf"(?:\.(?=\s|[*_`~<])|(?=[\s*_`<]|$))",
        re.IGNORECASE,
    )
    matches: List[tuple[int, int]] = []
    for sid_match in sid_rx.finditer(raw_line):
        start = sid_match.start()
        end = _title_prefix_match_end(
            raw_line[start:],
            section_id,
            section_title,
        )
        if end is not None:
            matches.append((start, start + end))
    return matches


def _markdown_heading_start_for_match(raw_line: str, match_start: int) -> int:
    """Include a same-line Markdown heading prefix when it belongs to the match."""
    prefix = re.match(r"^[ \t]{0,3}#{1,6}[ \t]+", raw_line)
    if prefix is not None and prefix.end() == match_start:
        return prefix.start()
    return match_start


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
        if not line.strip():
            pos = line_end + 1
            continue

        exact_prefixes = _iter_exact_prefix_matches(
            raw_line=line,
            section_id=section_id,
            section_title=section_title,
            allow_embedded=True,
        )
        for rel_start, rel_end in exact_prefixes:
            rel_header_start = _markdown_heading_start_for_match(line, rel_start)
            absolute_start = pos + rel_header_start
            absolute_end = pos + rel_end
            if absolute_start < search_start:
                continue
            if is_toc_entry_line(line) or _looks_like_printed_toc_suffix(line, rel_end):
                continue

            found_candidate = True
            header_end_for_content = absolute_end
            if not line[rel_end:].strip():
                header_end_for_content = line_end

            if 1.0 > best_score:
                best = (absolute_start, header_end_for_content)
                best_score = 1.0

        match = header_rx.match(line)
        if not match:
            pos = line_end + 1
            continue

        found_candidate = True
        absolute_start = pos + match.start()
        absolute_end = pos + match.end()
        if absolute_start < search_start:
            pos = line_end + 1
            continue
        if (
            is_toc_entry_line(line)
            or (
                is_table_or_summary_line(line)
                and not exact_prefixes
            )
        ):
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
    search_after: int = 0,
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
    window_start = max(window_start, body_start, search_after)

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
            search_start=max(body_start, search_after),
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
            search_start=max(body_start, search_after),
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
    else:
        if header_pos[0] < search_after:
            flags.append("header_non_monotonic")
        if _header_inside_html_table(markdown, header_pos[0]):
            flags.append("header_in_html_table")

    return {
        "section_id": sec.get("id"),
        "header_pos": header_pos,
        "matched_heading_id": matched_heading_id,
        "win_start": window_start,
        "win_end": window_end,
        "quality_flags": sorted(set(flags)),
    }



# Content extraction with TOC boundaries

def _pdf_heading_pattern(section_id: str, title: Optional[str]) -> Optional[re.Pattern]:
    title_words = re.findall(r"\w+", title or "")
    if not section_id or not title_words:
        return None
    return re.compile(
        rf"(?<![\d.]){re.escape(section_id)}(?:\.)?\s+"
        + r"\W+".join(re.escape(word) for word in title_words),
        re.IGNORECASE,
    )


def _local_pdf_fallback_text(
    pdf_path: Optional[Any],
    sec: Dict[str, Any],
    later_sections: List[Dict[str, Any]],
    max_pages: int = 3,
) -> tuple[str, List[str]]:
    """Extract a small PDF page-range fallback for a missing leaf heading."""
    if pdf_path is None:
        return "", ["pdf_fallback_unavailable"]

    try:
        import fitz  # type: ignore
    except Exception:
        return "", ["pdf_fallback_unavailable"]

    section_id = section_match_id(sec)
    heading_rx = _pdf_heading_pattern(section_id, sec.get("title"))
    if heading_rx is None:
        return "", ["pdf_fallback_unavailable"]

    page_start = int(sec.get("page_start") or 1)
    page_end = int(sec.get("page_end") or page_start)
    if page_end < page_start:
        page_end = page_start
    page_end = min(page_end, page_start + max_pages - 1)

    try:
        with fitz.open(str(pdf_path)) as document:
            page_count = document.page_count
            parts = []
            for page_number in range(page_start, min(page_end, page_count) + 1):
                page = document.load_page(page_number - 1)
                parts.append(page.get_text("text"))
    except Exception as exc:
        logger.warning(
            "Local PDF fallback failed for %s %s: %s",
            getattr(pdf_path, "name", pdf_path),
            section_id,
            exc,
        )
        return "", ["pdf_fallback_failed"]

    page_text = "\n".join(parts)
    start_match = heading_rx.search(page_text)
    if start_match is None:
        return "", ["pdf_fallback_heading_not_found"]

    content_start = start_match.end()
    content_end = len(page_text)
    for candidate in later_sections:
        candidate_rx = _pdf_heading_pattern(
            section_match_id(candidate),
            candidate.get("title"),
        )
        if candidate_rx is None:
            continue
        candidate_match = candidate_rx.search(page_text, content_start)
        if candidate_match is not None:
            content_end = candidate_match.start()
            break

    fallback_text = page_text[content_start:content_end].strip()
    fallback_text = re.sub(r"[ \t]+", " ", fallback_text)
    fallback_text = re.sub(r"\n{3,}", "\n\n", fallback_text)
    if not fallback_text:
        return "", ["pdf_fallback_empty"]
    return fallback_text, ["pdf_fallback"]


def extract_section_text(
    markdown: str,
    sec: Dict[str, Any],
    section_index: int,
    ordered_sections: List[Dict[str, Any]],
    header_lookup: Dict[str, Dict[str, Any]],
    pdf_path: Optional[Any] = None,
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
        later_sections = ordered_sections[section_index + 1:]
        if not sec.get("_has_children"):
            fallback_text, fallback_flags = _local_pdf_fallback_text(
                pdf_path=pdf_path,
                sec=sec,
                later_sections=later_sections,
            )
            if fallback_text:
                return {
                    "text": fallback_text,
                    "quality_flags": sorted(set(flags + fallback_flags)),
                    "header_found": False,
                    "boundary_source": "local_pdf_fallback",
                }
            flags.extend(fallback_flags)
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
        if boundary and boundary[0] >= content_start:
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
        "text": (
            _extract_false_table_text(markdown, content_start, content_end)
            if "header_in_html_table" in flags
            else None
        ) or markdown[content_start:content_end].strip(),
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


def _contains_canonical_heading(
    text: str,
    section_id: str,
    section_title: Optional[str],
) -> bool:
    if not text or not section_id or not section_title:
        return False
    normalized_text = normalize_heading_for_matching(text)
    normalized_title = normalize_heading_for_matching(section_title)
    title_tokens = re.findall(r"\w+", normalized_title)
    if not title_tokens:
        return False
    pattern = (
        rf"(?<![\d.]){re.escape(section_id)}(?:\.)?\s+"
        + r"\W+".join(re.escape(token) for token in title_tokens)
    )
    return bool(re.search(pattern, normalized_text, re.IGNORECASE))


def validate_section_boundaries(
    toc_tree: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    markdown: str,
    doc_id: str,
) -> Dict[str, Any]:
    """Build a deterministic validation report for TOC-based section chunks."""
    sections = collect_sections(toc_tree)
    ordered = [section for section in sections if section.get("id")]
    expected_ids = [section["id"] for section in ordered]
    expected_set = set(expected_ids)
    chunk_ids = [chunk.get("section_id") for chunk in chunks]
    chunk_set = set(chunk_id for chunk_id in chunk_ids if chunk_id)

    missing_chunk_ids = [section_id for section_id in expected_ids if section_id not in chunk_set]
    extra_chunk_ids = [section_id for section_id in chunk_ids if section_id not in expected_set]
    duplicate_chunk_ids = sorted(
        section_id
        for section_id in chunk_set
        if chunk_ids.count(section_id) > 1
    )
    order_matches = chunk_ids == expected_ids

    body_start = find_body_start(markdown, ordered)
    cursor = body_start
    anchor_positions: Dict[str, int] = {}
    anchor_missing: List[str] = []
    anchor_non_monotonic: List[str] = []
    for section in ordered:
        info = locate_section_header(
            markdown=markdown,
            anchors={},
            sec=section,
            body_start=body_start,
            search_after=cursor,
            window=1,
        )
        header_pos = info.get("header_pos")
        if header_pos is None:
            anchor_missing.append(section["id"])
            continue
        if header_pos[0] < cursor:
            anchor_non_monotonic.append(section["id"])
        anchor_positions[section["id"]] = int(header_pos[0])
        cursor = max(cursor, int(header_pos[1]))

    chunk_by_id = {chunk.get("section_id"): chunk for chunk in chunks}
    empty_leaf_sections: List[str] = []
    parent_text_contains_child_heading: List[Dict[str, str]] = []
    child_text_also_in_parent: List[Dict[str, str]] = []
    heading_flags: Dict[str, List[str]] = {}

    for section in ordered:
        section_id = section["id"]
        chunk = chunk_by_id.get(section_id)
        if chunk is None:
            continue
        flags = chunk.get("quality_flags") or []
        if flags:
            heading_flags[section_id] = flags

        if (
            chunk.get("is_empty")
            and not section.get("_has_children")
            and not is_effectively_excluded_section(section)
        ):
            empty_leaf_sections.append(section_id)

        text = chunk.get("text") or ""
        if not text:
            continue
        descendants = [
            candidate
            for candidate in ordered
            if _is_descendant_id(candidate.get("id") or "", section_id)
        ]
        for child in descendants:
            child_id = child.get("id") or ""
            if _contains_canonical_heading(text, section_match_id(child), child.get("title")):
                parent_text_contains_child_heading.append(
                    {"parent_section_id": section_id, "child_section_id": child_id}
                )
            child_text = (chunk_by_id.get(child_id) or {}).get("text") or ""
            if (
                child_text
                and not is_effectively_empty(child_text)
                and child_text in text
            ):
                child_text_also_in_parent.append(
                    {"parent_section_id": section_id, "child_section_id": child_id}
                )

    boundary_uncertain = [
        chunk["section_id"]
        for chunk in chunks
        if "boundary_uncertain" in (chunk.get("quality_flags") or [])
    ]
    pdf_fallback = [
        chunk["section_id"]
        for chunk in chunks
        if "pdf_fallback" in (chunk.get("quality_flags") or [])
    ]
    header_not_found = [
        chunk["section_id"]
        for chunk in chunks
        if "header_not_found" in (chunk.get("quality_flags") or [])
    ]

    return {
        "doc_id": doc_id,
        "toc_section_count": len(expected_ids),
        "chunk_count": len(chunks),
        "section_ids_match_toc": not missing_chunk_ids and not extra_chunk_ids,
        "chunk_order_matches_toc": order_matches,
        "one_chunk_per_section": not duplicate_chunk_ids and len(chunks) == len(chunk_set),
        "anchors_monotonic": not anchor_non_monotonic,
        "anchor_positions_found": len(anchor_positions),
        "missing_chunk_section_ids": missing_chunk_ids,
        "extra_chunk_section_ids": extra_chunk_ids,
        "duplicate_chunk_section_ids": duplicate_chunk_ids,
        "anchor_missing_section_ids": anchor_missing,
        "anchor_non_monotonic_section_ids": anchor_non_monotonic,
        "empty_leaf_sections": empty_leaf_sections,
        "header_not_found_sections": header_not_found,
        "boundary_uncertain_sections": boundary_uncertain,
        "pdf_fallback_sections": pdf_fallback,
        "parent_text_contains_child_heading": parent_text_contains_child_heading,
        "child_text_also_in_parent": child_text_also_in_parent,
        "quality_flags_by_section": heading_flags,
    }



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
    header_lookup: Dict[str, Dict[str, Any]] = {}
    search_after = body_start
    for section in ordered:
        info = locate_section_header(
            markdown=markdown_manager.text,
            anchors=anchors,
            sec=section,
            body_start=body_start,
            search_after=search_after,
            window=1,
        )
        header_lookup[section["id"]] = info
        header_pos = info.get("header_pos")
        if header_pos is not None:
            search_after = max(search_after, int(header_pos[1]))

    known_titles: Dict[str, List[str]] = {}
    for section in ordered:
        match_id = section_match_id(section)
        if match_id and section.get("title"):
            known_titles.setdefault(match_id, []).append(section["title"])

    chunks: List[Dict[str, Any]] = []

    for section_index, sec in enumerate(ordered):
        sec_id = sec["id"]

        extraction = extract_section_text(
            markdown=markdown_manager.text,
            sec=sec,
            section_index=section_index,
            ordered_sections=ordered,
            header_lookup=header_lookup,
            pdf_path=markdown_manager.filepath,
        )
        raw_text = extraction["text"]
        quality_flags = list(extraction["quality_flags"])
        excluded = is_effectively_excluded_section(sec)
        if excluded:
            quality_flags.append("excluded_section")

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

        text_parts = [""] if empty else [raw_text]

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
                "excluded": excluded,
                # Preserve every TOC node; non-body/excluded sections remain
                # structural but are not sent to embedding by default.
                "embed": (not empty and not excluded),
                "part_index": part_idx,
                "part_count": len(text_parts),
                "quality_flags": part_flags,
                "boundary_source": extraction["boundary_source"],
            })

    logger.info("Built %d chunks", len(chunks))
    return chunks
