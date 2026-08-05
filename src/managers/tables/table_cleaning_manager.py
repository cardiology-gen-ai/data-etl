#!/usr/bin/env python3
"""Conservative rendering of raw table-catalogue JSONL records.

Input
-----
``*_tables_raw.jsonl`` files produced by :mod:`table_catalog_manager`.

Output
------
For every document:

* ``*_tables_clean.jsonl``: source records enriched with parsed rows,
  conservative rendered text, classification, and quality flags;
* ``*_logical_tables.jsonl``: recommendation continuations grouped only when
  adjacency and section provenance agree;
* ``*_recommendations.jsonl``: structured Recommendation/Class/Level rows;
* ``*_table_cleaning_summary.json`` and ``*_rendering_preview.md``.

The manager never modifies ``raw_html`` and never writes into chunk files.  It
performs only structure-backed normalization: HTML entity decoding, whitespace
collapse, explicit block/list breaks, and explicit superscript preservation.
It deliberately does not infer word boundaries, acronym boundaries, citations,
or footnote suffixes inside ordinary words.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

VERSION = "table_render_conservative_v1_4"
LOG = logging.getLogger(__name__)

SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
ROMAN_CLASS_RE = re.compile(r"^(?:I|IIa|IIb|III)$", re.I)
LEVEL_RE = re.compile(r"^[ABC]$", re.I)
RECOMMENDATION_LANGUAGE_RE = re.compile(
    r"\b(?:is|are|should|may|must|can)\s+(?:not\s+)?"
    r"(?:recommended|considered|indicated|performed|used|offered|avoided)\b"
    r"|\b(?:recommend(?:ed|ation)|should be considered|may be considered)\b",
    re.I,
)
GUIDANCE_SEMANTICS_RE = re.compile(
    r"\b(?:guidance|advice|considerations?|implications?|counselling|"
    r"interventions?|treatment goals?|practical options?)\b",
    re.I,
)
DEPENDENT_RECOMMENDATION_RE = re.compile(
    r"^\s*[•·\-–—]?\s*(?:is|are|was|were|should|may|must|can)\b",
    re.I,
)
BULLET_PREFIX_RE = re.compile(r"^\s*[•·\-–—]\s*")
PREPOSITIONAL_FRAGMENT_RE = re.compile(
    r"^\s*[•·\-–—]\s*(?:in|for|to|when|after|before|among|with|without)\b",
    re.I,
)
CONTEXT_FOOTNOTE_MARKER_RE = re.compile(
    r":\s*(?:\^\{)?[a-z](?:\})?\s*$",
    re.I,
)
SHORT_DISCOURSE_CONTEXT_RE = re.compile(
    r"^(?:in addition|additionally|alternatively|otherwise|therefore|thus):$",
    re.I,
)
CITATION_TOKEN_RE = re.compile(r"^\d{1,4}(?:[-–]\d{1,4})?$")
FOOTNOTE_TOKEN_RE = re.compile(r"^[a-z]$", re.I)
TERMINAL_SUPERSCRIPT_ANNOTATION_RE = re.compile(
    r"\s*\^\{(?P<suffix>[^{}]+)\}\s*$",
    re.I | re.S,
)
TERMINAL_PLAIN_ANNOTATION_RE = re.compile(
    r"(?P<body>.*?[.!?;])\s*(?P<suffix>"
    r"(?:[a-z](?:\s*,\s*[a-z]){0,2})?"
    r"(?:\s*,?\s*\d{1,4}(?:[-–]\d{1,4})?"
    r"(?:\s*,\s*\d{1,4}(?:[-–]\d{1,4})?)*)?"
    r")\s*$",
    re.I | re.S,
)
ADJACENT_MARKER_REFERENCE_RE = re.compile(
    r"^(?P<marker>[a-z])(?P<first_ref>\d{1,4}(?:[-–]\d{1,4})?)"
    r"(?P<rest>(?:\s*,\s*\d{1,4}(?:[-–]\d{1,4})?)*)$",
    re.I,
)
SQUARED_UNIT_GLUE_RE = re.compile(
    r"(?P<unit>\b(?:mg|g|µg|μg|kg|mmol|ml|l|cm|mm)/?m(?:2|²))"
    r"(?P<word>[A-Za-z]{2,})",
    re.I,
)
INTERNAL_COMMA_MARKER_GLUE_RE = re.compile(
    r"(?<=,)(?P<marker>[c-g])(?=[a-z]{4,})"
)
INTERNAL_MARKER_BEFORE_IS_RE = re.compile(
    r"\b[A-Za-z0-9/_-]{3,}(?P<marker>[c-g])(?=is\b)",
    re.I,
)
INTERNAL_ACRONYM_MARKER_OR_RE = re.compile(
    r"\b[A-Z][A-Z0-9/_-]{1,20}(?P<marker>[c-g])(?=or\b)"
)
INTERNAL_CITATION_BEFORE_FUNCTION_RE = re.compile(
    r"(?<=[a-z])\d{1,4}\s+(?="
    r"(?:before|after|during|when|is|are|should|may|must|can)\b)",
    re.I,
)
EXPLICIT_BULLET_RE = re.compile(r"\s*([•·])\s*")
COLON_GLUE_RE = re.compile(r"(?<=[A-Za-z0-9)])\:(?=[A-Za-z])")
SENTENCE_GLUE_RE = re.compile(r"(?<=[.!?])(?=[A-Z])")

RECOMMENDATION_ALIASES = {
    "recommendation",
    "recommendations",
    "recommendation text",
}
CLASS_ALIASES = {
    "class",
    "class of recommendation",
    "class of recommendations",
    "cor",
}
LEVEL_ALIASES = {"level", "level of evidence", "loe"}
SOURCE_ALIASES = {
    "source",
    "guideline",
    "guideline source",
    "year",
    "reference guideline",
}
BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "blockquote",
    "ul",
    "ol",
    "dl",
    "dt",
    "dd",
}


def normalize_space(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "")).strip()


def normalize_header(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"[\^\[\]\{\}]", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return normalize_space(text).casefold()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def normalize_explicit_boundaries(value: Any) -> str:
    """Normalize only separators that are explicit in the source text.

    The function does not infer ordinary word boundaries.  It restores spacing
    around visible bullets, after a colon glued to alphabetic text, after
    sentence punctuation followed by an uppercase letter, and after an
    explicit squared-dose unit such as ``mg/m2`` when a following word is
    visibly attached.  Ambiguous internal footnote glue is only flagged, never
    rewritten.
    """

    text = normalize_space(value)
    text = EXPLICIT_BULLET_RE.sub(r" \1 ", text)
    text = COLON_GLUE_RE.sub(": ", text)
    text = SENTENCE_GLUE_RE.sub(" ", text)
    text = SQUARED_UNIT_GLUE_RE.sub(r"\g<unit> \g<word>", text)
    return normalize_space(text)


def _annotation_tokens(suffix: str) -> tuple[list[str], list[str]]:
    """Parse a terminal annotation suffix into references and footnotes.

    Accepted tokens are deliberately narrow: lowercase single-letter footnote
    markers and numeric citation tokens/ranges.  Mixed or unfamiliar suffixes
    are rejected as a whole rather than partially consumed.
    """

    compact = normalize_space(suffix).strip()
    if not compact:
        return [], []

    adjacent = ADJACENT_MARKER_REFERENCE_RE.fullmatch(compact)
    if adjacent:
        refs = [adjacent.group("first_ref")]
        rest = adjacent.group("rest")
        if rest:
            refs.extend(
                normalize_space(item)
                for item in rest.lstrip(" ,").split(",")
                if normalize_space(item)
            )
        return refs, [adjacent.group("marker").casefold()]

    tokens = [normalize_space(item) for item in compact.split(",")]
    tokens = [item for item in tokens if item]
    if not tokens:
        return [], []
    markers = [item.casefold() for item in tokens if FOOTNOTE_TOKEN_RE.fullmatch(item)]
    refs = [item for item in tokens if CITATION_TOKEN_RE.fullmatch(item)]
    if len(markers) + len(refs) != len(tokens):
        return [], []
    # A terminal annotation must contain at least one numeric reference or one
    # explicit footnote marker.  This guard avoids consuming arbitrary prose.
    return refs, markers


def split_terminal_annotations(value: Any) -> tuple[str, list[str], list[str]]:
    """Split only unambiguous terminal citations and footnote markers.

    General supported forms include ``.^{12-14}``, ``.12,14``, ``.c12,14``,
    ``.c,12,14``, ``.d,e`` and ``.^{f}``.  Numeric material and alphabetic
    characters inside the sentence are preserved.
    """

    text = normalize_space(value)
    superscript = TERMINAL_SUPERSCRIPT_ANNOTATION_RE.search(text)
    if superscript:
        refs, markers = _annotation_tokens(superscript.group("suffix"))
        if refs or markers:
            return normalize_space(text[: superscript.start()]), refs, markers

    plain = TERMINAL_PLAIN_ANNOTATION_RE.fullmatch(text)
    if plain:
        suffix = normalize_space(plain.group("suffix"))
        refs, markers = _annotation_tokens(suffix)
        if refs or markers:
            return normalize_space(plain.group("body")), refs, markers
    return text, [], []


def split_terminal_citations(value: Any) -> tuple[str, list[str], Optional[str]]:
    """Backward-compatible wrapper around :func:`split_terminal_annotations`."""

    text, refs, markers = split_terminal_annotations(value)
    marker = ",".join(markers) if markers else None
    return text, refs, marker


def detect_text_quality_flags(value: Any) -> list[str]:
    """Return conservative diagnostics for ambiguous internal text glue.

    These rules only flag high-confidence structural anomalies.  They do not
    alter the source text because a letter attached inside a sentence may be a
    footnote marker or a genuine first/last letter of a word.
    """

    text = normalize_space(value)
    flags: list[str] = []
    if INTERNAL_COMMA_MARKER_GLUE_RE.search(text):
        flags.append("possible_footnote_marker_after_comma")
    if INTERNAL_MARKER_BEFORE_IS_RE.search(text):
        flags.append("possible_footnote_marker_before_is")
    if INTERNAL_ACRONYM_MARKER_OR_RE.search(text):
        flags.append("possible_footnote_marker_before_or")
    if SQUARED_UNIT_GLUE_RE.search(text):
        flags.append("missing_space_after_squared_unit")
    if INTERNAL_CITATION_BEFORE_FUNCTION_RE.search(text):
        flags.append("possible_embedded_citation")
    if flags:
        flags.insert(0, "possible_internal_text_glue")
    return ordered_unique(flags)


def normalize_group_header(value: Any) -> str:
    header = normalize_explicit_boundaries(value)
    # MinerU sometimes glues a single footnote marker to the colon terminating
    # a group header (for example ``recommended:d``).  Remove only that exact
    # structural suffix and keep the colon itself.
    return CONTEXT_FOOTNOTE_MARKER_RE.sub(":", header)


def recommendation_dependency_kind(
    group_header: Any,
    recommendation: Any,
) -> Optional[str]:
    header = normalize_group_header(group_header)
    rec = normalize_explicit_boundaries(recommendation)
    if not header:
        return None
    if DEPENDENT_RECOMMENDATION_RE.match(rec):
        return "auxiliary_verb"
    if (
        header.endswith(":")
        and RECOMMENDATION_LANGUAGE_RE.search(header)
        and PREPOSITIONAL_FRAGMENT_RE.match(rec)
    ):
        return "prepositional_list_item"
    if (
        header.endswith(":")
        and RECOMMENDATION_LANGUAGE_RE.search(header)
        and BULLET_PREFIX_RE.match(rec)
    ):
        return "recommendation_header_list_item"
    if header.endswith(":") and PREPOSITIONAL_FRAGMENT_RE.match(rec):
        return "prepositional_list_item"
    if SHORT_DISCOURSE_CONTEXT_RE.fullmatch(header) and BULLET_PREFIX_RE.match(rec):
        return "discourse_list_item"
    return None


def context_dependent_recommendation(
    recommendation: Any,
    group_header: Any = None,
) -> bool:
    return recommendation_dependency_kind(group_header, recommendation) is not None


def compose_normalized_recommendation(
    group_header: Any,
    recommendation: Any,
) -> tuple[str, bool, Optional[str]]:
    """Return a retrieval-ready sentence without inventing missing content.

    Context is prepended only when a general syntactic signal shows that the
    row depends on its preceding header: an auxiliary-verb continuation, a
    bullet governed by a recommendation-bearing header, a prepositional list
    item, or a short discourse continuation such as ``In addition:``.
    """

    rec = normalize_explicit_boundaries(recommendation)
    header = normalize_group_header(group_header)
    kind = recommendation_dependency_kind(header, rec)
    if not kind:
        return rec, False, None
    rec = BULLET_PREFIX_RE.sub("", rec)
    if kind == "discourse_list_item":
        header = header.rstrip(" :;,-–—") + ","
    elif kind == "recommendation_header_list_item":
        header = header.rstrip(" :;,-–—") + ":"
    else:
        header = header.rstrip(" :;,-–—")
    return normalize_space(f"{header} {rec}"), True, kind


def enrich_recommendation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Add annotation-separated, normalized, and diagnostic text fields."""

    output = dict(record)
    raw = normalize_space(output.get("raw_recommendation") or output.get("recommendation"))
    annotation_free, citations, footnote_markers = split_terminal_annotations(raw)
    annotation_free = normalize_explicit_boundaries(annotation_free)
    normalized, dependent, dependency_kind = compose_normalized_recommendation(
        output.get("group_header"),
        annotation_free,
    )

    # A bullet with no local group header is not expanded using the section
    # title, because that would invent prose.  It is nevertheless marked as an
    # orphan list item and receives the section title as a non-generative hint.
    context_hint = normalize_space(output.get("context_hint"))
    if (
        not dependent
        and not normalize_space(output.get("group_header"))
        and BULLET_PREFIX_RE.match(annotation_free)
    ):
        section_hint = normalize_space(output.get("section_title"))
        dependent = True
        dependency_kind = "orphan_list_item"
        context_hint = context_hint or section_hint

    output["raw_recommendation"] = raw
    output["recommendation"] = annotation_free
    output["normalized_recommendation"] = normalized
    output["citation_numbers"] = citations
    output["footnote_markers"] = footnote_markers
    output["text_quality_flags"] = detect_text_quality_flags(raw)
    output["context_dependent"] = dependent
    if dependency_kind:
        output["context_dependency_kind"] = dependency_kind
    else:
        output.pop("context_dependency_kind", None)
    if context_hint:
        output["context_hint"] = context_hint
        output["context_hint_source"] = "section_title"
    else:
        output.pop("context_hint", None)
        output.pop("context_hint_source", None)
    # Backward compatibility for consumers that previously read one marker.
    if footnote_markers:
        output["citation_marker"] = ",".join(footnote_markers)
    else:
        output.pop("citation_marker", None)
    return output


def retrieval_state(record: Mapping[str, Any]) -> tuple[str, bool]:
    """Return catalogue status and whether a table is active for retrieval."""
    linked = str(record.get("link_status") or "").startswith("matched")
    excluded = bool(record.get("excluded", False))
    embed = bool(record.get("embed", True))
    if linked and embed and not excluded:
        return "linked_active", True
    if linked:
        return "linked_excluded", False
    return "catalog_only_unlinked", False


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


@dataclass
class Cell:
    raw_text: str
    text: str
    header: bool = False
    colspan: int = 1
    rowspan: int = 1
    parts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "text": self.text,
            "header": self.header,
            "colspan": self.colspan,
            "rowspan": self.rowspan,
            "parts": self.parts,
        }


class ConservativeTableHTMLParser(HTMLParser):
    """Parse HTML rows/cells while preserving only explicit layout breaks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[Cell]] = []
        self._row: Optional[list[Cell]] = None
        self._parts: Optional[list[str]] = None
        self._header = False
        self._colspan = 1
        self._rowspan = 1
        self._sup_depth = 0

    @staticmethod
    def _positive_int(value: Optional[str]) -> int:
        try:
            return max(1, int(value or "1"))
        except (TypeError, ValueError):
            return 1

    def _break(self) -> None:
        if self._parts is not None and (not self._parts or self._parts[-1] != "\n"):
            self._parts.append("\n")

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
    ) -> None:
        tag = tag.casefold()
        attr = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._parts = []
            self._header = tag == "th"
            self._colspan = self._positive_int(attr.get("colspan"))
            self._rowspan = self._positive_int(attr.get("rowspan"))
        elif tag == "br":
            self._break()
        elif tag == "li":
            self._break()
            if self._parts is not None:
                self._parts.append("• ")
        elif tag in BLOCK_TAGS:
            self._break()
        elif tag == "sup" and self._parts is not None:
            self._parts.append("^{")
            self._sup_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "sup" and self._parts is not None and self._sup_depth:
            self._parts.append("}")
            self._sup_depth -= 1
        elif tag in BLOCK_TAGS or tag == "li":
            self._break()
        elif tag in {"td", "th"} and self._parts is not None:
            raw = "".join(self._parts)
            parts = [normalize_space(part) for part in raw.split("\n")]
            parts = [part for part in parts if part]
            text = "; ".join(parts)
            if self._row is not None:
                self._row.append(
                    Cell(
                        raw_text=raw,
                        text=text,
                        header=self._header,
                        colspan=self._colspan,
                        rowspan=self._rowspan,
                        parts=parts,
                    )
                )
            self._parts = None
            self._header = False
            self._colspan = 1
            self._rowspan = 1
            self._sup_depth = 0
        elif tag == "tr" and self._row is not None:
            if any(cell.text for cell in self._row):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._parts is not None:
            self._parts.append(data)


@dataclass
class GridSlot:
    cell: Cell
    origin_row: int
    origin_col: int


@dataclass
class RowShape:
    class_index: int
    level_index: int
    recommendation_indices: list[int]
    source_indices: list[int]
    class_value: str
    level_value: str


def expanded_grid(rows: Sequence[Sequence[Cell]]) -> list[list[Optional[GridSlot]]]:
    grid: list[list[Optional[GridSlot]]] = []

    def ensure(row_index: int, columns: int) -> None:
        while len(grid) <= row_index:
            grid.append([])
        while len(grid[row_index]) < columns:
            grid[row_index].append(None)

    for row_index, row in enumerate(rows):
        ensure(row_index, 0)
        column_index = 0
        for cell in row:
            while True:
                ensure(row_index, column_index + 1)
                if grid[row_index][column_index] is None:
                    break
                column_index += 1
            for row_offset in range(cell.rowspan):
                for column_offset in range(cell.colspan):
                    target_row = row_index + row_offset
                    target_column = column_index + column_offset
                    ensure(target_row, target_column + 1)
                    if grid[target_row][target_column] is not None:
                        raise ValueError("Overlapping rowspan/colspan in table")
                    grid[target_row][target_column] = GridSlot(
                        cell=cell,
                        origin_row=row_index,
                        origin_col=column_index,
                    )
            column_index += cell.colspan

    width = max((len(row) for row in grid), default=0)
    for row in grid:
        row.extend([None] * (width - len(row)))
    return grid


def grid_values(row: Sequence[Optional[GridSlot]]) -> list[str]:
    values: list[str] = []
    previous_origin: Optional[tuple[int, int]] = None
    for slot in row:
        if slot is None:
            values.append("")
            previous_origin = None
            continue
        origin = (slot.origin_row, slot.origin_col)
        if origin == previous_origin:
            continue
        values.append(slot.cell.text)
        previous_origin = origin
    while values and not values[-1]:
        values.pop()
    return values


def canonical_header(value: str) -> Optional[str]:
    text = normalize_header(value)
    if text in RECOMMENDATION_ALIASES:
        return "recommendation"
    if text in CLASS_ALIASES or re.fullmatch(r"class\s*[a-z]?", text):
        return "class"
    if text in LEVEL_ALIASES or re.fullmatch(r"level\s*[a-z]?", text):
        return "level"
    if text in SOURCE_ALIASES:
        return "source"
    return None


def normalize_class(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]", "", normalize_space(value)).upper()
    aliases = {
        "I": "I",
        "II": "II",
        "IIA": "IIa",
        "IIB": "IIb",
        "III": "III",
        "1": "I",
    }
    return aliases.get(compact, normalize_space(value))


def normalize_level(value: str) -> str:
    compact = re.sub(r"[^A-Za-z]", "", normalize_space(value)).upper()
    return compact if compact in {"A", "B", "C"} else normalize_space(value)


def valid_class(value: str) -> bool:
    return bool(ROMAN_CLASS_RE.fullmatch(normalize_class(value)))


def valid_level(value: str) -> bool:
    return bool(LEVEL_RE.fullmatch(normalize_level(value)))


def find_header_map(rows: Sequence[Sequence[Cell]]) -> tuple[Optional[int], dict[str, int]]:
    best_index: Optional[int] = None
    best_map: dict[str, int] = {}
    for row_index, row in enumerate(rows[:6]):
        current: dict[str, int] = {}
        logical_index = 0
        for cell in row:
            key = canonical_header(cell.text)
            if key and key not in current:
                current[key] = logical_index
            logical_index += cell.colspan
        if len(current) > len(best_map):
            best_index, best_map = row_index, current
        if {"recommendation", "class", "level"} <= set(current):
            return row_index, current
        if {"class", "level"} <= set(current):
            best_index, best_map = row_index, current
    return best_index, best_map


def row_shape(values: Sequence[str]) -> Optional[RowShape]:
    for class_index in range(max(0, len(values) - 1)):
        level_index = class_index + 1
        if not (valid_class(values[class_index]) and valid_level(values[level_index])):
            continue
        recommendation_indices = [
            index for index in range(class_index) if normalize_space(values[index])
        ]
        if not recommendation_indices:
            continue
        source_indices = [
            index
            for index in range(level_index + 1, len(values))
            if normalize_space(values[index])
        ]
        return RowShape(
            class_index=class_index,
            level_index=level_index,
            recommendation_indices=recommendation_indices,
            source_indices=source_indices,
            class_value=normalize_class(values[class_index]),
            level_value=normalize_level(values[level_index]),
        )
    return None


def strict_acronym_like(value: str) -> bool:
    text = normalize_space(value).strip(" .,:;()[]{}")
    if not (2 <= len(text) <= 24) or len(text.split()) > 3:
        return False
    if not re.fullmatch(r"[A-Za-z0-9.+\-/]+(?:\s+[A-Za-z0-9.+\-/]+){0,2}", text):
        return False
    uppercase = sum(char.isupper() for char in text)
    lowercase = sum(char.islower() for char in text)
    digits = sum(char.isdigit() for char in text)
    return (uppercase + digits) >= 2 and uppercase >= lowercase


def glossary_score(rows: Sequence[Sequence[Cell]]) -> float:
    grid = expanded_grid(rows)
    usable: list[list[str]] = []
    for row in grid:
        values = [value for value in grid_values(row) if value]
        if len(values) == 2:
            usable.append(values)
    if len(usable) < 5:
        return 0.0
    acronym_left = sum(strict_acronym_like(values[0]) for values in usable)
    return acronym_left / len(usable)


def structured_guidance_score(
    rows: Sequence[Sequence[Cell]],
    caption: Sequence[str] | str | None = None,
) -> tuple[float, bool, float]:
    """Measure a general label-to-guidance table structure.

    A guidance table has repeated short labels in the first column and longer
    explanatory text in the second.  Semantic guidance terms are considered in
    captions and the first row, but no document title or table identifier is
    used.
    """

    grid = expanded_grid(rows)
    pairs: list[tuple[str, str]] = []
    for row in grid:
        values = [value for value in grid_values(row) if value]
        if len(values) == 2:
            pairs.append((values[0], values[1]))
    if len(pairs) < 3:
        return 0.0, False, 0.0

    # The first pair often contains column headers, so score all rows but do not
    # require that row itself to look like a data record.
    data_pairs = pairs[1:] if len(pairs) >= 4 else pairs
    label_description = sum(
        len(normalize_space(left)) <= 100
        and len(normalize_space(left).split()) <= 14
        and len(normalize_space(right)) >= 18
        for left, right in data_pairs
    )
    structure_score = label_description / max(1, len(data_pairs))
    recommendation_ratio = sum(
        bool(RECOMMENDATION_LANGUAGE_RE.search(right)) for _, right in data_pairs
    ) / max(1, len(data_pairs))

    caption_text = (
        " ".join(str(item) for item in caption)
        if isinstance(caption, (list, tuple))
        else str(caption or "")
    )
    first_row_text = " ".join(pairs[0]) if pairs else ""
    semantic_signal = bool(
        GUIDANCE_SEMANTICS_RE.search(f"{caption_text} {first_row_text}")
    )
    return structure_score, semantic_signal, recommendation_ratio


def classify_rows(
    rows: Sequence[Sequence[Cell]],
    source_classification: str,
    caption: Sequence[str] | str | None = None,
) -> tuple[str, list[str], Optional[int], dict[str, int], int]:
    header_index, header_map = find_header_map(rows)
    grid = expanded_grid(rows)
    shaped = sum(row_shape(grid_values(row)) is not None for row in grid)
    max_columns = max((len(row) for row in grid), default=0)
    nonempty_rows = sum(any(grid_values(row)) for row in grid)
    reasons: list[str] = []

    if {"recommendation", "class", "level"} <= set(header_map):
        return (
            "recommendation_table",
            ["recommendation/class/level headers"],
            header_index,
            header_map,
            shaped,
        )
    if {"class", "level"} <= set(header_map) and shaped:
        return (
            "recommendation_table",
            ["class/level headers and recommendation-shaped rows"],
            header_index,
            header_map,
            shaped,
        )
    if shaped:
        return (
            "recommendation_continuation_candidate",
            [f"recommendation_shaped_rows={shaped}"],
            header_index,
            header_map,
            shaped,
        )

    caption_text = (
        " ".join(str(item) for item in caption)
        if isinstance(caption, (list, tuple))
        else str(caption or "")
    ).casefold()
    matrix_semantics = bool(
        re.search(r"\b(?:gene|genes|genotype|phenotype|phenotypes)\b", caption_text)
    )
    # Sparse clinical matrices can accidentally resemble acronym glossaries when
    # only their two non-empty cells are inspected.  Tables wider than four
    # columns, and explicitly gene/phenotype-labelled tables, must remain
    # clinical matrices.
    if nonempty_rows and (max_columns > 4 or matrix_semantics):
        reasons = [f"rows={nonempty_rows}", f"max_columns={max_columns}"]
        if matrix_semantics:
            reasons.append("clinical_matrix_caption")
        return (
            "clinical_table",
            reasons,
            header_index,
            header_map,
            shaped,
        )

    if (
        max_columns <= 4
        and (source_classification == "acronym_or_glossary" or glossary_score(rows) >= 0.8)
    ):
        return (
            "acronym_or_glossary",
            ["two-column short-left structure"],
            header_index,
            header_map,
            shaped,
        )

    visible = " ".join(cell.text for row in rows for cell in row)
    guidance_score, guidance_semantics, guidance_recommendation_ratio = (
        structured_guidance_score(rows, caption)
    )
    if (
        max_columns <= 2
        and nonempty_rows >= 3
        and guidance_score >= 0.60
        and (guidance_semantics or guidance_recommendation_ratio >= 0.25)
    ):
        return (
            "structured_guidance_table",
            [
                f"label_description_score={guidance_score:.2f}",
                f"recommendation_language_ratio={guidance_recommendation_ratio:.2f}",
                f"guidance_semantics={str(guidance_semantics).lower()}",
            ],
            header_index,
            header_map,
            shaped,
        )
    if max_columns <= 2 and RECOMMENDATION_LANGUAGE_RE.search(visible):
        return (
            "recommendation_text_fragment",
            ["recommendation language without Class/Level"],
            header_index,
            header_map,
            shaped,
        )
    if nonempty_rows and max_columns <= 1:
        return (
            "structured_list",
            ["single-column table"],
            header_index,
            header_map,
            shaped,
        )
    if nonempty_rows:
        return (
            "clinical_table",
            [f"rows={nonempty_rows}", f"max_columns={max_columns}"],
            header_index,
            header_map,
            shaped,
        )
    return "empty_table", ["no non-empty rows"], header_index, header_map, shaped


def is_group_header(row: Sequence[Cell], width: int) -> bool:
    nonempty = [cell for cell in row if cell.text]
    if len(nonempty) != 1:
        return False
    return len(row) == 1 or nonempty[0].colspan >= max(1, width)


def recommendation_rows_for_table(
    *,
    record: Mapping[str, Any],
    rows: Sequence[Sequence[Cell]],
    header_index: Optional[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    grid = expanded_grid(rows)
    width = max((len(row) for row in grid), default=0)
    context: Optional[str] = None
    output: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for row_index, row in enumerate(grid):
        values = grid_values(row)
        if not any(values):
            continue
        if header_index is not None and row_index == header_index:
            continue

        original = rows[row_index] if row_index < len(rows) else []
        if original and is_group_header(original, width):
            context = next((cell.text for cell in original if cell.text), context)
            continue

        shape = row_shape(values)
        if shape is None:
            joined = " | ".join(value for value in values if value)
            if len([value for value in values if value]) == 1:
                context = joined
            elif RECOMMENDATION_LANGUAGE_RE.search(joined):
                unresolved.append(f"row_{row_index}:recommendation_without_class_level")
            elif joined:
                unresolved.append(f"row_{row_index}:unrecognized_recommendation_row")
            continue

        rec_values = [values[index] for index in shape.recommendation_indices]
        local_context = " | ".join(rec_values[:-1]) if len(rec_values) > 1 else ""
        recommendation = rec_values[-1]
        sources = [values[index] for index in shape.source_indices]
        table_id = str(record.get("table_id") or "")
        output.append(
            enrich_recommendation_record(
                {
                    "version": VERSION,
                    "record_type": "recommendation_with_class_level",
                    "recommendation_id": f"{table_id}::row::{row_index:04d}",
                    "table_id": table_id,
                    "doc_id": record.get("doc_id"),
                    "section_id": record.get("section_id"),
                    "section_title": record.get("section_title"),
                    "page_idx": record.get("page_idx"),
                    "page": record.get("page"),
                    "source_row_index": row_index,
                    "group_header": local_context or context,
                    "recommendation": recommendation,
                    "class": shape.class_value,
                    "level": shape.level_value,
                    "source_values": sources,
                    "link_status": record.get("link_status"),
                    "chunk_id": record.get("chunk_id"),
                    "excluded": bool(record.get("excluded", False)),
                    "embed": bool(record.get("embed", True)),
                    "catalog_status": retrieval_state(record)[0],
                    "active_for_retrieval": retrieval_state(record)[1],
                }
            )
        )
    return output, unresolved


def recommendation_fragments_for_table(
    *,
    record: Mapping[str, Any],
    rows: Sequence[Sequence[Cell]],
) -> list[dict[str, Any]]:
    """Extract prescriptive text lacking explicit Class/Level cells.

    These records are kept separate from fully structured recommendations.
    Missing Class and Level values remain ``None`` and are never inferred from
    neighbouring tables or document-specific knowledge.
    """

    grid = expanded_grid(rows)
    context: Optional[str] = None
    output: list[dict[str, Any]] = []
    table_id = str(record.get("table_id") or "")
    has_embedded_image = bool(re.search(r"<img\b", str(record.get("raw_html") or ""), re.I))

    for row_index, row in enumerate(grid):
        values = [value for value in grid_values(row) if normalize_space(value)]
        if not values:
            continue
        joined = " | ".join(values)
        if RECOMMENDATION_LANGUAGE_RE.search(joined):
            fragment = enrich_recommendation_record(
                {
                    "version": VERSION,
                    "record_type": "recommendation_text_fragment",
                    "recommendation_fragment_id": (
                        f"{table_id}::fragment::{row_index:04d}"
                    ),
                    "recommendation_id": f"{table_id}::fragment::{row_index:04d}",
                    "table_id": table_id,
                    "doc_id": record.get("doc_id"),
                    "section_id": record.get("section_id"),
                    "section_title": record.get("section_title"),
                    "page_idx": record.get("page_idx"),
                    "page": record.get("page"),
                    "source_row_index": row_index,
                    "group_header": context,
                    "recommendation": joined,
                    "class": None,
                    "level": None,
                    "source_values": [],
                    "link_status": record.get("link_status"),
                    "chunk_id": record.get("chunk_id"),
                    "excluded": bool(record.get("excluded", False)),
                    "embed": bool(record.get("embed", True)),
                    "catalog_status": retrieval_state(record)[0],
                    "active_for_retrieval": retrieval_state(record)[1],
                    "quality_flags": ordered_unique(
                        [
                            "missing_class_level",
                            *(
                                ["embedded_image_present"]
                                if has_embedded_image
                                else []
                            ),
                        ]
                    ),
                }
            )
            output.append(fragment)
        elif len(values) == 1:
            context = values[0]
    return output


def render_table(
    classification: str,
    rows: Sequence[Sequence[Cell]],
    recommendation_rows: Sequence[Mapping[str, Any]],
    recommendation_fragments: Sequence[Mapping[str, Any]] = (),
) -> str:
    label = {
        "recommendation_table": "Recommendation table",
        "recommendation_continuation": "Recommendation continuation",
        "recommendation_continuation_candidate": "Recommendation continuation candidate",
        "recommendation_text_fragment": "Recommendation text fragment",
        "structured_guidance_table": "Structured guidance table",
        "acronym_or_glossary": "Glossary table",
        "structured_list": "Structured list",
        "clinical_table": "Clinical table",
        "empty_table": "Empty table",
    }.get(classification, classification)
    lines = [f"[{label}]"]

    if classification.startswith("recommendation") and recommendation_rows:
        for row in recommendation_rows:
            if row.get("group_header"):
                lines.append(f"Context: {row['group_header']}")
            elif row.get("context_hint"):
                lines.append(f"Context hint: {row['context_hint']}")
            lines.append(f"Recommendation: {row['recommendation']}")
            normalized = row.get("normalized_recommendation") or row["recommendation"]
            if normalized != row["recommendation"]:
                lines.append(f"Normalized: {normalized}")
            lines.append(f"Class: {row['class']}")
            lines.append(f"Level: {row['level']}")
            if row.get("citation_numbers"):
                lines.append("Citations: " + ", ".join(row["citation_numbers"]))
            if row.get("footnote_markers"):
                lines.append("Footnotes: " + ", ".join(row["footnote_markers"]))
            if row.get("text_quality_flags"):
                lines.append("Text flags: " + ", ".join(row["text_quality_flags"]))
            if row.get("source_values"):
                lines.append("Source: " + " | ".join(row["source_values"]))
            lines.append("")
        return "\n".join(lines).strip()

    if classification == "recommendation_text_fragment" and recommendation_fragments:
        for row in recommendation_fragments:
            if row.get("group_header"):
                lines.append(f"Context: {row['group_header']}")
            elif row.get("context_hint"):
                lines.append(f"Context hint: {row['context_hint']}")
            lines.append(f"Recommendation fragment: {row['recommendation']}")
            lines.append("Class: null")
            lines.append("Level: null")
            if row.get("citation_numbers"):
                lines.append("Citations: " + ", ".join(row["citation_numbers"]))
            if row.get("footnote_markers"):
                lines.append("Footnotes: " + ", ".join(row["footnote_markers"]))
            if row.get("text_quality_flags"):
                lines.append("Text flags: " + ", ".join(row["text_quality_flags"]))
            lines.append("")
        return "\n".join(lines).strip()

    grid = expanded_grid(rows)
    if classification == "acronym_or_glossary":
        # ESC acronym pages often contain two acronym-definition pairs per row.
        # Emit each pair independently and avoid repeating rowspan-origin pairs.
        seen_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for row in grid:
            for column in range(0, max(0, len(row) - 1), 2):
                left = row[column]
                right = row[column + 1]
                if left is None or right is None:
                    continue
                term = left.cell.text
                definition = right.cell.text
                if not term or not definition:
                    continue
                key = (
                    (left.origin_row, left.origin_col),
                    (right.origin_row, right.origin_col),
                )
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                lines.append(f"{term}: {definition}")
        return "\n".join(lines).strip()

    if classification == "structured_guidance_table":
        for row_index, row in enumerate(grid):
            values = [value for value in grid_values(row) if value]
            if not values:
                continue
            first_row_looks_like_header = (
                row_index == 0
                and len(values) == 2
                and len(normalize_space(values[0])) <= 100
                and len(normalize_space(values[1])) <= 100
            )
            if first_row_looks_like_header:
                lines.append(" | ".join(values))
            elif len(values) == 2:
                lines.append(f"{values[0]}: {normalize_explicit_boundaries(values[1])}")
            else:
                lines.append(" | ".join(values))
        return "\n".join(lines).strip()

    for row in grid:
        values = [value for value in grid_values(row) if value]
        if not values:
            continue
        if classification == "structured_list":
            lines.append("- " + " — ".join(values))
        else:
            lines.append(" | ".join(values))
    return "\n".join(lines).strip()


def parse_catalog_record(record: Mapping[str, Any]) -> dict[str, Any]:
    raw_html = str(record.get("raw_html") or "")
    expected_hash = record.get("raw_html_sha256")
    actual_hash = sha256_text(raw_html) if raw_html else None
    integrity_ok = expected_hash in {None, actual_hash}

    parser = ConservativeTableHTMLParser()
    parser.feed(raw_html)
    parser.close()

    source_classification = str(record.get("classification") or "")
    classification, reasons, header_index, header_map, shaped = classify_rows(
        parser.rows,
        source_classification,
        record.get("caption"),
    )
    structured_recommendation = classification in {
        "recommendation_table",
        "recommendation_continuation",
        "recommendation_continuation_candidate",
    }
    if structured_recommendation:
        recommendation_rows, unresolved = recommendation_rows_for_table(
            record=record,
            rows=parser.rows,
            header_index=header_index,
        )
    else:
        recommendation_rows, unresolved = [], []

    recommendation_fragments = (
        recommendation_fragments_for_table(record=record, rows=parser.rows)
        if classification == "recommendation_text_fragment"
        else []
    )

    # Drop derived flags from a previous processing pass so reruns are
    # idempotent even when a processed catalogue is used for diagnostics.
    flags = [
        flag
        for flag in (record.get("quality_flags") or [])
        if flag not in {
            "raw_html_hash_mismatch",
            "unresolved_recommendation_rows",
        }
    ]
    if not integrity_ok:
        flags.append("raw_html_hash_mismatch")
    if unresolved:
        flags.append("unresolved_recommendation_rows")
    flags = ordered_unique(flags)

    catalog_status, active_for_retrieval = retrieval_state(record)
    output = dict(record)
    output.update(
        {
            "processing_version": VERSION,
            "catalog_status": catalog_status,
            "active_for_retrieval": active_for_retrieval,
            "source_classification": record.get("classification"),
            "source_classification_reasons": record.get("classification_reasons"),
            "classification": classification,
            "classification_reasons": reasons,
            "header_row_index": header_index,
            "header_map": header_map or None,
            "row_count": len(parser.rows),
            "max_columns": max((len(row) for row in expanded_grid(parser.rows)), default=0),
            "recommendation_shaped_row_count": shaped,
            "recommendation_rows": recommendation_rows,
            "recommendation_fragments": recommendation_fragments,
            "unresolved_patterns": unresolved,
            "quality_flags": flags,
            "rows": [[cell.as_dict() for cell in row] for row in parser.rows],
            "rendered_text": render_table(
                classification,
                parser.rows,
                recommendation_rows,
                recommendation_fragments,
            ),
            "raw_html_integrity_ok": integrity_ok,
            "raw_html_unchanged": True,
        }
    )
    output["rendered_text_sha256"] = sha256_text(output["rendered_text"])
    return output


def _recommendation_render(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["[Recommendation table]"]
    for row in rows:
        if row.get("group_header"):
            lines.append(f"Context: {row['group_header']}")
        elif row.get("context_hint"):
            lines.append(f"Context hint: {row['context_hint']}")
        recommendation = row.get("recommendation") or ""
        lines.append(f"Recommendation: {recommendation}")
        normalized = row.get("normalized_recommendation") or recommendation
        if normalized != recommendation:
            lines.append(f"Normalized: {normalized}")
        lines.append(f"Class: {row.get('class') or ''}")
        lines.append(f"Level: {row.get('level') or ''}")
        if row.get("citation_numbers"):
            lines.append("Citations: " + ", ".join(row["citation_numbers"]))
        if row.get("footnote_markers"):
            lines.append("Footnotes: " + ", ".join(row["footnote_markers"]))
        if row.get("text_quality_flags"):
            lines.append("Text flags: " + ", ".join(row["text_quality_flags"]))
        if row.get("source_values"):
            lines.append("Source: " + " | ".join(row["source_values"]))
        lines.append("")
    return "\n".join(lines).strip()


def _fragment_render(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["[Recommendation text fragment]"]
    for row in rows:
        if row.get("group_header"):
            lines.append(f"Context: {row['group_header']}")
        elif row.get("context_hint"):
            lines.append(f"Context hint: {row['context_hint']}")
        lines.append(f"Recommendation fragment: {row.get('recommendation') or ''}")
        lines.append("Class: null")
        lines.append("Level: null")
        if row.get("citation_numbers"):
            lines.append("Citations: " + ", ".join(row["citation_numbers"]))
        if row.get("footnote_markers"):
            lines.append("Footnotes: " + ", ".join(row["footnote_markers"]))
        if row.get("text_quality_flags"):
            lines.append("Text flags: " + ", ".join(row["text_quality_flags"]))
        lines.append("")
    return "\n".join(lines).strip()


def _logical_from_fragments(
    fragments: Sequence[dict[str, Any]],
    *,
    doc_id: str,
    section_key: str,
) -> dict[str, Any]:
    ordered = sorted(
        fragments,
        key=lambda item: (
            int(item.get("fragment_index") or 10**9),
            int(item.get("source_index") or 0),
        ),
    )
    first = ordered[0]
    recommendations: list[dict[str, Any]] = []
    recommendation_fragments: list[dict[str, Any]] = []
    inherited_context: Optional[str] = None
    inherited_from_table: Optional[str] = None

    def append_with_context(
        original_row: Mapping[str, Any],
        *,
        table_id: str,
        destination: list[dict[str, Any]],
    ) -> None:
        nonlocal inherited_context, inherited_from_table
        row = dict(original_row)
        current_context = normalize_space(row.get("group_header"))
        if current_context:
            inherited_context = current_context
            inherited_from_table = table_id
        elif inherited_context:
            row["group_header"] = inherited_context
            row["group_header_inherited"] = True
            row["group_header_inherited_from_table_id"] = inherited_from_table
        # Recompute context-dependent normalized text after inheritance.
        destination.append(enrich_recommendation_record(row))

    for item in ordered:
        table_id = str(item.get("table_id") or "")
        for original_row in item.get("recommendation_rows") or []:
            append_with_context(
                original_row,
                table_id=table_id,
                destination=recommendations,
            )
        for original_row in item.get("recommendation_fragments") or []:
            append_with_context(
                original_row,
                table_id=table_id,
                destination=recommendation_fragments,
            )

    classifications = [str(item.get("classification") or "") for item in ordered]
    if recommendations:
        classification = (
            "recommendation_table_with_continuations"
            if len(ordered) > 1
            else classifications[0]
        )
        rendered_text = _recommendation_render(recommendations)
    elif recommendation_fragments:
        classification = "recommendation_text_fragment"
        rendered_text = _fragment_render(recommendation_fragments)
    else:
        classification = classifications[0]
        rendered_text = "\n\n".join(
            str(item.get("rendered_text") or "") for item in ordered
        ).strip()

    active = all(bool(item.get("active_for_retrieval")) for item in ordered)
    statuses = ordered_unique(str(item.get("catalog_status") or "") for item in ordered)
    link_statuses = ordered_unique(str(item.get("link_status") or "") for item in ordered)
    return {
        "version": VERSION,
        "doc_id": doc_id,
        "section_key": section_key,
        "section_id": first.get("section_id"),
        "section_title": first.get("section_title"),
        "chunk_id": first.get("chunk_id"),
        "classification": classification,
        "link_status": link_statuses[0] if len(link_statuses) == 1 else "mixed",
        "catalog_status": statuses[0] if len(statuses) == 1 else "mixed",
        "active_for_retrieval": active,
        "fragment_group_id": first.get("fragment_group_id"),
        "fragment_ids": [item.get("table_id") for item in ordered],
        "fragment_count": len(ordered),
        "recommendation_rows": recommendations,
        "recommendation_fragments": recommendation_fragments,
        "rendered_text": rendered_text,
        "_sort_order": min(int(item.get("source_index") or 0) for item in ordered),
    }


def group_logical_tables(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build logical tables from one-to-one records and page fragments."""

    logical_temp: list[dict[str, Any]] = []
    consumed_ids: set[str] = set()

    # Explicit fragment groups created by the catalogue linker take priority and
    # work for both recommendation and clinical multipage tables.
    fragment_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_id = normalize_space(record.get("fragment_group_id"))
        if group_id:
            fragment_groups[group_id].append(record)

    for group_id, fragments in fragment_groups.items():
        first = min(fragments, key=lambda item: int(item.get("source_index") or 0))
        doc_id = str(first.get("doc_id") or "")
        section_key = str(first.get("section_id") or f"page:{first.get('page_idx')}")
        logical_temp.append(
            _logical_from_fragments(fragments, doc_id=doc_id, section_key=section_key)
        )
        consumed_ids.update(str(item.get("table_id") or "") for item in fragments)

    # Preserve the previous continuation grouping for ordinary one-to-one linked
    # records that do not carry an explicit fragment group.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record.get("table_id") or "") in consumed_ids:
            continue
        section_key = str(record.get("section_id") or f"page:{record.get('page_idx')}")
        grouped[(str(record.get("doc_id") or ""), section_key)].append(record)

    for (doc_id, section_key), section_records in sorted(grouped.items()):
        section_records.sort(key=lambda item: int(item.get("source_index") or 0))
        current: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal current
            if current:
                logical_temp.append(
                    _logical_from_fragments(
                        current,
                        doc_id=doc_id,
                        section_key=section_key,
                    )
                )
                current = []

        for record in section_records:
            classification = str(record.get("classification") or "")
            if not current:
                current = [record]
                continue
            previous = current[-1]
            consecutive = (
                int(record.get("source_index") or 0)
                == int(previous.get("source_index") or 0) + 1
            )
            if (
                str(current[0].get("classification") or "") == "recommendation_table"
                and classification == "recommendation_continuation_candidate"
                and consecutive
            ):
                record["classification"] = "recommendation_continuation"
                current.append(record)
                continue
            flush()
            current = [record]
        flush()

    logical_temp.sort(
        key=lambda item: (
            str(item.get("doc_id") or ""),
            int(item.get("_sort_order") or 0),
        )
    )
    logical: list[dict[str, Any]] = []
    per_section_counter: Counter[tuple[str, str]] = Counter()
    for item in logical_temp:
        doc_id = str(item.get("doc_id") or "")
        section_key = str(item.pop("section_key", ""))
        item.pop("_sort_order", None)
        key = (doc_id, section_key)
        per_section_counter[key] += 1
        item["logical_table_id"] = (
            f"{doc_id}::{section_key}::logical::{per_section_counter[key]:04d}"
        )
        logical.append(item)
    return logical


def markdown_preview(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Conservative table rendering preview",
        "",
        f"Version: `{VERSION}`",
        "",
        "> Read-only output. Raw HTML and chunk files are not modified.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record.get('doc_id')} — {record.get('table_id')}",
                "",
                f"Classification: `{record.get('classification')}`",
                "",
                f"Page: `{record.get('page')}` — Section: `{record.get('section_id')}`",
                "",
                "```text",
                str(record.get("rendered_text") or ""),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def process_catalog_file(path: Path, output_dir: Path, *, force: bool = False) -> dict[str, Any]:
    records = load_jsonl(path)
    processed = [parse_catalog_record(record) for record in records]
    logical = group_logical_tables(processed)
    # Flatten from logical tables so inherited group headers from continuation
    # fragments are present in the public recommendation artefacts.
    recommendations = [
        row
        for record in logical
        for row in record.get("recommendation_rows") or []
    ]
    recommendation_fragments = [
        row
        for record in logical
        for row in record.get("recommendation_fragments") or []
    ]
    active_recommendations = [
        row for row in recommendations if bool(row.get("active_for_retrieval"))
    ]
    active_recommendation_fragments = [
        row
        for row in recommendation_fragments
        if bool(row.get("active_for_retrieval"))
    ]
    text_quality_review = [
        row for row in recommendations if row.get("text_quality_flags")
    ]
    active_text_quality_review = [
        row
        for row in text_quality_review
        if bool(row.get("active_for_retrieval"))
    ]
    active_logical = [
        record for record in logical if bool(record.get("active_for_retrieval"))
    ]
    structured_guidance_fragments = [
        record
        for record in processed
        if record.get("classification") == "structured_guidance_table"
    ]
    active_structured_guidance_fragments = [
        record
        for record in structured_guidance_fragments
        if bool(record.get("active_for_retrieval"))
    ]
    structured_guidance_logical = [
        record
        for record in logical
        if record.get("classification") == "structured_guidance_table"
    ]
    active_structured_guidance_logical = [
        record
        for record in structured_guidance_logical
        if bool(record.get("active_for_retrieval"))
    ]

    doc_ids = ordered_unique(str(record.get("doc_id") or "") for record in processed)
    doc_id = doc_ids[0] if len(doc_ids) == 1 else path.name.removesuffix("_tables_raw.jsonl")
    clean_path = output_dir / f"{doc_id}_tables_clean.jsonl"
    logical_path = output_dir / f"{doc_id}_logical_tables.jsonl"
    active_logical_path = output_dir / f"{doc_id}_logical_tables_active.jsonl"
    recommendation_path = output_dir / f"{doc_id}_recommendations.jsonl"
    active_recommendation_path = output_dir / f"{doc_id}_recommendations_active.jsonl"
    recommendation_fragment_path = (
        output_dir / f"{doc_id}_recommendation_fragments.jsonl"
    )
    active_recommendation_fragment_path = (
        output_dir / f"{doc_id}_recommendation_fragments_active.jsonl"
    )
    text_quality_review_path = (
        output_dir / f"{doc_id}_recommendation_text_quality_review.jsonl"
    )
    active_text_quality_review_path = (
        output_dir / f"{doc_id}_recommendation_text_quality_review_active.jsonl"
    )
    summary_path = output_dir / f"{doc_id}_table_cleaning_summary.json"
    preview_path = output_dir / f"{doc_id}_rendering_preview.md"

    targets = (
        clean_path,
        logical_path,
        active_logical_path,
        recommendation_path,
        active_recommendation_path,
        recommendation_fragment_path,
        active_recommendation_fragment_path,
        text_quality_review_path,
        active_text_quality_review_path,
        summary_path,
        preview_path,
    )
    for target in targets:
        if target.exists() and not force:
            raise FileExistsError(f"Output exists: {target}. Use --force.")

    write_jsonl(clean_path, processed)
    write_jsonl(logical_path, logical)
    write_jsonl(active_logical_path, active_logical)
    write_jsonl(recommendation_path, recommendations)
    write_jsonl(active_recommendation_path, active_recommendations)
    write_jsonl(recommendation_fragment_path, recommendation_fragments)
    write_jsonl(
        active_recommendation_fragment_path,
        active_recommendation_fragments,
    )
    write_jsonl(text_quality_review_path, text_quality_review)
    write_jsonl(active_text_quality_review_path, active_text_quality_review)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(markdown_preview(processed), encoding="utf-8")

    classification_counts = Counter(record["classification"] for record in processed)
    active_classification_counts = Counter(
        record["classification"]
        for record in processed
        if bool(record.get("active_for_retrieval"))
    )
    catalog_status_counts = Counter(
        str(record.get("catalog_status") or "unknown") for record in processed
    )
    flag_counts = Counter(
        flag for record in processed for flag in record.get("quality_flags") or []
    )
    linked_recommendations = [
        row
        for row in recommendations
        if str(row.get("link_status") or "").startswith("matched")
    ]
    active_chunk_keys: set[tuple[str, str]] = set()
    for record in processed:
        if not bool(record.get("active_for_retrieval")):
            continue
        source_path = str(record.get("chunk_source_path") or "")
        source_order = record.get("chunk_source_order")
        if source_order is not None:
            active_chunk_keys.add((source_path, str(source_order)))
        else:
            active_chunk_keys.add(
                (
                    str(record.get("chunk_id") or ""),
                    str(record.get("chunk_table_index") or ""),
                )
            )
    summary = {
        "version": VERSION,
        "doc_id": doc_id,
        "input_path": str(path.resolve()),
        "table_count": len(processed),
        # Backward-compatible fragment count plus the retrieval-facing logical
        # table count.  Multipage source fragments can map to one chunk table.
        "active_table_count": sum(
            bool(record.get("active_for_retrieval")) for record in processed
        ),
        "active_table_fragment_count": sum(
            bool(record.get("active_for_retrieval")) for record in processed
        ),
        "logical_table_count": len(logical),
        "active_logical_table_count": len(active_logical),
        "active_retrieval_table_count": len(active_logical),
        "active_linked_chunk_table_count": len(active_chunk_keys),
        "recommendation_row_count": len(recommendations),
        "linked_recommendation_row_count": len(linked_recommendations),
        "active_recommendation_row_count": len(active_recommendations),
        "recommendation_fragment_count": len(recommendation_fragments),
        "active_recommendation_fragment_count": len(
            active_recommendation_fragments
        ),
        "structured_guidance_fragment_count": len(structured_guidance_fragments),
        "active_structured_guidance_fragment_count": len(
            active_structured_guidance_fragments
        ),
        "structured_guidance_logical_table_count": len(
            structured_guidance_logical
        ),
        "active_structured_guidance_logical_table_count": len(
            active_structured_guidance_logical
        ),
        "context_dependent_recommendation_count": sum(
            bool(row.get("context_dependent")) for row in recommendations
        ),
        "active_context_dependent_recommendation_count": sum(
            bool(row.get("context_dependent")) for row in active_recommendations
        ),
        "recommendations_with_citations_count": sum(
            bool(row.get("citation_numbers")) for row in recommendations
        ),
        "active_recommendations_with_citations_count": sum(
            bool(row.get("citation_numbers")) for row in active_recommendations
        ),
        "recommendations_with_footnotes_count": sum(
            bool(row.get("footnote_markers")) for row in recommendations
        ),
        "active_recommendations_with_footnotes_count": sum(
            bool(row.get("footnote_markers")) for row in active_recommendations
        ),
        "orphan_list_item_count": sum(
            row.get("context_dependency_kind") == "orphan_list_item"
            for row in recommendations
        ),
        "active_orphan_list_item_count": sum(
            row.get("context_dependency_kind") == "orphan_list_item"
            for row in active_recommendations
        ),
        "suspicious_internal_text_glue_count": sum(
            "possible_internal_text_glue" in (row.get("text_quality_flags") or [])
            for row in recommendations
        ),
        "active_suspicious_internal_text_glue_count": sum(
            "possible_internal_text_glue" in (row.get("text_quality_flags") or [])
            for row in active_recommendations
        ),
        "catalog_only_recommendation_row_count": len(recommendations)
        - len(linked_recommendations),
        "classification_counts": dict(sorted(classification_counts.items())),
        "active_classification_counts": dict(
            sorted(active_classification_counts.items())
        ),
        "catalog_status_counts": dict(sorted(catalog_status_counts.items())),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "raw_html_integrity_failures": sum(
            not bool(record.get("raw_html_integrity_ok")) for record in processed
        ),
        "raw_html_modified": False,
        "chunk_files_modified": False,
        "output_files": {
            "tables": str(clean_path.resolve()),
            "logical_tables": str(logical_path.resolve()),
            "active_logical_tables": str(active_logical_path.resolve()),
            "recommendations": str(recommendation_path.resolve()),
            "active_recommendations": str(active_recommendation_path.resolve()),
            "recommendation_fragments": str(
                recommendation_fragment_path.resolve()
            ),
            "active_recommendation_fragments": str(
                active_recommendation_fragment_path.resolve()
            ),
            "recommendation_text_quality_review": str(
                text_quality_review_path.resolve()
            ),
            "active_recommendation_text_quality_review": str(
                active_text_quality_review_path.resolve()
            ),
            "preview": str(preview_path.resolve()),
        },
    }
    write_json(summary_path, summary)
    return summary


def load_table_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("The config root must be a JSON object")

    tables: Optional[Mapping[str, Any]] = None
    protocols = payload.get("cardiology_protocols")
    if isinstance(protocols, Mapping):
        preprocessing = protocols.get("preprocessing")
        if isinstance(preprocessing, Mapping) and isinstance(preprocessing.get("tables"), Mapping):
            tables = preprocessing["tables"]
    if tables is None:
        preprocessing = payload.get("preprocessing")
        if isinstance(preprocessing, Mapping) and isinstance(preprocessing.get("tables"), Mapping):
            tables = preprocessing["tables"]
    if tables is None:
        raise KeyError("Missing cardiology_protocols.preprocessing.tables")

    def resolve(*keys: str) -> Optional[Path]:
        for key in keys:
            value = tables.get(key)
            if isinstance(value, str) and value.strip():
                path = Path(value)
                return path if path.is_absolute() else config_path.parent / path
        return None

    return {
        "enabled": bool(tables.get("enabled", True)),
        "input_dir": resolve("catalog_dir", "input_dir"),
        "output_dir": resolve("processed_dir", "cleaned_catalog_dir", "preview_dir", "output_dir"),
        "force": bool(tables.get("force", False)),
        "render_mode": str(tables.get("render_mode", "conservative")),
        "write_back_to_chunks": bool(tables.get("write_back_to_chunks", False)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render raw table-catalogue JSONL records conservatively."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    settings: dict[str, Any] = {}
    if args.config:
        settings = load_table_config(args.config)
        if not settings.get("enabled", True):
            LOG.info("Table preprocessing is disabled in the configuration.")
            return 0
        if settings.get("render_mode") != "conservative":
            raise SystemExit("Only render_mode=conservative is supported in this version.")
        if settings.get("write_back_to_chunks"):
            raise SystemExit("write_back_to_chunks must remain false in this version.")

    input_dir = args.input_dir or settings.get("input_dir")
    output_dir = args.output_dir or settings.get("output_dir")
    force = bool(args.force or settings.get("force", False))
    if input_dir is None or output_dir is None:
        raise SystemExit("Provide --input-dir/--output-dir or --config.")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    paths = sorted(input_dir.glob("*_tables_raw.jsonl"))
    if not paths:
        raise SystemExit(f"No *_tables_raw.jsonl files found in {input_dir}")

    summaries = []
    for path in paths:
        summary = process_catalog_file(path, output_dir, force=force)
        summaries.append(summary)
        LOG.info(
            "%s | source_fragments=%d | active_fragments=%d | "
            "active_chunk_tables=%d | logical_all=%d | logical_active=%d | "
            "recommendations=%d | active_recommendations=%d | "
            "active_ungraded_fragments=%d | citations=%d | footnotes=%d | "
            "orphans=%d | text_review=%d | integrity_failures=%d",
            path.name,
            summary["table_count"],
            summary["active_table_fragment_count"],
            summary["active_linked_chunk_table_count"],
            summary["logical_table_count"],
            summary["active_logical_table_count"],
            summary["recommendation_row_count"],
            summary["active_recommendation_row_count"],
            summary["active_recommendation_fragment_count"],
            summary["active_recommendations_with_citations_count"],
            summary["active_recommendations_with_footnotes_count"],
            summary["active_orphan_list_item_count"],
            summary["active_suspicious_internal_text_glue_count"],
            summary["raw_html_integrity_failures"],
        )

    aggregate = {
        "version": VERSION,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "document_count": len(summaries),
        "table_count": sum(item["table_count"] for item in summaries),
        "recommendation_row_count": sum(
            item["recommendation_row_count"] for item in summaries
        ),
        "linked_recommendation_row_count": sum(
            item["linked_recommendation_row_count"] for item in summaries
        ),
        "active_recommendation_row_count": sum(
            item["active_recommendation_row_count"] for item in summaries
        ),
        "recommendation_fragment_count": sum(
            item["recommendation_fragment_count"] for item in summaries
        ),
        "active_recommendation_fragment_count": sum(
            item["active_recommendation_fragment_count"] for item in summaries
        ),
        "structured_guidance_fragment_count": sum(
            item["structured_guidance_fragment_count"] for item in summaries
        ),
        "active_structured_guidance_fragment_count": sum(
            item["active_structured_guidance_fragment_count"] for item in summaries
        ),
        "structured_guidance_logical_table_count": sum(
            item["structured_guidance_logical_table_count"] for item in summaries
        ),
        "active_structured_guidance_logical_table_count": sum(
            item["active_structured_guidance_logical_table_count"] for item in summaries
        ),
        "context_dependent_recommendation_count": sum(
            item["context_dependent_recommendation_count"] for item in summaries
        ),
        "active_context_dependent_recommendation_count": sum(
            item["active_context_dependent_recommendation_count"]
            for item in summaries
        ),
        "recommendations_with_citations_count": sum(
            item["recommendations_with_citations_count"] for item in summaries
        ),
        "active_recommendations_with_citations_count": sum(
            item["active_recommendations_with_citations_count"]
            for item in summaries
        ),
        "recommendations_with_footnotes_count": sum(
            item["recommendations_with_footnotes_count"] for item in summaries
        ),
        "active_recommendations_with_footnotes_count": sum(
            item["active_recommendations_with_footnotes_count"] for item in summaries
        ),
        "orphan_list_item_count": sum(
            item["orphan_list_item_count"] for item in summaries
        ),
        "active_orphan_list_item_count": sum(
            item["active_orphan_list_item_count"] for item in summaries
        ),
        "suspicious_internal_text_glue_count": sum(
            item["suspicious_internal_text_glue_count"] for item in summaries
        ),
        "active_suspicious_internal_text_glue_count": sum(
            item["active_suspicious_internal_text_glue_count"] for item in summaries
        ),
        "catalog_only_recommendation_row_count": sum(
            item["catalog_only_recommendation_row_count"] for item in summaries
        ),
        "active_table_count": sum(
            item["active_table_count"] for item in summaries
        ),
        "active_table_fragment_count": sum(
            item["active_table_fragment_count"] for item in summaries
        ),
        "active_retrieval_table_count": sum(
            item["active_retrieval_table_count"] for item in summaries
        ),
        "active_linked_chunk_table_count": sum(
            item["active_linked_chunk_table_count"] for item in summaries
        ),
        "raw_html_integrity_failures": sum(
            item["raw_html_integrity_failures"] for item in summaries
        ),
        "documents": summaries,
    }
    write_json(output_dir / "summary.json", aggregate)
    return 0 if aggregate["raw_html_integrity_failures"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
