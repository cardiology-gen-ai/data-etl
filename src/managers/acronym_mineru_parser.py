"""MinerU parser for document-level acronym caches.

The parser reads front-matter acronym/glossary sections from MinerU artefacts.
It supports the layouts observed across ESC guidelines without relying on
per-document or per-acronym allowlists:

* text lines with separate short/definition spans and indented continuations;
* one inline ``SHORT definition`` pair per line;
* grouped text blocks containing ``N`` short forms followed by ``N`` definitions;
* HTML tables with one or more short/definition streams, rowspans, colspans,
  definition-only continuation rows, shared definitions, and numeric suffix rows.

The module is read-only.  It returns source-backed candidates plus explicit
blocking/review diagnostics.  Consumers must use MinerU output only when
``result.usable`` is true; otherwise the existing PDF extractor should be used
as a fallback.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from managers.tables.table_cleaning_manager import (
    ConservativeTableHTMLParser,
    expanded_grid,
)

PARSER_VERSION = "mineru_acronym_parser_v2_0"

SPACE_RE = re.compile(r"\s+")
TIMESTAMP_SUFFIX_RE = re.compile(r"__(?P<stamp>\d{8,14})$")
KNOWN_SUFFIXES = (
    "_content_list_v2",
    "_content_list",
    "_middle",
    "_model",
    "_tables",
)

ACRONYM_HEADING_RE = re.compile(
    r"(?i)^\s*(?:(?:list\s+of\s+)?abbreviations?"
    r"(?:\s+and\s+acronyms?)?|acronyms?)\s*$"
)
BODY_HEADING_RE = re.compile(
    r"(?i)^\s*\d+(?:\.\d+)*\.?\s+"
    r"(?:preamble|introduction|background|methods?|recommendations?)\b"
)
GENERIC_NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z]")
TEX_SUP_RE = re.compile(r"\^\{([^{}]+)\}")
TEX_SUB_RE = re.compile(r"_\{([^{}]+)\}")
HTML_SCRIPT_TAG_RE = re.compile(r"</?(?:sup|sub)\b[^>]*>", re.I)
ALPHA_RE = re.compile(r"[A-Za-zΑ-Ωα-ω]")


@dataclass(frozen=True)
class MinerUAcronymCandidate:
    short: str
    definition: str
    page_idx: Optional[int]
    source_kind: str
    block_index: Optional[int] = None
    stream_index: Optional[int] = None
    source_order: int = 0
    parser_strategy: str = "unknown"
    row_index: Optional[int] = None


@dataclass
class MinerUAcronymResult:
    doc_id: str
    source_path: Path
    candidates: list[MinerUAcronymCandidate] = field(default_factory=list)
    page_start_idx: Optional[int] = None
    page_end_idx: Optional[int] = None
    heading_found: bool = False
    source_format: str = "unknown"
    candidate_file_count: int = 1
    warnings: list[str] = field(default_factory=list)
    blocking_issues: list[dict[str, Any]] = field(default_factory=list)
    review_issues: list[dict[str, Any]] = field(default_factory=list)
    layout_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text_pair_count(self) -> int:
        return sum(item.source_kind == "mineru_text" for item in self.candidates)

    @property
    def table_pair_count(self) -> int:
        return sum(item.source_kind == "mineru_table" for item in self.candidates)

    @property
    def strategy_counts(self) -> dict[str, int]:
        return dict(Counter(item.parser_strategy for item in self.candidates))

    @property
    def usable(self) -> bool:
        return (
            self.heading_found
            and bool(self.candidates)
            and not self.blocking_issues
        )


def normalize_space(value: Any) -> str:
    """Normalize MinerU text without document-specific substitutions."""
    text = html.unescape(str(value or ""))
    text = HTML_SCRIPT_TAG_RE.sub("", text)
    text = re.sub(r"\\uparrow\s*", "↑", text)
    text = re.sub(r"\\downarrow\s*", "↓", text)
    text = re.sub(r"\\prime\s*", "′", text)
    # MinerU uses TeX-like superscript/subscript fragments in otherwise plain
    # text.  Flattening them is appropriate for acronym matching, e.g.
    # ``^{99m}Tc`` -> ``99mTc`` and ``P2Y_{12}`` -> ``P2Y12``.
    previous = None
    while previous != text:
        previous = text
        text = TEX_SUP_RE.sub(r"\1", text)
        text = TEX_SUB_RE.sub(r"\1", text)
    return SPACE_RE.sub(" ", text).strip()


def infer_doc_id(path: Path) -> str:
    stem = path.stem
    if stem.startswith("MinerU_"):
        stem = stem[len("MinerU_") :]
    stem = TIMESTAMP_SUFFIX_RE.sub("", stem)
    for suffix in KNOWN_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.strip("_-")


def _timestamp_key(path: Path) -> tuple[str, int, str]:
    match = TIMESTAMP_SUFFIX_RE.search(path.stem)
    stamp = match.group("stamp") if match else ""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return stamp, mtime, path.name


def find_mineru_artifact(
    content_list_dir: Path,
    doc_id: str,
    *,
    explicit_file: Optional[Path] = None,
) -> tuple[Optional[Path], int]:
    """Return the newest exact document match and the candidate count."""
    if explicit_file is not None:
        path = Path(explicit_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        inferred = infer_doc_id(path)
        if inferred != doc_id:
            raise ValueError(
                f"MinerU document mismatch: expected {doc_id!r}, got {inferred!r}"
            )
        return path, 1

    root = Path(content_list_dir).expanduser().resolve()
    if not root.exists():
        return None, 0

    candidates: list[Path] = []
    for pattern in ("*.json", "*.jsonl", "*.json_l"):
        for path in root.glob(pattern):
            if path.is_file() and infer_doc_id(path) == doc_id:
                candidates.append(path)

    unique = sorted(set(candidates), key=_timestamp_key, reverse=True)
    return (unique[0], len(unique)) if unique else (None, 0)


def load_mineru_payload(path: Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
        return records


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_bbox(value: Any) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _SpanRecord:
    text: str
    bbox: Optional[tuple[float, float, float, float]]


@dataclass(frozen=True)
class _LineRecord:
    spans: tuple[_SpanRecord, ...]
    bbox: Optional[tuple[float, float, float, float]]

    @property
    def text(self) -> str:
        # Adjacent spans on the same visual line are parts of one string; do not
        # inject a newline between e.g. ``P2Y`` and ``_{12}``.
        return normalize_space("".join(span.text for span in self.spans))


def _line_records(block: Mapping[str, Any]) -> list[_LineRecord]:
    records: list[_LineRecord] = []
    for line in block.get("lines") or []:
        if not isinstance(line, Mapping):
            continue
        spans: list[_SpanRecord] = []
        for span in line.get("spans") or []:
            if not isinstance(span, Mapping):
                continue
            content = span.get("content")
            if not isinstance(content, str) or not content:
                continue
            spans.append(
                _SpanRecord(
                    text=content,
                    bbox=_normalize_bbox(span.get("bbox")),
                )
            )
        if spans:
            records.append(
                _LineRecord(
                    spans=tuple(spans),
                    bbox=_normalize_bbox(line.get("bbox")),
                )
            )
    return records


def _block_text_lines(block: Mapping[str, Any]) -> list[str]:
    """Return logical text lines while preserving intra-line span adjacency."""
    values: list[str] = []
    records = _line_records(block)
    if records:
        for record in records:
            joined = "".join(span.text for span in record.spans)
            values.extend(normalize_space(part) for part in joined.splitlines())
        return [value for value in values if value]

    for key in ("text", "content"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return [normalize_space(part) for part in value.splitlines() if normalize_space(part)]
    return []


def _block_text(block: Mapping[str, Any]) -> str:
    return "\n".join(_block_text_lines(block))


def _iter_spans(block: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for line in block.get("lines") or []:
        if not isinstance(line, Mapping):
            continue
        for span in line.get("spans") or []:
            if isinstance(span, Mapping):
                yield span


def _table_html_values(block: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    direct = block.get("table_body") or block.get("html")
    if isinstance(direct, str) and direct.strip():
        values.append(direct.strip())

    for child in block.get("blocks") or []:
        if not isinstance(child, Mapping):
            continue
        for span in _iter_spans(child):
            value = span.get("html")
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    return values


def _page_blocks(payload: Any) -> tuple[list[tuple[int, list[Mapping[str, Any]]]], str]:
    if isinstance(payload, Mapping) and isinstance(payload.get("pdf_info"), list):
        output = []
        for position, page in enumerate(payload["pdf_info"]):
            if not isinstance(page, Mapping):
                continue
            page_idx = _optional_int(page.get("page_idx"))
            if page_idx is None:
                page_idx = position
            blocks = [
                block
                for block in (page.get("preproc_blocks") or [])
                if isinstance(block, Mapping)
            ]
            output.append((page_idx, blocks))
        return output, "mineru_pdf_info"

    if isinstance(payload, list):
        grouped: dict[int, list[Mapping[str, Any]]] = {}
        order: list[int] = []
        for block in payload:
            if not isinstance(block, Mapping):
                continue
            page_idx = _optional_int(block.get("page_idx"))
            if page_idx is None:
                page_idx = 0
            if page_idx not in grouped:
                grouped[page_idx] = []
                order.append(page_idx)
            grouped[page_idx].append(block)
        return [(idx, grouped[idx]) for idx in order], "mineru_content_list"

    raise ValueError(
        "Unsupported MinerU shape: expected a pdf_info object or a flat list"
    )


def _is_acronym_heading(value: Any) -> bool:
    return bool(ACRONYM_HEADING_RE.fullmatch(normalize_space(value)))


def _is_body_heading(value: Any) -> bool:
    text = normalize_space(value)
    return bool(BODY_HEADING_RE.match(text) or GENERIC_NUMBERED_HEADING_RE.match(text))


def _looks_like_structural_short(
    value: Any,
    *,
    short_detector: Callable[[str], bool],
) -> bool:
    """Validate a term isolated by glossary structure."""
    text = normalize_space(value).strip(" :;,")
    if not text or len(text) > 64 or len(text.split()) > 5:
        return False
    if _is_acronym_heading(text) or _is_body_heading(text):
        return False
    if re.match(r"(?i)^(?:figure|table|recommendation)\b", text):
        return False
    if re.search(r"[.!?]$", text) and not re.fullmatch(r"(?:[A-Za-z]\.){2,}", text):
        return False
    if not re.search(r"[A-Za-zΑ-Ωα-ω0-9↑↓′']", text):
        return False
    if short_detector(text):
        return True
    # Structure provides enough evidence for one-character glossary labels and
    # short lowercase forms such as ``tx``.
    if re.fullmatch(r"[A-ZΑ-Ω0-9]", text):
        return True
    if (
        re.fullmatch(r"[a-zα-ω]{2,4}", text)
        and text.casefold()
        not in {"and", "or", "of", "to", "in", "on", "by", "as", "at", "vs"}
    ):
        return True
    if re.fullmatch(r"[A-Za-zΑ-Ωα-ω][′']", text):
        return True
    # A dedicated glossary cell is strong evidence for compact title-case and
    # multi-token labels that the body-text detector intentionally rejects,
    # such as ``Echo`` and ``ROCKET AF``.
    if re.fullmatch(r"[A-Z][a-z]{1,7}", text):
        return True
    parts = text.split()
    if 2 <= len(parts) <= 4 and all(
        re.fullmatch(r"[A-Za-z0-9.+/\-]{1,24}", part)
        and (any(char.isupper() for char in part) or any(char.isdigit() for char in part))
        for part in parts
    ):
        return True
    return False


def _definition_is_structural(value: Any) -> bool:
    text = normalize_space(value)
    if not text or len(text) > 2000:
        return False
    if _is_acronym_heading(text) or _is_body_heading(text):
        return False
    return True


def _definition_looks_like_short(
    value: str,
    *,
    short_detector: Callable[[str], bool],
) -> bool:
    text = normalize_space(value).strip(" :;,")
    if not text or len(text.split()) > 4 or len(text) > 48:
        return False
    if re.fullmatch(r"(?i)(?:yes|no|months?|treatment|intravenous|intracoronary)", text):
        return False
    return short_detector(text)


def _same_origin(left: Any, right: Any) -> bool:
    return (
        left is not None
        and right is not None
        and left.origin_row == right.origin_row
        and left.origin_col == right.origin_col
    )


def _slot_text(slot: Any) -> str:
    return normalize_space(slot.cell.text) if slot is not None else ""


def _pairing_score(
    grid: Sequence[Sequence[Any]],
    pairs: Sequence[tuple[int, int]],
    *,
    short_detector: Callable[[str], bool],
) -> tuple[float, int]:
    score = 0.0
    usable = 0
    for row in grid:
        for short_col, def_col in pairs:
            left = row[short_col] if short_col < len(row) else None
            right = row[def_col] if def_col < len(row) else None
            if left is None or right is None or _same_origin(left, right):
                continue
            short = _slot_text(left)
            definition = _slot_text(right)
            if not short or not definition:
                continue
            if _is_acronym_heading(short) or _is_acronym_heading(definition):
                continue
            if _is_body_heading(short) or _is_body_heading(definition):
                continue
            usable += 1
            if _looks_like_structural_short(short, short_detector=short_detector):
                score += 3.0
            else:
                score -= 3.0
            if _definition_looks_like_short(definition, short_detector=short_detector):
                score -= 4.0
            elif _definition_is_structural(definition):
                score += 2.0
    return score, usable


def _choose_table_layout(
    grid: Sequence[Sequence[Any]],
    *,
    short_detector: Callable[[str], bool],
) -> tuple[list[tuple[int, int]], dict[str, Any], Optional[dict[str, Any]]]:
    width = max((len(row) for row in grid), default=0)
    if width < 2:
        return [], {"width": width, "layout": "none"}, None
    if width == 2:
        return [(0, 1)], {"width": width, "layout": "two_column"}, None

    interleaved = [(index, index + 1) for index in range(0, width - 1, 2)]
    candidates: list[tuple[str, list[tuple[int, int]]]] = [("interleaved", interleaved)]
    if width == 4:
        candidates.append(("grouped", [(0, 2), (1, 3)]))

    scored = []
    for name, pairs in candidates:
        score, usable = _pairing_score(
            grid,
            pairs,
            short_detector=short_detector,
        )
        scored.append({"name": name, "pairs": pairs, "score": score, "usable": usable})
    scored.sort(key=lambda item: (item["score"], item["usable"]), reverse=True)
    best = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None
    diagnostic = {
        "width": width,
        "layout": best["name"],
        "score": best["score"],
        "usable_pair_count": best["usable"],
        "alternatives": [
            {"layout": item["name"], "score": item["score"], "usable": item["usable"]}
            for item in scored
        ],
    }
    issue = None
    if best["usable"] == 0 or best["score"] <= 0:
        issue = {
            "code": "no_plausible_table_pairing_layout",
            "severity": "blocking",
            "details": diagnostic,
        }
    elif runner_up is not None and abs(best["score"] - runner_up["score"]) < 3.0:
        issue = {
            "code": "ambiguous_table_pairing_layout",
            "severity": "blocking",
            "details": diagnostic,
        }
    return list(best["pairs"]), diagnostic, issue


@dataclass
class _MutableCandidate:
    short: str
    definition: str
    page_idx: int
    block_index: Optional[int]
    stream_index: Optional[int]
    source_order: int
    parser_strategy: str
    row_index: Optional[int]
    short_origin: Optional[tuple[int, int]] = None
    definition_origin: Optional[tuple[int, int]] = None

    def frozen(self) -> MinerUAcronymCandidate:
        return MinerUAcronymCandidate(
            short=normalize_space(self.short),
            definition=normalize_space(self.definition),
            page_idx=self.page_idx,
            source_kind="mineru_table",
            block_index=self.block_index,
            stream_index=self.stream_index,
            source_order=self.source_order,
            parser_strategy=self.parser_strategy,
            row_index=self.row_index,
        )


def _stream_candidates_from_table(
    grid: Sequence[Sequence[Any]],
    *,
    pair: tuple[int, int],
    page_idx: int,
    block_index: Optional[int],
    stream_index: int,
    active_before: bool,
    table_has_heading: bool,
    short_detector: Callable[[str], bool],
    order_start: int,
    review_issues: list[dict[str, Any]],
) -> tuple[list[MinerUAcronymCandidate], bool, bool, int]:
    short_col, def_col = pair
    stream_active = active_before or table_has_heading
    heading_seen = table_has_heading
    body_seen = False
    output: list[_MutableCandidate] = []
    order = order_start

    for row_index, row in enumerate(grid):
        left = row[short_col] if short_col < len(row) else None
        right = row[def_col] if def_col < len(row) else None
        short = _slot_text(left)
        definition = _slot_text(right)
        visible = [item for item in (short, definition) if item]

        if any(_is_acronym_heading(item) for item in visible):
            stream_active = True
            heading_seen = True
            continue
        if any(_is_body_heading(item) for item in visible):
            stream_active = False
            body_seen = True
            continue
        if not stream_active:
            continue
        if left is not None and right is not None and _same_origin(left, right):
            continue

        short_origin = (
            (left.origin_row, left.origin_col) if left is not None else None
        )
        definition_origin = (
            (right.origin_row, right.origin_col) if right is not None else None
        )

        # A definition-only row continues the previous glossary entry.
        if not short and definition:
            if output:
                output[-1].definition = normalize_space(
                    f"{output[-1].definition} {definition}"
                )
                review_issues.append(
                    {
                        "code": "table_definition_continuation_merged",
                        "severity": "info",
                        "page_idx": page_idx,
                        "block_index": block_index,
                        "stream_index": stream_index,
                        "row_index": row_index,
                    }
                )
            continue

        if not short or not definition:
            continue

        # A shared rowspan definition attached to two consecutive short cells
        # represents a multi-token short label, e.g. GLOBAL LEADERS.
        if (
            output
            and definition_origin is not None
            and output[-1].definition_origin == definition_origin
            and short_origin != output[-1].short_origin
            and _looks_like_structural_short(short, short_detector=short_detector)
        ):
            output[-1].short = normalize_space(f"{output[-1].short} {short}")
            review_issues.append(
                {
                    "code": "table_shared_definition_short_merged",
                    "severity": "info",
                    "page_idx": page_idx,
                    "block_index": block_index,
                    "stream_index": stream_index,
                    "row_index": row_index,
                    "merged_short": output[-1].short,
                }
            )
            continue

        # A numeric-only row immediately after an acronym entry is normally a
        # suffix split by MinerU (e.g. PEGASUS-TIMI / 54) and its right cell is
        # the continuation of the same definition.
        if re.fullmatch(r"\d+", short) and output:
            output[-1].short = normalize_space(f"{output[-1].short} {short}")
            output[-1].definition = normalize_space(
                f"{output[-1].definition} {definition}"
            )
            review_issues.append(
                {
                    "code": "table_numeric_suffix_merged",
                    "severity": "info",
                    "page_idx": page_idx,
                    "block_index": block_index,
                    "stream_index": stream_index,
                    "row_index": row_index,
                    "merged_short": output[-1].short,
                }
            )
            continue

        if not _looks_like_structural_short(short, short_detector=short_detector):
            continue
        if not _definition_is_structural(definition):
            continue

        output.append(
            _MutableCandidate(
                short=short,
                definition=definition,
                page_idx=page_idx,
                block_index=block_index,
                stream_index=stream_index,
                source_order=order,
                parser_strategy="table_structured",
                row_index=row_index,
                short_origin=short_origin,
                definition_origin=definition_origin,
            )
        )
        order += 1

    return [item.frozen() for item in output], heading_seen, body_seen, order


def _pairs_from_table(
    raw_html: str,
    *,
    page_idx: int,
    block_index: Optional[int],
    active_before: bool,
    short_detector: Callable[[str], bool],
    order_start: int,
    blocking_issues: list[dict[str, Any]],
    review_issues: list[dict[str, Any]],
    layout_diagnostics: list[dict[str, Any]],
) -> tuple[list[MinerUAcronymCandidate], bool, bool, int]:
    parser = ConservativeTableHTMLParser()
    parser.feed(raw_html)
    parser.close()
    grid = expanded_grid(parser.rows)
    pairs, diagnostic, layout_issue = _choose_table_layout(
        grid,
        short_detector=short_detector,
    )
    diagnostic.update({"page_idx": page_idx, "block_index": block_index})
    layout_diagnostics.append(diagnostic)
    if layout_issue:
        layout_issue.update({"page_idx": page_idx, "block_index": block_index})
        blocking_issues.append(layout_issue)
        return [], False, False, order_start

    table_has_heading = any(
        slot is not None and _is_acronym_heading(slot.cell.text)
        for row in grid
        for slot in row
    )
    candidates: list[MinerUAcronymCandidate] = []
    heading_seen = table_has_heading
    body_seen = False
    order = order_start
    for stream_index, pair in enumerate(pairs):
        stream_candidates, stream_heading, stream_body, order = (
            _stream_candidates_from_table(
                grid,
                pair=pair,
                page_idx=page_idx,
                block_index=block_index,
                stream_index=stream_index,
                active_before=active_before,
                table_has_heading=table_has_heading,
                short_detector=short_detector,
                order_start=order,
                review_issues=review_issues,
            )
        )
        candidates.extend(stream_candidates)
        heading_seen = heading_seen or stream_heading
        body_seen = body_seen or stream_body
    return candidates, heading_seen, body_seen, order



def _looks_like_inline_short(
    value: str,
    *,
    short_detector: Callable[[str], bool],
) -> bool:
    """Stricter short detector for unstructured ``SHORT definition`` lines."""
    text = normalize_space(value).strip(" :;,")
    if short_detector(text):
        return True
    if re.fullmatch(r"[A-ZΑ-Ω0-9]", text):
        return True
    if (
        re.fullmatch(r"[a-zα-ω]{2,4}", text)
        and text.casefold()
        not in {"and", "or", "of", "to", "in", "on", "by", "as", "at", "vs"}
    ):
        return True
    return False

def _split_inline_pair(
    line: str,
    *,
    short_detector: Callable[[str], bool],
) -> Optional[tuple[str, str]]:
    text = normalize_space(line)
    if not text or _is_acronym_heading(text) or _is_body_heading(text):
        return None
    tokens = text.split()
    if len(tokens) < 2:
        return None
    max_prefix = min(5, len(tokens) - 1)
    for size in range(max_prefix, 0, -1):
        short = " ".join(tokens[:size])
        definition = " ".join(tokens[size:])
        if not _looks_like_inline_short(short, short_detector=short_detector):
            continue
        if not _definition_is_structural(definition):
            continue
        if _definition_looks_like_short(definition, short_detector=short_detector):
            continue
        return short, definition
    return None


def _parse_geometric_text_block(
    block: Mapping[str, Any],
    *,
    page_idx: int,
    block_index: Optional[int],
    short_detector: Callable[[str], bool],
    order_start: int,
) -> tuple[list[MinerUAcronymCandidate], int]:
    records = _line_records(block)
    two_span_lines = 0
    for record in records:
        nonempty = [span for span in record.spans if normalize_space(span.text)]
        if len(nonempty) >= 2 and all(span.bbox is not None for span in nonempty[:2]):
            if nonempty[1].bbox[0] - nonempty[0].bbox[0] >= 20:  # type: ignore[index]
                two_span_lines += 1
    if two_span_lines < 3:
        return [], order_start

    output: list[MinerUAcronymCandidate] = []
    current_index: Optional[int] = None
    definition_x0: Optional[float] = None
    order = order_start

    for row_index, record in enumerate(records):
        spans = [
            _SpanRecord(normalize_space(span.text), span.bbox)
            for span in record.spans
            if normalize_space(span.text)
        ]
        if not spans:
            continue
        line_text = normalize_space("".join(span.text for span in spans))
        if _is_body_heading(line_text):
            break

        first = spans[0]
        first_x0 = first.bbox[0] if first.bbox else None
        if (
            len(spans) >= 2
            and _looks_like_structural_short(first.text, short_detector=short_detector)
        ):
            definition = normalize_space(" ".join(span.text for span in spans[1:]))
            if _definition_is_structural(definition):
                candidate = MinerUAcronymCandidate(
                    short=first.text,
                    definition=definition,
                    page_idx=page_idx,
                    source_kind="mineru_text",
                    block_index=block_index,
                    stream_index=0,
                    source_order=order,
                    parser_strategy="text_geometric_spans",
                    row_index=row_index,
                )
                output.append(candidate)
                current_index = len(output) - 1
                definition_x0 = spans[1].bbox[0] if spans[1].bbox else None
                order += 1
                continue

        if current_index is not None and len(spans) == 1:
            continuation = spans[0].text
            x0 = first_x0
            if (
                definition_x0 is None
                or x0 is None
                or x0 >= definition_x0 - 8
            ):
                output[current_index] = replace(
                    output[current_index],
                    definition=normalize_space(
                        f"{output[current_index].definition} {continuation}"
                    ),
                )
                continue

    return output, order


def _grouped_text_pairs(
    lines: Sequence[str],
    *,
    page_idx: int,
    block_index: Optional[int],
    short_detector: Callable[[str], bool],
    order_start: int,
) -> tuple[list[MinerUAcronymCandidate], int, Optional[dict[str, Any]]]:
    clean = [normalize_space(line) for line in lines if normalize_space(line)]
    if len(clean) < 6 or len(clean) % 2:
        return [], order_start, None
    half = len(clean) // 2
    left = clean[:half]
    right = clean[half:]
    left_flags = [
        _looks_like_structural_short(item, short_detector=short_detector)
        for item in left
    ]
    right_short_flags = [
        _looks_like_structural_short(item, short_detector=short_detector)
        for item in right
    ]
    if sum(left_flags) / len(left_flags) < 0.85:
        return [], order_start, None
    if sum(right_short_flags) / len(right_short_flags) > 0.35:
        return [], order_start, None
    if not all(_definition_is_structural(item) for item in right):
        return [], order_start, None

    output = []
    order = order_start
    for row_index, (short, definition) in enumerate(zip(left, right)):
        output.append(
            MinerUAcronymCandidate(
                short=short,
                definition=definition,
                page_idx=page_idx,
                source_kind="mineru_text",
                block_index=block_index,
                stream_index=0,
                source_order=order,
                parser_strategy="text_grouped_halves",
                row_index=row_index,
            )
        )
        order += 1
    diagnostic = {
        "code": "grouped_text_halves_detected",
        "severity": "info",
        "page_idx": page_idx,
        "block_index": block_index,
        "pair_count": len(output),
    }
    return output, order, diagnostic


def _parse_linear_text_lines(
    lines: Sequence[str],
    *,
    page_idx: int,
    block_index: Optional[int],
    short_detector: Callable[[str], bool],
    order_start: int,
) -> tuple[list[MinerUAcronymCandidate], int]:
    clean = [normalize_space(line) for line in lines if normalize_space(line)]
    output: list[MinerUAcronymCandidate] = []
    current_index: Optional[int] = None
    order = order_start

    for row_index, line in enumerate(clean):
        if _is_acronym_heading(line):
            continue
        if _is_body_heading(line):
            break

        inline = _split_inline_pair(line, short_detector=short_detector)
        if inline is not None:
            short, definition = inline
            output.append(
                MinerUAcronymCandidate(
                    short=short,
                    definition=definition,
                    page_idx=page_idx,
                    source_kind="mineru_text",
                    block_index=block_index,
                    stream_index=0,
                    source_order=order,
                    parser_strategy="text_inline",
                    row_index=row_index,
                )
            )
            current_index = len(output) - 1
            order += 1
            continue

        if _looks_like_structural_short(line, short_detector=short_detector):
            # Start an isolated short; its definition arrives on following lines.
            output.append(
                MinerUAcronymCandidate(
                    short=line,
                    definition="",
                    page_idx=page_idx,
                    source_kind="mineru_text",
                    block_index=block_index,
                    stream_index=0,
                    source_order=order,
                    parser_strategy="text_isolated",
                    row_index=row_index,
                )
            )
            current_index = len(output) - 1
            order += 1
            continue

        if current_index is not None:
            output[current_index] = replace(
                output[current_index],
                definition=normalize_space(
                    f"{output[current_index].definition} {line}"
                ),
            )

    return [item for item in output if item.definition], order


def _parse_text_block(
    block: Mapping[str, Any],
    *,
    page_idx: int,
    block_index: Optional[int],
    short_detector: Callable[[str], bool],
    order_start: int,
    review_issues: list[dict[str, Any]],
) -> tuple[list[MinerUAcronymCandidate], int]:
    geometric, order = _parse_geometric_text_block(
        block,
        page_idx=page_idx,
        block_index=block_index,
        short_detector=short_detector,
        order_start=order_start,
    )
    if geometric:
        return geometric, order

    lines = _block_text_lines(block)
    grouped, order, diagnostic = _grouped_text_pairs(
        lines,
        page_idx=page_idx,
        block_index=block_index,
        short_detector=short_detector,
        order_start=order_start,
    )
    if grouped:
        if diagnostic:
            review_issues.append(diagnostic)
        return grouped, order

    return _parse_linear_text_lines(
        lines,
        page_idx=page_idx,
        block_index=block_index,
        short_detector=short_detector,
        order_start=order_start,
    )


def _short_letters(value: str) -> str:
    return "".join(char.upper() for char in normalize_space(value) if char.isalpha())


def _definition_initials(value: str) -> str:
    tokens = re.findall(r"[A-Za-zΑ-Ωα-ω]+", normalize_space(value))
    return "".join(token[0].upper() for token in tokens if token)


def _is_subsequence(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    iterator = iter(haystack)
    return all(any(char == candidate for candidate in iterator) for char in needle)


def _deduplicate_candidates(
    candidates: Sequence[MinerUAcronymCandidate],
) -> list[MinerUAcronymCandidate]:
    output: list[MinerUAcronymCandidate] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        normalized_item = replace(
            item,
            short=normalize_space(item.short).strip(" :;,"),
            definition=normalize_space(item.definition),
        )
        key = (normalized_item.short, normalized_item.definition)
        if not all(key) or key in seen:
            continue
        seen.add(key)
        output.append(normalized_item)
    return output


def _validate_candidates(
    candidates: Sequence[MinerUAcronymCandidate],
    *,
    short_detector: Callable[[str], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocking: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    short_index = {normalize_space(item.short).casefold(): item for item in candidates}

    for item in candidates:
        short = normalize_space(item.short)
        definition = normalize_space(item.definition)
        if re.fullmatch(r"\d+", short):
            blocking.append(
                {
                    "code": "numeric_only_short",
                    "severity": "blocking",
                    "short": short,
                    "definition": definition,
                    "page_idx": item.page_idx,
                    "block_index": item.block_index,
                }
            )
        definition_key = definition.casefold()
        if (
            definition_key in short_index
            and definition_key != short.casefold()
            and _definition_looks_like_short(definition, short_detector=short_detector)
        ):
            blocking.append(
                {
                    "code": "definition_is_another_short_form",
                    "severity": "blocking",
                    "short": short,
                    "definition": definition,
                    "other_short": short_index[definition_key].short,
                    "page_idx": item.page_idx,
                    "block_index": item.block_index,
                }
            )

        embedded_short_count = sum(
            1
            for other_key in short_index
            if other_key != short.casefold()
            and len(other_key) >= 3
            and re.search(rf"(?<!\w){re.escape(other_key)}(?!\w)", definition_key)
        )
        if len(definition) > 180 and embedded_short_count >= 3:
            blocking.append(
                {
                    "code": "possible_multi_entry_definition",
                    "severity": "blocking",
                    "short": short,
                    "definition": definition,
                    "embedded_short_count": embedded_short_count,
                    "page_idx": item.page_idx,
                    "block_index": item.block_index,
                }
            )

    # Detect a shifted chain within one source stream: definition initials of
    # one row equal the next short, repeatedly. This catches sequences such as
    # THEMIS -> Transient ischaemic dilatation; TID -> TIMI expansion.
    groups: dict[tuple[Any, ...], list[MinerUAcronymCandidate]] = defaultdict(list)
    for item in candidates:
        groups[
            (
                item.source_kind,
                item.page_idx,
                item.block_index,
                item.stream_index,
            )
        ].append(item)
    for key, group in groups.items():
        ordered = sorted(group, key=lambda item: item.source_order)
        links: list[dict[str, Any]] = []
        for current, following in zip(ordered, ordered[1:]):
            initials = _definition_initials(current.definition)
            next_letters = _short_letters(following.short)
            if len(next_letters) >= 2 and (
                initials == next_letters
                or (
                    len(initials) >= len(next_letters)
                    and initials.startswith(next_letters)
                )
            ):
                links.append(
                    {
                        "from_short": current.short,
                        "definition": current.definition,
                        "to_short": following.short,
                    }
                )
            else:
                if len(links) >= 2:
                    blocking.append(
                        {
                            "code": "shifted_definition_chain",
                            "severity": "blocking",
                            "stream": key,
                            "links": links,
                        }
                    )
                links = []
        if len(links) >= 2:
            blocking.append(
                {
                    "code": "shifted_definition_chain",
                    "severity": "blocking",
                    "stream": key,
                    "links": links,
                }
            )

    # Weak alignment is review-only unless it participates in a chain.
    for item in candidates:
        letters = _short_letters(item.short)
        initials = _definition_initials(item.definition)
        if (
            3 <= len(letters) <= 8
            and len(item.definition.split()) <= 8
            and initials
            and not _is_subsequence(letters, initials)
        ):
            review.append(
                {
                    "code": "weak_short_definition_initial_alignment",
                    "severity": "review",
                    "short": item.short,
                    "definition": item.definition,
                    "page_idx": item.page_idx,
                    "block_index": item.block_index,
                }
            )

    # Deduplicate diagnostics deterministically.
    def marker(issue: Mapping[str, Any]) -> str:
        return json.dumps(issue, ensure_ascii=False, sort_keys=True, default=str)

    blocking = list({marker(item): item for item in blocking}.values())
    review = list({marker(item): item for item in review}.values())
    return blocking, review


def extract_mineru_acronym_candidates(
    source_path: Path,
    *,
    doc_id: Optional[str] = None,
    short_detector: Callable[[str], bool],
    candidate_file_count: int = 1,
) -> MinerUAcronymResult:
    source_path = Path(source_path).expanduser().resolve()
    resolved_doc_id = doc_id or infer_doc_id(source_path)
    payload = load_mineru_payload(source_path)
    pages, source_format = _page_blocks(payload)

    result = MinerUAcronymResult(
        doc_id=resolved_doc_id,
        source_path=source_path,
        source_format=source_format,
        candidate_file_count=candidate_file_count,
    )

    globally_active = False
    stop_after_page = False
    used_pages: list[int] = []
    source_order = 0

    for page_idx, blocks in pages:
        if stop_after_page:
            break
        page_used = False
        for position, block in enumerate(blocks):
            block_type = str(block.get("type") or "").casefold()
            block_index = _optional_int(block.get("index"))
            if block_index is None:
                block_index = position
            text = _block_text(block)

            if block_type == "title":
                if _is_acronym_heading(text):
                    globally_active = True
                    result.heading_found = True
                    page_used = True
                    continue
                if globally_active and _is_body_heading(text):
                    globally_active = False
                    stop_after_page = True
                    break

            html_values = _table_html_values(block) if block_type == "table" else []
            if html_values:
                table_body_seen = False
                for raw_html in html_values:
                    pairs, heading_seen, body_seen, source_order = _pairs_from_table(
                        raw_html,
                        page_idx=page_idx,
                        block_index=block_index,
                        active_before=globally_active,
                        short_detector=short_detector,
                        order_start=source_order,
                        blocking_issues=result.blocking_issues,
                        review_issues=result.review_issues,
                        layout_diagnostics=result.layout_diagnostics,
                    )
                    if heading_seen:
                        result.heading_found = True
                        globally_active = True
                    if pairs:
                        result.candidates.extend(pairs)
                        page_used = True
                    table_body_seen = table_body_seen or body_seen
                if table_body_seen:
                    globally_active = False
                    stop_after_page = True
                continue

            if globally_active and block_type in {"text", "list"} and text:
                pairs, source_order = _parse_text_block(
                    block,
                    page_idx=page_idx,
                    block_index=block_index,
                    short_detector=short_detector,
                    order_start=source_order,
                    review_issues=result.review_issues,
                )
                if pairs:
                    result.candidates.extend(pairs)
                    page_used = True

        if page_used:
            used_pages.append(page_idx)

    result.candidates = _deduplicate_candidates(result.candidates)
    intrinsic_blocking, intrinsic_review = _validate_candidates(
        result.candidates,
        short_detector=short_detector,
    )
    result.blocking_issues.extend(intrinsic_blocking)
    result.review_issues.extend(intrinsic_review)

    if used_pages:
        result.page_start_idx = min(used_pages)
        result.page_end_idx = max(used_pages)
    if result.candidate_file_count > 1:
        result.warnings.append(
            f"multiple_mineru_candidates:{result.candidate_file_count}"
        )
    if not result.heading_found:
        result.warnings.append("acronym_heading_not_found")
    if not result.candidates:
        result.warnings.append("no_acronym_candidates")
    if result.blocking_issues:
        result.warnings.append(
            f"mineru_blocking_issues:{len(result.blocking_issues)}"
        )
    return result
