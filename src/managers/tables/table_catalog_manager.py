"""Build a read-only table catalogue from MinerU artefacts.

The manager supports two MinerU shapes:

1. A flat ``content_list``: a JSON list containing blocks with
   ``type == \"table\"`` and fields such as ``table_body`` and ``page_idx``.
2. A page-level MinerU artefact: a JSON object containing ``pdf_info``.  Only
   ``preproc_blocks`` are read, because the same tables may also be repeated in
   ``para_blocks``.

The source HTML is immutable.  Derived classification, quality flags, and
chunk links are stored in separate fields.  This first version never modifies
chunk files and never writes cleaned table text back into them.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import logging
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence
from html.parser import HTMLParser

VERSION = "table_catalog_v2_2"
LOGGER = logging.getLogger(__name__)

TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
KNOWN_SUFFIXES = (
    "_content_list_v2",
    "_content_list",
    "_middle",
    "_model",
    "_tables",
)
TIMESTAMP_SUFFIX_RE = re.compile(r"__\d{8,14}$")
TAG_GAP_RE = re.compile(r">\s+<")
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
CAMEL_GLUE_RE = re.compile(r"(?<=[a-z]{3})(?=[A-Z][a-z]{2})")


@dataclass(frozen=True)
class SourceTable:
    """A table block extracted from a MinerU artefact."""

    source_index: int
    page_idx: Optional[int]
    block_index: Optional[int]
    bbox: Optional[list[float]]
    caption: list[str]
    footnotes: list[str]
    raw_html: str
    image_source: Optional[str]
    source_format: str
    quality_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChunkTableOccurrence:
    """A table occurrence found in a canonical or clean chunk file."""

    doc_id: str
    chunk_id: Optional[str]
    section_id: Optional[str]
    section_title: Optional[str]
    excluded: bool
    embed: bool
    chunk_table_index: int
    start_offset: int
    end_offset: int
    raw_html: str
    exact_sha256: str
    canonical_sha256: str
    source_path: str
    source_order: int


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_table_element(value: str) -> str:
    """Return the first complete ``<table>`` element, without outer wrappers."""

    match = TABLE_RE.search(value or "")
    return match.group(0).strip() if match else (value or "").strip()


def canonicalize_table_html(value: str) -> str:
    """Canonicalize only insignificant whitespace for deterministic matching.

    This function is deliberately conservative: it does not reorder elements,
    rewrite text, infer missing separators, or alter attributes.
    """

    table = extract_table_element(value)
    table = TAG_GAP_RE.sub("><", table)
    table = table.strip()
    return table


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


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
            elif item is not None:
                text = str(item).strip()
                if text:
                    result.append(text)
        return result
    text = str(value).strip()
    return [text] if text else []


def _iter_spans(block: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for line in block.get("lines") or []:
        if not isinstance(line, Mapping):
            continue
        for span in line.get("spans") or []:
            if isinstance(span, Mapping):
                yield span


def _block_text(block: Mapping[str, Any]) -> str:
    values: list[str] = []
    for span in _iter_spans(block):
        content = span.get("content")
        if isinstance(content, str) and content.strip():
            values.append(content.strip())
    return " ".join(values).strip()


def _extract_rich_table(
    table_block: Mapping[str, Any],
    *,
    source_index: int,
    page_idx: Optional[int],
) -> SourceTable:
    captions: list[str] = []
    footnotes: list[str] = []
    html_values: list[str] = []
    image_values: list[str] = []

    for child in table_block.get("blocks") or []:
        if not isinstance(child, Mapping):
            continue
        child_type = str(child.get("type") or "")
        text = _block_text(child)
        if child_type == "table_caption" and text:
            captions.append(text)
        elif child_type == "table_footnote" and text:
            footnotes.append(text)

        for span in _iter_spans(child):
            html_value = span.get("html")
            if isinstance(html_value, str) and html_value.strip():
                html_values.append(html_value.strip())
            image_path = span.get("image_path") or span.get("img_path")
            if isinstance(image_path, str) and image_path.strip():
                image_values.append(image_path.strip())

    # MinerU should expose one body span.  Joining is safer than silently
    # dropping a second fragment, while still preserving each fragment exactly.
    raw_html = "\n".join(html_values).strip()
    flags: list[str] = []
    if not raw_html:
        flags.append("missing_html")
    if len(html_values) > 1:
        flags.append("multiple_html_fragments")
    if not captions:
        flags.append("missing_caption")
    if not image_values:
        flags.append("missing_image")

    return SourceTable(
        source_index=source_index,
        page_idx=page_idx,
        block_index=_optional_int(table_block.get("index")),
        bbox=_normalize_bbox(table_block.get("bbox")),
        caption=captions,
        footnotes=footnotes,
        raw_html=raw_html,
        image_source=image_values[0] if image_values else None,
        source_format="mineru_pdf_info",
        quality_flags=flags,
    )


def _extract_flat_table(block: Mapping[str, Any], source_index: int) -> SourceTable:
    raw_html = str(block.get("table_body") or block.get("html") or "").strip()
    image_source = block.get("img_path") or block.get("image_path")
    flags: list[str] = []
    if not raw_html:
        flags.append("missing_html")
    if not block.get("table_caption"):
        flags.append("missing_caption")
    if not image_source:
        flags.append("missing_image")

    return SourceTable(
        source_index=source_index,
        page_idx=_optional_int(block.get("page_idx")),
        block_index=_optional_int(block.get("index")),
        bbox=_normalize_bbox(block.get("bbox")),
        caption=_as_text_list(block.get("table_caption")),
        footnotes=_as_text_list(block.get("table_footnote")),
        raw_html=raw_html,
        image_source=str(image_source).strip() if image_source else None,
        source_format="mineru_content_list",
        quality_flags=flags,
    )


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_bbox(value: Any) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return None
    return result


def extract_mineru_tables(payload: Any) -> tuple[list[SourceTable], dict[str, Any]]:
    """Extract tables from either supported MinerU JSON shape."""

    tables: list[SourceTable] = []

    if isinstance(payload, list):
        for block in payload:
            if not isinstance(block, Mapping) or block.get("type") != "table":
                continue
            tables.append(_extract_flat_table(block, len(tables)))
        return tables, {
            "source_format": "mineru_content_list",
            "page_count": None,
        }

    if isinstance(payload, Mapping) and isinstance(payload.get("pdf_info"), list):
        pages = payload["pdf_info"]
        for page_position, page in enumerate(pages):
            if not isinstance(page, Mapping):
                continue
            page_idx = _optional_int(page.get("page_idx"))
            if page_idx is None:
                page_idx = page_position
            # Deliberately ignore para_blocks: tables may be duplicated there.
            for block in page.get("preproc_blocks") or []:
                if not isinstance(block, Mapping) or block.get("type") != "table":
                    continue
                tables.append(
                    _extract_rich_table(
                        block,
                        source_index=len(tables),
                        page_idx=page_idx,
                    )
                )
        return tables, {
            "source_format": "mineru_pdf_info",
            "page_count": len(pages),
            "mineru_backend": payload.get("_backend"),
            "mineru_version": payload.get("_version_name"),
            "ocr_enabled": payload.get("_ocr_enable"),
        }

    raise ValueError(
        "Unsupported MinerU JSON shape: expected a flat list or an object "
        "containing a pdf_info list."
    )


def _visible_text(raw_html: str) -> str:
    text = TAG_RE.sub(" ", raw_html)
    text = html_lib.unescape(text)
    return SPACE_RE.sub(" ", text).strip()


class _RowSignatureParser(HTMLParser):
    """Extract conservative row/cell text signatures for fragment matching."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, ...]] = []
        self._row: Optional[list[str]] = None
        self._cell_parts: Optional[list[str]] = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
    ) -> None:
        tag = tag.casefold()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in {"td", "th"} and self._cell_parts is not None:
            value = SPACE_RE.sub(" ", "".join(self._cell_parts)).strip()
            if self._row is not None:
                self._row.append(value)
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(tuple(self._row))
            self._row = None


def table_row_signatures(raw_html: str) -> list[tuple[str, ...]]:
    """Return ordered row signatures without inventing cell boundaries."""

    parser = _RowSignatureParser()
    parser.feed(extract_table_element(raw_html))
    parser.close()
    return parser.rows


def _fragment_group_id(occurrence: ChunkTableOccurrence) -> str:
    token = f"{occurrence.doc_id}|{occurrence.source_order}|{occurrence.canonical_sha256}"
    return f"{occurrence.doc_id}::fragment_group::{sha256_text(token)[:16]}"


def strict_acronym_like(value: str) -> bool:
    """Return True for short acronym/code labels, not ordinary category words."""

    text = SPACE_RE.sub(" ", TAG_RE.sub(" ", value or "")).strip(" .,:;()[]{}")
    if not (2 <= len(text) <= 24) or len(text.split()) > 3:
        return False
    if not re.fullmatch(r"[A-Za-z0-9.+\-/]+(?:\s+[A-Za-z0-9.+\-/]+){0,2}", text):
        return False
    uppercase = sum(char.isupper() for char in text)
    lowercase = sum(char.islower() for char in text)
    digits = sum(char.isdigit() for char in text)
    return (uppercase + digits) >= 2 and uppercase >= lowercase


def classify_table(table: SourceTable) -> tuple[str, list[str]]:
    caption = " ".join(table.caption)
    visible = _visible_text(table.raw_html)
    combined = f"{caption} {visible}".casefold()
    reasons: list[str] = []

    caption_has_recommendation = "recommendation" in caption.casefold()
    body_has_recommendation = "recommendation" in visible.casefold()
    has_class = bool(re.search(r"\bclass(?:es)?\b", combined))
    has_level = bool(re.search(r"\blevel(?:s)?(?: of evidence)?\b", combined))
    if caption_has_recommendation or (body_has_recommendation and (has_class or has_level)):
        if caption_has_recommendation:
            reasons.append("recommendation_caption")
        if body_has_recommendation:
            reasons.append("recommendation_keyword")
        if has_class or has_level:
            reasons.append("class_or_level_keyword")
        # Candidate only: exact recommendation/continuation classification remains
        # the responsibility of the relocated table cleaner/parser.
        return "recommendation_candidate", reasons

    # Conservative glossary detector: require acronym-like left cells rather
    # than merely short labels.  This avoids classifying tables such as
    # ``Topics | Content`` as acronym glossaries.
    rows = re.findall(r"<tr\b[^>]*>(.*?)</tr\s*>", table.raw_html, re.I | re.S)
    acronym_left = 0
    two_cell_rows = 0
    for row in rows[:80]:
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]\s*>", row, re.I | re.S)
        if len(cells) == 2:
            two_cell_rows += 1
            left = html_lib.unescape(SPACE_RE.sub(" ", TAG_RE.sub(" ", cells[0])).strip())
            if strict_acronym_like(left):
                acronym_left += 1
    if two_cell_rows >= 5 and acronym_left / two_cell_rows >= 0.75:
        reasons.append("two_column_acronym_left_cells")
        return "acronym_or_glossary", reasons

    return "table_unclassified", reasons


def detect_quality_flags(table: SourceTable) -> list[str]:
    flags = list(table.quality_flags)
    visible = _visible_text(table.raw_html)
    if CAMEL_GLUE_RE.search(visible):
        flags.append("possible_concatenated_words")
    return sorted(set(flags))


def _load_chunk_records(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("chunks", "sections", "records", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _record_text(record: Mapping[str, Any]) -> str:
    for key in ("text", "body", "content", "chunk_text"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def _record_id(record: Mapping[str, Any]) -> Optional[str]:
    for key in ("chunk_id", "id", "section_uid"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def _find_chunk_files(chunk_dir: Path, doc_id: str) -> list[Path]:
    candidates = sorted(chunk_dir.glob(f"{doc_id}*.json"))
    if candidates:
        return candidates
    # Filename-derived doc IDs may contain '&' or differ from the repository's
    # chosen spelling.  Fall back to token overlap, but never silently use more
    # than one unrelated document.
    tokens = [token.casefold() for token in re.split(r"[_\W]+", doc_id) if len(token) >= 4]
    scored: list[tuple[int, Path]] = []
    for path in chunk_dir.glob("*.json"):
        name = path.stem.casefold()
        score = sum(token in name for token in tokens)
        if score:
            scored.append((score, path))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    return sorted(path for score, path in scored if score == best)


def extract_chunk_tables(chunk_dir: Path, doc_id: str) -> list[ChunkTableOccurrence]:
    occurrences: list[ChunkTableOccurrence] = []
    source_order = 0
    for path in _find_chunk_files(chunk_dir, doc_id):
        for record in _load_chunk_records(path):
            text = _record_text(record)
            if not text:
                continue
            table_index = 0
            for match in TABLE_RE.finditer(text):
                table_index += 1
                raw_html = match.group(0).strip()
                canonical = canonicalize_table_html(raw_html)
                occurrences.append(
                    ChunkTableOccurrence(
                        doc_id=doc_id,
                        chunk_id=_record_id(record),
                        section_id=(
                            str(record.get("section_id"))
                            if record.get("section_id") is not None
                            else None
                        ),
                        section_title=(
                            str(record.get("section_title"))
                            if record.get("section_title") is not None
                            else None
                        ),
                        excluded=bool(record.get("excluded", False)),
                        embed=bool(record.get("embed", True)),
                        chunk_table_index=table_index,
                        start_offset=match.start(),
                        end_offset=match.end(),
                        raw_html=raw_html,
                        exact_sha256=sha256_text(raw_html),
                        canonical_sha256=sha256_text(canonical),
                        source_path=str(path.resolve()),
                        source_order=source_order,
                    )
                )
                source_order += 1
    return occurrences


def _link_tables(
    source_tables: Sequence[SourceTable],
    chunk_tables: Sequence[ChunkTableOccurrence],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Link source fragments to chunk tables.

    The first pass performs one-to-one exact/canonical hash matching.  The
    second pass links consecutive MinerU page fragments when their ordered row
    signatures concatenate exactly to one unmatched chunk table.  This handles
    the common MinerU shape ``N page fragments -> 1 Markdown table`` without
    fuzzy text matching.
    """

    exact_queues: dict[str, deque[ChunkTableOccurrence]] = defaultdict(deque)
    canonical_queues: dict[str, deque[ChunkTableOccurrence]] = defaultdict(deque)
    for occurrence in chunk_tables:
        exact_queues[occurrence.exact_sha256].append(occurrence)
        canonical_queues[occurrence.canonical_sha256].append(occurrence)

    used_chunk_orders: set[int] = set()
    links: list[dict[str, Any]] = [{"link_status": "not_linked"} for _ in source_tables]

    def link_payload(
        occurrence: ChunkTableOccurrence,
        *,
        status: str,
        fragment_index: Optional[int] = None,
        fragment_count: Optional[int] = None,
        fragment_group_id: Optional[str] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "link_status": status,
            "chunk_id": occurrence.chunk_id,
            "section_id": occurrence.section_id,
            "section_title": occurrence.section_title,
            "chunk_table_index": occurrence.chunk_table_index,
            "chunk_start_offset": occurrence.start_offset,
            "chunk_end_offset": occurrence.end_offset,
            "excluded": occurrence.excluded,
            "embed": occurrence.embed,
            "chunk_source_path": occurrence.source_path,
            "chunk_source_order": occurrence.source_order,
        }
        if fragment_group_id is not None:
            payload.update(
                {
                    "fragment_group_id": fragment_group_id,
                    "fragment_index": fragment_index,
                    "fragment_count": fragment_count,
                    "fragment_match_mode": "exact_row_concatenation",
                }
            )
        return payload

    # Pass 1: exact/canonical one-to-one matching.
    for source_position, table in enumerate(source_tables):
        raw = extract_table_element(table.raw_html)
        exact_hash = sha256_text(raw) if raw else None
        canonical = canonicalize_table_html(raw) if raw else ""
        canonical_hash = sha256_text(canonical) if canonical else None
        matched: Optional[ChunkTableOccurrence] = None
        status = "not_linked"

        if exact_hash:
            while (
                exact_queues[exact_hash]
                and exact_queues[exact_hash][0].source_order in used_chunk_orders
            ):
                exact_queues[exact_hash].popleft()
            if exact_queues[exact_hash]:
                matched = exact_queues[exact_hash].popleft()
                status = "matched_exact"

        if matched is None and canonical_hash:
            while (
                canonical_queues[canonical_hash]
                and canonical_queues[canonical_hash][0].source_order in used_chunk_orders
            ):
                canonical_queues[canonical_hash].popleft()
            if canonical_queues[canonical_hash]:
                matched = canonical_queues[canonical_hash].popleft()
                status = "matched_canonical"

        if matched is not None:
            used_chunk_orders.add(matched.source_order)
            links[source_position] = link_payload(matched, status=status)

    # Pass 2: exact ordered row-concatenation for consecutive unlinked source
    # fragments.  Restricting the match to complete ordered rows prevents
    # accidental substring/fuzzy matches.
    unlinked_positions = [
        index for index, link in enumerate(links) if link.get("link_status") == "not_linked"
    ]
    unlinked_set = set(unlinked_positions)
    fragment_sequence_count = 0
    fragment_linked_source_table_count = 0

    for occurrence in sorted(chunk_tables, key=lambda item: item.source_order):
        if occurrence.source_order in used_chunk_orders:
            continue
        target_rows = table_row_signatures(occurrence.raw_html)
        if not target_rows:
            continue

        match_positions: Optional[list[int]] = None
        for start_position in unlinked_positions:
            if start_position not in unlinked_set:
                continue
            accumulated: list[tuple[str, ...]] = []
            candidate_positions: list[int] = []
            previous_source_index: Optional[int] = None
            for source_position in range(start_position, len(source_tables)):
                if source_position not in unlinked_set:
                    break
                source_index = source_tables[source_position].source_index
                if previous_source_index is not None and source_index != previous_source_index + 1:
                    break
                previous_source_index = source_index
                rows = table_row_signatures(source_tables[source_position].raw_html)
                if not rows:
                    break
                candidate_positions.append(source_position)
                accumulated.extend(rows)
                if len(candidate_positions) >= 2 and accumulated == target_rows:
                    match_positions = candidate_positions.copy()
                    break
                if len(accumulated) >= len(target_rows):
                    break
            if match_positions is not None:
                break

        if match_positions is None:
            continue

        group_id = _fragment_group_id(occurrence)
        fragment_count = len(match_positions)
        for fragment_index, source_position in enumerate(match_positions, start=1):
            links[source_position] = link_payload(
                occurrence,
                status="matched_fragment_sequence",
                fragment_index=fragment_index,
                fragment_count=fragment_count,
                fragment_group_id=group_id,
            )
            unlinked_set.remove(source_position)
        used_chunk_orders.add(occurrence.source_order)
        fragment_sequence_count += 1
        fragment_linked_source_table_count += fragment_count

    status_counts = Counter(str(link.get("link_status") or "not_linked") for link in links)
    unmatched_chunk_records = [
        {
            "version": VERSION,
            "doc_id": occurrence.doc_id,
            "chunk_id": occurrence.chunk_id,
            "section_id": occurrence.section_id,
            "section_title": occurrence.section_title,
            "excluded": occurrence.excluded,
            "embed": occurrence.embed,
            "chunk_table_index": occurrence.chunk_table_index,
            "chunk_start_offset": occurrence.start_offset,
            "chunk_end_offset": occurrence.end_offset,
            "raw_html": occurrence.raw_html,
            "raw_html_sha256": occurrence.exact_sha256,
            "canonical_html_sha256": occurrence.canonical_sha256,
            "chunk_source_path": occurrence.source_path,
            "chunk_source_order": occurrence.source_order,
            "link_status": "unmatched_chunk",
        }
        for occurrence in chunk_tables
        if occurrence.source_order not in used_chunk_orders
    ]

    return links, {
        "chunk_table_count": len(chunk_tables),
        "linked_chunk_table_count": len(used_chunk_orders),
        "source_linked_table_count": sum(
            str(link.get("link_status") or "").startswith("matched") for link in links
        ),
        "source_not_linked_table_count": status_counts.get("not_linked", 0),
        "unmatched_chunk_table_count": len(unmatched_chunk_records),
        "fragment_sequence_count": fragment_sequence_count,
        "fragment_linked_source_table_count": fragment_linked_source_table_count,
        "link_status_counts": dict(status_counts),
    }, unmatched_chunk_records


def _record_for_table(
    *,
    doc_id: str,
    table: SourceTable,
    source_path: Path,
    link: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    classification, reasons = classify_table(table)
    raw_table = extract_table_element(table.raw_html)
    exact_hash = sha256_text(raw_table) if raw_table else None
    canonical = canonicalize_table_html(raw_table) if raw_table else ""
    canonical_hash = sha256_text(canonical) if canonical else None

    record: dict[str, Any] = {
        "version": VERSION,
        "doc_id": doc_id,
        "table_id": f"{doc_id}::table::{table.source_index + 1:04d}",
        "source_index": table.source_index,
        "source_format": table.source_format,
        "source_path": str(source_path.resolve()),
        "page_idx": table.page_idx,
        "page": table.page_idx + 1 if table.page_idx is not None else None,
        "block_index": table.block_index,
        "bbox": table.bbox,
        "caption": table.caption,
        "footnotes": table.footnotes,
        "image_source": table.image_source,
        "raw_html": table.raw_html,
        "raw_html_sha256": exact_hash,
        "canonical_html_sha256": canonical_hash,
        "classification": classification,
        "classification_reasons": reasons,
        "quality_flags": detect_quality_flags(table),
        "link_status": "not_requested",
    }
    if link:
        record.update(link)
    return record


def build_catalog_for_file(
    source_path: Path,
    output_dir: Path,
    *,
    chunk_dir: Optional[Path] = None,
    doc_id: Optional[str] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build one raw catalogue and return its summary."""

    source_path = source_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_doc_id = doc_id or infer_doc_id(source_path)
    output_path = output_dir / f"{resolved_doc_id}_tables_raw.jsonl"
    summary_path = output_dir / f"{resolved_doc_id}_table_catalog_summary.json"

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --force to overwrite."
        )

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    tables, source_metadata = extract_mineru_tables(payload)

    links: list[dict[str, Any]] = [{} for _ in tables]
    unmatched_chunk_records: list[dict[str, Any]] = []
    link_summary: dict[str, Any] = {
        "chunk_table_count": 0,
        "linked_chunk_table_count": 0,
        "source_linked_table_count": 0,
        "source_not_linked_table_count": 0,
        "unmatched_chunk_table_count": 0,
        "fragment_sequence_count": 0,
        "fragment_linked_source_table_count": 0,
        "link_status_counts": {"not_requested": len(tables)},
    }
    chunk_files: list[str] = []
    if chunk_dir is not None:
        chunk_dir = chunk_dir.resolve()
        files = _find_chunk_files(chunk_dir, resolved_doc_id)
        chunk_files = [str(path.resolve()) for path in files]
        chunk_tables = extract_chunk_tables(chunk_dir, resolved_doc_id)
        links, link_summary, unmatched_chunk_records = _link_tables(tables, chunk_tables)

    records = [
        _record_for_table(
            doc_id=resolved_doc_id,
            table=table,
            source_path=source_path,
            link=link,
        )
        for table, link in zip(tables, links)
    ]

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    unlinked_source_path = output_dir / f"{resolved_doc_id}_unlinked_source_tables.jsonl"
    unmatched_chunk_path = output_dir / f"{resolved_doc_id}_unmatched_chunk_tables.jsonl"
    with unlinked_source_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.get("link_status") == "not_linked":
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
    with unmatched_chunk_path.open("w", encoding="utf-8") as handle:
        for record in unmatched_chunk_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    classification_counts = Counter(record["classification"] for record in records)
    quality_flag_counts = Counter(
        flag for record in records for flag in record.get("quality_flags", [])
    )
    summary: dict[str, Any] = {
        "version": VERSION,
        "doc_id": resolved_doc_id,
        "source_path": str(source_path),
        "source_sha256": sha256_text(source_path.read_text(encoding="utf-8")),
        "output_path": str(output_path.resolve()),
        "source_format": source_metadata.get("source_format"),
        "page_count": source_metadata.get("page_count"),
        "mineru_backend": source_metadata.get("mineru_backend"),
        "mineru_version": source_metadata.get("mineru_version"),
        "ocr_enabled": source_metadata.get("ocr_enabled"),
        "table_count": len(records),
        "tables_with_html": sum(bool(record["raw_html"]) for record in records),
        "tables_with_image": sum(bool(record["image_source"]) for record in records),
        "tables_with_caption": sum(bool(record["caption"]) for record in records),
        "tables_with_footnotes": sum(bool(record["footnotes"]) for record in records),
        "classification_counts": dict(classification_counts),
        "quality_flag_counts": dict(quality_flag_counts),
        "chunk_dir": str(chunk_dir) if chunk_dir else None,
        "chunk_files": chunk_files,
        **link_summary,
        "diagnostic_files": {
            "unlinked_source_tables": str(unlinked_source_path.resolve()),
            "unmatched_chunk_tables": str(unmatched_chunk_path.resolve()),
        },
        "source_html_modified": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _iter_source_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.glob("*.json")
        if path.is_file() and not path.name.endswith("_summary.json")
    )


def build_catalogs(
    input_dir: Path,
    output_dir: Path,
    *,
    chunk_dir: Optional[Path] = None,
    force: bool = False,
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    files = _iter_source_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    summaries: list[dict[str, Any]] = []
    for path in files:
        LOGGER.info("Building table catalogue | source=%s", path)
        summaries.append(
            build_catalog_for_file(
                path,
                output_dir,
                chunk_dir=chunk_dir,
                force=force,
            )
        )

    aggregate = {
        "version": VERSION,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "chunk_dir": str(chunk_dir.resolve()) if chunk_dir else None,
        "document_count": len(summaries),
        "table_count": sum(item["table_count"] for item in summaries),
        "tables_with_html": sum(item["tables_with_html"] for item in summaries),
        "linked_chunk_table_count": sum(
            item["linked_chunk_table_count"] for item in summaries
        ),
        "source_linked_table_count": sum(
            item.get("source_linked_table_count", 0) for item in summaries
        ),
        "fragment_sequence_count": sum(
            item.get("fragment_sequence_count", 0) for item in summaries
        ),
        "fragment_linked_source_table_count": sum(
            item.get("fragment_linked_source_table_count", 0) for item in summaries
        ),
        "source_not_linked_table_count": sum(
            item.get("source_not_linked_table_count", 0) for item in summaries
        ),
        "unmatched_chunk_table_count": sum(
            item.get("unmatched_chunk_table_count", 0) for item in summaries
        ),
        "documents": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


def load_table_config(config_path: Path) -> dict[str, Any]:
    """Load table preprocessing settings from the project configuration.

    Preferred location::

        cardiology_protocols.preprocessing.tables

    The loader also accepts ``preprocessing.tables`` and the two legacy
    locations ``knowledge_graph.tables`` / top-level ``tables`` so older local
    configurations continue to work.
    Relative paths are resolved against the directory containing the config.
    """

    config_path = config_path.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("The config root must be a JSON object.")

    candidates: list[Mapping[str, Any]] = []

    protocols = payload.get("cardiology_protocols")
    if isinstance(protocols, Mapping):
        preprocessing = protocols.get("preprocessing")
        if isinstance(preprocessing, Mapping):
            candidate = preprocessing.get("tables")
            if isinstance(candidate, Mapping):
                candidates.append(candidate)

    preprocessing = payload.get("preprocessing")
    if isinstance(preprocessing, Mapping):
        candidate = preprocessing.get("tables")
        if isinstance(candidate, Mapping):
            candidates.append(candidate)

    knowledge_graph = payload.get("knowledge_graph")
    if isinstance(knowledge_graph, Mapping):
        candidate = knowledge_graph.get("tables")
        if isinstance(candidate, Mapping):
            candidates.append(candidate)

    if isinstance(payload.get("tables"), Mapping):
        candidates.append(payload["tables"])

    if not candidates:
        raise KeyError(
            "Missing table configuration: expected "
            "cardiology_protocols.preprocessing.tables."
        )
    tables = candidates[0]

    def resolve(*keys: str) -> Optional[Path]:
        for key in keys:
            value = tables.get(key)
            if isinstance(value, str) and value.strip():
                path = Path(value)
                return path if path.is_absolute() else config_path.parent / path
        return None

    return {
        "enabled": bool(tables.get("enabled", True)),
        "input_dir": resolve("source_dir", "content_list_dir", "input_dir"),
        "output_dir": resolve("catalog_dir", "output_dir"),
        "chunk_dir": resolve("chunk_dir"),
        "link_to_chunks": bool(tables.get("link_to_chunks", True)),
        "force": bool(tables.get("force", False)),
    }

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a read-only table catalogue from MinerU JSON artefacts."
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--input-file", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional project config containing preprocessing.tables paths.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        help="Optional directory containing canonical or clean hierarchical chunks.",
    )
    parser.add_argument(
        "--doc-id",
        help="Override the document ID. Valid only with --input-file.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config_paths: dict[str, Optional[Path]] = {}
    if args.config:
        config_paths = load_table_config(args.config)

    if config_paths and not config_paths.get("enabled", True):
        LOGGER.info("Table preprocessing is disabled in the configuration.")
        return 0

    input_file = args.input_file
    input_dir = args.input_dir or config_paths.get("input_dir")
    output_dir = args.output_dir or config_paths.get("output_dir")
    configured_chunk_dir = (
        config_paths.get("chunk_dir")
        if config_paths.get("link_to_chunks", True)
        else None
    )
    chunk_dir = args.chunk_dir or configured_chunk_dir
    force = bool(args.force or config_paths.get("force", False))

    if input_file is None and input_dir is None:
        raise SystemExit(
            "Provide --input-file, --input-dir, or --config with a table input directory."
        )
    if output_dir is None:
        raise SystemExit(
            "Provide --output-dir or --config with preprocessing.tables.catalog_dir."
        )

    if input_dir:
        if args.doc_id:
            raise SystemExit("--doc-id can only be used with --input-file")
        summary = build_catalogs(
            input_dir,
            output_dir,
            chunk_dir=chunk_dir,
            force=force,
        )
        LOGGER.info(
            "Completed | documents=%d | tables=%d | linked=%d",
            summary["document_count"],
            summary["table_count"],
            summary["linked_chunk_table_count"],
        )
        return 0

    summary = build_catalog_for_file(
        input_file,
        output_dir,
        chunk_dir=chunk_dir,
        doc_id=args.doc_id,
        force=force,
    )
    LOGGER.info(
        "Completed | doc_id=%s | tables=%d | html=%d | linked=%d",
        summary["doc_id"],
        summary["table_count"],
        summary["tables_with_html"],
        summary["linked_chunk_table_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
