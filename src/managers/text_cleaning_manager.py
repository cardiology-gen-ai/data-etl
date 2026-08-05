#!/usr/bin/env python3
"""Conservative cleaner for canonical ``*_hier_chunks.json`` files.

Only the ``text`` field is rewritten. Complete HTML tables are validated,
masked, and restored byte-for-byte. Canonical chunk files are never
modified. The module also exposes a cache-aware API that can be called by the
graph pipeline in a later integration step.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any

LOG = logging.getLogger("text_cleaning_manager")
VERSION = "canonical_text_safe_v2_2"

TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.I | re.S)
TABLE_OPEN_RE = re.compile(r"<table\b", re.I)
TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.I)
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)\n]+)\)")
LINK_RE = re.compile(r"(?<!!)\[(?P<label>[^\]]+)\]\((?P<url>[^)\n]+)\)")
SUP_RE = re.compile(r"<sup\b[^>]*>(?P<body>.*?)</sup\s*>", re.I | re.S)
SUB_RE = re.compile(r"<sub\b[^>]*>(?P<body>.*?)</sub\s*>", re.I | re.S)
EQ_RE = re.compile(r"<eq\b[^>]*>(?P<body>.*?)</eq\s*>", re.I | re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
BLOCK_RE = re.compile(
    r"</?(?:p|div|section|article|h[1-6]|ul|ol|dl|dt|dd|blockquote)\b[^>]*>",
    re.I,
)
BR_RE = re.compile(r"<br\s*/?>", re.I)
LI_OPEN_RE = re.compile(r"<li\b[^>]*>", re.I)
LI_CLOSE_RE = re.compile(r"</li\s*>", re.I)
TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*\b[^>]*>", re.S)
LOWER_HYPHEN_BREAK_RE = re.compile(
    r"(?P<a>[a-z])[-\u2010\u2011\u2212]\n(?P<b>[a-z])"
)
SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+(?=[,.;:!?%)\]\}])")
NUMERIC_SUP_RE = re.compile(r"^[\d\s,;.\-–—]+$")
MULTI_REFERENCE_SUP_RE = re.compile(
    r"^\d+(?:\s*(?:,|;|-|–|—)\s*\d+)+$"
)
SINGLE_NUMERIC_SUP_RE = re.compile(r"^\d+$")
TRUNCATED_REFERENCE_SUP_RE = re.compile(r"^\d+\s*[-–—]\s*$")
RENDERED_TRUNCATED_REFERENCE_RE = re.compile(
    r"\^\{\s*\d+\s*[-–—]\s*\}"
)
TABLE_PLACEHOLDER_RE = re.compile(
    r"^TEXTCLEAN_TABLE_\d{6}_[0-9a-f]{12}$"
)
BRACKET_FRAGMENT_RE = re.compile(r"\[[^\]\n]{1,24}\]")
EMBEDDED_BRACKET_RE = re.compile(
    r"(?:[A-Za-z]\[[^\]\n]{1,24}\]|"
    r"\[[^\]\n]{1,24}\][A-Za-z])"
)
PUNCTUATION_CLUSTER_RE = re.compile(r"(?:\s*[;,]){4,}")
LONG_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]{28,}")

# Full-line page furniture only. Legitimate prose containing the phrase
# "ESC Guidelines" is not removed.
PAGE_FURNITURE_PATTERNS = (
    re.compile(r"^\s*ESC Guidelines\s+\d{1,6}\s*$", re.I),
    re.compile(r"^\s*\d{1,6}\s+ESC Guidelines\s*$", re.I),
    re.compile(r"^\s*©\s*ESC\s*\d{4}\s*$", re.I),
)
ESC_GUIDELINES_ONLY_RE = re.compile(r"^\s*ESC Guidelines\s*$", re.I)
PAGE_NUMBER_ONLY_RE = re.compile(r"^\s*\d{1,6}\s*$")
MATH_SINGLE_SYMBOL_BASE_RE = re.compile(
    r"(?:^|[\s(=+\-×*/])(?P<base>[A-Za-z])$"
)
POWER_OF_TEN_BASE_RE = re.compile(r"(?:^|[^0-9])10$")
DOWNLOAD_PREFIX = "downloadedfromhttp"
DOWNLOAD_GUEST_MARKER = "bygueston"
DOWNLOAD_DATE_RE = re.compile(r"\d{1,2}[a-z]+\d{4}", re.I)

GENERIC_ALT = {"", "image", "img", "figure", "picture", "photo", "diagram"}
STRUCTURAL_FIELDS = (
    "chunk_id",
    "doc_id",
    "section_id",
    "printed_section_id",
    "parent_section_id",
    "section_title",
    "section_level",
    "section_type",
    "page_start",
    "page_end",
    "is_empty",
    "excluded",
    "embed",
    "part_index",
    "part_count",
    "quality_flags",
    "boundary_source",
)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def tables(text: str) -> list[str]:
    return [match.group(0) for match in TABLE_RE.finditer(text or "")]


def validate_balanced_tables(text: str, *, context: str = "text") -> None:
    """Fail before cleaning when table HTML is structurally incomplete.

    The previous byte-for-byte comparison could not detect a malformed table
    that failed to match ``TABLE_RE`` in both the source and cleaned text.
    """
    open_count = len(TABLE_OPEN_RE.findall(text or ""))
    close_count = len(TABLE_CLOSE_RE.findall(text or ""))
    if open_count != close_count:
        raise ValueError(
            f"Unbalanced table HTML in {context}: "
            f"open_table_tags={open_count}, close_table_tags={close_count}"
        )


def mask_tables(text: str) -> tuple[str, list[str]]:
    validate_balanced_tables(text, context="chunk text")
    found: list[str] = []

    def repl(match: re.Match[str]) -> str:
        table = match.group(0)
        idx = len(found)
        found.append(table)
        return f"\n\nTEXTCLEAN_TABLE_{idx:06d}_{sha_text(table)[:12]}\n\n"

    return TABLE_RE.sub(repl, text or ""), found


def restore_tables(text: str, found: list[str]) -> str:
    result = text
    for idx, table in enumerate(found):
        token = f"TEXTCLEAN_TABLE_{idx:06d}_{sha_text(table)[:12]}"
        if result.count(token) != 1:
            raise ValueError(f"Invalid placeholder count for {token}")
        result = result.replace(token, table, 1)
    validate_balanced_tables(result, context="cleaned chunk text")
    return result


def _compact_line_fragment(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _download_watermark_end(lines: list[str], start: int) -> int | None:
    """Return the exclusive end of an obvious download watermark block.

    MinerU/PDF extraction may split ``Downloaded`` and the URL over several
    short lines. We accept only a bounded block that starts like
    ``Downloaded from http...`` and contains ``by guest on <date>``.
    """
    compact = ""
    for end in range(start, min(len(lines), start + 12)):
        compact += _compact_line_fragment(lines[end])
        if not compact:
            continue
        if not (
            DOWNLOAD_PREFIX.startswith(compact)
            or compact.startswith(DOWNLOAD_PREFIX)
        ):
            return None
        if (
            compact.startswith(DOWNLOAD_PREFIX)
            and DOWNLOAD_GUEST_MARKER in compact
            and DOWNLOAD_DATE_RE.search(compact)
        ):
            return end + 1
        if len(compact) > 1200:
            return None
    return None


def remove_page_furniture(text: str) -> tuple[str, dict[str, int]]:
    """Remove only high-confidence, line-scoped publisher/page furniture."""
    lines = text.split("\n")
    kept: list[str] = []
    removed_lines = 0
    removed_download_blocks = 0
    index = 0

    while index < len(lines):
        download_end = _download_watermark_end(lines, index)
        if download_end is not None:
            removed_lines += download_end - index
            removed_download_blocks += 1
            index = download_end
            continue

        line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else None

        # MinerU may split the page furniture over two adjacent lines, e.g.
        # ``ESC Guidelines`` followed by ``4237`` or the reverse order.
        # Remove only the exact two-line pair.
        if next_line is not None and (
            (
                ESC_GUIDELINES_ONLY_RE.fullmatch(line)
                and PAGE_NUMBER_ONLY_RE.fullmatch(next_line)
            )
            or (
                PAGE_NUMBER_ONLY_RE.fullmatch(line)
                and ESC_GUIDELINES_ONLY_RE.fullmatch(next_line)
            )
        ):
            removed_lines += 2
            index += 2
            continue

        if any(pattern.fullmatch(line) for pattern in PAGE_FURNITURE_PATTERNS):
            removed_lines += 1
            index += 1
            continue

        kept.append(line)
        index += 1

    return "\n".join(kept), {
        "publisher_noise_lines_removed": removed_lines,
        "download_watermark_blocks_removed": removed_download_blocks,
    }


def _corruption_features(paragraph: str) -> dict[str, int | bool]:
    return {
        "bracket_fragments": len(BRACKET_FRAGMENT_RE.findall(paragraph)),
        "embedded_brackets": len(EMBEDDED_BRACKET_RE.findall(paragraph)),
        "punctuation_cluster": bool(PUNCTUATION_CLUSTER_RE.search(paragraph)),
        "long_alpha_token": bool(LONG_ALPHA_TOKEN_RE.search(paragraph)),
        "word_count": len(re.findall(r"\b[A-Za-z0-9]+\b", paragraph)),
        "char_count": len(paragraph),
    }


def _corruption_reason(
    paragraph: str,
    *,
    previous_is_table: bool,
    next_is_table: bool,
) -> str | None:
    """Recognize only high-confidence extraction corruption.

    The rules intentionally require several independent structural signals and
    table adjacency. They do not contain document IDs, section titles, table
    captions, or corpus-specific text snippets.
    """
    features = _corruption_features(paragraph)
    bracket_count = int(features["bracket_fragments"])
    embedded_count = int(features["embedded_brackets"])
    word_count = int(features["word_count"])
    char_count = int(features["char_count"])

    # Short broken caption immediately before a table, e.g. multiple OCR
    # fragments inserted in or replacing words. Requiring table adjacency,
    # embedded brackets and a short span avoids treating normal bracketed prose
    # as corruption.
    if (
        next_is_table
        and bracket_count >= 3
        and embedded_count >= 1
        and word_count <= 20
        and char_count <= 240
    ):
        return "corrupted_caption_before_table"

    # Broken abbreviation/footnote block immediately after a table. This is
    # accepted only with heavy bracket fragmentation plus an additional signal
    # such as punctuation bursts, glued alphabetic text, or several brackets
    # embedded inside words.
    if (
        previous_is_table
        and bracket_count >= 8
        and (
            bool(features["punctuation_cluster"])
            or bool(features["long_alpha_token"])
            or embedded_count >= 2
        )
    ):
        return "corrupted_note_after_table"

    # Extremely damaged standalone OCR block. The thresholds are deliberately
    # high so ordinary clinical text, formulas and citations are preserved.
    if (
        bracket_count >= 12
        and bool(features["punctuation_cluster"])
        and bool(features["long_alpha_token"])
    ):
        return "severe_ocr_fragmentation"
    return None


def _find_corruption_in_masked_text(text: str) -> list[dict[str, Any]]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text)]
    findings: list[dict[str, Any]] = []
    for index, paragraph in enumerate(paragraphs):
        if not paragraph or TABLE_PLACEHOLDER_RE.fullmatch(paragraph):
            continue
        previous_is_table = bool(
            index > 0 and TABLE_PLACEHOLDER_RE.fullmatch(paragraphs[index - 1])
        )
        next_is_table = bool(
            index + 1 < len(paragraphs)
            and TABLE_PLACEHOLDER_RE.fullmatch(paragraphs[index + 1])
        )
        reason = _corruption_reason(
            paragraph,
            previous_is_table=previous_is_table,
            next_is_table=next_is_table,
        )
        if reason is None:
            continue
        findings.append(
            {
                "paragraph_index": index,
                "reason": reason,
                "sha256": sha_text(paragraph),
                "chars": len(paragraph),
                "preview": paragraph[:240],
                "features": _corruption_features(paragraph),
            }
        )
    return findings


def find_high_confidence_extraction_corruption(
    text: str,
) -> list[dict[str, Any]]:
    """Return high-confidence corrupt paragraphs without modifying ``text``."""
    masked, _ = mask_tables(text)
    return _find_corruption_in_masked_text(masked)


def remove_high_confidence_extraction_corruption(
    masked_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Remove high-confidence corrupt paragraphs from table-masked text."""
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", masked_text)]
    findings = _find_corruption_in_masked_text(masked_text)
    remove_indices = {int(item["paragraph_index"]) for item in findings}
    kept = [
        paragraph
        for index, paragraph in enumerate(paragraphs)
        if paragraph and index not in remove_indices
    ]
    return "\n\n".join(kept), findings


def clean_text(raw: str) -> tuple[str, dict[str, Any]]:
    masked, found_tables = mask_tables(raw)
    stats: dict[str, Any] = {
        "raw_chars": len(raw),
        "table_blocks_preserved": len(found_tables),
        "images_removed": 0,
        "informative_image_alts_preserved": 0,
        "links_unwrapped": 0,
        "numeric_superscripts_removed": 0,
        "single_reference_superscripts_removed": 0,
        "truncated_reference_superscripts_removed": 0,
        "rendered_truncated_reference_markers_removed": 0,
        "scientific_numeric_superscripts_preserved": 0,
        "ambiguous_numeric_superscripts_preserved": 0,
        "note_superscripts_preserved": 0,
        "exponent_superscripts_converted": 0,
        "subscripts_flattened": 0,
        "equation_tags_unwrapped": 0,
        "soft_hyphens_removed": 0,
        "hyphenated_line_breaks_joined": 0,
        "html_tags_removed": 0,
        "punctuation_spacing_fixes": 0,
        "publisher_noise_lines_removed": 0,
        "download_watermark_blocks_removed": 0,
        "high_confidence_corruption_blocks_removed": 0,
        "high_confidence_corruption_chars_removed": 0,
        "removed_corruption_blocks": [],
    }

    value = unicodedata.normalize(
        "NFC", masked.replace("\r\n", "\n").replace("\r", "\n")
    )
    value = COMMENT_RE.sub("", value)
    value, furniture_stats = remove_page_furniture(value)
    stats.update(furniture_stats)

    def image_repl(match: re.Match[str]) -> str:
        stats["images_removed"] += 1
        alt = re.sub(r"\s+", " ", match.group("alt") or "").strip()
        if alt.casefold() not in GENERIC_ALT:
            stats["informative_image_alts_preserved"] += 1
            return f"\n\n[Figure: {alt}]\n\n"
        return "\n\n"

    value = IMAGE_RE.sub(image_repl, value)

    def link_repl(match: re.Match[str]) -> str:
        stats["links_unwrapped"] += 1
        return match.group("label")

    value = LINK_RE.sub(link_repl, value)

    def sup_repl(match: re.Match[str]) -> str:
        body = re.sub(
            r"<[^>]+>", "", html.unescape(match.group("body") or "")
        )
        body = re.sub(r"\s+", " ", body).strip()
        prefix = match.string[max(0, match.start() - 24) : match.start()].casefold()
        unit_before = re.search(
            r"(?:^|[\s/(])(?:m|cm|mm|km|µm|μm|nm|in|ft)$",
            prefix,
        )
        if body in {"2", "3"} and unit_before:
            stats["exponent_superscripts_converted"] += 1
            return {"2": "²", "3": "³"}[body]

        if body and TRUNCATED_REFERENCE_SUP_RE.fullmatch(body):
            stats["numeric_superscripts_removed"] += 1
            stats["truncated_reference_superscripts_removed"] += 1
            return ""

        if body and MULTI_REFERENCE_SUP_RE.fullmatch(body):
            stats["numeric_superscripts_removed"] += 1
            return ""

        if body and SINGLE_NUMERIC_SUP_RE.fullmatch(body):
            # Single numeric superscripts in ESC prose are overwhelmingly
            # bibliography markers. Preserve only explicit scientific power
            # contexts that can be recognized without document-specific rules:
            #   10<sup>6</sup>  -> 10^{6}
            #   x<sup>2</sup>   -> x^{2}
            # Unit squares/cubes were already handled above.
            stripped_prefix = match.string[: match.start()].rstrip()
            explicit_power_context = bool(
                POWER_OF_TEN_BASE_RE.search(stripped_prefix)
                or (
                    body in {"2", "3"}
                    and MATH_SINGLE_SYMBOL_BASE_RE.search(stripped_prefix)
                )
            )
            if explicit_power_context:
                stats["scientific_numeric_superscripts_preserved"] += 1
                return f"^{{{body}}}"

            stats["numeric_superscripts_removed"] += 1
            stats["single_reference_superscripts_removed"] += 1
            return ""

        if body and NUMERIC_SUP_RE.fullmatch(body):
            # Numeric forms not covered by the strict multi-reference grammar
            # are retained explicitly for review rather than discarded.
            stats["ambiguous_numeric_superscripts_preserved"] += 1
            return f"^{{{body}}}"

        if body:
            stats["note_superscripts_preserved"] += 1
            return f"[{body}]"
        return ""

    value = SUP_RE.sub(sup_repl, value)
    value, rendered_truncated_count = RENDERED_TRUNCATED_REFERENCE_RE.subn(
        "", value
    )
    stats["rendered_truncated_reference_markers_removed"] = (
        rendered_truncated_count
    )

    def sub_repl(match: re.Match[str]) -> str:
        stats["subscripts_flattened"] += 1
        body = re.sub(
            r"<[^>]+>", "", html.unescape(match.group("body") or "")
        )
        return re.sub(r"\s+", "", body)

    value = SUB_RE.sub(sub_repl, value)

    def eq_repl(match: re.Match[str]) -> str:
        stats["equation_tags_unwrapped"] += 1
        body = html.unescape(match.group("body") or "")
        for source, target in {
            r"\leq": "≤",
            r"\geq": "≥",
            r"\pm": "±",
            r"\times": "×",
        }.items():
            body = body.replace(source, target)
        return re.sub(r"\s+", " ", body).strip()

    value = EQ_RE.sub(eq_repl, value)
    value = html.unescape(value)
    stats["soft_hyphens_removed"] = value.count("\u00ad")
    value = value.replace("\u00ad", "")

    def hyphen_repl(match: re.Match[str]) -> str:
        stats["hyphenated_line_breaks_joined"] += 1
        return match.group("a") + match.group("b")

    value = LOWER_HYPHEN_BREAK_RE.sub(hyphen_repl, value)
    value = BR_RE.sub("\n", value)
    value = LI_OPEN_RE.sub("\n- ", value)
    value = LI_CLOSE_RE.sub("\n", value)
    value = BLOCK_RE.sub("\n\n", value)
    value, stats["html_tags_removed"] = TAG_RE.subn(" ", value)
    value = re.sub(r"(\*\*|__)(?=\S)(.*?)(?<=\S)\1", r"\2", value, flags=re.S)
    value = re.sub(r"`([^`\n]+)`", r"\1", value)
    value = "\n".join(
        re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")
    )
    value, stats["punctuation_spacing_fixes"] = SPACE_BEFORE_PUNCT_RE.subn(
        "", value
    )
    value, corruption_findings = remove_high_confidence_extraction_corruption(
        value
    )
    stats["high_confidence_corruption_blocks_removed"] = len(
        corruption_findings
    )
    stats["high_confidence_corruption_chars_removed"] = sum(
        int(item["chars"]) for item in corruption_findings
    )
    stats["removed_corruption_blocks"] = corruption_findings
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    cleaned = restore_tables(value, found_tables).strip()

    if tables(raw) != tables(cleaned):
        raise ValueError("Table blocks changed during text cleaning")

    stats["clean_chars"] = len(cleaned)
    stats["changed"] = cleaned != raw
    return cleaned, stats


def load_payload(path: Path) -> tuple[Any, list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, [dict(item) for item in payload], "list"
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload, [dict(item) for item in payload["chunks"]], "dict"
    raise TypeError("Expected a list or an object containing a 'chunks' list")


def validate(
    raw_rows: list[dict[str, Any]], clean_rows: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    if len(raw_rows) != len(clean_rows):
        return ["record count changed"]

    for idx, (raw, clean) in enumerate(zip(raw_rows, clean_rows)):
        for field in STRUCTURAL_FIELDS:
            if raw.get(field) != clean.get(field):
                errors.append(f"record[{idx}] changed structural field {field}")

        raw_text = str(raw.get("text") or "")
        clean_text_value = str(clean.get("text") or "")
        try:
            validate_balanced_tables(raw_text, context=f"raw record[{idx}]")
            validate_balanced_tables(
                clean_text_value, context=f"clean record[{idx}]"
            )
        except ValueError as exc:
            errors.append(str(exc))

        if raw_text.strip() and not clean_text_value.strip():
            errors.append(f"record[{idx}] became empty")
        if tables(raw_text) != tables(clean_text_value):
            errors.append(f"record[{idx}] changed table blocks")

    if [item.get("section_id") for item in raw_rows] != [
        item.get("section_id") for item in clean_rows
    ]:
        errors.append("section order changed")
    return errors


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _json_text(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(serialized, encoding="utf-8")
    tmp.replace(path)
    return sha_text(serialized)


def clean_output_path(input_path: Path, output_dir: Path) -> Path:
    suffix = "_hier_chunks.json"
    if not input_path.name.endswith(suffix):
        raise ValueError(f"Unexpected canonical chunk filename: {input_path.name}")
    return output_dir / input_path.name.replace(
        suffix, "_hier_chunks_clean.json"
    )


def clean_audit_path(input_path: Path, audit_dir: Path) -> Path:
    suffix = "_hier_chunks.json"
    if not input_path.name.endswith(suffix):
        raise ValueError(f"Unexpected canonical chunk filename: {input_path.name}")
    return audit_dir / input_path.name.replace(
        suffix, "_text_cleaning_audit.json"
    )


def validate_clean_cache(
    input_path: Path,
    output_path: Path,
    audit_path: Path,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Validate a cached clean artifact against source, version and invariants."""
    if not output_path.exists():
        return False, "clean output missing", None
    if not audit_path.exists():
        return False, "clean audit missing", None

    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive boundary
        return False, f"invalid audit JSON: {exc}", None

    if audit.get("cleaning_version") != VERSION:
        return False, "cleaning version changed", audit
    if audit.get("source_sha256") != sha_file(input_path):
        return False, "source hash changed", audit
    if audit.get("output_sha256") != sha_file(output_path):
        return False, "clean output hash changed", audit

    try:
        _, raw_rows, _ = load_payload(input_path)
        _, clean_rows, _ = load_payload(output_path)
    except Exception as exc:
        return False, f"cache payload unreadable: {exc}", audit

    errors = validate(raw_rows, clean_rows)
    if errors:
        return False, "cache invariant failure: " + "; ".join(errors[:5]), audit
    return True, "valid", audit


def process_file(
    input_path: Path,
    output_dir: Path,
    audit_dir: Path,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_hash = sha_file(input_path)
    payload, raw_rows, shape = load_payload(input_path)
    clean_rows: list[dict[str, Any]] = []
    section_audit: list[dict[str, Any]] = []

    for row_index, raw in enumerate(raw_rows):
        clean = deepcopy(raw)
        raw_text = str(raw.get("text") or "")
        validate_balanced_tables(raw_text, context=f"record[{row_index}]")
        cleaned_text, stats = clean_text(raw_text)
        clean["text"] = cleaned_text
        clean_rows.append(clean)
        section_audit.append(
            {
                "doc_id": raw.get("doc_id"),
                "section_id": raw.get("section_id"),
                "section_title": raw.get("section_title"),
                "excluded": bool(raw.get("excluded")),
                "embed": bool(raw.get("embed")),
                "raw_text_sha256": sha_text(raw_text),
                "clean_text_sha256": sha_text(cleaned_text),
                "table_cleaning_status": "preserved_raw_pending_table_cleaner",
                **stats,
            }
        )

    errors = validate(raw_rows, clean_rows)
    if errors:
        raise ValueError("Invariant failures: " + "; ".join(errors[:20]))

    rebuilt = clean_rows if shape == "list" else {**payload, "chunks": clean_rows}
    output_path = clean_output_path(input_path, output_dir)
    audit_path = clean_audit_path(input_path, audit_dir)

    if output_path.exists() and not force and not dry_run:
        raise FileExistsError(f"Output exists; use --force: {output_path}")

    serialized_output = _json_text(rebuilt)
    report: dict[str, Any] = {
        "cleaning_version": VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "audit_path": str(audit_path),
        "chunk_count": len(raw_rows),
        "changed_chunk_count": sum(bool(item["changed"]) for item in section_audit),
        "raw_chars": sum(int(item["raw_chars"]) for item in section_audit),
        "clean_chars": sum(int(item["clean_chars"]) for item in section_audit),
        "table_blocks_preserved": sum(
            int(item["table_blocks_preserved"]) for item in section_audit
        ),
        "publisher_noise_lines_removed": sum(
            int(item["publisher_noise_lines_removed"]) for item in section_audit
        ),
        "download_watermark_blocks_removed": sum(
            int(item["download_watermark_blocks_removed"])
            for item in section_audit
        ),
        "numeric_superscripts_removed": sum(
            int(item["numeric_superscripts_removed"]) for item in section_audit
        ),
        "single_reference_superscripts_removed": sum(
            int(item["single_reference_superscripts_removed"])
            for item in section_audit
        ),
        "truncated_reference_superscripts_removed": sum(
            int(item["truncated_reference_superscripts_removed"])
            for item in section_audit
        ),
        "rendered_truncated_reference_markers_removed": sum(
            int(item["rendered_truncated_reference_markers_removed"])
            for item in section_audit
        ),
        "high_confidence_corruption_blocks_removed": sum(
            int(item["high_confidence_corruption_blocks_removed"])
            for item in section_audit
        ),
        "high_confidence_corruption_chars_removed": sum(
            int(item["high_confidence_corruption_chars_removed"])
            for item in section_audit
        ),
        "scientific_numeric_superscripts_preserved": sum(
            int(item["scientific_numeric_superscripts_preserved"])
            for item in section_audit
        ),
        "ambiguous_numeric_superscripts_preserved": sum(
            int(item["ambiguous_numeric_superscripts_preserved"])
            for item in section_audit
        ),
        "source_sha256": source_hash,
        "output_sha256": sha_text(serialized_output),
        "invariant_errors": [],
        "sections": section_audit,
    }

    if not dry_run:
        written_hash = write_json(output_path, rebuilt)
        if written_hash != report["output_sha256"]:
            raise RuntimeError("Serialized clean output hash changed unexpectedly")
        write_json(audit_path, report)
        if sha_file(input_path) != source_hash:
            raise RuntimeError("Raw source changed unexpectedly")

    return report


def load_or_build_clean_chunks(
    input_path: Path,
    output_dir: Path,
    audit_dir: Path,
    *,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Return a valid clean chunk artifact, rebuilding stale caches.

    This is the API intended for the later ``main_graph`` integration.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    audit_dir = Path(audit_dir)
    output_path = clean_output_path(input_path, output_dir)
    audit_path = clean_audit_path(input_path, audit_dir)

    if not force:
        valid, reason, audit = validate_clean_cache(
            input_path, output_path, audit_path
        )
        if valid and audit is not None:
            cached = dict(audit)
            cached["cache_status"] = "reused"
            cached["cache_reason"] = reason
            return output_path, cached
        LOG.info("Clean chunk cache invalid for %s: %s", input_path.name, reason)

    report = process_file(
        input_path=input_path,
        output_dir=output_dir,
        audit_dir=audit_dir,
        force=True,
        dry_run=False,
    )
    report["cache_status"] = "rebuilt"
    report["cache_reason"] = "forced" if force else "missing_or_stale"
    return output_path, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("mineru_test/chunks")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("mineru_test/clean_chunks")
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("mineru_test/text_cleaning_audit/safe_v2_2"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    inputs = sorted(args.input_dir.glob("*_hier_chunks.json"))
    if not inputs:
        LOG.error("No *_hier_chunks.json files found in %s", args.input_dir)
        return 2

    failures = 0
    for path in inputs:
        try:
            if args.dry_run:
                report = process_file(
                    path,
                    args.output_dir,
                    args.audit_dir,
                    force=args.force,
                    dry_run=True,
                )
                report["cache_status"] = "dry_run"
            else:
                _, report = load_or_build_clean_chunks(
                    path,
                    args.output_dir,
                    args.audit_dir,
                    force=args.force,
                )
            LOG.info(
                "%s | cache=%s | chunks=%d | changed=%d | tables=%d | "
                "publisher_noise_lines=%d | chars=%d -> %d | dry_run=%s",
                path.name,
                report.get("cache_status"),
                report["chunk_count"],
                report["changed_chunk_count"],
                report["table_blocks_preserved"],
                report["publisher_noise_lines_removed"],
                report["raw_chars"],
                report["clean_chars"],
                args.dry_run,
            )
        except Exception:
            failures += 1
            LOG.exception("Cleaning failed: %s", path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
