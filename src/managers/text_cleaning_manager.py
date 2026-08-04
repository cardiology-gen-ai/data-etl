#!/usr/bin/env python3
"""Safe v1 cleaner for canonical *_hier_chunks.json files.

Only the ``text`` field is rewritten. Complete HTML tables are masked and
restored byte-for-byte. Raw chunk files are never overwritten.
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
VERSION = "canonical_text_safe_v1"

TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.I | re.S)
IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)\n]+)\)")
LINK_RE = re.compile(r"(?<!!)\[(?P<label>[^\]]+)\]\((?P<url>[^)\n]+)\)")
SUP_RE = re.compile(r"<sup\b[^>]*>(?P<body>.*?)</sup\s*>", re.I | re.S)
SUB_RE = re.compile(r"<sub\b[^>]*>(?P<body>.*?)</sub\s*>", re.I | re.S)
EQ_RE = re.compile(r"<eq\b[^>]*>(?P<body>.*?)</eq\s*>", re.I | re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
BLOCK_RE = re.compile(r"</?(?:p|div|section|article|h[1-6]|ul|ol|dl|dt|dd|blockquote)\b[^>]*>", re.I)
BR_RE = re.compile(r"<br\s*/?>", re.I)
LI_OPEN_RE = re.compile(r"<li\b[^>]*>", re.I)
LI_CLOSE_RE = re.compile(r"</li\s*>", re.I)
TAG_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*\b[^>]*>", re.S)
LOWER_HYPHEN_BREAK_RE = re.compile(r"(?P<a>[a-z])[-\u2010\u2011\u2212]\n(?P<b>[a-z])")
SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+(?=[,.;:!?%)\]\}])")
NUMERIC_SUP_RE = re.compile(r"^[\d\s,;.\-–—]+$")
GENERIC_ALT = {"", "image", "img", "figure", "picture", "photo", "diagram"}

STRUCTURAL_FIELDS = (
    "chunk_id", "doc_id", "section_id", "printed_section_id",
    "parent_section_id", "section_title", "section_level", "section_type",
    "page_start", "page_end", "is_empty", "excluded", "embed",
    "part_index", "part_count", "quality_flags", "boundary_source",
)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tables(text: str) -> list[str]:
    return [m.group(0) for m in TABLE_RE.finditer(text or "")]


def mask_tables(text: str) -> tuple[str, list[str]]:
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
    return result


def clean_text(raw: str) -> tuple[str, dict[str, Any]]:
    masked, found_tables = mask_tables(raw)
    stats: dict[str, Any] = {
        "raw_chars": len(raw),
        "table_blocks_preserved": len(found_tables),
        "images_removed": 0,
        "informative_image_alts_preserved": 0,
        "links_unwrapped": 0,
        "numeric_superscripts_removed": 0,
        "note_superscripts_preserved": 0,
        "exponent_superscripts_converted": 0,
        "subscripts_flattened": 0,
        "equation_tags_unwrapped": 0,
        "soft_hyphens_removed": 0,
        "hyphenated_line_breaks_joined": 0,
        "html_tags_removed": 0,
        "punctuation_spacing_fixes": 0,
    }

    value = unicodedata.normalize("NFC", masked.replace("\r\n", "\n").replace("\r", "\n"))
    value = COMMENT_RE.sub("", value)

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
        body = re.sub(r"<[^>]+>", "", html.unescape(match.group("body") or ""))
        body = re.sub(r"\s+", " ", body).strip()
        prefix = match.string[max(0, match.start() - 16):match.start()].casefold()
        unit_before = re.search(
            r"(?:^|[\s/(])(?:m|cm|mm|km|µm|μm|nm|in|ft)$",
            prefix,
        )
        if body in {"2", "3"} and unit_before:
            stats["exponent_superscripts_converted"] += 1
            return {"2": "²", "3": "³"}[body]
        if body and NUMERIC_SUP_RE.fullmatch(body):
            stats["numeric_superscripts_removed"] += 1
            return ""
        if body:
            stats["note_superscripts_preserved"] += 1
            return f"[{body}]"
        return ""
    value = SUP_RE.sub(sup_repl, value)

    def sub_repl(match: re.Match[str]) -> str:
        stats["subscripts_flattened"] += 1
        body = re.sub(r"<[^>]+>", "", html.unescape(match.group("body") or ""))
        return re.sub(r"\s+", "", body)
    value = SUB_RE.sub(sub_repl, value)

    def eq_repl(match: re.Match[str]) -> str:
        stats["equation_tags_unwrapped"] += 1
        body = html.unescape(match.group("body") or "")
        for source, target in {r"\leq":"≤", r"\geq":"≥", r"\pm":"±", r"\times":"×"}.items():
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

    value = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n"))
    value, stats["punctuation_spacing_fixes"] = SPACE_BEFORE_PUNCT_RE.subn("", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    cleaned = restore_tables(value, found_tables).strip()
    stats["clean_chars"] = len(cleaned)
    stats["changed"] = cleaned != raw
    return cleaned, stats


def load_payload(path: Path) -> tuple[Any, list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, [dict(x) for x in payload], "list"
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload, [dict(x) for x in payload["chunks"]], "dict"
    raise TypeError("Expected a list or an object containing a 'chunks' list")


def validate(raw_rows: list[dict[str, Any]], clean_rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(raw_rows) != len(clean_rows):
        return ["record count changed"]
    for idx, (raw, clean) in enumerate(zip(raw_rows, clean_rows)):
        for field in STRUCTURAL_FIELDS:
            if raw.get(field) != clean.get(field):
                errors.append(f"record[{idx}] changed structural field {field}")
        raw_text = str(raw.get("text") or "")
        clean_text_value = str(clean.get("text") or "")
        if raw_text.strip() and not clean_text_value.strip():
            errors.append(f"record[{idx}] became empty")
        if tables(raw_text) != tables(clean_text_value):
            errors.append(f"record[{idx}] changed table blocks")
    if [x.get("section_id") for x in raw_rows] != [x.get("section_id") for x in clean_rows]:
        errors.append("section order changed")
    return errors


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def process_file(input_path: Path, output_dir: Path, audit_dir: Path, force: bool, dry_run: bool) -> dict[str, Any]:
    source_hash = sha_file(input_path)
    payload, raw_rows, shape = load_payload(input_path)
    clean_rows: list[dict[str, Any]] = []
    section_audit: list[dict[str, Any]] = []

    for raw in raw_rows:
        clean = deepcopy(raw)
        raw_text = str(raw.get("text") or "")
        cleaned_text, stats = clean_text(raw_text)
        clean["text"] = cleaned_text
        clean_rows.append(clean)
        section_audit.append({
            "doc_id": raw.get("doc_id"),
            "section_id": raw.get("section_id"),
            "section_title": raw.get("section_title"),
            "excluded": bool(raw.get("excluded")),
            "embed": bool(raw.get("embed")),
            "raw_text_sha256": sha_text(raw_text),
            "clean_text_sha256": sha_text(cleaned_text),
            "table_cleaning_status": "preserved_raw_pending_table_cleaner",
            **stats,
        })

    errors = validate(raw_rows, clean_rows)
    if errors:
        raise ValueError("Invariant failures: " + "; ".join(errors[:20]))

    rebuilt = clean_rows if shape == "list" else {**payload, "chunks": clean_rows}
    output_name = input_path.name.replace("_hier_chunks.json", "_hier_chunks_clean.json")
    audit_name = input_path.name.replace("_hier_chunks.json", "_text_cleaning_audit.json")
    output_path = output_dir / output_name
    audit_path = audit_dir / audit_name
    if output_path.exists() and not force and not dry_run:
        raise FileExistsError(f"Output exists; use --force: {output_path}")

    report = {
        "cleaning_version": VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "chunk_count": len(raw_rows),
        "changed_chunk_count": sum(bool(x["changed"]) for x in section_audit),
        "raw_chars": sum(int(x["raw_chars"]) for x in section_audit),
        "clean_chars": sum(int(x["clean_chars"]) for x in section_audit),
        "table_blocks_preserved": sum(int(x["table_blocks_preserved"]) for x in section_audit),
        "source_sha256": source_hash,
        "invariant_errors": [],
        "sections": section_audit,
    }
    if not dry_run:
        write_json(output_path, rebuilt)
        write_json(audit_path, report)
        if sha_file(input_path) != source_hash:
            raise RuntimeError("Raw source changed unexpectedly")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("mineru_test/chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("mineru_test/clean_chunks"))
    parser.add_argument("--audit-dir", type=Path, default=Path("mineru_test/text_cleaning_audit/safe_v1"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    inputs = sorted(args.input_dir.glob("*_hier_chunks.json"))
    if not inputs:
        LOG.error("No *_hier_chunks.json files found in %s", args.input_dir)
        return 2
    failures = 0
    for path in inputs:
        try:
            report = process_file(path, args.output_dir, args.audit_dir, args.force, args.dry_run)
            LOG.info(
                "%s | chunks=%d | changed=%d | tables=%d | chars=%d -> %d | dry_run=%s",
                path.name, report["chunk_count"], report["changed_chunk_count"],
                report["table_blocks_preserved"], report["raw_chars"],
                report["clean_chars"], args.dry_run,
            )
        except Exception:
            failures += 1
            LOG.exception("Cleaning failed: %s", path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
