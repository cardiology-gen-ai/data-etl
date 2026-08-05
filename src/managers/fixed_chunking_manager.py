#!/usr/bin/env python3
"""Build fixed-size retrieval chunks from cleaned hierarchical chunks.

This module is deliberately independent from the knowledge-graph pipeline.
It reads ``*_hier_chunks_clean.json`` artifacts and emits a separate fixed-size
corpus suitable for the classic vector-RAG baseline.

Design guarantees
-----------------
* canonical and cleaned hierarchical artifacts are never modified;
* only active source chunks (``embed=true``, ``excluded=false``, non-empty text)
  are emitted;
* splitting is performed inside each source section, so every output chunk has
  unambiguous section provenance for retrieval evaluation;
* complete HTML ``<table>...</table>`` blocks are atomic: they are never split
  or duplicated; an oversized table is emitted as one explicitly flagged chunk;
* output is deterministic and self-describing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LOG = logging.getLogger("fixed_chunking_manager")
VERSION = "fixed_within_section_v2"
STRATEGY = "fixed_within_section"
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 300
TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
TABLE_OPEN_RE = re.compile(r"<table\b", re.IGNORECASE)
TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
STANDALONE_ARTIFACT_RE = re.compile(r"^\s*continued[.:;,-]?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Block:
    """Atomic block used by the table-aware packer."""

    kind: str  # ``text`` or ``table``
    text: str



def _is_standalone_artifact(text: str) -> bool:
    """Return True for non-informative continuation labels emitted alone."""
    return bool(STANDALONE_ARTIFACT_RE.fullmatch(text or ""))


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _validate_balanced_tables(text: str, *, context: str) -> None:
    opens = len(TABLE_OPEN_RE.findall(text or ""))
    closes = len(TABLE_CLOSE_RE.findall(text or ""))
    if opens != closes:
        raise ValueError(
            f"Unbalanced table HTML in {context}: "
            f"open_table_tags={opens}, close_table_tags={closes}"
        )


def _load_rows(path: Path) -> tuple[Any, list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
        shape = "list"
    elif isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        rows = payload["chunks"]
        shape = "object_with_chunks"
    else:
        raise TypeError(
            "Clean chunk JSON must be a list or an object containing a 'chunks' list"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError("Every clean chunk record must be a JSON object")
    return payload, [dict(row) for row in rows], shape


def _is_eligible(row: Mapping[str, Any]) -> bool:
    text = str(row.get("text") or "")
    return (
        bool(row.get("embed"))
        and not bool(row.get("excluded"))
        and not bool(row.get("is_empty"))
        and bool(text.strip())
    )


def _preferred_split(text: str, limit: int) -> tuple[str, str]:
    """Split text at a useful boundary not exceeding ``limit``.

    The returned pieces contain all non-boundary content. Boundary whitespace is
    normalized away and reintroduced by the chunk packer as a blank line.
    """
    if len(text) <= limit:
        return text.strip(), ""

    window = text[:limit]
    lower_bound = max(1, limit // 2)
    candidates: list[int] = []

    # Prefer paragraph, line, sentence, then generic whitespace boundaries.
    for pattern in (r"\n\s*\n", r"\n", r"(?<=[.!?;:])\s+", r"\s+"):
        matches = [m for m in re.finditer(pattern, window) if m.start() >= lower_bound]
        if matches:
            candidates.append(matches[-1].start())
            break

    split_at = candidates[0] if candidates else limit
    if split_at <= 0:
        split_at = limit

    left = text[:split_at].strip()
    right = text[split_at:].lstrip()
    if not left:
        left = text[:limit]
        right = text[limit:]
    return left, right


def _split_text_piece(text: str, chunk_size: int) -> list[str]:
    remaining = (text or "").strip()
    pieces: list[str] = []
    while remaining:
        left, remaining = _preferred_split(remaining, chunk_size)
        if left:
            pieces.append(left)
    return pieces


def _atomic_blocks(text: str, chunk_size: int) -> list[Block]:
    """Return text pieces and complete table blocks in source order."""
    _validate_balanced_tables(text, context="source section")
    blocks: list[Block] = []
    cursor = 0
    for match in TABLE_RE.finditer(text):
        prefix = text[cursor : match.start()]
        blocks.extend(Block("text", piece) for piece in _split_text_piece(prefix, chunk_size))
        blocks.append(Block("table", match.group(0)))
        cursor = match.end()
    suffix = text[cursor:]
    blocks.extend(Block("text", piece) for piece in _split_text_piece(suffix, chunk_size))
    return [block for block in blocks if block.text.strip()]


def _join_blocks(blocks: Sequence[Block]) -> str:
    return "\n\n".join(block.text.strip() for block in blocks if block.text.strip()).strip()


def _text_overlap_block(blocks: Sequence[Block], overlap: int) -> Block | None:
    """Build overlap only from trailing prose after the last table.

    Tables are never duplicated into the next chunk. This makes table counts
    stable and keeps every HTML table attributable to a single retrieval chunk.
    """
    if overlap <= 0:
        return None

    trailing_text: list[str] = []
    for block in reversed(blocks):
        if block.kind == "table":
            break
        trailing_text.append(block.text)
    if not trailing_text:
        return None

    text = "\n\n".join(reversed(trailing_text)).strip()
    if not text:
        return None
    tail = text[-overlap:]
    # Avoid beginning in the middle of a word when a nearby boundary exists.
    match = re.search(r"\s+", tail)
    if match and match.end() < len(tail):
        tail = tail[match.end() :]
    tail = tail.strip()
    return Block("text", tail) if tail else None


def split_section_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Split one cleaned section into table-aware fixed-size chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    blocks = _atomic_blocks(text, chunk_size)
    if not blocks:
        return []

    emitted: list[dict[str, Any]] = []
    current: list[Block] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        chunk_text = _join_blocks(current)
        table_blocks = [block.text for block in current if block.kind == "table"]
        emitted.append(
            {
                "text": chunk_text,
                "char_count": len(chunk_text),
                "contains_table": bool(table_blocks),
                "table_count": len(table_blocks),
                "oversized_atomic_table": (
                    len(current) == 1
                    and current[0].kind == "table"
                    and len(current[0].text) > chunk_size
                ),
            }
        )
        overlap_block = _text_overlap_block(current, chunk_overlap)
        current = [overlap_block] if overlap_block is not None else []

    for block in blocks:
        # An oversized table is emitted alone and unchanged.
        if block.kind == "table" and len(block.text) > chunk_size:
            flush()
            current = [block]
            flush()
            current = []  # no overlap may cross an atomic table
            continue

        candidate = _join_blocks([*current, block])
        if current and len(candidate) > chunk_size:
            flush()
            candidate = _join_blocks([*current, block])

        # Overlap can consume too much room. Drop it rather than split a table
        # or create an avoidably oversized normal chunk.
        if current and len(candidate) > chunk_size:
            current = []
            candidate = block.text.strip()

        if len(candidate) > chunk_size and block.kind != "table":
            # Defensive fallback. Text blocks should already fit.
            for piece in _split_text_piece(block.text, chunk_size):
                if current and len(_join_blocks([*current, Block("text", piece)])) > chunk_size:
                    flush()
                current.append(Block("text", piece))
                flush()
            continue

        current.append(block)

    flush()

    # MinerU/ESC tables can contain a standalone continuation marker between
    # two table fragments. It has no retrieval value and must not become an
    # independent embedding document. Do not remove the word when it occurs
    # inside a meaningful sentence or a larger chunk.
    return [part for part in emitted if not _is_standalone_artifact(part["text"])]


def _fixed_output_name(input_path: Path, chunk_size: int, chunk_overlap: int) -> str:
    suffix = "_hier_chunks_clean.json"
    if not input_path.name.endswith(suffix):
        raise ValueError(
            "Expected an input filename ending in '_hier_chunks_clean.json': "
            f"{input_path.name}"
        )
    stem = input_path.name[: -len(suffix)]
    return f"{stem}_fixed_chunks_c{chunk_size}_o{chunk_overlap}.json"


def build_fixed_corpus(
    input_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Build a fixed-size corpus payload from one clean hierarchical artifact."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    _, rows, source_shape = _load_rows(input_path)
    output_rows: list[dict[str, Any]] = []
    skipped = {"excluded": 0, "not_embeddable": 0, "empty": 0}
    eligible_source_ids: list[str] = []
    document_ids = {
        str(row.get("doc_id")) for row in rows if row.get("doc_id") is not None
    }
    if len(document_ids) != 1:
        raise ValueError(
            "Expected exactly one doc_id in the clean chunk artifact; "
            f"found {sorted(document_ids)}"
        )
    doc_id = next(iter(document_ids))

    for source_order, row in enumerate(rows):
        text = str(row.get("text") or "")
        if bool(row.get("excluded")):
            skipped["excluded"] += 1
            continue
        if not bool(row.get("embed")):
            skipped["not_embeddable"] += 1
            continue
        if bool(row.get("is_empty")) or not text.strip():
            skipped["empty"] += 1
            continue
        if not _is_eligible(row):  # defensive consistency check
            continue

        source_chunk_id = str(row.get("chunk_id") or "")
        source_section_id = str(row.get("section_id") or "")
        if not source_chunk_id or not source_section_id:
            raise ValueError(
                f"Eligible source record at index {source_order} lacks chunk_id/section_id"
            )
        eligible_source_ids.append(source_chunk_id)
        section_chunks = split_section_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not section_chunks:
            raise ValueError(f"Eligible source chunk produced no output: {source_chunk_id}")

        part_count = len(section_chunks)
        for part_index, part in enumerate(section_chunks):
            fixed_chunk_id = (
                f"{doc_id}::fixed_c{chunk_size}_o{chunk_overlap}::"
                f"{source_section_id}::{part_index:04d}"
            )
            output_rows.append(
                {
                    "chunk_id": fixed_chunk_id,
                    "fixed_chunk_id": fixed_chunk_id,
                    "doc_id": doc_id,
                    "strategy": STRATEGY,
                    "text": part["text"],
                    "char_count": part["char_count"],
                    "chunk_size_unit": "characters",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "fixed_part_index": part_index,
                    "fixed_part_count": part_count,
                    "source_order": source_order,
                    "source_chunk_id": source_chunk_id,
                    "source_chunk_ids": [source_chunk_id],
                    "source_section_id": source_section_id,
                    "source_section_ids": [source_section_id],
                    "section_title": row.get("section_title"),
                    "section_level": row.get("section_level"),
                    "parent_section_id": row.get("parent_section_id"),
                    "printed_section_id": row.get("printed_section_id"),
                    "page_start": row.get("page_start"),
                    "page_end": row.get("page_end"),
                    "source_text_sha256": _sha_text(text),
                    "contains_table": part["contains_table"],
                    "table_count": part["table_count"],
                    "oversized_atomic_table": part["oversized_atomic_table"],
                    "excluded": False,
                    "embed": True,
                }
            )

    payload: dict[str, Any] = {
        "version": VERSION,
        "strategy": STRATEGY,
        "doc_id": doc_id,
        "source_path": str(input_path),
        "source_sha256": _sha_file(input_path),
        "source_shape": source_shape,
        "chunk_size_unit": "characters",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "overlap_policy": "target_max_preserve_boundaries_and_tables",
        "source_record_count": len(rows),
        "eligible_source_record_count": len(eligible_source_ids),
        "output_chunk_count": len(output_rows),
        "skipped_source_records": skipped,
        "chunks": output_rows,
    }
    validate_fixed_corpus(payload, source_rows=rows)
    return payload


def validate_fixed_corpus(
    payload: Mapping[str, Any],
    *,
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate provenance, limits, ordering, coverage and table preservation."""
    errors: list[str] = []
    chunks_raw = payload.get("chunks")
    if not isinstance(chunks_raw, list):
        return {"valid": False, "errors": ["payload.chunks must be a list"]}
    chunks = [dict(chunk) for chunk in chunks_raw]
    chunk_size = int(payload.get("chunk_size") or 0)

    eligible = [row for row in source_rows if _is_eligible(row)]
    eligible_ids = [str(row.get("chunk_id") or "") for row in eligible]
    observed_source_ids = [str(chunk.get("source_chunk_id") or "") for chunk in chunks]

    if len({str(chunk.get("chunk_id") or "") for chunk in chunks}) != len(chunks):
        errors.append("fixed chunk IDs are not unique")
    if any(not str(chunk.get("text") or "").strip() for chunk in chunks):
        errors.append("one or more fixed chunks are empty")
    if any(bool(chunk.get("excluded")) for chunk in chunks):
        errors.append("an excluded fixed chunk was emitted")
    if any(not bool(chunk.get("embed")) for chunk in chunks):
        errors.append("a non-embeddable fixed chunk was emitted")
    if any(_is_standalone_artifact(str(chunk.get("text") or "")) for chunk in chunks):
        errors.append("a standalone continuation artifact was emitted")

    for source_id in eligible_ids:
        if source_id not in observed_source_ids:
            errors.append(f"eligible source chunk has no fixed chunk: {source_id}")

    by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        source_id = str(chunk.get("source_chunk_id") or "")
        by_source.setdefault(source_id, []).append(chunk)
        char_count = len(str(chunk.get("text") or ""))
        if char_count != int(chunk.get("char_count") or -1):
            errors.append(f"char_count mismatch for {chunk.get('chunk_id')}")
        if char_count > chunk_size and not bool(chunk.get("oversized_atomic_table")):
            errors.append(f"unflagged oversized fixed chunk: {chunk.get('chunk_id')}")
        try:
            _validate_balanced_tables(
                str(chunk.get("text") or ""), context=str(chunk.get("chunk_id"))
            )
        except ValueError as exc:
            errors.append(str(exc))

    source_by_id = {str(row.get("chunk_id") or ""): row for row in eligible}
    for source_id, source in source_by_id.items():
        source_tables = TABLE_RE.findall(str(source.get("text") or ""))
        emitted_tables: list[str] = []
        parts = by_source.get(source_id, [])
        for part in parts:
            emitted_tables.extend(TABLE_RE.findall(str(part.get("text") or "")))
        if source_tables != emitted_tables:
            errors.append(f"table blocks changed, split, reordered or duplicated: {source_id}")

        expected_indices = list(range(len(parts)))
        observed_indices = [int(part.get("fixed_part_index", -1)) for part in parts]
        if observed_indices != expected_indices:
            errors.append(f"fixed part indices are not sequential: {source_id}")
        if any(int(part.get("fixed_part_count", -1)) != len(parts) for part in parts):
            errors.append(f"fixed part count mismatch: {source_id}")

    if int(payload.get("eligible_source_record_count") or -1) != len(eligible):
        errors.append("eligible source count mismatch")
    if int(payload.get("output_chunk_count") or -1) != len(chunks):
        errors.append("output chunk count mismatch")

    if errors:
        raise ValueError("Fixed corpus validation failed: " + "; ".join(errors[:20]))
    return {
        "valid": True,
        "errors": [],
        "eligible_source_record_count": len(eligible),
        "output_chunk_count": len(chunks),
        "oversized_atomic_table_count": sum(
            bool(chunk.get("oversized_atomic_table")) for chunk in chunks
        ),
        "table_chunk_count": sum(bool(chunk.get("contains_table")) for chunk in chunks),
    }


def write_fixed_corpus(
    input_path: Path,
    output_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    payload = build_fixed_corpus(
        input_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _fixed_output_name(
        Path(input_path), chunk_size, chunk_overlap
    )
    if output_path.exists() and not force:
        raise FileExistsError(f"Output exists; use --force: {output_path}")
    serialized = _json_text(payload)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(serialized, encoding="utf-8")
    tmp_path.replace(output_path)
    return output_path, payload


def _select_inputs(input_dir: Path, doc_id: str | None) -> list[Path]:
    paths = sorted(Path(input_dir).glob("*_hier_chunks_clean.json"))
    if doc_id:
        paths = [path for path in paths if path.name.startswith(f"{doc_id}_")]
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build fixed-size RAG chunks from cleaned hierarchical chunks."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input-file", type=Path)
    source_group.add_argument("--input-dir", type=Path)
    parser.add_argument("--doc-id")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("mineru_test/fixed_chunks")
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum prose chunk size in characters (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=(
            "Target maximum prose overlap in characters; boundaries and atomic "
            f"tables take precedence (default: {DEFAULT_CHUNK_OVERLAP})."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.input_file is not None:
        inputs = [args.input_file]
    else:
        inputs = _select_inputs(args.input_dir, args.doc_id)
    if not inputs:
        LOG.error("No *_hier_chunks_clean.json inputs found")
        return 2

    failures = 0
    for input_path in inputs:
        try:
            output_path, payload = write_fixed_corpus(
                input_path=input_path,
                output_dir=args.output_dir,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                force=args.force,
            )
            LOG.info(
                "%s -> %s | eligible_sections=%d | fixed_chunks=%d | "
                "table_chunks=%d | oversized_tables=%d",
                input_path.name,
                output_path,
                payload["eligible_source_record_count"],
                payload["output_chunk_count"],
                sum(bool(row.get("contains_table")) for row in payload["chunks"]),
                sum(bool(row.get("oversized_atomic_table")) for row in payload["chunks"]),
            )
        except Exception:
            failures += 1
            LOG.exception("Fixed chunking failed: %s", input_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
