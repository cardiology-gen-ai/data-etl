"""
Build retrieval units from canonical hierarchical section chunks.

The canonical ``*_hier_chunks.json`` files are never modified. This module
creates a derived retrieval representation in which sections deeper than an
optional maximum level are merged into their nearest ancestor at that level.

Semantics
---------
- max_level=None:
    No aggregation. Every canonical chunk with ``embed=True`` remains a
    separate retrieval unit.
- max_level=N:
    Every embeddable section with ``section_level <= N`` remains separate.
    Every embeddable section with ``section_level > N`` is assigned to its
    nearest ancestor whose ``section_level == N``.

The output keeps the canonical chunk fields used by downstream consumers and
adds provenance fields describing the source sections represented by each unit.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "doc_id",
    "section_id",
    "parent_section_id",
    "section_title",
    "section_level",
    "text",
    "is_empty",
    "excluded",
    "embed",
    "quality_flags",
}


def _strategy_name(max_level: Optional[int]) -> str:
    return "sections" if max_level is None else f"max_level_{max_level}"


def _strategy_suffix(max_level: Optional[int]) -> str:
    return "sections" if max_level is None else f"L{max_level}"


def _validate_max_level(max_level: Optional[int]) -> None:
    if max_level is None:
        return
    if isinstance(max_level, bool) or not isinstance(max_level, int):
        raise TypeError("max_level must be None or an integer")
    if max_level < 1:
        raise ValueError("max_level must be None or an integer >= 1")


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


def _sorted_unique_strings(values: Iterable[Any]) -> List[str]:
    return sorted(
        {
            str(value)
            for value in values
            if value is not None and str(value).strip()
        }
    )


def _normalize_payload(payload: Any) -> List[Dict[str, Any]]:
    """Accept a plain list or an object containing ``chunks``."""
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        records = payload["chunks"]
    else:
        raise TypeError(
            "Chunk JSON must contain a list or an object with a 'chunks' list"
        )

    if not all(isinstance(record, dict) for record in records):
        raise TypeError("Every chunk record must be a JSON object")

    return [dict(record) for record in records]


def load_hierarchical_chunks(path: Path | str) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return _normalize_payload(json.load(handle))


def write_json_atomic(payload: Any, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    temporary_path.replace(path)
    return path


def _validate_canonical_chunks(
    chunks: Sequence[Mapping[str, Any]],
) -> Tuple[str, Dict[str, Mapping[str, Any]], Dict[str, int]]:
    """Validate the current one-chunk-per-section canonical schema."""
    if not chunks:
        raise ValueError("Cannot build retrieval units from an empty list")

    doc_ids = {
        str(chunk.get("doc_id"))
        for chunk in chunks
        if chunk.get("doc_id") is not None
    }
    if len(doc_ids) != 1:
        raise ValueError(
            "All chunks must belong to one document; "
            f"found doc_ids={sorted(doc_ids)}"
        )
    doc_id = next(iter(doc_ids))

    seen_chunk_ids: set[str] = set()
    seen_section_ids: set[str] = set()
    chunk_by_section_id: Dict[str, Mapping[str, Any]] = {}
    order_by_section_id: Dict[str, int] = {}

    for index, chunk in enumerate(chunks):
        missing = REQUIRED_CHUNK_FIELDS - set(chunk)
        if missing:
            raise ValueError(
                f"Chunk at index {index} is missing fields: {sorted(missing)}"
            )

        chunk_id = str(chunk["chunk_id"])
        section_id = str(chunk["section_id"])
        level = chunk["section_level"]

        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
        if section_id in seen_section_ids:
            raise ValueError(
                "retrieval_unit_manager currently expects one canonical "
                f"chunk per section; duplicate section_id: {section_id}"
            )
        if isinstance(level, bool) or not isinstance(level, int) or level < 1:
            raise ValueError(
                f"Invalid section_level for {chunk_id}: {level!r}"
            )

        if bool(chunk["embed"]):
            if bool(chunk["excluded"]):
                raise ValueError(
                    f"Embeddable chunk is also excluded: {chunk_id}"
                )
            if bool(chunk["is_empty"]):
                raise ValueError(
                    f"Embeddable chunk is marked empty: {chunk_id}"
                )
            if not str(chunk.get("text") or "").strip():
                raise ValueError(
                    f"Embeddable chunk has no usable text: {chunk_id}"
                )

        seen_chunk_ids.add(chunk_id)
        seen_section_ids.add(section_id)
        chunk_by_section_id[section_id] = chunk
        order_by_section_id[section_id] = index

    for chunk in chunks:
        section_id = str(chunk["section_id"])
        parent_id = chunk.get("parent_section_id")
        child_level = int(chunk["section_level"])

        if parent_id is None:
            if child_level != 1:
                raise ValueError(
                    f"Non-root section has no parent: {section_id}"
                )
            continue

        parent_id = str(parent_id)
        parent = chunk_by_section_id.get(parent_id)
        if parent is None:
            raise ValueError(
                f"Missing parent {parent_id!r} for section {section_id!r}"
            )
        if int(parent["section_level"]) >= child_level:
            raise ValueError(
                f"Invalid hierarchy: {parent_id} is not above {section_id}"
            )

    return doc_id, chunk_by_section_id, order_by_section_id


def _resolve_root_section_id(
    source_chunk: Mapping[str, Any],
    chunk_by_section_id: Mapping[str, Mapping[str, Any]],
    max_level: Optional[int],
) -> str:
    """Resolve the retrieval root for one embeddable source chunk."""
    section_id = str(source_chunk["section_id"])
    source_level = int(source_chunk["section_level"])

    if max_level is None or source_level <= max_level:
        return section_id

    current = source_chunk
    visited = {section_id}

    while int(current["section_level"]) > max_level:
        parent_id = current.get("parent_section_id")
        if parent_id is None:
            raise ValueError(
                f"Could not reach level {max_level} from {section_id}"
            )

        parent_id = str(parent_id)
        if parent_id in visited:
            raise ValueError(
                f"Cycle detected while resolving ancestors of {section_id}"
            )
        visited.add(parent_id)

        parent = chunk_by_section_id.get(parent_id)
        if parent is None:
            raise ValueError(
                f"Missing ancestor {parent_id!r} for {section_id!r}"
            )
        current = parent

    root_level = int(current["section_level"])
    if root_level != max_level:
        raise ValueError(
            f"Section {section_id!r} has no ancestor exactly at level "
            f"{max_level}; resolved {current['section_id']!r} at "
            f"level {root_level}"
        )
    if bool(current.get("excluded")):
        raise ValueError(
            "Embeddable descendant resolved to an excluded root: "
            f"{current['section_id']}"
        )

    return str(current["section_id"])


def _path_below_root(
    source_section_id: str,
    root_section_id: str,
    chunk_by_section_id: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    """Return the path from the first node below root to the source."""
    if source_section_id == root_section_id:
        return []

    reverse_path: List[str] = []
    current_id = source_section_id
    visited: set[str] = set()

    while current_id != root_section_id:
        if current_id in visited:
            raise ValueError(
                f"Cycle detected while building path for {source_section_id}"
            )
        visited.add(current_id)
        reverse_path.append(current_id)

        current = chunk_by_section_id.get(current_id)
        if current is None:
            raise ValueError(f"Missing section in path: {current_id}")

        parent_id = current.get("parent_section_id")
        if parent_id is None:
            raise ValueError(
                f"{source_section_id!r} is not below {root_section_id!r}"
            )
        current_id = str(parent_id)

    reverse_path.reverse()
    return reverse_path


def _format_descendant_heading(
    chunk: Mapping[str, Any],
    include_section_ids: bool,
) -> str:
    section_id = str(chunk.get("section_id") or "").strip()
    title = str(chunk.get("section_title") or "").strip()

    if include_section_ids and section_id and title:
        return f"{section_id} {title}"
    return title or section_id


def _compose_unit_text(
    all_chunks: Sequence[Mapping[str, Any]],
    root_section_id: str,
    source_chunks: Sequence[Mapping[str, Any]],
    chunk_by_section_id: Mapping[str, Mapping[str, Any]],
    include_descendant_titles: bool,
    include_section_ids_in_titles: bool,
) -> Tuple[str, List[str], List[str]]:
    """
    Compose one retrieval text in canonical TOC order.

    Empty intermediate descendants are represented by their heading when they
    are required to preserve the path to a deeper source section.
    """
    source_section_ids = {
        str(chunk["section_id"])
        for chunk in source_chunks
    }

    represented_ids = {root_section_id}
    for source_chunk in source_chunks:
        represented_ids.update(
            _path_below_root(
                source_section_id=str(source_chunk["section_id"]),
                root_section_id=root_section_id,
                chunk_by_section_id=chunk_by_section_id,
            )
        )

    represented_chunks = [
        chunk
        for chunk in all_chunks
        if str(chunk["section_id"]) in represented_ids
    ]

    blocks: List[str] = []
    structural_context_ids: List[str] = []

    for chunk in represented_chunks:
        section_id = str(chunk["section_id"])
        is_source = section_id in source_section_ids

        if section_id == root_section_id:
            if is_source:
                root_text = str(chunk.get("text") or "").strip()
                if root_text:
                    blocks.append(root_text)
            continue

        if not is_source:
            structural_context_ids.append(section_id)

        if include_descendant_titles:
            heading = _format_descendant_heading(
                chunk,
                include_section_ids=include_section_ids_in_titles,
            )
            if heading:
                blocks.append(heading)

        if is_source:
            source_text = str(chunk.get("text") or "").strip()
            if source_text:
                blocks.append(source_text)

    text = "\n\n".join(block for block in blocks if block.strip()).strip()
    represented_section_ids = [
        str(chunk["section_id"])
        for chunk in represented_chunks
    ]
    return text, represented_section_ids, structural_context_ids


def _source_metadata(
    source_chunks: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": str(chunk["chunk_id"]),
            "section_id": str(chunk["section_id"]),
            "section_title": chunk.get("section_title"),
            "section_level": int(chunk["section_level"]),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "text_chars": len(str(chunk.get("text") or "")),
            "quality_flags": list(chunk.get("quality_flags") or []),
        }
        for chunk in source_chunks
    ]


def _page_bounds(
    source_chunks: Sequence[Mapping[str, Any]],
    root_chunk: Mapping[str, Any],
) -> Tuple[Optional[int], Optional[int]]:
    starts = [
        int(chunk["page_start"])
        for chunk in source_chunks
        if chunk.get("page_start") is not None
    ]
    ends = [
        int(chunk["page_end"])
        for chunk in source_chunks
        if chunk.get("page_end") is not None
    ]
    return (
        min(starts) if starts else root_chunk.get("page_start"),
        max(ends) if ends else root_chunk.get("page_end"),
    )


def _build_unit(
    all_chunks: Sequence[Mapping[str, Any]],
    root_chunk: Mapping[str, Any],
    source_chunks: Sequence[Mapping[str, Any]],
    chunk_by_section_id: Mapping[str, Mapping[str, Any]],
    max_level: Optional[int],
    include_descendant_titles: bool,
    include_section_ids_in_titles: bool,
) -> Dict[str, Any]:
    root_section_id = str(root_chunk["section_id"])
    root_chunk_id = str(root_chunk["chunk_id"])
    source_section_ids = [
        str(chunk["section_id"])
        for chunk in source_chunks
    ]
    source_chunk_ids = [
        str(chunk["chunk_id"])
        for chunk in source_chunks
    ]

    text, represented_section_ids, structural_context_ids = _compose_unit_text(
        all_chunks=all_chunks,
        root_section_id=root_section_id,
        source_chunks=source_chunks,
        chunk_by_section_id=chunk_by_section_id,
        include_descendant_titles=include_descendant_titles,
        include_section_ids_in_titles=include_section_ids_in_titles,
    )
    if not text:
        raise ValueError(
            f"Retrieval unit rooted at {root_section_id!r} has no text"
        )

    is_aggregated = (
        len(source_chunks) > 1
        or any(sid != root_section_id for sid in source_section_ids)
    )

    quality_flags = _sorted_unique_strings(
        flag
        for chunk in source_chunks
        for flag in (chunk.get("quality_flags") or [])
    )
    if is_aggregated:
        quality_flags.append("aggregated_retrieval_unit")
    quality_flags = sorted(set(quality_flags))

    page_start, page_end = _page_bounds(source_chunks, root_chunk)
    strategy = _strategy_name(max_level)

    return {
        # Schema-compatible fields used by current consumers.
        "chunk_id": root_chunk_id,
        "doc_id": str(root_chunk["doc_id"]),
        "section_id": root_section_id,
        "printed_section_id": root_chunk.get("printed_section_id"),
        "parent_section_id": root_chunk.get("parent_section_id"),
        "section_title": root_chunk.get("section_title"),
        "section_level": int(root_chunk["section_level"]),
        "section_type": root_chunk.get("section_type"),
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
        "is_empty": False,
        "excluded": False,
        "embed": True,
        "part_index": 0,
        "part_count": 1,
        "quality_flags": quality_flags,
        "boundary_source": (
            "retrieval_unit_aggregation"
            if is_aggregated
            else root_chunk.get("boundary_source")
        ),

        # Retrieval strategy.
        "retrieval_unit_id": (
            f"{root_chunk_id}::retrieval::{strategy}"
        ),
        "retrieval_strategy": strategy,
        "aggregation_mode": (
            "none" if max_level is None else "merge_below_level"
        ),
        "aggregation_max_level": max_level,
        "is_aggregated": is_aggregated,
        "root_has_local_text": root_section_id in source_section_ids,

        # Provenance.
        "root_section_id": root_section_id,
        "root_chunk_id": root_chunk_id,
        "root_page_start": root_chunk.get("page_start"),
        "root_page_end": root_chunk.get("page_end"),
        "root_quality_flags": list(root_chunk.get("quality_flags") or []),
        "source_section_ids": source_section_ids,
        "source_chunk_ids": source_chunk_ids,
        "source_sections": _source_metadata(source_chunks),
        "represented_section_ids": represented_section_ids,
        "structural_context_section_ids": structural_context_ids,
        "source_count": len(source_chunks),
        "represented_section_count": len(represented_section_ids),

        # Audit statistics.
        "text_chars": len(text),
        "text_words": _word_count(text),
        "source_text_chars": sum(
            len(str(chunk.get("text") or ""))
            for chunk in source_chunks
        ),
    }


def build_retrieval_units(
    chunks: Sequence[Mapping[str, Any]],
    max_level: Optional[int] = None,
    include_descendant_titles: bool = True,
    include_section_ids_in_titles: bool = True,
    validate: bool = True,
) -> List[Dict[str, Any]]:
    """Build retrieval units without mutating canonical chunks."""
    _validate_max_level(max_level)
    canonical_chunks = [dict(chunk) for chunk in chunks]

    doc_id, chunk_by_section_id, order_by_section_id = (
        _validate_canonical_chunks(canonical_chunks)
    )

    source_chunks = [
        chunk
        for chunk in canonical_chunks
        if bool(chunk.get("embed"))
    ]

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for source_chunk in source_chunks:
        root_section_id = _resolve_root_section_id(
            source_chunk,
            chunk_by_section_id,
            max_level,
        )
        grouped[root_section_id].append(source_chunk)

    ordered_root_ids = sorted(
        grouped,
        key=lambda section_id: order_by_section_id[section_id],
    )

    units = [
        _build_unit(
            all_chunks=canonical_chunks,
            root_chunk=chunk_by_section_id[root_section_id],
            source_chunks=grouped[root_section_id],
            chunk_by_section_id=chunk_by_section_id,
            max_level=max_level,
            include_descendant_titles=include_descendant_titles,
            include_section_ids_in_titles=include_section_ids_in_titles,
        )
        for root_section_id in ordered_root_ids
    ]

    if validate:
        report = validate_retrieval_units(
            chunks=canonical_chunks,
            retrieval_units=units,
            max_level=max_level,
        )
        if not report["valid"]:
            raise ValueError(
                "Retrieval-unit validation failed: "
                + "; ".join(report["errors"])
            )

    logger.info(
        "Built %d retrieval units for %s from %d embeddable chunks "
        "(strategy=%s)",
        len(units),
        doc_id,
        len(source_chunks),
        _strategy_name(max_level),
    )
    return units


def validate_retrieval_units(
    chunks: Sequence[Mapping[str, Any]],
    retrieval_units: Sequence[Mapping[str, Any]],
    max_level: Optional[int],
) -> Dict[str, Any]:
    """Check coverage, uniqueness, hierarchy assignment, and TOC order."""
    _validate_max_level(max_level)
    errors: List[str] = []

    canonical_chunks = [dict(chunk) for chunk in chunks]
    try:
        doc_id, chunk_by_section_id, _ = _validate_canonical_chunks(
            canonical_chunks
        )
    except (TypeError, ValueError) as exc:
        return {"valid": False, "errors": [str(exc)]}

    expected_sources = [
        chunk
        for chunk in canonical_chunks
        if bool(chunk.get("embed"))
    ]
    expected_source_ids = [
        str(chunk["chunk_id"])
        for chunk in expected_sources
    ]
    source_by_chunk_id = {
        str(chunk["chunk_id"]): chunk
        for chunk in expected_sources
    }

    unit_ids: List[str] = []
    root_chunk_ids: List[str] = []
    observed_source_ids: List[str] = []

    for index, unit in enumerate(retrieval_units):
        unit_id = str(unit.get("retrieval_unit_id") or "")
        root_section_id = str(unit.get("root_section_id") or "")
        root_chunk_id = str(unit.get("root_chunk_id") or "")
        unit_ids.append(unit_id)
        root_chunk_ids.append(root_chunk_id)

        if not unit_id:
            errors.append(f"Unit at index {index} has no retrieval_unit_id")
        if not bool(unit.get("embed")):
            errors.append(f"Unit is not embeddable: {unit_id}")
        if bool(unit.get("excluded")):
            errors.append(f"Unit is excluded: {unit_id}")
        if bool(unit.get("is_empty")):
            errors.append(f"Unit is marked empty: {unit_id}")
        if not str(unit.get("text") or "").strip():
            errors.append(f"Unit has no text: {unit_id}")

        root_chunk = chunk_by_section_id.get(root_section_id)
        if root_chunk is None:
            errors.append(f"Unknown root section in {unit_id}")
            continue

        source_ids = [
            str(value)
            for value in (unit.get("source_chunk_ids") or [])
        ]
        if not source_ids:
            errors.append(f"Unit has no source chunks: {unit_id}")
        observed_source_ids.extend(source_ids)

        for source_id in source_ids:
            source = source_by_chunk_id.get(source_id)
            if source is None:
                errors.append(
                    f"Unknown/non-embeddable source {source_id} in {unit_id}"
                )
                continue
            expected_root = _resolve_root_section_id(
                source,
                chunk_by_section_id,
                max_level,
            )
            if expected_root != root_section_id:
                errors.append(
                    f"Source {source_id} assigned to {root_section_id}, "
                    f"expected {expected_root}"
                )

    if len(unit_ids) != len(set(unit_ids)):
        errors.append("Duplicate retrieval_unit_id values")
    if len(root_chunk_ids) != len(set(root_chunk_ids)):
        errors.append("Duplicate root_chunk_id values")
    if len(observed_source_ids) != len(set(observed_source_ids)):
        errors.append("A source chunk appears in multiple retrieval units")

    missing = sorted(set(expected_source_ids) - set(observed_source_ids))
    extra = sorted(set(observed_source_ids) - set(expected_source_ids))
    if missing:
        errors.append(f"Missing source chunks: {missing[:20]}")
    if extra:
        errors.append(f"Unexpected source chunks: {extra[:20]}")
    if observed_source_ids != expected_source_ids:
        errors.append("Source order does not match canonical TOC order")

    if max_level is None:
        if len(retrieval_units) != len(expected_sources):
            errors.append(
                "max_level=None must produce one unit per embeddable chunk"
            )
        if any(bool(unit.get("is_aggregated")) for unit in retrieval_units):
            errors.append("max_level=None produced an aggregated unit")

    return {
        "valid": not errors,
        "errors": errors,
        "doc_id": doc_id,
        "strategy": _strategy_name(max_level),
        "max_level": max_level,
        "canonical_chunk_count": len(canonical_chunks),
        "embeddable_source_chunk_count": len(expected_sources),
        "retrieval_unit_count": len(retrieval_units),
        "aggregated_retrieval_unit_count": sum(
            1
            for unit in retrieval_units
            if bool(unit.get("is_aggregated"))
        ),
        "source_coverage_complete": (
            set(observed_source_ids) == set(expected_source_ids)
        ),
        "source_order_preserved": observed_source_ids == expected_source_ids,
    }


def retrieval_units_output_path(
    output_dir: Path | str,
    doc_id: str,
    max_level: Optional[int],
) -> Path:
    _validate_max_level(max_level)
    return (
        Path(output_dir)
        / f"{doc_id}_retrieval_units_{_strategy_suffix(max_level)}.json"
    )


def retrieval_units_validation_path(output_path: Path | str) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(
        f"{output_path.stem}_validation.json"
    )


def build_retrieval_units_file(
    input_path: Path | str,
    output_dir: Path | str,
    max_level: Optional[int] = None,
    include_descendant_titles: bool = True,
    include_section_ids_in_titles: bool = True,
    force: bool = False,
    write_validation: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """Load canonical chunks, build units, write units and validation."""
    input_path = Path(input_path)
    chunks = load_hierarchical_chunks(input_path)

    doc_ids = {
        str(chunk.get("doc_id"))
        for chunk in chunks
        if chunk.get("doc_id") is not None
    }
    if len(doc_ids) != 1:
        raise ValueError(f"Could not determine one doc_id from {input_path}")
    doc_id = next(iter(doc_ids))

    output_path = retrieval_units_output_path(
        output_dir=output_dir,
        doc_id=doc_id,
        max_level=max_level,
    )
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --force to replace it."
        )

    units = build_retrieval_units(
        chunks=chunks,
        max_level=max_level,
        include_descendant_titles=include_descendant_titles,
        include_section_ids_in_titles=include_section_ids_in_titles,
        validate=True,
    )
    report = validate_retrieval_units(chunks, units, max_level)

    # Plain list, matching the current hierarchical-chunk artifact style.
    write_json_atomic(units, output_path)
    if write_validation:
        write_json_atomic(
            report,
            retrieval_units_validation_path(output_path),
        )

    return output_path, report


def _parse_max_level(raw_value: str) -> Optional[int]:
    normalized = raw_value.strip().lower()
    if normalized in {"none", "null", ""}:
        return None

    try:
        value = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--max-level must be none/null or an integer >= 1"
        ) from exc

    if value < 1:
        raise argparse.ArgumentTypeError(
            "--max-level must be none/null or an integer >= 1"
        )
    return value


def _default_output_dir(input_path: Path) -> Path:
    # Expected layout:
    #   <work_root>/chunks/<doc>_hier_chunks.json
    #   <work_root>/retrieval_units/
    if input_path.parent.name == "chunks":
        return input_path.parent.parent / "retrieval_units"
    return input_path.parent / "retrieval_units"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build retrieval units from *_hier_chunks.json."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-level",
        type=_parse_max_level,
        default=None,
        help="none for baseline, otherwise an integer such as 4 or 5",
    )
    parser.add_argument("--no-descendant-titles", action="store_true")
    parser.add_argument("--no-section-ids-in-titles", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-validation-file", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    input_path = args.input.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_output_dir(input_path).resolve()
    )

    output_path, report = build_retrieval_units_file(
        input_path=input_path,
        output_dir=output_dir,
        max_level=args.max_level,
        include_descendant_titles=not args.no_descendant_titles,
        include_section_ids_in_titles=(
            not args.no_section_ids_in_titles
        ),
        force=args.force,
        write_validation=not args.no_validation_file,
    )

    logger.info("Retrieval units written to %s", output_path)
    logger.info(
        "Validation: valid=%s | units=%d | aggregated=%d | "
        "coverage=%s | order=%s",
        report["valid"],
        report["retrieval_unit_count"],
        report["aggregated_retrieval_unit_count"],
        report["source_coverage_complete"],
        report["source_order_preserved"],
    )


if __name__ == "__main__":
    main()
