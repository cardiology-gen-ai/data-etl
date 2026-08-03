"""
Build a retrieval-oriented Section view from canonical hierarchical chunks.

The canonical ``*_hier_chunks.json`` files are never modified. The module
first derives retrieval units and then emits the exact ``Section`` records that
should be loaded by retrieval systems and by the knowledge graph.

The resulting view contains only:

1. retrievable sections, one for each effective retrieval chunk;
2. empty structural ancestors required to preserve ``HAS_CHILD`` paths.

Excluded sections are omitted. Sections absorbed by an aggregation are omitted
as nodes because their content is represented by the owner section through
``source_section_ids`` and related provenance fields.

Semantics
---------
- ``max_level=None``:
    No aggregation. Every non-excluded canonical chunk with ``embed=True``
    remains a separate retrievable Section. Empty non-excluded ancestors needed
    to connect those Sections are retained as structural Sections.

- ``max_level=N``:
    Every embeddable section with ``section_level <= N`` remains separate.
    Every embeddable section with ``section_level > N`` is merged into its
    nearest ancestor at level ``N``. The absorbed descendants are not emitted
    as nodes. Necessary ancestors above the retained retrieval roots remain as
    empty structural Sections.

All emitted records keep the ``Section`` schema expected by downstream
consumers and add provenance fields describing the active retrieval strategy.
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
    """Resolve the retrieval owner for one embeddable source chunk."""
    section_id = str(source_chunk["section_id"])
    source_level = int(source_chunk["section_level"])

    if bool(source_chunk.get("excluded")):
        raise ValueError(
            f"Excluded section cannot be a retrieval source: {section_id}"
        )

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
        if bool(parent.get("excluded")):
            raise ValueError(
                f"Retrieval source {section_id!r} is below excluded "
                f"section {parent_id!r}"
            )

        current = parent

    root_level = int(current["section_level"])
    if root_level != max_level:
        raise ValueError(
            f"Section {section_id!r} has no ancestor exactly at level "
            f"{max_level}; resolved {current['section_id']!r} at "
            f"level {root_level}"
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
        if bool(current.get("excluded")):
            raise ValueError(
                f"Path from {root_section_id!r} to "
                f"{source_section_id!r} crosses excluded section "
                f"{current_id!r}"
            )

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



def _collect_required_ancestor_ids(
    content_root_ids: Sequence[str],
    chunk_by_section_id: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """
    Return every non-excluded ancestor required to connect retained roots.

    The returned set may contain other content roots. Excluded ancestors are
    rejected because silently re-parenting around them would alter the
    canonical document hierarchy.
    """
    required: set[str] = set()

    for root_id in content_root_ids:
        current_id = str(root_id)
        visited = {current_id}

        while True:
            current = chunk_by_section_id.get(current_id)
            if current is None:
                raise ValueError(
                    f"Missing section while resolving ancestors: {current_id}"
                )

            parent_id = current.get("parent_section_id")
            if parent_id is None:
                break

            parent_id = str(parent_id)
            if parent_id in visited:
                raise ValueError(
                    f"Cycle detected while resolving ancestors of {root_id}"
                )
            visited.add(parent_id)

            parent = chunk_by_section_id.get(parent_id)
            if parent is None:
                raise ValueError(
                    f"Missing ancestor {parent_id!r} for {root_id!r}"
                )
            if bool(parent.get("excluded")):
                raise ValueError(
                    f"Retained retrieval section {root_id!r} is below "
                    f"excluded ancestor {parent_id!r}"
                )

            required.add(parent_id)
            current_id = parent_id

    return required


def _build_structural_view_section(
    canonical_chunk: Mapping[str, Any],
    max_level: Optional[int],
) -> Dict[str, Any]:
    """Create one empty structural Section for the derived view."""
    section_id = str(canonical_chunk["section_id"])
    chunk_id = str(canonical_chunk["chunk_id"])
    strategy = _strategy_name(max_level)
    canonical_text = str(canonical_chunk.get("text") or "")

    record = dict(canonical_chunk)
    record.update(
        {
            "text": "",
            "is_empty": True,
            "excluded": False,
            "embed": False,
            "retrieval_unit_id": None,
            "retrieval_strategy": strategy,
            "aggregation_mode": (
                "none" if max_level is None else "merge_below_level"
            ),
            "aggregation_max_level": max_level,
            "is_aggregated": False,
            "root_has_local_text": False,
            "root_section_id": section_id,
            "root_chunk_id": chunk_id,
            "root_page_start": canonical_chunk.get("page_start"),
            "root_page_end": canonical_chunk.get("page_end"),
            "root_quality_flags": list(
                canonical_chunk.get("quality_flags") or []
            ),
            "source_section_ids": [],
            "source_chunk_ids": [],
            "source_sections": [],
            "represented_section_ids": [],
            "structural_context_section_ids": [],
            "source_count": 0,
            "represented_section_count": 0,
            "text_chars": 0,
            "text_words": 0,
            "source_text_chars": 0,
            "section_view_role": "structural",
            "content_owner_section_id": None,
            "absorbed_section_ids": [],
            "absorbed_source_section_ids": [],
            "canonical_text_chars": len(canonical_text),
            "canonical_is_empty": bool(
                canonical_chunk.get("is_empty")
            ),
            "canonical_embed": bool(canonical_chunk.get("embed")),
        }
    )
    return record


def _build_retrieval_view_section(
    unit: Mapping[str, Any],
    canonical_root: Mapping[str, Any],
) -> Dict[str, Any]:
    """Enrich one retrieval unit as the retained content Section."""
    root_section_id = str(unit["root_section_id"])
    represented = [
        str(value)
        for value in (unit.get("represented_section_ids") or [])
    ]
    source_ids = [
        str(value)
        for value in (unit.get("source_section_ids") or [])
    ]

    record = dict(canonical_root)
    record.update(dict(unit))
    record.update(
        {
            "section_view_role": "retrieval",
            "content_owner_section_id": root_section_id,
            "absorbed_section_ids": [
                section_id
                for section_id in represented
                if section_id != root_section_id
            ],
            "absorbed_source_section_ids": [
                section_id
                for section_id in source_ids
                if section_id != root_section_id
            ],
            "canonical_text_chars": len(
                str(canonical_root.get("text") or "")
            ),
            "canonical_is_empty": bool(
                canonical_root.get("is_empty")
            ),
            "canonical_embed": bool(canonical_root.get("embed")),
        }
    )
    return record


def build_retrieval_section_view(
    chunks: Sequence[Mapping[str, Any]],
    max_level: Optional[int] = None,
    include_descendant_titles: bool = True,
    include_section_ids_in_titles: bool = True,
    validate: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build the complete Section view used by retrieval and graph loading.

    The output contains one content Section per effective retrieval unit plus
    only the empty ancestors required to preserve parent-child paths.
    Excluded sections and absorbed descendants are omitted.
    """
    _validate_max_level(max_level)
    canonical_chunks = [dict(chunk) for chunk in chunks]

    doc_id, chunk_by_section_id, _ = _validate_canonical_chunks(
        canonical_chunks
    )

    units = build_retrieval_units(
        chunks=canonical_chunks,
        max_level=max_level,
        include_descendant_titles=include_descendant_titles,
        include_section_ids_in_titles=include_section_ids_in_titles,
        validate=True,
    )
    unit_by_root_id = {
        str(unit["root_section_id"]): unit
        for unit in units
    }
    content_root_ids = list(unit_by_root_id)

    required_ancestor_ids = _collect_required_ancestor_ids(
        content_root_ids=content_root_ids,
        chunk_by_section_id=chunk_by_section_id,
    )
    retained_ids = set(content_root_ids) | required_ancestor_ids

    section_view: List[Dict[str, Any]] = []

    for canonical_chunk in canonical_chunks:
        section_id = str(canonical_chunk["section_id"])

        if section_id not in retained_ids:
            continue
        if bool(canonical_chunk.get("excluded")):
            raise ValueError(
                f"Excluded section unexpectedly retained: {section_id}"
            )

        unit = unit_by_root_id.get(section_id)
        if unit is not None:
            section_view.append(
                _build_retrieval_view_section(
                    unit=unit,
                    canonical_root=canonical_chunk,
                )
            )
        else:
            section_view.append(
                _build_structural_view_section(
                    canonical_chunk=canonical_chunk,
                    max_level=max_level,
                )
            )

    if validate:
        report = validate_retrieval_section_view(
            chunks=canonical_chunks,
            section_view=section_view,
            max_level=max_level,
        )
        if not report["valid"]:
            raise ValueError(
                "Retrieval Section-view validation failed: "
                + "; ".join(report["errors"])
            )

    logger.info(
        "Built Section view for %s: sections=%d, retrieval=%d, "
        "structural=%d, strategy=%s",
        doc_id,
        len(section_view),
        sum(
            1
            for section in section_view
            if section.get("section_view_role") == "retrieval"
        ),
        sum(
            1
            for section in section_view
            if section.get("section_view_role") == "structural"
        ),
        _strategy_name(max_level),
    )
    return section_view


def validate_retrieval_section_view(
    chunks: Sequence[Mapping[str, Any]],
    section_view: Sequence[Mapping[str, Any]],
    max_level: Optional[int],
) -> Dict[str, Any]:
    """
    Validate coverage, parent closure, pruning, ordering, and node roles.
    """
    _validate_max_level(max_level)
    errors: List[str] = []

    canonical_chunks = [dict(chunk) for chunk in chunks]
    try:
        doc_id, chunk_by_section_id, order_by_section_id = (
            _validate_canonical_chunks(canonical_chunks)
        )
    except (TypeError, ValueError) as exc:
        return {"valid": False, "errors": [str(exc)]}

    expected_sources = [
        chunk
        for chunk in canonical_chunks
        if bool(chunk.get("embed"))
    ]

    source_to_owner: Dict[str, str] = {}
    owner_to_sources: Dict[str, List[str]] = defaultdict(list)

    for source in expected_sources:
        source_section_id = str(source["section_id"])
        try:
            owner_id = _resolve_root_section_id(
                source_chunk=source,
                chunk_by_section_id=chunk_by_section_id,
                max_level=max_level,
            )
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue

        source_to_owner[source_section_id] = owner_id
        owner_to_sources[owner_id].append(source_section_id)

    expected_root_ids: List[str] = []
    seen_roots: set[str] = set()
    for source in expected_sources:
        source_section_id = str(source["section_id"])
        owner_id = source_to_owner.get(source_section_id)
        if owner_id is None or owner_id in seen_roots:
            continue
        seen_roots.add(owner_id)
        expected_root_ids.append(owner_id)

    try:
        required_ancestor_ids = _collect_required_ancestor_ids(
            content_root_ids=expected_root_ids,
            chunk_by_section_id=chunk_by_section_id,
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        required_ancestor_ids = set()

    expected_root_set = set(expected_root_ids)
    expected_structural_ids = required_ancestor_ids - expected_root_set
    expected_view_ids = expected_root_set | required_ancestor_ids

    expected_ordered_view_ids = [
        str(chunk["section_id"])
        for chunk in canonical_chunks
        if str(chunk["section_id"]) in expected_view_ids
    ]

    observed_section_ids: List[str] = []
    observed_chunk_ids: List[str] = []
    observed_retrieval_ids: List[str] = []
    observed_structural_ids: List[str] = []
    retrieval_sections: List[Mapping[str, Any]] = []

    for index, section in enumerate(section_view):
        section_id = str(section.get("section_id") or "")
        chunk_id = str(section.get("chunk_id") or "")
        role = str(section.get("section_view_role") or "")

        observed_section_ids.append(section_id)
        observed_chunk_ids.append(chunk_id)

        if not section_id:
            errors.append(f"Section-view record at index {index} has no id")
            continue

        canonical = chunk_by_section_id.get(section_id)
        if canonical is None:
            errors.append(f"Unknown Section-view id: {section_id}")
            continue

        if bool(section.get("excluded")):
            errors.append(
                f"Excluded section present in Section view: {section_id}"
            )
        if str(section.get("retrieval_strategy")) != _strategy_name(
            max_level
        ):
            errors.append(
                f"Wrong retrieval strategy on section {section_id}"
            )

        parent_id = section.get("parent_section_id")
        if parent_id is not None:
            parent_id = str(parent_id)
            if parent_id not in expected_view_ids:
                errors.append(
                    f"Section {section_id} has missing retained parent "
                    f"{parent_id}"
                )

        if max_level is not None:
            level = int(section.get("section_level") or 0)
            if level > max_level:
                errors.append(
                    f"Section {section_id} exceeds max_level={max_level}"
                )

        if role == "retrieval":
            observed_retrieval_ids.append(section_id)
            retrieval_sections.append(section)

            if not bool(section.get("embed")):
                errors.append(
                    f"Retrieval Section is not embeddable: {section_id}"
                )
            if bool(section.get("is_empty")):
                errors.append(
                    f"Retrieval Section is marked empty: {section_id}"
                )
            if not str(section.get("text") or "").strip():
                errors.append(
                    f"Retrieval Section has no text: {section_id}"
                )
            if str(section.get("root_section_id") or "") != section_id:
                errors.append(
                    f"Retrieval Section root mismatch: {section_id}"
                )
            if str(
                section.get("content_owner_section_id") or ""
            ) != section_id:
                errors.append(
                    f"Content owner mismatch on Section {section_id}"
                )

        elif role == "structural":
            observed_structural_ids.append(section_id)

            if bool(section.get("embed")):
                errors.append(
                    f"Structural Section is embeddable: {section_id}"
                )
            if not bool(section.get("is_empty")):
                errors.append(
                    f"Structural Section is not empty: {section_id}"
                )
            if str(section.get("text") or ""):
                errors.append(
                    f"Structural Section contains text: {section_id}"
                )
            if section.get("source_section_ids"):
                errors.append(
                    f"Structural Section has source ids: {section_id}"
                )

        else:
            errors.append(
                f"Unknown section_view_role={role!r} on {section_id}"
            )

    if len(observed_section_ids) != len(set(observed_section_ids)):
        errors.append("Duplicate section_id values in Section view")
    if len(observed_chunk_ids) != len(set(observed_chunk_ids)):
        errors.append("Duplicate chunk_id values in Section view")

    if observed_section_ids != expected_ordered_view_ids:
        errors.append("Section-view order does not match canonical TOC order")

    if set(observed_retrieval_ids) != expected_root_set:
        missing = sorted(expected_root_set - set(observed_retrieval_ids))
        extra = sorted(set(observed_retrieval_ids) - expected_root_set)
        if missing:
            errors.append(
                f"Missing retrieval Sections: {missing[:20]}"
            )
        if extra:
            errors.append(
                f"Unexpected retrieval Sections: {extra[:20]}"
            )

    if set(observed_structural_ids) != expected_structural_ids:
        missing = sorted(
            expected_structural_ids - set(observed_structural_ids)
        )
        extra = sorted(
            set(observed_structural_ids) - expected_structural_ids
        )
        if missing:
            errors.append(
                f"Missing structural Sections: {missing[:20]}"
            )
        if extra:
            errors.append(
                f"Unexpected structural Sections: {extra[:20]}"
            )

    observed_id_set = set(observed_section_ids)
    for section in section_view:
        parent_id = section.get("parent_section_id")
        if parent_id is None:
            continue
        if str(parent_id) not in observed_id_set:
            errors.append(
                f"Parent closure failed for {section.get('section_id')}: "
                f"{parent_id}"
            )

    retrieval_report = validate_retrieval_units(
        chunks=canonical_chunks,
        retrieval_units=retrieval_sections,
        max_level=max_level,
    )
    errors.extend(
        f"Retrieval-unit check: {message}"
        for message in retrieval_report.get("errors", [])
    )

    absorbed_source_ids = {
        source_id
        for source_id, owner_id in source_to_owner.items()
        if source_id != owner_id
    }
    absorbed_node_ids = {
        str(section_id)
        for section in retrieval_sections
        for section_id in (
            section.get("absorbed_section_ids") or []
        )
    }

    absorbed_sources_still_present = sorted(
        absorbed_source_ids & observed_id_set
    )
    if absorbed_sources_still_present:
        errors.append(
            "Absorbed source Sections are still present as nodes: "
            f"{absorbed_sources_still_present[:20]}"
        )

    excluded_ids = {
        str(chunk["section_id"])
        for chunk in canonical_chunks
        if bool(chunk.get("excluded"))
    }
    excluded_present = sorted(excluded_ids & observed_id_set)
    if excluded_present:
        errors.append(
            f"Excluded Sections present in view: {excluded_present[:20]}"
        )

    if max_level is None:
        if absorbed_source_ids:
            errors.append(
                "max_level=None unexpectedly absorbed source Sections"
            )
        if any(
            bool(section.get("is_aggregated"))
            for section in retrieval_sections
        ):
            errors.append(
                "max_level=None produced aggregated retrieval Sections"
            )

    non_excluded_ids = {
        str(chunk["section_id"])
        for chunk in canonical_chunks
        if not bool(chunk.get("excluded"))
    }
    pruned_non_required_ids = (
        non_excluded_ids
        - observed_id_set
        - absorbed_node_ids
    )

    return {
        "valid": not errors,
        "errors": errors,
        "doc_id": doc_id,
        "strategy": _strategy_name(max_level),
        "max_level": max_level,
        "canonical_chunk_count": len(canonical_chunks),
        "canonical_excluded_chunk_count": len(excluded_ids),
        "embeddable_source_chunk_count": len(expected_sources),
        "section_view_count": len(section_view),
        "retrievable_section_count": len(retrieval_sections),
        "structural_section_count": len(observed_structural_ids),
        "aggregated_retrievable_section_count": sum(
            1
            for section in retrieval_sections
            if bool(section.get("is_aggregated"))
        ),
        "absorbed_source_section_count": len(absorbed_source_ids),
        "absorbed_node_count": len(absorbed_node_ids),
        "pruned_non_required_section_count": len(
            pruned_non_required_ids
        ),
        "source_coverage_complete": bool(
            retrieval_report.get("source_coverage_complete")
        ),
        "source_order_preserved": bool(
            retrieval_report.get("source_order_preserved")
        ),
        "toc_order_preserved": (
            observed_section_ids == expected_ordered_view_ids
        ),
        "parent_closure_complete": all(
            section.get("parent_section_id") is None
            or str(section.get("parent_section_id")) in observed_id_set
            for section in section_view
        ),
        "source_to_owner_section_id": source_to_owner,
        "owner_to_source_section_ids": {
            owner_id: source_ids
            for owner_id, source_ids in owner_to_sources.items()
        },
        "pruned_non_required_section_ids": sorted(
            pruned_non_required_ids,
            key=lambda section_id: order_by_section_id[section_id],
        ),
    }


def section_view_output_path(
    output_dir: Path | str,
    doc_id: str,
    max_level: Optional[int],
) -> Path:
    _validate_max_level(max_level)
    return (
        Path(output_dir)
        / f"{doc_id}_section_view_{_strategy_suffix(max_level)}.json"
    )


def section_view_validation_path(output_path: Path | str) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(
        f"{output_path.stem}_validation.json"
    )


def build_retrieval_section_view_file(
    input_path: Path | str,
    output_dir: Path | str,
    max_level: Optional[int] = None,
    include_descendant_titles: bool = True,
    include_section_ids_in_titles: bool = True,
    force: bool = False,
    write_validation: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """Load canonical chunks, build the Section view, and write its audit."""
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

    output_path = section_view_output_path(
        output_dir=output_dir,
        doc_id=doc_id,
        max_level=max_level,
    )
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use --force to replace it."
        )

    section_view = build_retrieval_section_view(
        chunks=chunks,
        max_level=max_level,
        include_descendant_titles=include_descendant_titles,
        include_section_ids_in_titles=include_section_ids_in_titles,
        validate=True,
    )
    report = validate_retrieval_section_view(
        chunks=chunks,
        section_view=section_view,
        max_level=max_level,
    )

    write_json_atomic(section_view, output_path)
    if write_validation:
        write_json_atomic(
            report,
            section_view_validation_path(output_path),
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
    #   <work_root>/section_views/
    if input_path.parent.name == "chunks":
        return input_path.parent.parent / "section_views"
    return input_path.parent / "section_views"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the retrieval-oriented Section view from "
            "*_hier_chunks.json."
        )
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
    parser.add_argument(
        "--no-section-ids-in-titles",
        action="store_true",
    )
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

    output_path, report = build_retrieval_section_view_file(
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

    logger.info("Section view written to %s", output_path)
    logger.info(
        "Validation: valid=%s | sections=%d | retrieval=%d | "
        "structural=%d | aggregated=%d | coverage=%s | order=%s",
        report["valid"],
        report["section_view_count"],
        report["retrievable_section_count"],
        report["structural_section_count"],
        report["aggregated_retrievable_section_count"],
        report["source_coverage_complete"],
        report["source_order_preserved"],
    )


if __name__ == "__main__":
    main()
