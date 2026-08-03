from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

from neo4j import Driver

from knowledge_graph.relationship_metadata import (
    build_structural_relationship_metadata,
)


logger = logging.getLogger(__name__)

DEFAULT_MIN_TEXT_CHARS_TO_EMBED = int(
    os.getenv("MIN_TEXT_CHARS_TO_EMBED", "20")
)

SECTION_VIEW_SCHEMA_VERSION = "1"
RETRIEVAL_ROLE = "retrieval"
STRUCTURAL_ROLE = "structural"
VALID_SECTION_VIEW_ROLES = {RETRIEVAL_ROLE, STRUCTURAL_ROLE}

T = TypeVar("T")


def chunked(items: Sequence[T], batch_size: int) -> Iterable[List[T]]:
    """Yield successive batches from a sequence."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    for i in range(0, len(items), batch_size):
        yield list(items[i:i + batch_size])


def make_section_uid(doc_id: str, section_id: str) -> str:
    """Build the globally unique Section UID."""
    return f"{doc_id}::{section_id}"


def infer_should_embed(
    section: Dict[str, Any],
    min_text_chars_to_embed: int = DEFAULT_MIN_TEXT_CHARS_TO_EMBED,
) -> bool:
    """
    Decide whether a Section is eligible for embeddings.

    A valid Section view always provides an explicit boolean ``embed`` value.
    The text-length fallback is retained only for API compatibility with older
    callers and is not used after Section-view validation.
    """
    embed_flag = section.get("embed")
    if embed_flag is not None:
        return bool(embed_flag)

    if section.get("is_empty") is True:
        return False

    text = str(section.get("text") or "").strip()
    return len(text) >= min_text_chars_to_embed


def _non_empty_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _validate_string_list(
    section: Dict[str, Any],
    field_name: str,
    *,
    record_label: str,
    errors: List[str],
) -> List[str]:
    value = section.get(field_name, [])

    if value is None:
        value = []

    if not isinstance(value, list):
        errors.append(
            f"{record_label}: {field_name} must be a list, "
            f"got {type(value).__name__}"
        )
        return []

    normalized: List[str] = []
    for position, item in enumerate(value):
        item_value = _non_empty_string(item)
        if item_value is None:
            errors.append(
                f"{record_label}: {field_name}[{position}] must be a "
                "non-empty string"
            )
            continue
        normalized.append(item_value)

    if len(normalized) != len(set(normalized)):
        errors.append(
            f"{record_label}: {field_name} contains duplicate values"
        )

    return normalized


def validate_section_view_records(
    sections: List[Dict[str, Any]],
    *,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Validate a retrieval-oriented Section view before any Neo4j write.

    The loader intentionally rejects canonical chunk files. Every record must
    explicitly declare ``section_view_role`` as either ``retrieval`` or
    ``structural`` and must satisfy the invariants produced by
    ``retrieval_unit_manager.py``.

    Returns a compact summary when valid; raises ``ValueError`` otherwise.
    """
    source_label = str(source_path) if source_path is not None else "<memory>"

    if not isinstance(sections, list):
        raise ValueError(
            f"Section view must contain a JSON list: {source_label}"
        )

    if not sections:
        raise ValueError(f"Section view is empty: {source_label}")

    errors: List[str] = []

    doc_ids: set[str] = set()
    strategies: set[str] = set()
    aggregation_modes: set[str] = set()
    aggregation_max_levels: set[Optional[int]] = set()
    schema_versions: set[str] = set()

    seen_section_ids: Dict[str, int] = {}
    seen_chunk_ids: Dict[str, int] = {}
    seen_retrieval_unit_ids: Dict[str, int] = {}

    record_metadata: List[Dict[str, Any]] = []
    source_owner: Dict[str, str] = {}

    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            errors.append(
                f"record[{index}]: expected an object, "
                f"got {type(section).__name__}"
            )
            continue

        section_id = _non_empty_string(section.get("section_id"))
        record_label = (
            f"record[{index}]"
            if section_id is None
            else f"record[{index}] section_id={section_id!r}"
        )

        doc_id = _non_empty_string(section.get("doc_id"))
        if doc_id is None:
            errors.append(f"{record_label}: missing non-empty doc_id")
        else:
            doc_ids.add(doc_id)

        if section_id is None:
            errors.append(f"{record_label}: missing non-empty section_id")
        else:
            previous_index = seen_section_ids.get(section_id)
            if previous_index is not None:
                errors.append(
                    f"{record_label}: duplicate section_id; first seen at "
                    f"record[{previous_index}]"
                )
            else:
                seen_section_ids[section_id] = index

        chunk_id = _non_empty_string(section.get("chunk_id"))
        if chunk_id is None:
            errors.append(f"{record_label}: missing non-empty chunk_id")
        else:
            previous_index = seen_chunk_ids.get(chunk_id)
            if previous_index is not None:
                errors.append(
                    f"{record_label}: duplicate chunk_id; first seen at "
                    f"record[{previous_index}]"
                )
            else:
                seen_chunk_ids[chunk_id] = index

        if section.get("excluded") is not False:
            errors.append(
                f"{record_label}: excluded must be explicitly false"
            )

        role = _non_empty_string(section.get("section_view_role"))
        if role not in VALID_SECTION_VIEW_ROLES:
            errors.append(
                f"{record_label}: section_view_role must be one of "
                f"{sorted(VALID_SECTION_VIEW_ROLES)}, got {role!r}"
            )

        strategy = _non_empty_string(section.get("retrieval_strategy"))
        if strategy is None:
            errors.append(
                f"{record_label}: missing non-empty retrieval_strategy"
            )
        else:
            strategies.add(strategy)

        aggregation_mode = _non_empty_string(
            section.get("aggregation_mode")
        )
        if aggregation_mode is None:
            errors.append(
                f"{record_label}: missing non-empty aggregation_mode"
            )
        else:
            aggregation_modes.add(aggregation_mode)

        aggregation_max_level = section.get("aggregation_max_level")
        if aggregation_max_level is not None and (
            isinstance(aggregation_max_level, bool)
            or not isinstance(aggregation_max_level, int)
            or aggregation_max_level <= 0
        ):
            errors.append(
                f"{record_label}: aggregation_max_level must be null or a "
                "positive integer"
            )
        else:
            aggregation_max_levels.add(aggregation_max_level)

        schema_version_raw = section.get(
            "section_view_schema_version",
            SECTION_VIEW_SCHEMA_VERSION,
        )
        schema_version = (
            None
            if schema_version_raw is None
            else _non_empty_string(str(schema_version_raw))
        )
        if schema_version is None:
            errors.append(
                f"{record_label}: section_view_schema_version cannot be empty"
            )
        else:
            schema_versions.add(schema_version)

        level = section.get("section_level")
        if (
            isinstance(level, bool)
            or not isinstance(level, int)
            or level <= 0
        ):
            errors.append(
                f"{record_label}: section_level must be a positive integer"
            )

        parent_section_id = section.get("parent_section_id")
        if parent_section_id is not None:
            parent_section_id = _non_empty_string(parent_section_id)
            if parent_section_id is None:
                errors.append(
                    f"{record_label}: parent_section_id must be null or a "
                    "non-empty string"
                )

        text = section.get("text")
        if not isinstance(text, str):
            errors.append(f"{record_label}: text must be a string")
            text = ""

        embed = section.get("embed")
        if not isinstance(embed, bool):
            errors.append(
                f"{record_label}: embed must be an explicit boolean"
            )

        is_empty = section.get("is_empty")
        if not isinstance(is_empty, bool):
            errors.append(
                f"{record_label}: is_empty must be an explicit boolean"
            )

        is_aggregated = section.get("is_aggregated")
        if not isinstance(is_aggregated, bool):
            errors.append(
                f"{record_label}: is_aggregated must be an explicit boolean"
            )
            is_aggregated = False

        root_has_local_text = section.get("root_has_local_text")
        if not isinstance(root_has_local_text, bool):
            errors.append(
                f"{record_label}: root_has_local_text must be an explicit "
                "boolean"
            )

        canonical_is_empty = section.get("canonical_is_empty")
        if not isinstance(canonical_is_empty, bool):
            errors.append(
                f"{record_label}: canonical_is_empty must be an explicit "
                "boolean"
            )

        canonical_embed = section.get("canonical_embed")
        if not isinstance(canonical_embed, bool):
            errors.append(
                f"{record_label}: canonical_embed must be an explicit boolean"
            )

        source_section_ids = _validate_string_list(
            section,
            "source_section_ids",
            record_label=record_label,
            errors=errors,
        )
        source_chunk_ids = _validate_string_list(
            section,
            "source_chunk_ids",
            record_label=record_label,
            errors=errors,
        )
        represented_section_ids = _validate_string_list(
            section,
            "represented_section_ids",
            record_label=record_label,
            errors=errors,
        )
        structural_context_section_ids = _validate_string_list(
            section,
            "structural_context_section_ids",
            record_label=record_label,
            errors=errors,
        )
        absorbed_section_ids = _validate_string_list(
            section,
            "absorbed_section_ids",
            record_label=record_label,
            errors=errors,
        )
        absorbed_source_section_ids = _validate_string_list(
            section,
            "absorbed_source_section_ids",
            record_label=record_label,
            errors=errors,
        )
        quality_flags = _validate_string_list(
            section,
            "quality_flags",
            record_label=record_label,
            errors=errors,
        )
        root_quality_flags = _validate_string_list(
            section,
            "root_quality_flags",
            record_label=record_label,
            errors=errors,
        )

        # These values are validated here because they are stored as Neo4j
        # string-list properties during ingestion.
        del (
            structural_context_section_ids,
            quality_flags,
            root_quality_flags,
        )

        source_count = section.get("source_count")
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or source_count < 0
        ):
            errors.append(
                f"{record_label}: source_count must be a non-negative integer"
            )
        elif source_count != len(source_section_ids):
            errors.append(
                f"{record_label}: source_count={source_count} does not match "
                f"len(source_section_ids)={len(source_section_ids)}"
            )

        represented_section_count = section.get(
            "represented_section_count"
        )
        if (
            isinstance(represented_section_count, bool)
            or not isinstance(represented_section_count, int)
            or represented_section_count < 0
        ):
            errors.append(
                f"{record_label}: represented_section_count must be a "
                "non-negative integer"
            )
        elif represented_section_count != len(represented_section_ids):
            errors.append(
                f"{record_label}: represented_section_count="
                f"{represented_section_count} does not match "
                f"len(represented_section_ids)="
                f"{len(represented_section_ids)}"
            )

        for integer_field in (
            "text_chars",
            "text_words",
            "source_text_chars",
            "canonical_text_chars",
        ):
            value = section.get(integer_field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                errors.append(
                    f"{record_label}: {integer_field} must be a "
                    "non-negative integer"
                )

        text_chars = section.get("text_chars")
        if isinstance(text_chars, int) and not isinstance(text_chars, bool):
            if text_chars != len(text):
                errors.append(
                    f"{record_label}: text_chars={text_chars} does not match "
                    f"len(text)={len(text)}"
                )

        root_section_id = _non_empty_string(section.get("root_section_id"))
        if root_section_id is None:
            errors.append(
                f"{record_label}: missing non-empty root_section_id"
            )
        elif section_id is not None and root_section_id != section_id:
            errors.append(
                f"{record_label}: root_section_id must equal section_id"
            )

        content_owner_section_id = section.get(
            "content_owner_section_id"
        )
        if content_owner_section_id is not None:
            content_owner_section_id = _non_empty_string(
                content_owner_section_id
            )
            if content_owner_section_id is None:
                errors.append(
                    f"{record_label}: content_owner_section_id must be null "
                    "or a non-empty string"
                )

        retrieval_unit_id = section.get("retrieval_unit_id")
        if retrieval_unit_id is not None:
            retrieval_unit_id = _non_empty_string(retrieval_unit_id)
            if retrieval_unit_id is None:
                errors.append(
                    f"{record_label}: retrieval_unit_id must be null or a "
                    "non-empty string"
                )

        if role == RETRIEVAL_ROLE:
            if not text.strip():
                errors.append(
                    f"{record_label}: retrieval Section must have non-empty text"
                )
            if embed is not True:
                errors.append(
                    f"{record_label}: retrieval Section must have embed=true"
                )
            if is_empty is not False:
                errors.append(
                    f"{record_label}: retrieval Section must have "
                    "is_empty=false"
                )
            if not source_section_ids:
                errors.append(
                    f"{record_label}: retrieval Section must represent at "
                    "least one source_section_id"
                )
            if len(source_chunk_ids) != len(source_section_ids):
                errors.append(
                    f"{record_label}: source_chunk_ids and "
                    "source_section_ids must have the same length"
                )
            if section_id is not None and (
                section_id not in represented_section_ids
            ):
                errors.append(
                    f"{record_label}: represented_section_ids must include "
                    "the owner section_id"
                )
            if not set(source_section_ids).issubset(
                set(represented_section_ids)
            ):
                errors.append(
                    f"{record_label}: every source_section_id must also be "
                    "represented"
                )
            if (
                content_owner_section_id is None
                or content_owner_section_id != section_id
            ):
                errors.append(
                    f"{record_label}: content_owner_section_id must equal "
                    "section_id for retrieval Sections"
                )
            if retrieval_unit_id is None:
                errors.append(
                    f"{record_label}: retrieval Section must have a "
                    "retrieval_unit_id"
                )
            else:
                previous_index = seen_retrieval_unit_ids.get(
                    retrieval_unit_id
                )
                if previous_index is not None:
                    errors.append(
                        f"{record_label}: duplicate retrieval_unit_id; first "
                        f"seen at record[{previous_index}]"
                    )
                else:
                    seen_retrieval_unit_ids[retrieval_unit_id] = index

            if is_aggregated:
                if not absorbed_section_ids:
                    errors.append(
                        f"{record_label}: aggregated retrieval Section must "
                        "have absorbed_section_ids"
                    )
                if not set(absorbed_source_section_ids).issubset(
                    set(source_section_ids)
                ):
                    errors.append(
                        f"{record_label}: absorbed_source_section_ids must be "
                        "a subset of source_section_ids"
                    )
                if not set(absorbed_section_ids).issubset(
                    set(represented_section_ids)
                ):
                    errors.append(
                        f"{record_label}: absorbed_section_ids must be a "
                        "subset of represented_section_ids"
                    )
            else:
                if absorbed_section_ids or absorbed_source_section_ids:
                    errors.append(
                        f"{record_label}: non-aggregated retrieval Section "
                        "cannot contain absorbed IDs"
                    )
                if section_id is not None and source_section_ids != [section_id]:
                    errors.append(
                        f"{record_label}: non-aggregated retrieval Section "
                        "must map only its own section_id"
                    )
                if (
                    section_id is not None
                    and represented_section_ids != [section_id]
                ):
                    errors.append(
                        f"{record_label}: non-aggregated retrieval Section "
                        "must represent only its own section_id"
                    )

            if section_id is not None:
                for source_section_id in source_section_ids:
                    previous_owner = source_owner.get(source_section_id)
                    if (
                        previous_owner is not None
                        and previous_owner != section_id
                    ):
                        errors.append(
                            f"{record_label}: source section "
                            f"{source_section_id!r} is already owned by "
                            f"{previous_owner!r}"
                        )
                    else:
                        source_owner[source_section_id] = section_id

        elif role == STRUCTURAL_ROLE:
            if text.strip():
                errors.append(
                    f"{record_label}: structural Section must have empty text"
                )
            if embed is not False:
                errors.append(
                    f"{record_label}: structural Section must have embed=false"
                )
            if is_empty is not True:
                errors.append(
                    f"{record_label}: structural Section must have "
                    "is_empty=true"
                )
            if is_aggregated is not False:
                errors.append(
                    f"{record_label}: structural Section cannot be aggregated"
                )
            if retrieval_unit_id is not None:
                errors.append(
                    f"{record_label}: structural Section cannot have a "
                    "retrieval_unit_id"
                )
            if content_owner_section_id is not None:
                errors.append(
                    f"{record_label}: structural Section cannot have a "
                    "content_owner_section_id"
                )
            if (
                source_section_ids
                or source_chunk_ids
                or represented_section_ids
                or absorbed_section_ids
                or absorbed_source_section_ids
            ):
                errors.append(
                    f"{record_label}: structural Section cannot contain "
                    "retrieval provenance IDs"
                )

        record_metadata.append(
            {
                "index": index,
                "section_id": section_id,
                "parent_section_id": parent_section_id,
                "level": level,
                "role": role,
                "is_aggregated": bool(is_aggregated),
                "absorbed_section_ids": absorbed_section_ids,
            }
        )

    if len(doc_ids) != 1:
        errors.append(
            f"Section view must contain exactly one doc_id, got "
            f"{sorted(doc_ids)}"
        )

    if len(strategies) != 1:
        errors.append(
            f"Section view must contain exactly one retrieval_strategy, got "
            f"{sorted(strategies)}"
        )

    if len(aggregation_modes) != 1:
        errors.append(
            f"Section view must contain exactly one aggregation_mode, got "
            f"{sorted(aggregation_modes)}"
        )

    if len(aggregation_max_levels) != 1:
        sorted_max_levels = sorted(
            aggregation_max_levels,
            key=lambda value: (
                value is not None,
                value if value is not None else -1,
            ),
        )
        errors.append(
            "Section view must contain exactly one aggregation_max_level, "
            f"got {sorted_max_levels}"
        )

    if len(schema_versions) != 1:
        errors.append(
            f"Section view must contain exactly one schema version, got "
            f"{sorted(schema_versions)}"
        )

    all_section_ids = set(seen_section_ids)
    level_by_section_id = {
        item["section_id"]: item["level"]
        for item in record_metadata
        if item["section_id"] is not None
        and isinstance(item["level"], int)
        and not isinstance(item["level"], bool)
    }

    for item in record_metadata:
        section_id = item["section_id"]
        parent_section_id = item["parent_section_id"]

        if section_id is None or parent_section_id is None:
            continue

        if parent_section_id == section_id:
            errors.append(
                f"section_id={section_id!r}: parent_section_id cannot refer "
                "to itself"
            )
            continue

        if parent_section_id not in all_section_ids:
            errors.append(
                f"section_id={section_id!r}: parent_section_id "
                f"{parent_section_id!r} is not present in the Section view"
            )
            continue

        parent_level = level_by_section_id.get(parent_section_id)
        child_level = level_by_section_id.get(section_id)
        if (
            parent_level is not None
            and child_level is not None
            and parent_level >= child_level
        ):
            errors.append(
                f"section_id={section_id!r}: parent level {parent_level} "
                f"must be lower than child level {child_level}"
            )

    for item in record_metadata:
        owner_section_id = item["section_id"]
        for absorbed_section_id in item["absorbed_section_ids"]:
            if absorbed_section_id in all_section_ids:
                errors.append(
                    f"section_id={owner_section_id!r}: absorbed Section "
                    f"{absorbed_section_id!r} is still present as a node"
                )

    strategy = next(iter(strategies)) if len(strategies) == 1 else None
    aggregation_mode = (
        next(iter(aggregation_modes))
        if len(aggregation_modes) == 1
        else None
    )
    aggregation_max_level = (
        next(iter(aggregation_max_levels))
        if len(aggregation_max_levels) == 1
        else None
    )

    aggregated_count = sum(
        1
        for item in record_metadata
        if item["role"] == RETRIEVAL_ROLE
        and item["is_aggregated"]
    )

    if strategy == "sections":
        if aggregation_mode != "none":
            errors.append(
                "strategy='sections' requires aggregation_mode='none'"
            )
        if aggregation_max_level is not None:
            errors.append(
                "strategy='sections' requires aggregation_max_level=null"
            )
        if aggregated_count != 0:
            errors.append(
                "strategy='sections' cannot contain aggregated Sections"
            )
    elif strategy is not None:
        match = re.fullmatch(r"max_level_(\d+)", strategy)
        if match is None:
            errors.append(
                f"Unsupported retrieval_strategy: {strategy!r}"
            )
        else:
            expected_max_level = int(match.group(1))
            if aggregation_mode != "merge_below_level":
                errors.append(
                    f"strategy={strategy!r} requires "
                    "aggregation_mode='merge_below_level'"
                )
            if aggregation_max_level != expected_max_level:
                errors.append(
                    f"strategy={strategy!r} requires "
                    f"aggregation_max_level={expected_max_level}"
                )
            for item in record_metadata:
                level = item["level"]
                if (
                    isinstance(level, int)
                    and not isinstance(level, bool)
                    and level > expected_max_level
                ):
                    errors.append(
                        f"section_id={item['section_id']!r}: level={level} "
                        f"exceeds active max level {expected_max_level}"
                    )

    if errors:
        preview_limit = 50
        preview = errors[:preview_limit]
        if len(errors) > preview_limit:
            preview.append(
                f"... and {len(errors) - preview_limit} additional errors"
            )
        details = "\n - ".join(preview)
        raise ValueError(
            f"Invalid retrieval Section view {source_label} "
            f"({len(errors)} error(s)):\n - {details}"
        )

    doc_id = next(iter(doc_ids))
    schema_version = next(iter(schema_versions))
    retrieval_count = sum(
        1
        for item in record_metadata
        if item["role"] == RETRIEVAL_ROLE
    )
    structural_count = sum(
        1
        for item in record_metadata
        if item["role"] == STRUCTURAL_ROLE
    )
    parent_child_count = sum(
        1
        for item in record_metadata
        if item["parent_section_id"] is not None
    )

    return {
        "doc_id": doc_id,
        "retrieval_strategy": strategy,
        "aggregation_mode": aggregation_mode,
        "aggregation_max_level": aggregation_max_level,
        "section_view_schema_version": schema_version,
        "section_count": len(sections),
        "retrieval_section_count": retrieval_count,
        "structural_section_count": structural_count,
        "aggregated_section_count": aggregated_count,
        "parent_child_count": parent_child_count,
        "next_count": max(retrieval_count - 1, 0),
        "source_section_count": len(source_owner),
    }


def load_and_validate_section_view(
    section_view_file: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load a Section-view JSON file and return records plus validation summary."""
    section_view_file = Path(section_view_file)

    try:
        raw_text = section_view_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(
            f"Unable to read Section view file: {section_view_file}"
        ) from exc

    try:
        sections = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in Section view file {section_view_file}: {exc}"
        ) from exc

    summary = validate_section_view_records(
        sections,
        source_path=section_view_file,
    )
    return sections, summary


def load_section_view_from_file(
    section_view_file: Path,
) -> List[Dict[str, Any]]:
    """Load and validate a Section-view JSON file."""
    sections, _ = load_and_validate_section_view(section_view_file)
    return sections


def load_chunks_from_file(chunk_file: Path) -> List[Dict[str, Any]]:
    """
    Backward-compatible alias.

    Despite the historical name, canonical chunk files are intentionally
    rejected: the graph loader now requires a validated Section view.
    """
    return load_section_view_from_file(chunk_file)


def normalize_section_record(
    section: Dict[str, Any],
    min_text_chars_to_embed: int = DEFAULT_MIN_TEXT_CHARS_TO_EMBED,
) -> Dict[str, Any]:
    """Normalize one validated Section-view record for Neo4j ingestion."""
    doc_id = str(section["doc_id"]).strip()
    section_id = str(section["section_id"]).strip()

    return {
        "uid": make_section_uid(doc_id, section_id),
        "chunk_id": section.get("chunk_id"),
        "doc_id": doc_id,
        "section_id": section_id,
        "printed_section_id": section.get("printed_section_id"),
        "title": section.get("section_title"),
        "section_type": section.get("section_type"),
        "level": section.get("section_level"),
        "text": section.get("text") or "",
        "is_empty": bool(section.get("is_empty")),
        "excluded": False,
        "embed": infer_should_embed(
            section,
            min_text_chars_to_embed=min_text_chars_to_embed,
        ),
        "page_start": section.get("page_start"),
        "page_end": section.get("page_end"),
        "parent_section_id": section.get("parent_section_id"),
        "part_index": section.get("part_index"),
        "part_count": section.get("part_count"),
        "quality_flags": list(section.get("quality_flags") or []),
        "boundary_source": section.get("boundary_source"),
        "section_view_role": section.get("section_view_role"),
        "section_view_schema_version": str(
            section.get(
                "section_view_schema_version",
                SECTION_VIEW_SCHEMA_VERSION,
            )
        ),
        "retrieval_unit_id": section.get("retrieval_unit_id"),
        "retrieval_strategy": section.get("retrieval_strategy"),
        "aggregation_mode": section.get("aggregation_mode"),
        "aggregation_max_level": section.get("aggregation_max_level"),
        "is_aggregated": bool(section.get("is_aggregated")),
        "root_has_local_text": bool(
            section.get("root_has_local_text")
        ),
        "root_section_id": section.get("root_section_id"),
        "root_chunk_id": section.get("root_chunk_id"),
        "root_page_start": section.get("root_page_start"),
        "root_page_end": section.get("root_page_end"),
        "root_quality_flags": list(
            section.get("root_quality_flags") or []
        ),
        "content_owner_section_id": section.get(
            "content_owner_section_id"
        ),
        "source_section_ids": list(
            section.get("source_section_ids") or []
        ),
        "source_chunk_ids": list(
            section.get("source_chunk_ids") or []
        ),
        "represented_section_ids": list(
            section.get("represented_section_ids") or []
        ),
        "structural_context_section_ids": list(
            section.get("structural_context_section_ids") or []
        ),
        "absorbed_section_ids": list(
            section.get("absorbed_section_ids") or []
        ),
        "absorbed_source_section_ids": list(
            section.get("absorbed_source_section_ids") or []
        ),
        "source_count": int(section.get("source_count", 0)),
        "represented_section_count": int(
            section.get("represented_section_count", 0)
        ),
        "text_chars": int(section.get("text_chars", 0)),
        "text_words": int(section.get("text_words", 0)),
        "source_text_chars": int(
            section.get("source_text_chars", 0)
        ),
        "canonical_text_chars": int(
            section.get("canonical_text_chars", 0)
        ),
        "canonical_is_empty": bool(
            section.get("canonical_is_empty")
        ),
        "canonical_embed": bool(section.get("canonical_embed")),
        "section_view_order": None,
        "retrieval_order": None,
    }


def setup_schema(tx) -> None:
    """Create constraints and indexes needed for graph loading."""
    tx.run(
        """
        CREATE CONSTRAINT document_id IF NOT EXISTS
        FOR (d:Document)
        REQUIRE d.doc_id IS UNIQUE
        """
    )

    tx.run(
        """
        CREATE CONSTRAINT section_uid IF NOT EXISTS
        FOR (s:Section)
        REQUIRE s.uid IS UNIQUE
        """
    )

    tx.run(
        """
        CREATE INDEX section_doc_id IF NOT EXISTS
        FOR (s:Section)
        ON (s.doc_id)
        """
    )

    tx.run(
        """
        CREATE INDEX section_doc_role IF NOT EXISTS
        FOR (s:Section)
        ON (s.doc_id, s.section_view_role)
        """
    )

    tx.run(
        """
        CREATE INDEX section_doc_embed IF NOT EXISTS
        FOR (s:Section)
        ON (s.doc_id, s.embed)
        """
    )


def create_document(
    tx,
    doc_id: str,
    metadata: Dict[str, Any],
) -> None:
    """Create or reuse a Document node and record the active Section view."""
    tx.run(
        """
        MERGE (d:Document {doc_id: $doc_id})
        SET d.retrieval_strategy = $metadata.retrieval_strategy,
            d.retrieval_max_level = $metadata.aggregation_max_level,
            d.aggregation_mode = $metadata.aggregation_mode,
            d.section_view_schema_version =
                $metadata.section_view_schema_version,
            d.section_view_file = $metadata.section_view_file,
            d.section_view_section_count = $metadata.section_count,
            d.section_view_retrieval_count =
                $metadata.retrieval_section_count,
            d.section_view_structural_count =
                $metadata.structural_section_count,
            d.section_view_aggregated_count =
                $metadata.aggregated_section_count,
            d.section_view_source_section_count =
                $metadata.source_section_count,
            d.section_view_loaded_at = datetime()
        """,
        doc_id=doc_id,
        metadata=metadata,
    )


def document_sections_exist(tx, doc_id: str) -> bool:
    """Return whether any Section node already exists for the document."""
    result = tx.run(
        """
        MATCH (s:Section {doc_id: $doc_id})
        RETURN count(s) > 0 AS has_sections
        """,
        doc_id=doc_id,
    )
    record = result.single()
    return bool(record["has_sections"]) if record is not None else False


def delete_existing_document_sections(tx, doc_id: str) -> None:
    """
    Remove every Section node for one document before reloading it.

    Matching by ``doc_id`` also cleans up Sections left behind by an interrupted
    or legacy load that did not create ``HAS_SECTION`` correctly.
    """
    tx.run(
        """
        MATCH (s:Section {doc_id: $doc_id})
        DETACH DELETE s
        """,
        doc_id=doc_id,
    )


def delete_orphan_concepts(tx) -> None:
    """Remove Concept nodes no longer mentioned by any Section."""
    tx.run(
        """
        MATCH (c:Concept)
        WHERE NOT (:Section)-[:MENTIONS]->(c)
        DETACH DELETE c
        """
    )


def create_sections_batch(
    tx,
    sections: List[Dict[str, Any]],
) -> None:
    """
    Create Section nodes in batch.

    ``source_sections`` is deliberately not persisted because it is a list of
    maps and therefore is not a valid Neo4j property value. The JSON audit file
    remains the complete source of that nested provenance.
    """
    tx.run(
        """
        UNWIND $sections AS section
        CREATE (s:Section {
            uid: section.uid,
            chunk_id: section.chunk_id,
            doc_id: section.doc_id,
            section_id: section.section_id,
            printed_section_id: section.printed_section_id,
            title: section.title,
            section_type: section.section_type,
            level: section.level,
            text: section.text,
            is_empty: section.is_empty,
            excluded: section.excluded,
            embed: section.embed,
            page_start: section.page_start,
            page_end: section.page_end,
            parent_section_id: section.parent_section_id,
            part_index: section.part_index,
            part_count: section.part_count,
            quality_flags: section.quality_flags,
            boundary_source: section.boundary_source,

            section_view_role: section.section_view_role,
            section_view_schema_version:
                section.section_view_schema_version,
            section_view_order: section.section_view_order,
            retrieval_order: section.retrieval_order,

            retrieval_unit_id: section.retrieval_unit_id,
            retrieval_strategy: section.retrieval_strategy,
            aggregation_mode: section.aggregation_mode,
            aggregation_max_level: section.aggregation_max_level,
            is_aggregated: section.is_aggregated,
            root_has_local_text: section.root_has_local_text,
            root_section_id: section.root_section_id,
            root_chunk_id: section.root_chunk_id,
            root_page_start: section.root_page_start,
            root_page_end: section.root_page_end,
            root_quality_flags: section.root_quality_flags,
            content_owner_section_id:
                section.content_owner_section_id,

            source_section_ids: section.source_section_ids,
            source_chunk_ids: section.source_chunk_ids,
            represented_section_ids:
                section.represented_section_ids,
            structural_context_section_ids:
                section.structural_context_section_ids,
            absorbed_section_ids: section.absorbed_section_ids,
            absorbed_source_section_ids:
                section.absorbed_source_section_ids,

            source_count: section.source_count,
            represented_section_count:
                section.represented_section_count,
            text_chars: section.text_chars,
            text_words: section.text_words,
            source_text_chars: section.source_text_chars,
            canonical_text_chars: section.canonical_text_chars,
            canonical_is_empty: section.canonical_is_empty,
            canonical_embed: section.canonical_embed,

            has_embedding: false,
            embedding: null,
            embedding_model: null,
            embedding_dim: null,
            embedding_updated_at: null,
            embedding_status: null,
            embedding_failed_at: null,

            entity_extracted: false,
            entity_extracted_at: null,
            entity_extraction_status: null,
            entity_extraction_failed_at: null
        })
        """,
        sections=sections,
    )


def link_document_sections_batch(
    tx,
    doc_id: str,
    section_uids: List[str],
) -> None:
    """Link one Document node to many Section nodes."""
    tx.run(
        """
        MATCH (d:Document {doc_id: $doc_id})
        UNWIND $section_uids AS uid
        MATCH (s:Section {uid: uid})
        MERGE (d)-[r:HAS_SECTION]->(s)
        SET r += $relationship_metadata
        """,
        doc_id=doc_id,
        section_uids=section_uids,
        relationship_metadata=build_structural_relationship_metadata(
            "HAS_SECTION",
            doc_id=doc_id,
        ),
    )


def link_parent_child_batch(
    tx,
    pairs: List[Dict[str, str]],
) -> None:
    """
    Create HAS_CHILD relationships in batch.

    Each pair is ``{"parent_uid": ..., "child_uid": ...}``.
    """
    if not pairs:
        return

    tx.run(
        """
        UNWIND $pairs AS pair
        MATCH (p:Section {uid: pair.parent_uid})
        MATCH (c:Section {uid: pair.child_uid})
        WITH p, c,
             coalesce(trim(toString(p.doc_id)), '') AS parent_doc_id,
             coalesce(trim(toString(c.doc_id)), '') AS child_doc_id
        WITH p, c,
             CASE
                 WHEN parent_doc_id <> '' AND child_doc_id <> ''
                      AND parent_doc_id <> child_doc_id
                 THEN null
                 WHEN parent_doc_id <> '' THEN parent_doc_id
                 WHEN child_doc_id <> '' THEN child_doc_id
                 ELSE null
             END AS relationship_doc_id
        MERGE (p)-[r:HAS_CHILD]->(c)
        SET r += $relationship_metadata
        FOREACH (_ IN CASE
            WHEN relationship_doc_id IS NULL THEN []
            ELSE [1]
        END |
            SET r.doc_id = relationship_doc_id
        )
        """,
        pairs=pairs,
        relationship_metadata=build_structural_relationship_metadata(
            "HAS_CHILD"
        ),
    )


def link_next_batch(
    tx,
    pairs: List[Dict[str, str]],
) -> None:
    """
    Create NEXT relationships between retrievable Sections only.

    Each pair is ``{"prev_uid": ..., "next_uid": ...}``.
    """
    if not pairs:
        return

    tx.run(
        """
        UNWIND $pairs AS pair
        MATCH (a:Section {uid: pair.prev_uid})
        MATCH (b:Section {uid: pair.next_uid})
        WHERE a.section_view_role = $retrieval_role
          AND b.section_view_role = $retrieval_role
        WITH a, b,
             coalesce(trim(toString(a.doc_id)), '') AS prev_doc_id,
             coalesce(trim(toString(b.doc_id)), '') AS next_doc_id
        WITH a, b,
             CASE
                 WHEN prev_doc_id <> '' AND next_doc_id <> ''
                      AND prev_doc_id <> next_doc_id
                 THEN null
                 WHEN prev_doc_id <> '' THEN prev_doc_id
                 WHEN next_doc_id <> '' THEN next_doc_id
                 ELSE null
             END AS relationship_doc_id
        MERGE (a)-[r:NEXT]->(b)
        SET r += $relationship_metadata
        FOREACH (_ IN CASE
            WHEN relationship_doc_id IS NULL THEN []
            ELSE [1]
        END |
            SET r.doc_id = relationship_doc_id
        )
        """,
        pairs=pairs,
        retrieval_role=RETRIEVAL_ROLE,
        relationship_metadata=build_structural_relationship_metadata(
            "NEXT"
        ),
    )


def build_graph_from_chunks(
    driver: Driver,
    chunk_file: Path,
    batch_size: int = 200,
    min_text_chars_to_embed: int = DEFAULT_MIN_TEXT_CHARS_TO_EMBED,
    replace_existing_document: bool = True,
) -> Optional[str]:
    """
    Build one document graph from a validated retrieval Section view.

    The historical function name is retained so current callers do not break.
    ``chunk_file`` must now point to a ``*_section_view_*.json`` artifact, not
    to the canonical ``*_hier_chunks.json`` file.

    ``NEXT`` links only retrievable Sections. ``HAS_CHILD`` preserves the full
    retained hierarchy, including structural Sections.
    """
    chunk_file = Path(chunk_file)
    logger.info("Loading retrieval Section view from %s", chunk_file)

    sections, validation = load_and_validate_section_view(chunk_file)
    if not sections:
        logger.warning("Empty Section view file: %s", chunk_file)
        return None

    doc_id = str(validation["doc_id"])
    logger.info(
        "Validated Section view for %s | strategy=%s | sections=%d | "
        "retrieval=%d | structural=%d | aggregated=%d",
        doc_id,
        validation["retrieval_strategy"],
        validation["section_count"],
        validation["retrieval_section_count"],
        validation["structural_section_count"],
        validation["aggregated_section_count"],
    )

    normalized_sections: List[Dict[str, Any]] = []
    retrieval_order = 0

    for section_view_order, section in enumerate(sections):
        normalized = normalize_section_record(
            section,
            min_text_chars_to_embed=min_text_chars_to_embed,
        )
        normalized["section_view_order"] = section_view_order

        if normalized["section_view_role"] == RETRIEVAL_ROLE:
            normalized["retrieval_order"] = retrieval_order
            retrieval_order += 1

        normalized_sections.append(normalized)

    section_uids = [section["uid"] for section in normalized_sections]

    parent_child_pairs: List[Dict[str, str]] = []
    for section in normalized_sections:
        parent_id = section.get("parent_section_id")
        if parent_id:
            parent_child_pairs.append(
                {
                    "parent_uid": make_section_uid(doc_id, parent_id),
                    "child_uid": section["uid"],
                }
            )

    retrievable_sections = [
        section
        for section in normalized_sections
        if section["section_view_role"] == RETRIEVAL_ROLE
    ]
    next_pairs = [
        {
            "prev_uid": previous["uid"],
            "next_uid": current["uid"],
        }
        for previous, current in zip(
            retrievable_sections,
            retrievable_sections[1:],
        )
    ]

    document_metadata = dict(validation)
    document_metadata["section_view_file"] = chunk_file.name

    logger.info(
        "Building graph structure for document: %s | "
        "replace_existing_document=%s",
        doc_id,
        replace_existing_document,
    )

    with driver.session() as session:
        session.execute_write(setup_schema)

        if replace_existing_document:
            session.execute_write(
                delete_existing_document_sections,
                doc_id,
            )
            session.execute_write(delete_orphan_concepts)
        elif session.execute_read(document_sections_exist, doc_id):
            raise ValueError(
                f"Document {doc_id} already has Sections in the graph. "
                "Use replace_existing_document=True to reload it."
            )

        session.execute_write(
            create_document,
            doc_id,
            document_metadata,
        )

        for batch in chunked(normalized_sections, batch_size):
            session.execute_write(create_sections_batch, batch)

        for uid_batch in chunked(section_uids, batch_size):
            session.execute_write(
                link_document_sections_batch,
                doc_id,
                uid_batch,
            )

        for batch in chunked(parent_child_pairs, batch_size):
            session.execute_write(link_parent_child_batch, batch)

        for batch in chunked(next_pairs, batch_size):
            session.execute_write(link_next_batch, batch)

    logger.info(
        "Graph structure built for document: %s | sections=%d | "
        "retrieval=%d | structural=%d | parent_child=%d | next=%d",
        doc_id,
        len(normalized_sections),
        len(retrievable_sections),
        validation["structural_section_count"],
        len(parent_child_pairs),
        len(next_pairs),
    )

    return doc_id


# Clearer alias for new code. Existing code may continue importing
# ``build_graph_from_chunks`` until build_graph.py is updated.
build_graph_from_section_view = build_graph_from_chunks
