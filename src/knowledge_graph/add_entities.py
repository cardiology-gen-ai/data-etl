"""
add_entities.py

Entity extraction pipeline for Section nodes in the knowledge graph.

Main notes:
- Concepts are normalized before being written.
- We keep ONE Concept node per normalized concept name.
- Type ambiguity is preserved at relationship level through MENTIONS.observed_types.
- Concept-level type state is later finalized by entity_disambiguation.py.
- During extraction, Concept.canonical_type is intentionally left unset/pending.
- Type aliases are normalized once at load time so alias lookup is consistent
  with normalize_type().
- Raw LLM name/type fields are preserved as raw_name/raw_type on MENTIONS
  relationships when available, so later validation/debugging can distinguish
  cases such as "AS" from normalized "as".
- When replace_section_mentions=True, stale MENTIONS are also cleared on failed
  or skipped-empty sections so section state stays consistent across reruns.
- Before writing, extracted concepts are deterministically validated against the
  section source text through validate_entities.py so unsupported outputs are not
  inserted into the KG.
- Entity validation can optionally use cached per-document acronym dictionaries:
  if a section contains an acronym short form and the cached acronym definition
  matches the extracted concept long form, the concept can be accepted even when
  the long form itself is absent from the section.
- If the LLM extracts an acronym short form itself, validation can expand it to
  the cached long form before writing, so Concept nodes store the meaningful
  normalized long-form concept while raw_name preserves the original acronym.
- Entity validation decisions are optionally exported to JSONL review files so
  accepted/rejected candidates can be inspected outside Neo4j.
- Only retrieval-role, embeddable, non-excluded Section-view nodes are processed;
  structural hierarchy nodes are never sent to the LLM or marked as skipped.
- Oversized retrieval Sections are segmented without dropping text; segment-level
  concepts are merged and validated against the original full Section before write.
- Orphan Concept nodes with no incoming MENTIONS edges are removed after the run.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neo4j import Driver

from knowledge_graph.acronym_utils import load_acronyms_by_doc_id
from knowledge_graph.entity_review_exports import (
    clear_entity_review_exports,
    utc_now_iso,
    write_entity_review_summary,
    write_section_entity_review_records,
)
from knowledge_graph.entity_schema import (
    ALLOWED_TYPES,
    BLOCKLIST_NAMES,
    deduplicate_concepts,
    normalize_concept,
)
from knowledge_graph.llm_utils import (
    generate_chat_text,
    get_chat_model_name,
)
from knowledge_graph.prompts import (
    ENTITY_EXTRACTION_SINGLE_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_BATCH_SYSTEM_PROMPT,
    build_entity_extraction_single_user_prompt,
    build_entity_extraction_batch_user_prompt,
)
from knowledge_graph.relationship_metadata import build_mention_relationship_metadata
from knowledge_graph.validate_entities import (
    validate_concepts_against_source,
    summarize_rejections,
)


logger = logging.getLogger(__name__)


CONCEPT_DEBUG_LOG_LIMIT = 30


def _concept_json_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {
                "type": "string",
                "enum": sorted(ALLOWED_TYPES),
            },
        },
        "required": ["name", "type"],
        "additionalProperties": False,
    }


SINGLE_ENTITY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": _concept_json_schema(),
        },
    },
    "required": ["concepts"],
    "additionalProperties": False,
}


BATCH_ENTITY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string"},
                    "concepts": {
                        "type": "array",
                        "items": _concept_json_schema(),
                    },
                },
                "required": ["uid", "concepts"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sections"],
    "additionalProperties": False,
}


def truncate_for_log(text: str, max_chars: int = 2000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_llm_json(text: str) -> Any:
    text = text.strip()
    candidates = [text, strip_code_fences(text)]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()

    for candidate in candidates:
        for i, ch in enumerate(candidate):
            if ch not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[i:])
                return parsed
            except json.JSONDecodeError:
                continue

    raise ValueError("Could not parse JSON from LLM output")


def stringify_raw_llm_value(value: Any) -> Optional[str]:
    """
    Convert a raw LLM field into a safe string value for validation metadata
    and Neo4j relationship properties.

    Neo4j properties cannot safely store arbitrary nested objects, so raw values
    are kept as strings.
    """
    if value is None:
        return None

    return str(value)


def normalize_llm_concept_preserving_raw(raw_concept: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize one raw LLM concept while preserving the original name/type fields.

    The normalized fields remain:
        name
        type

    The original LLM surface fields are kept as:
        raw_name
        raw_type

    This lets downstream validation/debugging distinguish cases such as:
        raw_name = "AS"
        name = "as"
    """
    normalized = normalize_concept(raw_concept)

    if normalized is None:
        return None

    if isinstance(raw_concept, dict):
        normalized["raw_name"] = stringify_raw_llm_value(raw_concept.get("name"))
        normalized["raw_type"] = stringify_raw_llm_value(raw_concept.get("type"))

    return normalized


def has_section_body(row: Dict[str, Any]) -> bool:
    """
    Return True when the Section has non-empty body text.

    When entity extraction is configured with use_section_text=True, title-only
    sections should be skipped. They remain useful as graph hierarchy nodes, but
    they should not create normal body-grounded entity mentions.
    """
    return bool((row.get("text") or "").strip())


def build_source_text(row: Dict[str, Any], use_section_text: bool) -> str:
    parts = []

    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()

    if title:
        parts.append(f"Title: {title}")
    if use_section_text and body:
        parts.append(f"Body:\n{body}")

    return "\n\n".join(parts).strip()


DEFAULT_ENTITY_SEGMENT_OVERLAP_CHARS = 500
RETRIEVAL_ROLE = "retrieval"


def split_text_for_llm(
    text: str,
    max_chars: int,
    overlap_chars: int = DEFAULT_ENTITY_SEGMENT_OVERLAP_CHARS,
) -> List[str]:
    """
    Split text deterministically into overlapping, boundary-aware segments.

    This function never drops source text. It prefers paragraph, line, and
    sentence boundaries near the end of the available window and falls back to
    a hard character boundary only when no suitable separator is available.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    normalized = (text or "").strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    segments: List[str] = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
        hard_end = min(start + max_chars, text_len)
        end = hard_end

        if hard_end < text_len:
            minimum_boundary = start + max(1, int(max_chars * 0.60))
            boundary_candidates: List[int] = []

            for separator in ("\n\n", "\n", ". ", "; ", ", "):
                position = normalized.rfind(
                    separator,
                    minimum_boundary,
                    hard_end,
                )
                if position >= minimum_boundary:
                    boundary_candidates.append(position + len(separator))

            if boundary_candidates:
                end = max(boundary_candidates)

        segment = normalized[start:end].strip()
        if not segment:
            end = hard_end
            segment = normalized[start:end].strip()

        if not segment:
            raise RuntimeError(
                "Unable to create a non-empty entity-extraction segment"
            )

        segments.append(segment)

        if end >= text_len:
            break

        next_start = max(start + 1, end - overlap_chars)

        # Avoid restarting in the middle of a word when a nearby whitespace
        # boundary is available.
        if (
            0 < next_start < text_len
            and normalized[next_start - 1].isalnum()
            and normalized[next_start].isalnum()
        ):
            whitespace = normalized.find(
                " ",
                next_start,
                min(end, next_start + 120),
            )
            if whitespace != -1:
                next_start = whitespace + 1

        if next_start <= start:
            raise RuntimeError(
                "Entity-extraction segmentation made no forward progress"
            )

        start = next_start

    return segments


def build_llm_units(
    rows: List[Dict[str, Any]],
    max_segment_chars: int,
    overlap_chars: int = DEFAULT_ENTITY_SEGMENT_OVERLAP_CHARS,
) -> List[Dict[str, Any]]:
    """
    Convert retrieval Sections into LLM request units.

    Normal Sections produce one request unit. Oversized Sections are split into
    multiple overlapping units while retaining the original Section UID. The
    title is repeated in every body segment so each request remains grounded in
    the same section context.

    The original full ``source_text`` remains on the Section row and is used for
    deterministic validation after concepts from all segments are merged.
    """
    if max_segment_chars < 1:
        raise ValueError("max_segment_chars must be >= 1")

    effective_overlap = min(
        max(0, overlap_chars),
        max(0, max_segment_chars - 1),
    )

    units: List[Dict[str, Any]] = []

    for row in rows:
        full_source = (row.get("source_text") or "").strip()
        if not full_source:
            continue

        if len(full_source) <= max_segment_chars:
            unit = dict(row)
            unit["llm_uid"] = row["uid"]
            unit["segment_index"] = 1
            unit["segment_count"] = 1
            units.append(unit)
            continue

        title = (row.get("title") or "").strip()
        body = (row.get("text") or "").strip()

        prefix_parts: List[str] = []
        if title:
            prefix_parts.append(f"Title: {title}")
        if body:
            prefix_parts.append("Body:")

        prefix = "\n\n".join(prefix_parts)
        if prefix:
            prefix += "\n"

        body_budget = max_segment_chars - len(prefix)

        if body and body_budget >= 256:
            body_overlap = min(
                effective_overlap,
                max(0, body_budget - 1),
            )
            body_segments = split_text_for_llm(
                body,
                max_chars=body_budget,
                overlap_chars=body_overlap,
            )
            segment_texts = [
                f"{prefix}{body_segment}".strip()
                for body_segment in body_segments
            ]
        else:
            # Extremely long titles or title-only inputs are split as a last
            # deterministic fallback, still without dropping text.
            segment_texts = split_text_for_llm(
                full_source,
                max_chars=max_segment_chars,
                overlap_chars=effective_overlap,
            )

        segment_count = len(segment_texts)

        for segment_index, segment_text in enumerate(segment_texts, start=1):
            if len(segment_text) > max_segment_chars:
                raise RuntimeError(
                    "Entity-extraction segment exceeds configured limit | "
                    f"uid={row['uid']} | segment={segment_index}/"
                    f"{segment_count} | chars={len(segment_text)} | "
                    f"limit={max_segment_chars}"
                )

            unit = dict(row)
            unit["source_text"] = segment_text
            unit["llm_uid"] = (
                f"{row['uid']}::segment::{segment_index:04d}"
                f"-of-{segment_count:04d}"
            )
            unit["segment_index"] = segment_index
            unit["segment_count"] = segment_count
            units.append(unit)

    return units


def pack_rows_for_llm(
    rows: List[Dict[str, Any]],
    max_sections_per_batch: int,
    max_batch_chars: int,
    emergency_max_single_chars: Optional[int] = None,
) -> Iterable[List[Dict[str, Any]]]:
    """
    Pack already-sized LLM request units into batches.

    No truncation is performed here. Oversized Sections must first be converted
    into bounded request units with ``build_llm_units``. The historical
    ``emergency_max_single_chars`` parameter is retained for API compatibility
    but is no longer used to discard source text.
    """
    del emergency_max_single_chars

    if max_sections_per_batch < 1:
        raise ValueError("max_sections_per_batch must be >= 1")
    if max_batch_chars < 1:
        raise ValueError("max_batch_chars must be >= 1")

    batch: List[Dict[str, Any]] = []
    current_chars = 0

    for row in rows:
        row_text = row["source_text"]
        row_chars = len(row_text)

        if row_chars > max_batch_chars:
            raise ValueError(
                "LLM request unit exceeds max_batch_chars after segmentation | "
                f"uid={row.get('llm_uid', row.get('uid'))} | "
                f"chars={row_chars} | budget={max_batch_chars}"
            )

        would_exceed_count = len(batch) >= max_sections_per_batch
        would_exceed_chars = (current_chars + row_chars) > max_batch_chars

        if batch and (would_exceed_count or would_exceed_chars):
            yield batch
            batch = [row]
            current_chars = row_chars
        else:
            batch.append(row)
            current_chars += row_chars

    if batch:
        yield batch


def extract_concepts_single(text: str) -> Optional[List[Dict[str, Any]]]:
    messages = [
        {"role": "system", "content": ENTITY_EXTRACTION_SINGLE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_entity_extraction_single_user_prompt(text),
        },
    ]

    try:
        content = generate_chat_text(
            messages=messages,
            json_mode=True,
            json_schema=SINGLE_ENTITY_RESPONSE_SCHEMA,
            json_schema_name="entity_extraction_single",
        )
    except Exception as e:
        logger.exception("Single request failed: %s", e)
        return None

    if content is None:
        logger.error("Single extraction backend returned None")
        return None

    content = content.strip()

    try:
        data = parse_llm_json(content)
        if isinstance(data, dict) and "concepts" in data:
            data = data["concepts"]

        if not isinstance(data, list):
            raise ValueError("Single-section LLM output is not a list")

        concepts: List[Dict[str, Any]] = []

        for item in data:
            normalized = normalize_llm_concept_preserving_raw(item)
            if normalized is not None:
                concepts.append(normalized)

        return deduplicate_concepts(concepts)

    except Exception:
        logger.error(
            "Failed to parse single-section LLM output: %s",
            truncate_for_log(content),
        )
        return None


def extract_concepts_batch(
    batch_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Extract concepts for one batch of LLM request units.

    ``llm_uid`` is used when present so multiple segments originating from the
    same Section can coexist in the same overall extraction run without UID
    collisions. The returned mapping is keyed by that request UID.
    """
    if not batch_rows:
        return {}

    sections_payload = [
        {
            "uid": row.get("llm_uid", row["uid"]),
            "text": row["source_text"],
        }
        for row in batch_rows
    ]

    messages = [
        {"role": "system", "content": ENTITY_EXTRACTION_BATCH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_entity_extraction_batch_user_prompt(
                sections_payload
            ),
        },
    ]

    try:
        content = generate_chat_text(
            messages=messages,
            json_mode=True,
            json_schema=BATCH_ENTITY_RESPONSE_SCHEMA,
            json_schema_name="entity_extraction_batch",
        )
    except Exception as e:
        logger.exception("Batch request failed: %s", e)
        return None

    if content is None:
        logger.error("Batch extraction backend returned None")
        return None

    content = content.strip()

    try:
        data = parse_llm_json(content)
        if isinstance(data, dict) and "sections" in data:
            data = data["sections"]

        if not isinstance(data, list):
            raise ValueError("Batch LLM output is not a list")

        expected_uids = {
            row.get("llm_uid", row["uid"])
            for row in batch_rows
        }
        out: Dict[str, List[Dict[str, Any]]] = {}

        for item in data:
            if not isinstance(item, dict):
                continue

            uid = item.get("uid")
            concepts_raw = item.get("concepts")

            if uid not in expected_uids:
                continue
            if uid in out:
                raise ValueError(f"Duplicate uid in batch result: {uid}")
            if not isinstance(concepts_raw, list):
                raise ValueError(f"Invalid concepts field for uid={uid}")

            concepts: List[Dict[str, Any]] = []

            for concept_raw in concepts_raw:
                normalized = normalize_llm_concept_preserving_raw(
                    concept_raw
                )
                if normalized is not None:
                    concepts.append(normalized)

            out[uid] = deduplicate_concepts(concepts)

        if set(out.keys()) != expected_uids:
            missing = sorted(expected_uids - set(out.keys()))
            raise ValueError(f"Batch result missing uids: {missing}")

        return out

    except Exception:
        logger.error(
            "Failed to parse batch LLM output: %s",
            truncate_for_log(content),
        )
        return None


def setup_entity_schema(tx) -> None:
    tx.run(
        """
        CREATE CONSTRAINT concept_name IF NOT EXISTS
        FOR (c:Concept)
        REQUIRE c.name IS UNIQUE
        """
    )

    tx.run(
        """
        CREATE INDEX section_entity_extracted IF NOT EXISTS
        FOR (s:Section)
        ON (s.entity_extracted)
        """
    )

    tx.run(
        """
        CREATE INDEX section_view_role IF NOT EXISTS
        FOR (s:Section)
        ON (s.section_view_role)
        """
    )

    tx.run(
        """
        CREATE INDEX section_doc_retrieval_order IF NOT EXISTS
        FOR (s:Section)
        ON (s.doc_id, s.retrieval_order)
        """
    )

    tx.run(
        """
        CREATE INDEX concept_canonical_type IF NOT EXISTS
        FOR (c:Concept)
        ON (c.canonical_type)
        """
    )

    tx.run(
        """
        CREATE INDEX concept_type_resolution_status IF NOT EXISTS
        FOR (c:Concept)
        ON (c.type_resolution_status)
        """
    )

    tx.run(
        """
        CREATE INDEX concept_needs_type_review IF NOT EXISTS
        FOR (c:Concept)
        ON (c.needs_type_review)
        """
    )


def clear_section_mentions(tx, section_uid: str) -> None:
    tx.run(
        """
        MATCH (s:Section {uid: $uid})
        OPTIONAL MATCH (s)-[r:MENTIONS]->(:Concept)
        DELETE r
        """,
        uid=section_uid,
    )


def delete_orphan_concepts(tx) -> int:
    """
    Delete Concept nodes that are no longer mentioned by any Section.

    This is useful after reruns where old MENTIONS relationships are replaced,
    or when failed/skipped sections clear stale mentions.
    """
    record = tx.run(
        """
        MATCH (c:Concept)
        WHERE NOT EXISTS {
            MATCH (:Section)-[:MENTIONS]->(c)
        }
        RETURN count(c) AS deleted
        """
    ).single()

    deleted = int(record["deleted"] or 0) if record is not None else 0

    if deleted == 0:
        return 0

    tx.run(
        """
        MATCH (c:Concept)
        WHERE NOT EXISTS {
            MATCH (:Section)-[:MENTIONS]->(c)
        }
        DETACH DELETE c
        """
    )

    return deleted


def mark_section_extraction_failed(
    tx,
    section_uid: str,
    replace_section_mentions: bool = True,
) -> None:
    if replace_section_mentions:
        clear_section_mentions(tx, section_uid)

    tx.run(
        """
        MATCH (s:Section {uid: $uid})
        SET s.entity_extracted = false,
            s.entity_extraction_status = 'failed',
            s.entity_extraction_failed_at = datetime()
        REMOVE s.entity_extracted_at
        """,
        uid=section_uid,
    )


def mark_section_extraction_skipped_empty(
    tx,
    section_uid: str,
    replace_section_mentions: bool = True,
) -> None:
    if replace_section_mentions:
        clear_section_mentions(tx, section_uid)

    tx.run(
        """
        MATCH (s:Section {uid: $uid})
        SET s.entity_extracted = true,
            s.entity_extracted_at = datetime(),
            s.entity_extraction_status = 'skipped_empty'
        REMOVE s.entity_extraction_failed_at
        """,
        uid=section_uid,
    )


def write_section_concepts(
    tx,
    section_uid: str,
    concepts: List[Dict[str, Any]],
    replace_section_mentions: bool = True,
) -> None:
    """
    Mark a section as successfully processed and write accepted concepts.

    When replace_section_mentions=True, old MENTIONS edges are cleared before
    writing the newly validated set. This avoids stale section-level mentions
    after reruns with different validation/prompt settings.

    Concept-level type state is intentionally provisional here. The final
    canonical_type should be assigned later by entity_disambiguation.py.
    """
    if replace_section_mentions:
        clear_section_mentions(tx, section_uid)

    tx.run(
        """
        MATCH (s:Section {uid: $uid})
        SET s.entity_extracted = true,
            s.entity_extracted_at = datetime(),
            s.entity_extraction_status = 'success'
        REMOVE s.entity_extraction_failed_at
        """,
        uid=section_uid,
    )

    if not concepts:
        return

    tx.run(
        """
        MATCH (s:Section {uid: $uid})
        WITH s, $concepts AS concepts
        UNWIND concepts AS concept

        MERGE (c:Concept {name: concept.name})
        ON CREATE SET
            c.observed_types = [concept.type],
            c.type_resolution_status = 'pending',
            c.needs_type_review = false,
            c.created_at = datetime()

        WITH
            s,
            concept,
            c,
            coalesce(c.observed_types, []) AS old_observed_types,
            coalesce(c.type_resolution_status, 'pending') AS old_status,
            coalesce(c.needs_type_review, false) AS old_needs_type_review

        SET
            c.observed_types =
                CASE
                    WHEN concept.type IN old_observed_types THEN old_observed_types
                    ELSE old_observed_types + concept.type
                END,
            c.canonical_type =
                CASE
                    WHEN NOT (concept.type IN old_observed_types)
                    THEN NULL
                    ELSE c.canonical_type
                END,
            c.type_resolution_status =
                CASE
                    WHEN NOT (concept.type IN old_observed_types)
                    THEN 'pending'
                    ELSE old_status
                END,
            c.needs_type_review =
                CASE
                    WHEN NOT (concept.type IN old_observed_types)
                    THEN true
                    ELSE old_needs_type_review
                END,
            c.updated_at = datetime()

        MERGE (s)-[r:MENTIONS]->(c)
        ON CREATE SET
            r.observed_types = [concept.type],
            r.created_at = datetime(),
            r.validation_reason = concept.validation_reason,
            r.support_method = concept.support_method,
            r.matched_text = concept.matched_text,
            r.matched_pattern = concept.matched_pattern,
            r.acronym_short = concept.acronym_short,
            r.acronym_definition = concept.acronym_definition,
            r.acronym_match_method = concept.acronym_match_method,
            r.expanded_from_acronym = concept.expanded_from_acronym,
            r.raw_name = concept.raw_name,
            r.raw_type = concept.raw_type,
            r.quality_flags = coalesce(concept.quality_flags, [])
        ON MATCH SET
            r.observed_types =
                CASE
                    WHEN r.observed_types IS NULL THEN [concept.type]
                    WHEN concept.type IN r.observed_types THEN r.observed_types
                    ELSE r.observed_types + concept.type
                END,
            r.updated_at = datetime(),
            r.validation_reason = concept.validation_reason,
            r.support_method = concept.support_method,
            r.matched_text = concept.matched_text,
            r.matched_pattern = concept.matched_pattern,
            r.acronym_short = concept.acronym_short,
            r.acronym_definition = concept.acronym_definition,
            r.acronym_match_method = concept.acronym_match_method,
            r.expanded_from_acronym = concept.expanded_from_acronym,
            r.raw_name = concept.raw_name,
            r.raw_type = concept.raw_type,
            r.quality_flags = coalesce(concept.quality_flags, [])
        WITH s, r
        SET r += $relationship_metadata
        FOREACH (_ IN CASE
            WHEN s.doc_id IS NULL OR trim(toString(s.doc_id)) = '' THEN []
            ELSE [1]
        END |
            SET r.doc_id = trim(toString(s.doc_id))
        )
        """,
        uid=section_uid,
        concepts=concepts,
        relationship_metadata=build_mention_relationship_metadata(),
    )


def build_raw_field_lookup(
    concepts: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Build a lookup from normalized (name, type) to raw LLM metadata.

    This lets us reattach raw_name/raw_type after validate_entities.py returns
    accepted/rejected records.

    Note:
    For acronym-expanded concepts, validate_entities.py should already preserve
    raw_name/raw_type directly, because the normalized name may change from the
    acronym short form to the long form.
    """
    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}

    for concept in concepts or []:
        name = concept.get("name")
        concept_type = concept.get("type")

        if not name or not concept_type:
            continue

        key = (str(name), str(concept_type))

        if key in lookup:
            continue

        raw_fields: Dict[str, str] = {}

        raw_name = concept.get("raw_name")
        raw_type = concept.get("raw_type")

        if raw_name not in (None, ""):
            raw_fields["raw_name"] = str(raw_name)
        if raw_type not in (None, ""):
            raw_fields["raw_type"] = str(raw_type)

        if raw_fields:
            lookup[key] = raw_fields

    return lookup


def attach_raw_fields_to_validation_records(
    records: List[Dict[str, Any]],
    raw_lookup: Dict[Tuple[str, str], Dict[str, str]],
) -> None:
    """
    Add raw_name/raw_type to accepted or rejected validation records in-place
    when the original LLM surface form is available.
    """
    for record in records or []:
        name = record.get("name")
        concept_type = record.get("type")

        if not name or not concept_type:
            continue

        raw_fields = raw_lookup.get((str(name), str(concept_type)), {})

        for field, value in raw_fields.items():
            if record.get(field) in (None, ""):
                record[field] = value


def validate_and_log_concepts(
    row: Dict[str, Any],
    concepts: List[Dict[str, Any]],
    acronyms: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run deterministic pre-write validation against the section source text.

    Returns a dictionary with:
    - accepted: concepts safe to write
    - rejected: concepts discarded before write
    - stats: validation counters

    Normal INFO logging is intentionally compact for cluster runs.
    Detailed concept names remain available in DEBUG logs and JSONL review files.
    """
    validation = validate_concepts_against_source(
        concepts=concepts,
        source_text=row["source_text"],
        allowed_types=ALLOWED_TYPES,
        blocklist_names=BLOCKLIST_NAMES,
        acronyms=acronyms,
    )

    accepted = validation["accepted"]
    rejected = validation["rejected"]

    raw_lookup = build_raw_field_lookup(concepts)
    attach_raw_fields_to_validation_records(accepted, raw_lookup)
    attach_raw_fields_to_validation_records(rejected, raw_lookup)

    accepted_by_acronym = [
        c for c in accepted
        if c.get("support_method") == "acronym"
    ]

    logger.info(
        "Section doc=%s section=%s -> raw=%d validated=%d rejected=%d acronym_supported=%d",
        row["doc_id"],
        row["section_id"],
        len(concepts),
        len(accepted),
        len(rejected),
        len(accepted_by_acronym),
    )

    if accepted:
        accepted_sample = accepted[:CONCEPT_DEBUG_LOG_LIMIT]
        logger.debug(
            " -> accepted concepts sample (%d/%d): %s",
            len(accepted_sample),
            len(accepted),
            ", ".join(
                (
                    f"{c['name']} [{c['type']}]"
                    + (
                        f" via {c.get('acronym_short')}"
                        if c.get("support_method") == "acronym" and c.get("acronym_short")
                        else ""
                    )
                    + (
                        f" from raw {c.get('raw_name')!r}"
                        if c.get("expanded_from_acronym") and c.get("raw_name")
                        else ""
                    )
                )
                for c in accepted_sample
            ),
        )

        omitted_count = len(accepted) - len(accepted_sample)
        if omitted_count > 0:
            logger.debug(
                " -> accepted concepts omitted from debug log: %d",
                omitted_count,
            )

    if rejected:
        rejection_summary = summarize_rejections(rejected)
        logger.debug(
            " -> rejected summary: %s",
            ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(rejection_summary.items())
            ),
        )

        rejected_sample = rejected[:CONCEPT_DEBUG_LOG_LIMIT]
        logger.debug(
            " -> rejected concepts sample (%d/%d): %s",
            len(rejected_sample),
            len(rejected),
            ", ".join(
                f"{c['name']} [{c['type']}] ({c['reason']})"
                for c in rejected_sample
            ),
        )

        omitted_count = len(rejected) - len(rejected_sample)
        if omitted_count > 0:
            logger.debug(
                " -> rejected concepts omitted from debug log: %d",
                omitted_count,
            )

    return validation


def export_entity_review_records_safely(
    row: Dict[str, Any],
    accepted: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    stats: Dict[str, int],
    output_dir: Optional[Path],
    run_id: Optional[str],
    include_source_preview: bool,
) -> None:
    """
    Export accepted/rejected validation decisions for inspection.

    Export failures are logged but do not stop graph writing.
    """
    try:
        export_stats = write_section_entity_review_records(
            row=row,
            accepted=accepted,
            rejected=rejected,
            output_dir=output_dir,
            run_id=run_id,
            include_source_preview=include_source_preview,
        )

        stats["entity_review_accepted_records"] += export_stats["accepted_exported"]
        stats["entity_review_rejected_records"] += export_stats["rejected_exported"]

    except Exception as e:
        stats["entity_review_export_failures"] += 1
        logger.exception(
            "Failed to export entity review records | doc=%s section=%s | error=%s",
            row.get("doc_id"),
            row.get("section_id"),
            e,
        )


def process_extracted_concepts(
    session,
    row: Dict[str, Any],
    concepts: List[Dict[str, Any]],
    stats: Dict[str, int],
    replace_section_mentions: bool,
    export_entity_review: bool,
    entity_review_output_dir: Optional[Path],
    entity_review_run_id: Optional[str],
    include_source_preview_in_review: bool,
    acronyms: Optional[Dict[str, str]] = None,
) -> None:
    """
    Validate extracted concepts, update stats, optionally export validation
    decisions, and write only accepted concepts.
    """
    validation = validate_and_log_concepts(
        row=row,
        concepts=concepts,
        acronyms=acronyms,
    )

    accepted = validation["accepted"]
    rejected = validation["rejected"]

    accepted_by_acronym = [
        c for c in accepted
        if c.get("support_method") == "acronym"
    ]
    accepted_with_quality_flags = [
        c for c in accepted
        if c.get("quality_flags")
    ]

    stats["successful_sections"] += 1
    stats["concepts_rejected_by_validation"] += len(rejected)
    stats["concepts_accepted_by_acronym"] += len(accepted_by_acronym)
    stats["concepts_with_quality_flags"] += len(
        accepted_with_quality_flags
    )

    if accepted_with_quality_flags:
        stats["sections_with_quality_flags"] += 1

    if accepted_by_acronym:
        stats["sections_with_acronym_supported_concepts"] += 1

    if accepted:
        stats["sections_with_concepts"] += 1
        stats["concepts_written"] += len(accepted)

    if export_entity_review:
        export_entity_review_records_safely(
            row=row,
            accepted=accepted,
            rejected=rejected,
            stats=stats,
            output_dir=entity_review_output_dir,
            run_id=entity_review_run_id,
            include_source_preview=include_source_preview_in_review,
        )

    session.execute_write(
        write_section_concepts,
        row["uid"],
        accepted,
        replace_section_mentions,
    )


def write_entity_review_summaries_safely(
    doc_ids: List[str],
    output_dir: Optional[Path],
) -> None:
    """
    Write one entity review summary per processed document.

    Summary failures are logged but do not affect the already-written graph.
    """
    for review_doc_id in doc_ids:
        try:
            summary = write_entity_review_summary(
                doc_id=review_doc_id,
                output_dir=output_dir,
            )

            logger.info(
                "Entity review summary written | doc=%s | accepted=%d | rejected=%d | file=%s",
                review_doc_id,
                summary.get("accepted_entities", 0),
                summary.get("rejected_entities", 0),
                summary.get("accepted_file"),
            )

        except Exception as e:
            logger.exception(
                "Failed to write entity review summary | doc=%s | error=%s",
                review_doc_id,
                e,
            )


def clear_entity_review_exports_safely(
    doc_ids: List[str],
    output_dir: Optional[Path],
) -> None:
    """
    Clear previous entity review exports for processed documents.

    Clearing failures are logged but do not stop entity extraction.
    """
    for review_doc_id in doc_ids:
        try:
            clear_entity_review_exports(
                doc_id=review_doc_id,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.exception(
                "Failed to clear previous entity review exports | doc=%s | error=%s",
                review_doc_id,
                e,
            )


def add_entities_from_sections(
    driver: Driver,
    doc_id: Optional[str] = None,
    section_ids: Optional[List[str]] = None,
    use_section_text: bool = False,
    max_sections: Optional[int] = None,
    max_sections_per_batch: int = 2,
    max_batch_chars: int = 12000,
    emergency_max_single_chars: Optional[int] = 12000,
    skip_processed: bool = True,
    replace_section_mentions: bool = True,
    export_entity_review: bool = True,
    entity_review_output_dir: Optional[Path] = None,
    clear_previous_entity_review: bool = True,
    include_source_preview_in_review: bool = False,
    acronym_dir: Optional[Path] = None,
    use_acronym_validation: bool = True,
) -> Dict[str, int]:
    """
    Extract, validate, and write concepts for retrievable Section-view nodes.

    Eligibility is intentionally strict:
    - section_view_role must be ``retrieval``;
    - embed must be true;
    - excluded must be false.

    ``section_ids`` can restrict extraction to an explicit document-local
    subset for integration tests and pilot runs. The default ``None`` preserves
    production behaviour and processes every eligible Section selected by
    ``doc_id``.

    Structural Section nodes are never sent to the LLM and are not marked as
    skipped. They remain pure hierarchy/navigation nodes.

    Oversized retrieval Sections are segmented deterministically without dropping
    source text. Concepts from every successful segment are merged, deduplicated,
    validated against the original full Section text, and then written back to
    the single owner Section. If any segment fails, the whole Section is marked
    failed rather than writing a partial concept set.

    ``emergency_max_single_chars`` is retained for configuration compatibility,
    but now acts as the maximum size of one LLM segment instead of a truncation
    threshold.
    """
    if max_sections_per_batch < 1:
        raise ValueError("max_sections_per_batch must be >= 1")
    if max_batch_chars < 1:
        raise ValueError("max_batch_chars must be >= 1")
    if (
        emergency_max_single_chars is not None
        and emergency_max_single_chars < 1
    ):
        raise ValueError("emergency_max_single_chars must be >= 1 or None")
    if max_sections is not None and max_sections < 1:
        raise ValueError("max_sections must be >= 1 or None")

    normalized_section_ids: Optional[List[str]] = None
    if section_ids is not None:
        normalized_section_ids = list(
            dict.fromkeys(
                str(section_id).strip()
                for section_id in section_ids
                if str(section_id).strip()
            )
        )
        if not normalized_section_ids:
            raise ValueError(
                "section_ids must contain at least one non-empty section id"
            )
        if doc_id is None:
            raise ValueError(
                "section_ids can be used only together with a specific doc_id"
            )

    max_segment_chars = min(
        max_batch_chars,
        (
            emergency_max_single_chars
            if emergency_max_single_chars is not None
            else max_batch_chars
        ),
    )

    model_name = get_chat_model_name()
    entity_review_run_id = f"entity_extraction::{utc_now_iso()}"

    stats = {
        "requested_section_ids": (
            len(normalized_section_ids)
            if normalized_section_ids is not None
            else 0
        ),
        "matched_requested_section_ids": 0,
        "missing_requested_section_ids": 0,
        "eligible_retrieval_sections": 0,
        "processed_sections": 0,
        "successful_sections": 0,
        "failed_sections": 0,
        "skipped_sections": 0,
        "segmented_sections": 0,
        "llm_segments": 0,
        "failed_segments": 0,
        "sections_with_concepts": 0,
        "concepts_written": 0,
        "concepts_rejected_by_validation": 0,
        "concepts_accepted_by_acronym": 0,
        "sections_with_acronym_supported_concepts": 0,
        "concepts_with_quality_flags": 0,
        "sections_with_quality_flags": 0,
        "documents_with_acronym_cache": 0,
        "acronyms_loaded": 0,
        "entity_review_accepted_records": 0,
        "entity_review_rejected_records": 0,
        "entity_review_export_failures": 0,
        "orphan_concepts_deleted": 0,
    }

    with driver.session() as session:
        session.execute_write(setup_entity_schema)

        query = """
        MATCH (s:Section)
        WHERE ($doc_id IS NULL OR s.doc_id = $doc_id)
          AND (
                $section_ids IS NULL
                OR s.section_id IN $section_ids
              )
          AND s.section_view_role = $retrieval_role
          AND coalesce(s.embed, false) = true
          AND coalesce(s.excluded, false) = false
        """

        if skip_processed:
            query += """
            AND coalesce(s.entity_extracted, false) = false
            """

        query += """
        RETURN
            s.uid AS uid,
            s.doc_id AS doc_id,
            s.section_id AS section_id,
            s.title AS title,
            s.text AS text,
            s.retrieval_order AS retrieval_order,
            coalesce(s.is_aggregated, false) AS is_aggregated,
            coalesce(s.source_count, 0) AS source_count
        ORDER BY
            s.doc_id,
            coalesce(s.retrieval_order, s.section_view_order, 0),
            s.uid
        """

        if max_sections is not None:
            query += "\nLIMIT $max_sections"

        result = session.run(
            query,
            doc_id=doc_id,
            section_ids=normalized_section_ids,
            retrieval_role=RETRIEVAL_ROLE,
            max_sections=max_sections,
        )

        prepared_rows: List[Dict[str, Any]] = []
        matched_section_ids: set[str] = set()

        for record in result:
            row = {
                "uid": record["uid"],
                "doc_id": record["doc_id"],
                "section_id": record["section_id"],
                "title": record["title"],
                "text": record["text"],
                "retrieval_order": record["retrieval_order"],
                "is_aggregated": bool(record["is_aggregated"]),
                "source_count": int(record["source_count"] or 0),
            }

            stats["eligible_retrieval_sections"] += 1
            if row.get("section_id") is not None:
                matched_section_ids.add(str(row["section_id"]))

            if use_section_text and not has_section_body(row):
                stats["skipped_sections"] += 1
                session.execute_write(
                    mark_section_extraction_skipped_empty,
                    row["uid"],
                    replace_section_mentions,
                )
                logger.error(
                    "Retrieval Section unexpectedly has empty body; skipping | "
                    "doc=%s section=%s title=%r",
                    row["doc_id"],
                    row["section_id"],
                    row["title"],
                )
                continue

            source_text = build_source_text(
                row=row,
                use_section_text=use_section_text,
            )

            if not source_text.strip():
                stats["skipped_sections"] += 1
                session.execute_write(
                    mark_section_extraction_skipped_empty,
                    row["uid"],
                    replace_section_mentions,
                )
                logger.error(
                    "Retrieval Section produced empty entity source text; "
                    "skipping | doc=%s section=%s",
                    row["doc_id"],
                    row["section_id"],
                )
                continue

            row["source_text"] = source_text
            prepared_rows.append(row)

        if normalized_section_ids is not None:
            requested_set = set(normalized_section_ids)
            missing_section_ids = sorted(requested_set - matched_section_ids)
            stats["matched_requested_section_ids"] = len(
                requested_set & matched_section_ids
            )
            stats["missing_requested_section_ids"] = len(
                missing_section_ids
            )

            if missing_section_ids:
                logger.warning(
                    "Requested Section ids were not eligible or not found | "
                    "doc=%s | missing=%s",
                    doc_id,
                    missing_section_ids,
                )

        review_doc_ids = sorted(
            {
                row["doc_id"]
                for row in prepared_rows
                if row.get("doc_id")
            }
        )

        acronyms_by_doc_id: Dict[str, Dict[str, str]] = {}

        if use_acronym_validation and acronym_dir is not None and review_doc_ids:
            acronyms_by_doc_id = load_acronyms_by_doc_id(
                acronym_dir=Path(acronym_dir),
                doc_ids=review_doc_ids,
            )

            stats["documents_with_acronym_cache"] = sum(
                1 for acronyms in acronyms_by_doc_id.values() if acronyms
            )
            stats["acronyms_loaded"] = sum(
                len(acronyms)
                for acronyms in acronyms_by_doc_id.values()
            )

            logger.info(
                "Acronym validation enabled | docs_with_acronyms=%d/%d | "
                "acronyms_loaded=%d | acronym_dir=%s",
                stats["documents_with_acronym_cache"],
                len(review_doc_ids),
                stats["acronyms_loaded"],
                acronym_dir,
            )

        elif use_acronym_validation and acronym_dir is None:
            logger.info(
                "Acronym validation requested but acronym_dir is None; "
                "using direct source validation only"
            )

        else:
            logger.info("Acronym validation disabled")

        if export_entity_review and clear_previous_entity_review and review_doc_ids:
            clear_entity_review_exports_safely(
                doc_ids=review_doc_ids,
                output_dir=entity_review_output_dir,
            )

        llm_units = build_llm_units(
            prepared_rows,
            max_segment_chars=max_segment_chars,
            overlap_chars=DEFAULT_ENTITY_SEGMENT_OVERLAP_CHARS,
        )

        segmented_section_uids = {
            unit["uid"]
            for unit in llm_units
            if int(unit.get("segment_count", 1)) > 1
        }
        stats["segmented_sections"] = len(segmented_section_uids)
        stats["llm_segments"] = len(llm_units)

        logger.info(
            "Preparing entity extraction | eligible_retrieval=%d | "
            "prepared=%d | skipped=%d | segmented_sections=%d | "
            "llm_segments=%d | model=%s | max_sections_per_batch=%d | "
            "max_batch_chars=%d | max_segment_chars=%d",
            stats["eligible_retrieval_sections"],
            len(prepared_rows),
            stats["skipped_sections"],
            stats["segmented_sections"],
            stats["llm_segments"],
            model_name,
            max_sections_per_batch,
            max_batch_chars,
            max_segment_chars,
        )

        if export_entity_review:
            logger.info(
                "Entity review exports enabled | docs=%d | output_dir=%s | "
                "include_source_preview=%s",
                len(review_doc_ids),
                entity_review_output_dir or "default",
                include_source_preview_in_review,
            )

        concepts_by_section_uid: Dict[str, List[Dict[str, Any]]] = {
            row["uid"]: []
            for row in prepared_rows
        }
        failed_section_uids: set[str] = set()

        batch_count = 0

        for batch in pack_rows_for_llm(
            llm_units,
            max_sections_per_batch=max_sections_per_batch,
            max_batch_chars=max_batch_chars,
            emergency_max_single_chars=emergency_max_single_chars,
        ):
            batch_count += 1
            logger.info(
                "Extracting entities for LLM batch %d | units=%d | "
                "source_sections=%d",
                batch_count,
                len(batch),
                len({unit["uid"] for unit in batch}),
            )

            if len(batch) == 1:
                unit = batch[0]
                concepts = extract_concepts_single(unit["source_text"])

                if concepts is None:
                    stats["failed_segments"] += 1
                    failed_section_uids.add(unit["uid"])
                    logger.warning(
                        "Entity extraction failed for segment | "
                        "doc=%s section=%s segment=%d/%d",
                        unit["doc_id"],
                        unit["section_id"],
                        unit.get("segment_index", 1),
                        unit.get("segment_count", 1),
                    )
                else:
                    concepts_by_section_uid[unit["uid"]].extend(concepts)

                continue

            batch_result = extract_concepts_batch(batch)

            if batch_result is not None:
                for unit in batch:
                    request_uid = unit.get("llm_uid", unit["uid"])
                    concepts = batch_result.get(request_uid)

                    if concepts is None:
                        stats["failed_segments"] += 1
                        failed_section_uids.add(unit["uid"])
                        logger.warning(
                            "Batch result missing segment despite successful "
                            "parse | request_uid=%s",
                            request_uid,
                        )
                        continue

                    concepts_by_section_uid[unit["uid"]].extend(concepts)

            else:
                logger.warning(
                    "Batch extraction failed; falling back to single-unit "
                    "extraction for %d units",
                    len(batch),
                )

                for unit in batch:
                    concepts = extract_concepts_single(unit["source_text"])

                    if concepts is None:
                        stats["failed_segments"] += 1
                        failed_section_uids.add(unit["uid"])
                        logger.warning(
                            "Entity extraction failed for fallback segment | "
                            "doc=%s section=%s segment=%d/%d",
                            unit["doc_id"],
                            unit["section_id"],
                            unit.get("segment_index", 1),
                            unit.get("segment_count", 1),
                        )
                        continue

                    concepts_by_section_uid[unit["uid"]].extend(concepts)

        logger.info("Processed %d LLM batches", batch_count)

        for row in prepared_rows:
            stats["processed_sections"] += 1

            if row["uid"] in failed_section_uids:
                stats["failed_sections"] += 1
                session.execute_write(
                    mark_section_extraction_failed,
                    row["uid"],
                    replace_section_mentions,
                )
                logger.warning(
                    "Section marked failed because at least one segment failed | "
                    "doc=%s section=%s",
                    row["doc_id"],
                    row["section_id"],
                )
                continue

            merged_concepts = deduplicate_concepts(
                concepts_by_section_uid.get(row["uid"], [])
            )

            process_extracted_concepts(
                session=session,
                row=row,
                concepts=merged_concepts,
                stats=stats,
                replace_section_mentions=replace_section_mentions,
                export_entity_review=export_entity_review,
                entity_review_output_dir=entity_review_output_dir,
                entity_review_run_id=entity_review_run_id,
                include_source_preview_in_review=(
                    include_source_preview_in_review
                ),
                acronyms=acronyms_by_doc_id.get(row["doc_id"], {}),
            )

        if export_entity_review and review_doc_ids:
            write_entity_review_summaries_safely(
                doc_ids=review_doc_ids,
                output_dir=entity_review_output_dir,
            )

        stats["orphan_concepts_deleted"] = session.execute_write(
            delete_orphan_concepts
        )

        if stats["orphan_concepts_deleted"] > 0:
            logger.info(
                "Deleted orphan Concept nodes | count=%d",
                stats["orphan_concepts_deleted"],
            )

        logger.info(
            "Entity extraction completed | eligible_retrieval=%d | "
            "processed=%d | successful=%d | failed=%d | skipped=%d | "
            "segmented_sections=%d | llm_segments=%d | failed_segments=%d | "
            "sections_with_concepts=%d | concepts_written=%d | "
            "concepts_rejected_by_validation=%d | "
            "acronym_supported_concepts=%d | "
            "sections_with_acronym_supported_concepts=%d | "
            "concepts_with_quality_flags=%d | "
            "sections_with_quality_flags=%d | "
            "docs_with_acronym_cache=%d | acronyms_loaded=%d | "
            "review_accepted=%d | review_rejected=%d | "
            "review_export_failures=%d | orphan_concepts_deleted=%d",
            stats["eligible_retrieval_sections"],
            stats["processed_sections"],
            stats["successful_sections"],
            stats["failed_sections"],
            stats["skipped_sections"],
            stats["segmented_sections"],
            stats["llm_segments"],
            stats["failed_segments"],
            stats["sections_with_concepts"],
            stats["concepts_written"],
            stats["concepts_rejected_by_validation"],
            stats["concepts_accepted_by_acronym"],
            stats["sections_with_acronym_supported_concepts"],
            stats["concepts_with_quality_flags"],
            stats["sections_with_quality_flags"],
            stats["documents_with_acronym_cache"],
            stats["acronyms_loaded"],
            stats["entity_review_accepted_records"],
            stats["entity_review_rejected_records"],
            stats["entity_review_export_failures"],
            stats["orphan_concepts_deleted"],
        )

        return stats
