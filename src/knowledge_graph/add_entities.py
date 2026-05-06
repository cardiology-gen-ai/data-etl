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
- When use_section_text=True, title-only sections with empty body text are skipped:
  they remain useful as hierarchy/navigation nodes, but they should not create
  normal body-grounded entity mentions.
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
from knowledge_graph.validate_entities import (
    validate_concepts_against_source,
    summarize_rejections,
)


logger = logging.getLogger(__name__)


CONCEPT_DEBUG_LOG_LIMIT = 30


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


def emergency_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def pack_rows_for_llm(
    rows: List[Dict[str, Any]],
    max_sections_per_batch: int,
    max_batch_chars: int,
    emergency_max_single_chars: Optional[int] = None,
) -> Iterable[List[Dict[str, Any]]]:
    """
    Pack rows into batches without cutting sections apart just to make a batch fit.

    Policy:
    - keep sections intact when building batches
    - if adding the next section would exceed the batch budget, move it to the next batch
    - only truncate as a last resort if a single section is too large on its own
    """
    batch = []
    current_chars = 0

    for row in rows:
        row_text = row["source_text"]
        row_chars = len(row_text)

        if row_chars > max_batch_chars:
            logger.warning(
                "Single section exceeds batch budget | uid=%s | chars=%d | budget=%d",
                row["uid"],
                row_chars,
                max_batch_chars,
            )

            oversized_row = dict(row)

            if emergency_max_single_chars is not None:
                effective_single_limit = min(emergency_max_single_chars, max_batch_chars)
                oversized_row["source_text"] = emergency_truncate(
                    oversized_row["source_text"],
                    effective_single_limit,
                )
                logger.warning(
                    "Emergency truncation applied to oversized section | uid=%s | new_chars=%d | effective_limit=%d",
                    oversized_row["uid"],
                    len(oversized_row["source_text"]),
                    effective_single_limit,
                )

            if len(oversized_row["source_text"]) > max_batch_chars:
                logger.warning(
                    "Oversized section still exceeds batch budget after truncation | uid=%s | chars=%d | budget=%d",
                    oversized_row["uid"],
                    len(oversized_row["source_text"]),
                    max_batch_chars,
                )

            if batch:
                yield batch
                batch = []
                current_chars = 0

            yield [oversized_row]
            continue

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
        content = generate_chat_text(messages=messages, json_mode=True)
    except Exception as e:
        logger.exception("Single request failed: %s", e)
        return None

    if content is None:
        logger.error("Single extraction backend returned None")
        return None

    content = content.strip()

    try:
        data = parse_llm_json(content)
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
    if not batch_rows:
        return {}

    sections_payload = [
        {
            "uid": row["uid"],
            "text": row["source_text"],
        }
        for row in batch_rows
    ]

    messages = [
        {"role": "system", "content": ENTITY_EXTRACTION_BATCH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_entity_extraction_batch_user_prompt(sections_payload),
        },
    ]

    try:
        content = generate_chat_text(messages=messages, json_mode=True)
    except Exception as e:
        logger.exception("Batch request failed: %s", e)
        return None

    if content is None:
        logger.error("Batch extraction backend returned None")
        return None

    content = content.strip()

    try:
        data = parse_llm_json(content)
        if not isinstance(data, list):
            raise ValueError("Batch LLM output is not a list")

        expected_uids = {row["uid"] for row in batch_rows}
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
                normalized = normalize_llm_concept_preserving_raw(concept_raw)
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
            c.type_resolution_status AS previous_status,
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
                    WHEN previous_status IS NULL THEN NULL
                    WHEN old_status IN ['single_type', 'resolved']
                         AND NOT (concept.type IN old_observed_types)
                    THEN NULL
                    ELSE c.canonical_type
                END,
            c.type_resolution_status =
                CASE
                    WHEN old_status IN ['single_type', 'resolved']
                         AND NOT (concept.type IN old_observed_types)
                    THEN 'pending'
                    ELSE old_status
                END,
            c.needs_type_review =
                CASE
                    WHEN old_status IN ['single_type', 'resolved']
                         AND NOT (concept.type IN old_observed_types)
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
            r.raw_type = concept.raw_type
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
            r.raw_type = concept.raw_type
        """,
        uid=section_uid,
        concepts=concepts,
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

    stats["successful_sections"] += 1
    stats["concepts_rejected_by_validation"] += len(rejected)
    stats["concepts_accepted_by_acronym"] += len(accepted_by_acronym)

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
    Extract concepts from section titles and optionally section text,
    validate them against the section source text,
    optionally export accepted/rejected validation decisions for review,
    then attach the accepted concepts to the Neo4j graph.

    Acronym validation:
    - if acronym_dir is provided and use_acronym_validation=True, this loads
      cached per-document acronym JSON files;
    - validate_entities.py can then accept a concept when its long form is
      supported by an acronym short form present in the section text;
    - validate_entities.py can also expand a raw acronym short form extracted by
      the LLM into its cached long form before graph writing.

    Empty-body policy:
    - when use_section_text=True, title-only sections with empty body text are
      skipped and marked as skipped_empty;
    - when use_section_text=False, title-only extraction is still allowed.
    """
    if max_sections_per_batch < 1:
        raise ValueError("max_sections_per_batch must be >= 1")

    model_name = get_chat_model_name()
    entity_review_run_id = f"entity_extraction::{utc_now_iso()}"

    stats = {
        "processed_sections": 0,
        "successful_sections": 0,
        "failed_sections": 0,
        "skipped_sections": 0,
        "sections_with_concepts": 0,
        "concepts_written": 0,
        "concepts_rejected_by_validation": 0,
        "concepts_accepted_by_acronym": 0,
        "sections_with_acronym_supported_concepts": 0,
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
            s.text AS text
        ORDER BY s.uid
        """

        if max_sections is not None:
            query += "\nLIMIT $max_sections"

        result = session.run(
            query,
            doc_id=doc_id,
            max_sections=max_sections,
        )

        prepared_rows: List[Dict[str, Any]] = []

        for record in result:
            row = {
                "uid": record["uid"],
                "doc_id": record["doc_id"],
                "section_id": record["section_id"],
                "title": record["title"],
                "text": record["text"],
            }

            if use_section_text and not has_section_body(row):
                stats["skipped_sections"] += 1
                session.execute_write(
                    mark_section_extraction_skipped_empty,
                    row["uid"],
                    replace_section_mentions,
                )
                logger.info(
                    "Skipping section with empty body | doc=%s section=%s title=%r",
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
                logger.info(
                    "Skipping empty section | doc=%s section=%s",
                    row["doc_id"],
                    row["section_id"],
                )
                continue

            row["source_text"] = source_text
            prepared_rows.append(row)

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
                len(acronyms) for acronyms in acronyms_by_doc_id.values()
            )

            logger.info(
                "Acronym validation enabled | docs_with_acronyms=%d/%d | acronyms_loaded=%d | acronym_dir=%s",
                stats["documents_with_acronym_cache"],
                len(review_doc_ids),
                stats["acronyms_loaded"],
                acronym_dir,
            )

        elif use_acronym_validation and acronym_dir is None:
            logger.info(
                "Acronym validation requested but acronym_dir is None; using direct source validation only"
            )

        else:
            logger.info("Acronym validation disabled")

        if export_entity_review and clear_previous_entity_review and review_doc_ids:
            clear_entity_review_exports_safely(
                doc_ids=review_doc_ids,
                output_dir=entity_review_output_dir,
            )

        logger.info(
            "Preparing entity extraction for %d sections%s | skipped_empty=%d | model=%s | max_sections_per_batch=%d",
            len(prepared_rows),
            f" in document {doc_id}" if doc_id else "",
            stats["skipped_sections"],
            model_name,
            max_sections_per_batch,
        )

        if export_entity_review:
            logger.info(
                "Entity review exports enabled | docs=%d | output_dir=%s | include_source_preview=%s",
                len(review_doc_ids),
                entity_review_output_dir or "default",
                include_source_preview_in_review,
            )

        batch_count = 0

        for batch in pack_rows_for_llm(
            prepared_rows,
            max_sections_per_batch=max_sections_per_batch,
            max_batch_chars=max_batch_chars,
            emergency_max_single_chars=emergency_max_single_chars,
        ):
            batch_count += 1
            logger.info(
                "Extracting entities for batch %d of size %d sections",
                batch_count,
                len(batch),
            )

            # Direct single-section path: no fake "batch failed" warning when batch size is 1.
            if len(batch) == 1:
                row = batch[0]
                stats["processed_sections"] += 1

                concepts = extract_concepts_single(row["source_text"])

                if concepts is None:
                    stats["failed_sections"] += 1
                    session.execute_write(
                        mark_section_extraction_failed,
                        row["uid"],
                        replace_section_mentions,
                    )
                    logger.warning(
                        "Skipping section after failed extraction | doc=%s section=%s",
                        row["doc_id"],
                        row["section_id"],
                    )
                    continue

                process_extracted_concepts(
                    session=session,
                    row=row,
                    concepts=concepts,
                    stats=stats,
                    replace_section_mentions=replace_section_mentions,
                    export_entity_review=export_entity_review,
                    entity_review_output_dir=entity_review_output_dir,
                    entity_review_run_id=entity_review_run_id,
                    include_source_preview_in_review=include_source_preview_in_review,
                    acronyms=acronyms_by_doc_id.get(row["doc_id"], {}),
                )
                continue

            # True multi-section batch path.
            batch_result = extract_concepts_batch(batch)

            if batch_result is not None:
                for row in batch:
                    stats["processed_sections"] += 1
                    concepts = batch_result.get(row["uid"], [])

                    process_extracted_concepts(
                        session=session,
                        row=row,
                        concepts=concepts,
                        stats=stats,
                        replace_section_mentions=replace_section_mentions,
                        export_entity_review=export_entity_review,
                        entity_review_output_dir=entity_review_output_dir,
                        entity_review_run_id=entity_review_run_id,
                        include_source_preview_in_review=include_source_preview_in_review,
                        acronyms=acronyms_by_doc_id.get(row["doc_id"], {}),
                    )

            else:
                logger.warning(
                    "Batch extraction failed; falling back to single-section extraction for %d sections",
                    len(batch),
                )

                for row in batch:
                    stats["processed_sections"] += 1

                    concepts = extract_concepts_single(row["source_text"])

                    if concepts is None:
                        stats["failed_sections"] += 1
                        session.execute_write(
                            mark_section_extraction_failed,
                            row["uid"],
                            replace_section_mentions,
                        )
                        logger.warning(
                            "Skipping section after failed extraction | doc=%s section=%s",
                            row["doc_id"],
                            row["section_id"],
                        )
                        continue

                    process_extracted_concepts(
                        session=session,
                        row=row,
                        concepts=concepts,
                        stats=stats,
                        replace_section_mentions=replace_section_mentions,
                        export_entity_review=export_entity_review,
                        entity_review_output_dir=entity_review_output_dir,
                        entity_review_run_id=entity_review_run_id,
                        include_source_preview_in_review=include_source_preview_in_review,
                        acronyms=acronyms_by_doc_id.get(row["doc_id"], {}),
                    )

        logger.info("Processed %d LLM batches", batch_count)

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
            "Entity extraction completed | processed=%d | successful=%d | failed=%d | skipped=%d | sections_with_concepts=%d | concepts_written=%d | concepts_rejected_by_validation=%d | acronym_supported_concepts=%d | sections_with_acronym_supported_concepts=%d | docs_with_acronym_cache=%d | acronyms_loaded=%d | review_accepted=%d | review_rejected=%d | review_export_failures=%d | orphan_concepts_deleted=%d",
            stats["processed_sections"],
            stats["successful_sections"],
            stats["failed_sections"],
            stats["skipped_sections"],
            stats["sections_with_concepts"],
            stats["concepts_written"],
            stats["concepts_rejected_by_validation"],
            stats["concepts_accepted_by_acronym"],
            stats["sections_with_acronym_supported_concepts"],
            stats["documents_with_acronym_cache"],
            stats["acronyms_loaded"],
            stats["entity_review_accepted_records"],
            stats["entity_review_rejected_records"],
            stats["entity_review_export_failures"],
            stats["orphan_concepts_deleted"],
        )

        return stats