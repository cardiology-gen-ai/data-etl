"""
add_entities.py

Entity extraction pipeline for Section nodes in the knowledge graph.

Main notes:
- Concepts are normalized before being written.
- We keep ONE Concept node per normalized concept name.
- Type ambiguity is preserved at relationship level through MENTIONS.observed_types.
- Concept-level type state is later finalized by entity_disambiguation.py.
- TYPE_ALIASES are normalized once at load time so alias lookup is consistent
  with normalize_type().
- When replace_section_mentions=True, stale MENTIONS are also cleared on failed
  or skipped-empty sections so section state stays consistent across reruns.
"""

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from neo4j import Driver

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

logger = logging.getLogger(__name__)


ALLOWED_TYPES = {
    "disease",
    "clinical_finding",
    "risk_factor",
    "genetic_factor",
    "biomarker",
    "diagnostic_test",
    "imaging_modality",
    "score_or_risk_model",
    "drug_or_drug_class",
    "procedure_or_intervention",
    "device",
    "complication_or_comorbidity",
    "care_strategy",
    "anatomical_structure",
}


_RAW_TYPE_ALIASES = {
    "phenotype": "clinical_finding",
    "finding": "clinical_finding",
    "clinical finding": "clinical_finding",
    "sign": "clinical_finding",
    "symptom": "clinical_finding",
    "sign_or_symptom": "clinical_finding",

    "management": "care_strategy",
    "therapy": "care_strategy",
    "treatment_strategy": "care_strategy",
    "care plan": "care_strategy",
    "follow_up": "care_strategy",
    "follow-up": "care_strategy",

    "drug": "drug_or_drug_class",
    "drug class": "drug_or_drug_class",
    "drug_class": "drug_or_drug_class",
    "medication": "drug_or_drug_class",
    "medication class": "drug_or_drug_class",
    "pharmacotherapy": "drug_or_drug_class",

    "procedure": "procedure_or_intervention",
    "intervention": "procedure_or_intervention",
    "surgery": "procedure_or_intervention",
    "surgical procedure": "procedure_or_intervention",

    "test": "diagnostic_test",
    "lab test": "diagnostic_test",
    "laboratory test": "diagnostic_test",

    "imaging": "imaging_modality",
    "imaging test": "imaging_modality",
    "imaging modality": "imaging_modality",

    "biological_marker": "biomarker",
    "laboratory_marker": "biomarker",
    "lab_marker": "biomarker",
    "marker": "biomarker",
    "lab value": "biomarker",

    "score": "score_or_risk_model",
    "risk score": "score_or_risk_model",
    "risk model": "score_or_risk_model",
    "prediction rule": "score_or_risk_model",
    "clinical prediction rule": "score_or_risk_model",
    "clinical score": "score_or_risk_model",
    "calculator": "score_or_risk_model",

    "complication": "complication_or_comorbidity",
    "comorbidity": "complication_or_comorbidity",

    "gene": "genetic_factor",
    "genetic": "genetic_factor",
    "genetic marker": "genetic_factor",
    "genetic variant": "genetic_factor",
    "gene variant": "genetic_factor",
    "variant": "genetic_factor",
    "mutation": "genetic_factor",

    "anatomy": "anatomical_structure",
    "structure": "anatomical_structure",
    "anatomical structure": "anatomical_structure",
}


BLOCKLIST_NAMES = {
    "diagnosis",
    "treatment",
    "management",
    "therapy",
    "follow-up",
    "follow up",
    "recommendation",
    "recommendations",
    "patient",
    "patients",
    "disease",
    "risk",
    "test",
    "tests",
    "procedure",
    "procedures",
    "drug",
    "drugs",
    "care",
    "clinical finding",
    "clinical findings",
    "symptom",
    "symptoms",
    "sign",
    "signs",
    "biomarker",
    "biomarkers",
    "diagnostic test",
    "diagnostic tests",
    "imaging modality",
    "imaging modalities",
    "score",
    "scores",
    "model",
    "models",
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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_type_token(raw_type: Any) -> str:
    """
    Normalize type strings into the same canonical lookup format used for:
    - incoming LLM types
    - TYPE_ALIASES keys
    """
    concept_type = str(raw_type).strip().lower()
    concept_type = re.sub(r"^[\s,;:.()\[\]{}'\"`]+", "", concept_type)
    concept_type = re.sub(r"[\s,;:.()\[\]{}'\"`]+$", "", concept_type)
    concept_type = normalize_whitespace(concept_type)
    concept_type = concept_type.replace("-", "_")
    concept_type = concept_type.replace(" ", "_")
    return concept_type


TYPE_ALIASES = {
    normalize_type_token(alias): normalize_type_token(target)
    for alias, target in _RAW_TYPE_ALIASES.items()
}


def normalize_type(raw_type: Any) -> str:
    concept_type = normalize_type_token(raw_type)

    if concept_type in TYPE_ALIASES:
        concept_type = TYPE_ALIASES[concept_type]

    return concept_type


def normalize_name(raw_name: Any) -> str:
    name = str(raw_name).strip().lower()
    name = normalize_whitespace(name)

    # Remove only enclosing punctuation while preserving medically relevant
    # internal characters such as hyphens, slashes, and parentheses.
    name = re.sub(r"^[\s,;:.()\[\]{}'\"`]+", "", name)
    name = re.sub(r"[\s,;:.()\[\]{}'\"`]+$", "", name)
    name = normalize_whitespace(name)

    return name


def normalize_concept(raw: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if not isinstance(raw, dict):
        logger.debug("Discarding non-dict concept payload: %r", raw)
        return None

    if "name" not in raw or "type" not in raw:
        logger.debug("Discarding concept without required keys: %r", raw)
        return None

    name = normalize_name(raw["name"])
    concept_type = normalize_type(raw["type"])

    if not name or not concept_type:
        logger.debug("Discarding concept with empty normalized fields: %r", raw)
        return None

    if name in BLOCKLIST_NAMES:
        logger.debug("Discarding blocklisted concept name: %s", name)
        return None

    if concept_type not in ALLOWED_TYPES:
        logger.debug(
            "Discarding concept with non-allowed type | name=%s | type=%s",
            name,
            concept_type,
        )
        return None

    return {
        "name": name,
        "type": concept_type,
    }


def deduplicate_concepts(concepts: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Exact deduplication by (name, type).

    This intentionally preserves the case where the same normalized name is
    returned with different types, so type ambiguity can be preserved and
    later resolved in the disambiguation step.
    """
    seen = set()
    deduped = []

    for concept in concepts:
        key = (concept["name"], concept["type"])
        if key not in seen:
            seen.add(key)
            deduped.append(concept)

    return deduped


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


def extract_concepts_single(text: str) -> Optional[List[Dict[str, str]]]:
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

        concepts = []
        for item in data:
            normalized = normalize_concept(item)
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
) -> Optional[Dict[str, List[Dict[str, str]]]]:
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
        out = {}

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

            concepts = []
            for concept_raw in concepts_raw:
                normalized = normalize_concept(concept_raw)
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


def clear_section_mentions(tx, section_uid: str) -> None:
    tx.run(
        """
        MATCH (s:Section {uid: $uid})
        OPTIONAL MATCH (s)-[r:MENTIONS]->(:Concept)
        DELETE r
        """,
        uid=section_uid,
    )


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
    concepts: List[Dict[str, str]],
    replace_section_mentions: bool = True,
) -> None:
    tx.run(
        """
        MATCH (s:Section {uid: $uid})
        SET s.entity_extracted = true,
            s.entity_extracted_at = datetime(),
            s.entity_extraction_status = 'success'
        REMOVE s.entity_extraction_failed_at
        WITH s
        OPTIONAL MATCH (s)-[old:MENTIONS]->(:Concept)
        FOREACH (
            _ IN CASE WHEN $replace_section_mentions THEN [1] ELSE [] END |
            DELETE old
        )
        """,
        uid=section_uid,
        replace_section_mentions=replace_section_mentions,
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
            c.canonical_type = concept.type,
            c.observed_types = [concept.type],
            c.created_at = datetime()
        ON MATCH SET
            c.observed_types =
                CASE
                    WHEN c.observed_types IS NULL THEN [concept.type]
                    WHEN concept.type IN c.observed_types THEN c.observed_types
                    ELSE c.observed_types + concept.type
                END,
            c.updated_at = datetime()
        MERGE (s)-[r:MENTIONS]->(c)
        ON CREATE SET
            r.observed_types = [concept.type],
            r.created_at = datetime()
        ON MATCH SET
            r.observed_types =
                CASE
                    WHEN r.observed_types IS NULL THEN [concept.type]
                    WHEN concept.type IN r.observed_types THEN r.observed_types
                    ELSE r.observed_types + concept.type
                END,
            r.updated_at = datetime()
        """,
        uid=section_uid,
        concepts=concepts,
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
) -> Dict[str, int]:
    """
    Extract concepts from section titles and optionally section text,
    then attach them to the Neo4j graph.
    """
    if max_sections_per_batch < 1:
        raise ValueError("max_sections_per_batch must be >= 1")

    model_name = get_chat_model_name()

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

        prepared_rows = []

        for record in result:
            row = {
                "uid": record["uid"],
                "doc_id": record["doc_id"],
                "section_id": record["section_id"],
                "title": record["title"],
                "text": record["text"],
            }

            source_text = build_source_text(
                row=row,
                use_section_text=use_section_text,
            )

            if not source_text.strip():
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

        logger.info(
            "Preparing entity extraction for %d sections%s | model=%s | max_sections_per_batch=%d",
            len(prepared_rows),
            f" in document {doc_id}" if doc_id else "",
            model_name,
            max_sections_per_batch,
        )

        stats = {
            "processed_sections": 0,
            "successful_sections": 0,
            "failed_sections": 0,
            "sections_with_concepts": 0,
            "concepts_written": 0,
        }

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

            batch_result = None
            if len(batch) > 1:
                batch_result = extract_concepts_batch(batch)

            if batch_result is not None:
                for row in batch:
                    section_uid = row["uid"]
                    concepts = batch_result.get(section_uid, [])

                    stats["processed_sections"] += 1
                    stats["successful_sections"] += 1

                    if concepts:
                        stats["sections_with_concepts"] += 1
                        stats["concepts_written"] += len(concepts)

                    logger.info(
                        "Section doc=%s section=%s -> %d concepts",
                        row["doc_id"],
                        row["section_id"],
                        len(concepts),
                    )

                    if concepts:
                        logger.info(
                            " -> %s",
                            ", ".join(f"{c['name']} [{c['type']}]" for c in concepts),
                        )

                    session.execute_write(
                        write_section_concepts,
                        section_uid,
                        concepts,
                        replace_section_mentions,
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

                    stats["successful_sections"] += 1

                    if concepts:
                        stats["sections_with_concepts"] += 1
                        stats["concepts_written"] += len(concepts)

                    logger.info(
                        "Section doc=%s section=%s -> %d concepts",
                        row["doc_id"],
                        row["section_id"],
                        len(concepts),
                    )

                    if concepts:
                        logger.info(
                            " -> %s",
                            ", ".join(f"{c['name']} [{c['type']}]" for c in concepts),
                        )

                    session.execute_write(
                        write_section_concepts,
                        row["uid"],
                        concepts,
                        replace_section_mentions,
                    )

        logger.info("Processed %d LLM batches", batch_count)

        logger.info(
            "Entity extraction completed | processed=%d | successful=%d | failed=%d | sections_with_concepts=%d | concepts_written=%d",
            stats["processed_sections"],
            stats["successful_sections"],
            stats["failed_sections"],
            stats["sections_with_concepts"],
            stats["concepts_written"],
        )

        return stats