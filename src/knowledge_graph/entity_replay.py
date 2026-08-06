"""Deterministic replay of entity validation from existing review artifacts.

This module reuses accepted/rejected JSONL produced by the entity extraction
run. It never calls an LLM. It rebuilds the current validation decisions from
Section-view text and acronym caches, creates an auditable replay plan, and can
replace only Section-[:MENTIONS]->Concept relationships in Neo4j.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver
else:
    Driver = Any

from knowledge_graph.entity_schema import ALLOWED_TYPES, BLOCKLIST_NAMES
from knowledge_graph.relationship_metadata import build_mention_relationship_metadata
from knowledge_graph.validate_entities import (
    collapse_validated_concepts_by_name,
    deduplicate_validated_concepts,
    get_out_of_schema_rejection,
    normalize_mention_evidence,
    validate_single_concept,
)


REPLAY_VERSION = "entity_validation_replay_v1"
RETRIEVAL_ROLE = "retrieval"


@dataclass(frozen=True)
class SectionSource:
    uid: str
    doc_id: str
    section_id: str
    title: str
    text: str
    source_text: str


@dataclass
class ReplayPlan:
    sections: List[Dict[str, Any]]
    changed_decisions: List[Dict[str, Any]]
    newly_accepted: List[Dict[str, Any]]
    newly_rejected: List[Dict[str, Any]]
    summary: Dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected an object at {path}:{line_no}, got "
                    f"{type(payload).__name__}"
                )
            rows.append(payload)
    return rows


def load_review_records(
    review_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for path in sorted(review_dir.glob("*_accepted.jsonl")):
        accepted.extend(read_jsonl(path))
    for path in sorted(review_dir.glob("*_rejected.jsonl")):
        rejected.extend(read_jsonl(path))

    if not accepted:
        raise FileNotFoundError(
            f"No non-empty *_accepted.jsonl files found under {review_dir}"
        )
    if not rejected:
        raise FileNotFoundError(
            f"No non-empty *_rejected.jsonl files found under {review_dir}"
        )

    return accepted, rejected


def _build_source_text(title: Any, text: Any) -> str:
    parts: List[str] = []
    clean_title = str(title or "").strip()
    clean_text = str(text or "").strip()
    if clean_title:
        parts.append(f"Title: {clean_title}")
    if clean_text:
        parts.append(f"Body:\n{clean_text}")
    return "\n\n".join(parts).strip()


def load_section_sources(section_view_dir: Path) -> Dict[str, SectionSource]:
    sources: Dict[str, SectionSource] = {}

    for path in sorted(section_view_dir.glob("*_section_view_*.json")):
        if path.name.endswith("_validation.json"):
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a list in Section view: {path}")

        for row in payload:
            if not isinstance(row, dict):
                continue
            if row.get("section_view_role") != RETRIEVAL_ROLE:
                continue
            if not bool(row.get("embed", False)):
                continue
            if bool(row.get("excluded", False)):
                continue

            doc_id = str(row.get("doc_id") or "").strip()
            section_id = str(row.get("section_id") or "").strip()
            uid = str(row.get("uid") or "").strip()
            if not uid and doc_id and section_id:
                uid = f"{doc_id}::{section_id}"
            title = str(
                row.get("section_title")
                or row.get("title")
                or ""
            ).strip()
            text = str(row.get("text") or "").strip()

            if not uid or not doc_id or not section_id:
                raise ValueError(
                    f"Retrieval Section missing identity fields in {path}: {row}"
                )
            if uid in sources:
                raise ValueError(f"Duplicate retrieval Section UID: {uid}")
            if not text:
                raise ValueError(f"Retrieval Section has empty body: {uid}")

            sources[uid] = SectionSource(
                uid=uid,
                doc_id=doc_id,
                section_id=section_id,
                title=title,
                text=text,
                source_text=_build_source_text(title, text),
            )

    if not sources:
        raise FileNotFoundError(
            f"No active retrieval Section views found under {section_view_dir}"
        )

    return sources


def load_acronyms(acronym_dir: Path) -> Dict[str, Dict[str, str]]:
    acronyms_by_doc: Dict[str, Dict[str, str]] = {}

    for path in sorted(acronym_dir.glob("*_acronyms.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object in acronym cache: {path}")

        doc_id = str(payload.get("doc_id") or "").strip()
        acronyms = payload.get("acronyms") or {}
        if not doc_id or not isinstance(acronyms, dict):
            continue

        acronyms_by_doc[doc_id] = {
            str(short): str(definition)
            for short, definition in acronyms.items()
        }

    return acronyms_by_doc


def _record_uid(record: Mapping[str, Any]) -> str:
    uid = str(record.get("section_uid") or "").strip()
    if uid:
        return uid

    doc_id = str(record.get("doc_id") or "").strip()
    section_id = str(record.get("section_id") or "").strip()
    if doc_id and section_id:
        return f"{doc_id}::{section_id}"
    return ""


_CONCEPT_FIELDS = (
    "name",
    "type",
    "raw_name",
    "raw_type",
    "validation_reason",
    "support_method",
    "matched_text",
    "matched_pattern",
    "acronym_short",
    "acronym_definition",
    "acronym_match_method",
    "expanded_from_acronym",
    "quality_flags",
)


def _accepted_concept_from_review(
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    concept = {
        field: record.get(field)
        for field in _CONCEPT_FIELDS
        if record.get(field) not in (None, "")
    }
    concept["validation_reason"] = str(
        concept.get("validation_reason")
        or record.get("reason")
        or "accepted"
    )
    return normalize_mention_evidence(concept)


def _raw_candidate(record: Mapping[str, Any]) -> Dict[str, Any]:
    raw_name = record.get("raw_name") or record.get("name")
    raw_type = record.get("raw_type") or record.get("type")
    return {
        "name": raw_name,
        "type": raw_type,
        "raw_name": raw_name,
        "raw_type": raw_type,
    }


def _candidate_key(candidate: Mapping[str, Any]) -> Tuple[str, str]:
    return (
        str(candidate.get("raw_name") or candidate.get("name") or "").strip(),
        str(candidate.get("raw_type") or candidate.get("type") or "").strip(),
    )


def _assert_review_run_consistency(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, List[str]]:
    runs_by_doc: Dict[str, set[str]] = defaultdict(set)
    for row in records:
        doc_id = str(row.get("doc_id") or "").strip()
        run_id = str(row.get("run_id") or "UNKNOWN").strip() or "UNKNOWN"
        if doc_id:
            runs_by_doc[doc_id].add(run_id)

    inconsistent = {
        doc_id: sorted(run_ids)
        for doc_id, run_ids in runs_by_doc.items()
        if len(run_ids) != 1
    }
    if inconsistent:
        raise ValueError(
            "Review directory mixes multiple runs for the same document: "
            f"{inconsistent}"
        )

    return {
        doc_id: sorted(run_ids)
        for doc_id, run_ids in sorted(runs_by_doc.items())
    }


def _decision_key(
    section_uid: str,
    concept: Mapping[str, Any],
) -> Tuple[str, str, str]:
    return (
        section_uid,
        str(concept.get("name") or "").strip(),
        str(concept.get("type") or "").strip(),
    )


def _decision_record(
    section: SectionSource,
    concept: Mapping[str, Any],
    *,
    old_status: str,
    new_status: str,
) -> Dict[str, Any]:
    return {
        "section_uid": section.uid,
        "doc_id": section.doc_id,
        "section_id": section.section_id,
        "section_title": section.title,
        "name": concept.get("name"),
        "type": concept.get("type"),
        "raw_name": concept.get("raw_name"),
        "raw_type": concept.get("raw_type"),
        "old_status": old_status,
        "new_status": new_status,
        "reason": concept.get("reason") or concept.get("validation_reason"),
        "validation_reason": concept.get("validation_reason"),
        "support_method": concept.get("support_method"),
        "quality_flags": concept.get("quality_flags") or [],
    }


def build_replay_plan(
    *,
    accepted_records: Sequence[Dict[str, Any]],
    rejected_records: Sequence[Dict[str, Any]],
    section_sources: Mapping[str, SectionSource],
    acronyms_by_doc: Mapping[str, Mapping[str, str]],
) -> ReplayPlan:
    """Build a replay plan for the validation patch without re-running the LLM.

    The replay is intentionally delta-oriented:
    - previously accepted records are preserved, with evidence normalization;
    - the new research/publication exclusions are applied to accepted records;
    - only rejected records affected by the narrowed organization/research
      rules are revalidated against source text;
    - all other rejected decisions remain unchanged.

    This avoids treating review exports as if they were raw LLM output and keeps
    the replay limited to the behavior changed by the reviewed patch.
    """
    all_records: List[Dict[str, Any]] = [
        *accepted_records,
        *rejected_records,
    ]
    run_ids_by_doc = _assert_review_run_consistency(all_records)

    unknown_uids = sorted(
        {
            _record_uid(record)
            for record in all_records
            if _record_uid(record) not in section_sources
        }
    )
    if unknown_uids:
        raise ValueError(
            "Review records refer to unknown/non-retrieval Sections: "
            f"{unknown_uids[:20]}"
        )

    accepted_by_uid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rejected_by_uid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in accepted_records:
        accepted_by_uid[_record_uid(record)].append(record)
    for record in rejected_records:
        rejected_by_uid[_record_uid(record)].append(record)

    old_accepted_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for record in accepted_records:
        uid = _record_uid(record)
        old_accepted_by_key.setdefault(_decision_key(uid, record), record)

    affected_rejection_reasons = {
        "organization_not_supported_entity_type",
        "nonclinical_research_or_variable",
    }

    plan_sections: List[Dict[str, Any]] = []
    new_accepted_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    new_rejected_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    accepted_reasons: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    preserved_rejected_count = 0
    revalidated_rejected_count = 0
    type_assertion_count = 0
    relationship_count = 0
    sections_with_concepts = 0

    for uid, section in sorted(
        section_sources.items(),
        key=lambda item: (item[1].doc_id, item[1].section_id),
    ):
        accepted_for_section: List[Dict[str, Any]] = []
        rejected_for_section: List[Dict[str, Any]] = []

        for old_record in accepted_by_uid.get(uid, []):
            accepted_concept = _accepted_concept_from_review(old_record)
            name = str(accepted_concept.get("name") or "").strip()
            concept_type = str(accepted_concept.get("type") or "").strip()
            rejection = get_out_of_schema_rejection(
                name=name,
                concept_type=concept_type,
                blocklist_names=BLOCKLIST_NAMES,
            )

            if rejection is None:
                accepted_for_section.append(accepted_concept)
                continue

            rejected_concept = dict(accepted_concept)
            rejected_concept["reason"] = str(rejection["reason"])
            flags = list(rejection.get("quality_flags") or [])
            if flags:
                rejected_concept["quality_flags"] = flags
            rejected_for_section.append(rejected_concept)

        for old_record in rejected_by_uid.get(uid, []):
            old_reason = str(old_record.get("reason") or "").strip()

            if old_reason not in affected_rejection_reasons:
                preserved = dict(old_record)
                preserved["reason"] = old_reason or "unknown"
                rejected_for_section.append(preserved)
                preserved_rejected_count += 1
                continue

            revalidated_rejected_count += 1
            result = validate_single_concept(
                concept=_raw_candidate(old_record),
                source_text=section.source_text,
                allowed_types=ALLOWED_TYPES,
                blocklist_names=BLOCKLIST_NAMES,
                acronyms=dict(acronyms_by_doc.get(section.doc_id, {})),
            )

            concept = result["concept"]
            if result["accepted"]:
                accepted_for_section.append(concept)
            else:
                rejected_concept = dict(concept)
                rejected_concept["reason"] = str(result["reason"])
                rejected_for_section.append(rejected_concept)

        accepted = deduplicate_validated_concepts(accepted_for_section)
        collapsed = collapse_validated_concepts_by_name(accepted)

        if collapsed:
            sections_with_concepts += 1
        type_assertion_count += len(accepted)
        relationship_count += len(collapsed)

        for concept in accepted:
            key = _decision_key(uid, concept)
            new_accepted_by_key[key] = concept
            accepted_reasons[
                str(concept.get("validation_reason") or "UNKNOWN")
            ] += 1

        for concept in rejected_for_section:
            key = _decision_key(uid, concept)
            new_rejected_by_key[key] = concept
            rejection_reasons[str(concept.get("reason") or "UNKNOWN")] += 1

        plan_sections.append(
            {
                "uid": section.uid,
                "doc_id": section.doc_id,
                "section_id": section.section_id,
                "section_title": section.title,
                "input_accepted_records": len(accepted_by_uid.get(uid, [])),
                "input_rejected_records": len(rejected_by_uid.get(uid, [])),
                "accepted_type_assertions": len(accepted),
                "rejected_candidates": len(rejected_for_section),
                "concepts": collapsed,
            }
        )

    old_keys = set(old_accepted_by_key)
    new_keys = set(new_accepted_by_key)
    accepted_to_rejected_keys = sorted(old_keys - new_keys)
    rejected_to_accepted_keys = sorted(new_keys - old_keys)

    newly_rejected: List[Dict[str, Any]] = []
    for key in accepted_to_rejected_keys:
        uid, _, _ = key
        section = section_sources[uid]
        concept = new_rejected_by_key.get(key) or old_accepted_by_key[key]
        newly_rejected.append(
            _decision_record(
                section,
                concept,
                old_status="accepted",
                new_status="rejected",
            )
        )

    newly_accepted: List[Dict[str, Any]] = []
    for key in rejected_to_accepted_keys:
        uid, _, _ = key
        section = section_sources[uid]
        concept = new_accepted_by_key[key]
        newly_accepted.append(
            _decision_record(
                section,
                concept,
                old_status="rejected_or_absent",
                new_status="accepted",
            )
        )

    changed_decisions = sorted(
        [*newly_rejected, *newly_accepted],
        key=lambda row: (
            str(row.get("doc_id") or ""),
            str(row.get("section_id") or ""),
            str(row.get("name") or ""),
            str(row.get("type") or ""),
        ),
    )

    acronym_incoherent = 0
    gene_symbol_count = 0
    multi_type_relationships = 0
    for section_row in plan_sections:
        for concept in section_row["concepts"]:
            if len(concept.get("observed_types") or []) > 1:
                multi_type_relationships += 1
            if concept.get("validation_reason") == "accepted_gene_symbol_direct_source":
                gene_symbol_count += 1
            if concept.get("expanded_from_acronym") is True:
                raw_name = str(concept.get("raw_name") or "").strip().casefold()
                short = str(concept.get("acronym_short") or "").strip().casefold()
                if not raw_name or not short or raw_name != short:
                    acronym_incoherent += 1

    doc_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "sections": 0,
            "sections_with_concepts": 0,
            "type_assertions": 0,
            "mentions": 0,
        }
    )
    for row in plan_sections:
        doc = doc_counts[row["doc_id"]]
        doc["sections"] += 1
        if row["concepts"]:
            doc["sections_with_concepts"] += 1
        doc["type_assertions"] += row["accepted_type_assertions"]
        doc["mentions"] += len(row["concepts"])

    summary = {
        "replay_version": REPLAY_VERSION,
        "generated_at": utc_now_iso(),
        "review_run_ids_by_doc": run_ids_by_doc,
        "input_accepted_records": len(accepted_records),
        "input_rejected_records": len(rejected_records),
        "retrieval_sections": len(plan_sections),
        "sections_with_concepts": sections_with_concepts,
        "accepted_type_assertions": type_assertion_count,
        "planned_mentions": relationship_count,
        "accepted_to_rejected": len(newly_rejected),
        "rejected_to_accepted": len(newly_accepted),
        "changed_decisions": len(changed_decisions),
        "revalidated_rejected_records": revalidated_rejected_count,
        "preserved_rejected_records": preserved_rejected_count,
        "multi_type_mentions": multi_type_relationships,
        "acronym_metadata_incoherent": acronym_incoherent,
        "gene_symbol_mentions": gene_symbol_count,
        "accepted_by_reason": dict(accepted_reasons.most_common()),
        "rejected_by_reason": dict(rejection_reasons.most_common()),
        "by_document": dict(sorted(doc_counts.items())),
    }

    if acronym_incoherent:
        raise AssertionError(
            "Replay plan still contains incoherent acronym metadata: "
            f"{acronym_incoherent}"
        )

    return ReplayPlan(
        sections=plan_sections,
        changed_decisions=changed_decisions,
        newly_accepted=newly_accepted,
        newly_rejected=newly_rejected,
        summary=summary,
    )


def _json_default(value: Any) -> str:
    """Serialize Neo4j temporal/spatial values in audit artifacts.

    Neo4j returns values such as ``neo4j.time.DateTime`` inside
    ``properties(...)`` maps. The standard JSON encoder cannot serialize
    them directly, even though they have stable textual representations.
    Prefer ISO output when the object exposes it and fall back to ``str``
    for other driver-specific scalar values.
    """

    neo4j_iso_format = getattr(value, "iso_format", None)
    if callable(neo4j_iso_format):
        return str(neo4j_iso_format())

    python_isoformat = getattr(value, "isoformat", None)
    if callable(python_isoformat):
        return str(python_isoformat())

    return str(value)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )


def write_replay_artifacts(plan: ReplayPlan, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(plan.summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "section_plan.jsonl", plan.sections)
    write_jsonl(output_dir / "changed_decisions.jsonl", plan.changed_decisions)
    write_jsonl(output_dir / "newly_accepted.jsonl", plan.newly_accepted)
    write_jsonl(output_dir / "newly_rejected.jsonl", plan.newly_rejected)


def _graph_preflight(tx, section_uids: Sequence[str]) -> Dict[str, int]:
    record = tx.run(
        """
        MATCH (s:Section)
        WHERE s.uid IN $section_uids
        WITH collect(s) AS sections
        RETURN
            size(sections) AS matched_sections,
            size([
                s IN sections
                WHERE s.section_view_role = 'retrieval'
                  AND coalesce(s.embed, false) = true
                  AND coalesce(s.excluded, false) = false
            ]) AS eligible_sections,
            size([
                s IN sections
                WHERE s.entity_extraction_status = 'success'
                  AND coalesce(s.entity_extracted, false) = true
            ]) AS successful_sections
        """,
        section_uids=list(section_uids),
    ).single()

    entity_count_record = tx.run(
        """
        MATCH (s:Section)-[r:MENTIONS]->(:Concept)
        WHERE s.uid IN $section_uids
        WITH count(r) AS current_mentions
        MATCH (c:Concept)
        RETURN current_mentions, count(c) AS current_concepts
        """,
        section_uids=list(section_uids),
    ).single()

    normalization_record = tx.run(
        """
        MATCH (c:Concept)
        WITH
            count(CASE
                WHEN properties(c)['cui'] IS NOT NULL THEN 1
            END) AS concepts_with_cui,
            count(CASE
                WHEN properties(c)['normalization_status'] IS NOT NULL THEN 1
            END) AS concepts_with_normalization_status
        OPTIONAL MATCH ()-[r]->()
        WHERE type(r) = 'SAME_AS'
        RETURN
            concepts_with_cui,
            concepts_with_normalization_status,
            count(r) AS same_as_edges
        """
    ).single()

    return {
        "matched_sections": int(record["matched_sections"] or 0),
        "eligible_sections": int(record["eligible_sections"] or 0),
        "successful_sections": int(record["successful_sections"] or 0),
        "current_mentions": int(entity_count_record["current_mentions"] or 0),
        "current_concepts": int(entity_count_record["current_concepts"] or 0),
        "concepts_with_cui": int(normalization_record["concepts_with_cui"] or 0),
        "concepts_with_normalization_status": int(
            normalization_record["concepts_with_normalization_status"] or 0
        ),
        "same_as_edges": int(normalization_record["same_as_edges"] or 0),
    }


def graph_preflight(driver: Driver, plan: ReplayPlan) -> Dict[str, int]:
    section_uids = [row["uid"] for row in plan.sections]
    with driver.session() as session:
        result = session.execute_read(_graph_preflight, section_uids)

    expected = len(section_uids)
    for field in (
        "matched_sections",
        "eligible_sections",
        "successful_sections",
    ):
        if result[field] != expected:
            raise RuntimeError(
                f"Neo4j preflight failed: {field}={result[field]}, "
                f"expected={expected}"
            )

    normalized_state = (
        result["concepts_with_cui"]
        + result["concepts_with_normalization_status"]
        + result["same_as_edges"]
    )
    if normalized_state:
        raise RuntimeError(
            "Neo4j already contains normalization artifacts; refusing to "
            f"rewrite entity mentions: {result}"
        )

    return result


def _read_graph_backup(tx, section_uids: Sequence[str]) -> List[Dict[str, Any]]:
    rows = tx.run(
        """
        MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
        WHERE s.uid IN $section_uids
        RETURN
            s.uid AS section_uid,
            s.doc_id AS doc_id,
            s.section_id AS section_id,
            c.name AS concept_name,
            properties(r) AS mention_properties,
            properties(c) AS concept_properties
        ORDER BY s.doc_id, s.retrieval_order, c.name
        """,
        section_uids=list(section_uids),
    )
    return [dict(row) for row in rows]


def backup_graph_entity_layer(
    driver: Driver,
    plan: ReplayPlan,
    output_dir: Path,
) -> Dict[str, int]:
    section_uids = [row["uid"] for row in plan.sections]
    with driver.session() as session:
        rows = session.execute_read(_read_graph_backup, section_uids)

    backup_path = output_dir / "neo4j_before_replay.jsonl"
    write_jsonl(backup_path, rows)
    return {
        "backup_rows": len(rows),
        "backup_path": str(backup_path),
    }


def _rewrite_mentions(tx, sections: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    section_uids = [row["uid"] for row in sections]
    relationship_metadata = build_mention_relationship_metadata()

    tx.run(
        """
        MATCH (s:Section)-[r:MENTIONS]->(:Concept)
        WHERE s.uid IN $section_uids
        DELETE r
        """,
        section_uids=section_uids,
    ).consume()

    rows = tx.run(
        """
        UNWIND $sections AS section_row
        MATCH (s:Section {uid: section_row.uid})
        SET s.entity_extracted = true,
            s.entity_extraction_status = 'success',
            s.entity_revalidated_at = datetime(),
            s.entity_validation_version = $replay_version
        REMOVE s.entity_extraction_failed_at

        WITH s, section_row
        UNWIND section_row.concepts AS concept
        MERGE (c:Concept {name: concept.name})
        ON CREATE SET c.created_at = datetime()
        SET c.updated_at = datetime()

        MERGE (s)-[r:MENTIONS]->(c)
        SET r.observed_types = concept.observed_types,
            r.validation_reason = concept.validation_reason,
            r.support_method = concept.support_method,
            r.matched_text = concept.matched_text,
            r.matched_pattern = concept.matched_pattern,
            r.acronym_short = concept.acronym_short,
            r.acronym_definition = concept.acronym_definition,
            r.acronym_match_method = concept.acronym_match_method,
            r.expanded_from_acronym = coalesce(
                concept.expanded_from_acronym,
                false
            ),
            r.raw_name = concept.raw_name,
            r.raw_type = concept.raw_type,
            r.quality_flags = coalesce(concept.quality_flags, []),
            r.entity_validation_version = $replay_version,
            r.entity_revalidated_at = datetime(),
            r.created_at = coalesce(r.created_at, datetime()),
            r.updated_at = datetime()
        SET r += $relationship_metadata
        FOREACH (_ IN CASE
            WHEN s.doc_id IS NULL OR trim(toString(s.doc_id)) = '' THEN []
            ELSE [1]
        END |
            SET r.doc_id = trim(toString(s.doc_id))
        )
        RETURN count(r) AS mentions_written
        """,
        sections=list(sections),
        replay_version=REPLAY_VERSION,
        relationship_metadata=relationship_metadata,
    ).single()

    orphan_record = tx.run(
        """
        MATCH (c:Concept)
        WHERE NOT EXISTS {
            MATCH (:Section)-[:MENTIONS]->(c)
        }
        WITH collect(c) AS orphans, count(c) AS orphan_count
        FOREACH (c IN orphans | DETACH DELETE c)
        RETURN orphan_count
        """
    ).single()

    count_record = tx.run(
        """
        MATCH (s:Section)-[r:MENTIONS]->(:Concept)
        WHERE s.uid IN $section_uids
        RETURN count(r) AS mentions
        """,
        section_uids=section_uids,
    ).single()

    return {
        "mentions_written": int(count_record["mentions"] or 0),
        "orphans_deleted": int(orphan_record["orphan_count"] or 0),
    }


def rewrite_graph_mentions(driver: Driver, plan: ReplayPlan) -> Dict[str, int]:
    with driver.session() as session:
        return session.execute_write(_rewrite_mentions, plan.sections)
