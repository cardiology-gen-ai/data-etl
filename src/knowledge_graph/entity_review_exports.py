"""
entity_review_exports.py

Utilities for exporting entity extraction/validation decisions for manual review.

Purpose:
- save accepted and rejected entity candidates to JSONL files
- keep full validation details outside the normal runtime log
- help inspect entity extraction quality before/after changes such as acronym support
- preserve deterministic validation evidence such as support_method, matched_text,
  matched_pattern, acronym_short, and acronym_definition

Important:
- this module does NOT validate entities
- this module does NOT write to Neo4j
- this module does NOT change the KG
- it only writes review artifacts to disk

Typical output:
    test_data/entity_review/<doc_id>_accepted.jsonl
    test_data/entity_review/<doc_id>_rejected.jsonl
    test_data/entity_review/<doc_id>_summary.json
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)


ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = ROOT_DIR / "test_data"
DEFAULT_ENTITY_REVIEW_DIR = TEST_DATA_DIR / "entity_review"

ACCEPTED_SUFFIX = "_accepted.jsonl"
REJECTED_SUFFIX = "_rejected.jsonl"
SUMMARY_SUFFIX = "_summary.json"

MAX_TITLE_CHARS = 300
MAX_SOURCE_PREVIEW_CHARS = 500
MAX_EVIDENCE_TEXT_CHARS = 500


# Extra fields that may be produced by validate_entities.py and are useful
# for debugging accepted/rejected decisions.
REVIEW_EVIDENCE_FIELDS = [
    "validation_reason",
    "support_method",
    "matched_text",
    "matched_pattern",
    "acronym_short",
    "acronym_definition",
]


def utc_now_iso() -> str:
    """
    Return a UTC timestamp suitable for review records.
    """
    return datetime.now(timezone.utc).isoformat()


def safe_filename_component(value: Any, fallback: str = "unknown") -> str:
    """
    Convert a document id or similar value into a safe filename component.
    """
    text = str(value or "").strip()

    if not text:
        text = fallback

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text or fallback


def ensure_review_dir(output_dir: Optional[Path] = None) -> Path:
    """
    Ensure that the entity review output directory exists.
    """
    review_dir = Path(output_dir) if output_dir is not None else DEFAULT_ENTITY_REVIEW_DIR
    review_dir.mkdir(parents=True, exist_ok=True)
    return review_dir


def get_review_paths(
    doc_id: Any,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """
    Return accepted/rejected/summary paths for one document.
    """
    review_dir = ensure_review_dir(output_dir)
    safe_doc_id = safe_filename_component(doc_id, fallback="unknown_doc")

    accepted_path = review_dir / f"{safe_doc_id}{ACCEPTED_SUFFIX}"
    rejected_path = review_dir / f"{safe_doc_id}{REJECTED_SUFFIX}"
    summary_path = review_dir / f"{safe_doc_id}{SUMMARY_SUFFIX}"

    return accepted_path, rejected_path, summary_path


def truncate_text(value: Any, max_chars: int) -> str:
    """
    Safely stringify and truncate a value for review output.
    """
    text = str(value or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "...[truncated]"


def build_section_context(
    row: Dict[str, Any],
    include_source_preview: bool = False,
) -> Dict[str, Any]:
    """
    Build shared section-level metadata for accepted/rejected entity records.
    """
    if not isinstance(row, dict):
        row = {}

    source_text = str(row.get("source_text") or "")

    context = {
        "doc_id": row.get("doc_id"),
        "section_id": row.get("section_id"),
        "section_uid": row.get("uid"),
        "section_title": truncate_text(row.get("title"), MAX_TITLE_CHARS),
        "source_text_chars": len(source_text),
    }

    if include_source_preview:
        context["source_text_preview"] = truncate_text(
            source_text,
            MAX_SOURCE_PREVIEW_CHARS,
        )

    return context


def normalize_review_concept(
    concept: Dict[str, Any],
    default_reason: str,
) -> Optional[Dict[str, Any]]:
    """
    Normalize one accepted/rejected concept for export.

    Invalid non-dict records are skipped.

    For accepted concepts, validate_entities.py usually stores the reason under
    `validation_reason`, not `reason`, so we preserve both a normalized `reason`
    field and the original validation metadata.
    """
    if not isinstance(concept, dict):
        logger.debug("Skipping non-dict review concept: %r", concept)
        return None

    name = str(concept.get("name") or "").strip()
    concept_type = str(concept.get("type") or "").strip()

    reason = (
        concept.get("reason")
        or concept.get("validation_reason")
        or default_reason
    )
    reason = str(reason or default_reason).strip()

    if not name and not concept_type:
        logger.debug("Skipping empty review concept: %r", concept)
        return None

    normalized: Dict[str, Any] = {
        "name": name,
        "type": concept_type,
        "reason": reason or default_reason,
    }

    for field in REVIEW_EVIDENCE_FIELDS:
        value = concept.get(field)

        if value is None or value == "":
            continue

        if field in {"matched_text", "matched_pattern", "acronym_definition"}:
            normalized[field] = truncate_text(value, MAX_EVIDENCE_TEXT_CHARS)
        else:
            normalized[field] = value

    return normalized


def build_review_records(
    row: Dict[str, Any],
    concepts: Iterable[Dict[str, Any]],
    status: str,
    default_reason: str,
    run_id: Optional[str] = None,
    include_source_preview: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build JSON-serializable review records for accepted or rejected concepts.
    """
    section_context = build_section_context(
        row=row,
        include_source_preview=include_source_preview,
    )

    records: List[Dict[str, Any]] = []
    timestamp = utc_now_iso()

    for concept in concepts or []:
        normalized = normalize_review_concept(
            concept=concept,
            default_reason=default_reason,
        )

        if normalized is None:
            continue

        record = {
            "run_id": run_id,
            "exported_at": timestamp,
            "status": status,
            **section_context,
            **normalized,
        }

        records.append(record)

    return records


def append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Append JSON records to a JSONL file.

    Empty record lists are ignored.
    """
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_section_entity_review_records(
    row: Dict[str, Any],
    accepted: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    include_source_preview: bool = False,
) -> Dict[str, int]:
    """
    Export accepted and rejected entity validation decisions for one section.

    Returns:
        {
            "accepted_exported": ...,
            "rejected_exported": ...
        }
    """
    doc_id = row.get("doc_id") if isinstance(row, dict) else None

    accepted_path, rejected_path, _ = get_review_paths(
        doc_id=doc_id,
        output_dir=output_dir,
    )

    accepted_records = build_review_records(
        row=row,
        concepts=accepted,
        status="accepted",
        default_reason="accepted",
        run_id=run_id,
        include_source_preview=include_source_preview,
    )

    rejected_records = build_review_records(
        row=row,
        concepts=rejected,
        status="rejected",
        default_reason="unknown",
        run_id=run_id,
        include_source_preview=include_source_preview,
    )

    append_jsonl(accepted_path, accepted_records)
    append_jsonl(rejected_path, rejected_records)

    return {
        "accepted_exported": len(accepted_records),
        "rejected_exported": len(rejected_records),
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file.

    Malformed lines are skipped with a warning.
    """
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed JSONL line | path=%s | line=%d",
                    path,
                    line_number,
                )
                continue

            if isinstance(data, dict):
                records.append(data)

    return records


def count_by_field(records: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    """
    Count records by a given field.
    """
    counter = Counter()

    for record in records:
        value = record.get(field)

        if value is None or value == "":
            value = "UNKNOWN"

        counter[str(value)] += 1

    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def write_entity_review_summary(
    doc_id: Any,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build and write a summary JSON file from accepted/rejected JSONL exports.

    This can be called after entity extraction finishes for a document.
    """
    accepted_path, rejected_path, summary_path = get_review_paths(
        doc_id=doc_id,
        output_dir=output_dir,
    )

    accepted_records = read_jsonl(accepted_path)
    rejected_records = read_jsonl(rejected_path)

    section_ids = {
        record.get("section_uid")
        for record in accepted_records + rejected_records
        if record.get("section_uid")
    }

    summary = {
        "doc_id": doc_id,
        "generated_at": utc_now_iso(),
        "accepted_file": str(accepted_path),
        "rejected_file": str(rejected_path),
        "sections_with_review_records": len(section_ids),
        "accepted_entities": len(accepted_records),
        "rejected_entities": len(rejected_records),
        "accepted_by_type": count_by_field(accepted_records, "type"),
        "rejected_by_type": count_by_field(rejected_records, "type"),
        "accepted_by_reason": count_by_field(accepted_records, "reason"),
        "rejected_by_reason": count_by_field(rejected_records, "reason"),
        "accepted_by_support_method": count_by_field(accepted_records, "support_method"),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    return summary


def clear_entity_review_exports(
    doc_id: Any,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Remove existing accepted/rejected/summary review files for one document.

    Use this at the beginning of a clean rerun if you do not want JSONL files
    to accumulate records from previous runs.
    """
    paths = get_review_paths(
        doc_id=doc_id,
        output_dir=output_dir,
    )

    for path in paths:
        if path.exists():
            path.unlink()
            logger.info("Removed previous entity review export: %s", path)