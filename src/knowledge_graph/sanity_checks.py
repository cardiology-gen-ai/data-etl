import logging
from typing import Any, Dict, List, Set

from neo4j import Driver


logger = logging.getLogger(__name__)


ALLOWED_MODES = {"structure", "entities", "embeddings", "full"}

PHASE_EXPANSION: Dict[str, Set[str]] = {
    "structure": {"structure"},
    "entities": {"structure", "entities"},
    "embeddings": {"structure", "embeddings"},
    "full": {"structure", "entities", "embeddings"},
}


CHECKS: List[Dict[str, Any]] = [
    {
        "name": "documents_without_sections",
        "title": "Documents without sections",
        "group": "Document structure",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (d:Document)
            WHERE NOT (d)-[:HAS_SECTION]->(:Section)
            RETURN d.doc_id AS doc_id
            ORDER BY doc_id
        """,
    },
    {
        "name": "sections_linked_to_multiple_documents",
        "title": "Sections linked to multiple documents",
        "group": "Document structure",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (d1:Document)-[:HAS_SECTION]->(s:Section)<-[:HAS_SECTION]-(d2:Document)
            WHERE d1 <> d2
            RETURN DISTINCT s.uid AS uid, d1.doc_id AS doc_1, d2.doc_id AS doc_2
            ORDER BY uid, doc_1, doc_2
        """,
    },
    {
        "name": "orphan_sections",
        "title": "Orphan sections (no document)",
        "group": "Document structure",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE NOT (:Document)-[:HAS_SECTION]->(s)
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "duplicate_section_uids",
        "title": "Duplicate section UID values",
        "group": "Section identity",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WITH s.uid AS uid, count(*) AS n
            WHERE n > 1
            RETURN uid, n
            ORDER BY n DESC, uid
        """,
    },
    {
        "name": "uid_doc_id_mismatch",
        "title": "UID / doc_id mismatch",
        "group": "Section identity",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE NOT s.uid STARTS WITH s.doc_id + "::"
            RETURN s.uid AS uid, s.doc_id AS doc_id
            ORDER BY uid
        """,
    },
    {
        "name": "sections_with_multiple_parents",
        "title": "Sections with multiple parents",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (p:Section)-[:HAS_CHILD]->(c:Section)
            WITH c, count(DISTINCT p) AS parents
            WHERE parents > 1
            RETURN c.uid AS uid, parents
            ORDER BY parents DESC, uid
        """,
    },
    {
        "name": "cycles_in_has_child",
        "title": "Cycles in HAS_CHILD",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE (s)-[:HAS_CHILD*1..]->(s)
            RETURN DISTINCT s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "next_edges_crossing_documents",
        "title": "NEXT edges crossing documents",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (a:Section)-[:NEXT]->(b:Section)
            WHERE a.doc_id <> b.doc_id
            RETURN a.uid AS from_uid, b.uid AS to_uid
            ORDER BY from_uid, to_uid
        """,
    },
    {
        "name": "sections_missing_text",
        "title": "Sections missing text",
        "group": "Section content",
        "phases": {"structure"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(trim(s.text), '') = ''
            RETURN s.uid AS uid, s.title AS title
            ORDER BY uid
        """,
    },
    {
        "name": "empty_leaf_sections",
        "title": "Empty leaf sections",
        "group": "Section content",
        "phases": {"structure"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(s.is_empty, false) = true
              AND NOT (s)-[:HAS_CHILD]->(:Section)
            RETURN s.uid AS uid, s.title AS title
            ORDER BY uid
        """,
    },
    {
        "name": "non_empty_parent_sections",
        "title": "Non-empty parent sections",
        "group": "Section content",
        "phases": {"structure"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)-[:HAS_CHILD]->(:Section)
            WHERE coalesce(s.is_empty, false) = false
              AND coalesce(trim(s.text), '') <> ''
              AND size(s.text) > 100
            RETURN DISTINCT s.uid AS uid, size(s.text) AS text_len
            ORDER BY text_len DESC, uid
        """,
    },
    {
        "name": "orphan_concepts",
        "title": "Orphan concepts",
        "group": "Concepts",
        "phases": {"entities"},
        "level": "INFO",
        "query": """
            MATCH (c:Concept)
            WHERE NOT (:Section)-[:MENTIONS]->(c)
            RETURN c.name AS name
            ORDER BY name
        """,
    },
    {
        "name": "concepts_without_canonical_type",
        "title": "Concepts without canonical type",
        "group": "Concepts",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (c:Concept)
            WHERE c.canonical_type IS NULL
            RETURN c.name AS name
            ORDER BY name
        """,
    },
    {
        "name": "concepts_without_observed_types",
        "title": "Concepts without observed_types",
        "group": "Concepts",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE c.observed_types IS NULL OR size(c.observed_types) = 0
            RETURN c.name AS name, c.canonical_type AS canonical_type
            ORDER BY name
        """,
    },
    {
        "name": "canonical_type_not_in_observed_types",
        "title": "Canonical type not present in observed_types",
        "group": "Concepts",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE c.canonical_type IS NOT NULL
              AND c.observed_types IS NOT NULL
              AND NOT c.canonical_type IN c.observed_types
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types
            ORDER BY name
        """,
    },
    {
        "name": "ambiguous_concepts_needing_review",
        "title": "Ambiguous concepts needing review",
        "group": "Concepts",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE coalesce(c.needs_type_review, false) = true
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.type_support_pairs AS type_support_pairs,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "concepts_missing_type_resolution_status",
        "title": "Concepts missing type resolution status",
        "group": "Concepts",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE c.type_resolution_status IS NULL
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types
            ORDER BY name
        """,
    },
    {
        "name": "concepts_used_in_only_one_document",
        "title": "Concepts used in only one document",
        "group": "Concepts",
        "phases": {"entities"},
        "level": "INFO",
        "query": """
            MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
            WITH c, collect(DISTINCT s.doc_id) AS docs
            WHERE size(docs) = 1
            RETURN c.name AS name, docs
            ORDER BY name
        """,
    },
    {
        "name": "highly_overused_concepts",
        "title": "Highly overused concepts",
        "group": "Concepts",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)-[:MENTIONS]->(c:Concept)
            WITH c, count(DISTINCT s) AS n
            WHERE n > 30
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   n
            ORDER BY n DESC, name
        """,
    },
    {
        "name": "sections_missing_entity_extracted_flag",
        "title": "Sections missing entity_extracted flag",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.entity_extracted IS NULL
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "entity_extraction_status_summary",
        "title": "Entity extraction status summary",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH (s:Section)
            RETURN coalesce(s.entity_extraction_status, 'UNSET') AS status,
                   count(s) AS n
            ORDER BY n DESC, status ASC
        """,
    },
    {
        "name": "entity_status_success_but_not_extracted",
        "title": "Sections marked entity success but not extracted",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.entity_extraction_status = 'success'
              AND coalesce(s.entity_extracted, false) = false
            RETURN s.uid AS uid,
                   s.entity_extraction_status AS status,
                   s.entity_extracted AS entity_extracted
            ORDER BY uid
        """,
    },
    {
        "name": "entity_extracted_but_missing_status",
        "title": "Sections extracted but missing entity status",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(s.entity_extracted, false) = true
              AND s.entity_extraction_status IS NULL
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "entity_failed_without_timestamp",
        "title": "Sections with failed entity extraction but no timestamp",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE s.entity_extraction_status = 'failed'
              AND s.entity_extraction_failed_at IS NULL
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "sections_missing_has_embedding_flag",
        "title": "Sections missing has_embedding flag",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.has_embedding IS NULL
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "embedding_status_summary",
        "title": "Embedding status summary",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH (s:Section)
            RETURN coalesce(s.embedding_status, 'UNSET') AS status,
                   count(s) AS n
            ORDER BY n DESC, status ASC
        """,
    },
    {
        "name": "embedding_flag_inconsistencies",
        "title": "Embedding flag inconsistencies",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE (coalesce(s.has_embedding, false) = true AND s.embedding IS NULL)
               OR (coalesce(s.has_embedding, false) = false AND s.embedding IS NOT NULL)
            RETURN s.uid AS uid,
                   s.has_embedding AS has_embedding,
                   s.embedding IS NOT NULL AS has_embedding_vector
            ORDER BY uid
        """,
    },
    {
        "name": "embedding_metadata_inconsistencies",
        "title": "Embedding metadata inconsistencies",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE s.embedding IS NOT NULL
              AND (
                    s.embedding_dim IS NULL
                 OR s.embedding_model IS NULL
                 OR s.embedding_updated_at IS NULL
                 OR size(s.embedding) <> s.embedding_dim
              )
            RETURN s.uid AS uid,
                   s.embedding_dim AS embedding_dim,
                   size(s.embedding) AS actual_dim,
                   s.embedding_model AS embedding_model,
                   s.embedding_updated_at AS embedding_updated_at
            ORDER BY uid
        """,
    },
    {
        "name": "embedding_status_success_but_no_vector",
        "title": "Sections marked embedding success but with no vector",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.embedding_status = 'success'
              AND (s.embedding IS NULL OR coalesce(s.has_embedding, false) = false)
            RETURN s.uid AS uid,
                   s.embedding_status AS status,
                   s.has_embedding AS has_embedding
            ORDER BY uid
        """,
    },
    {
        "name": "embedding_failed_without_timestamp",
        "title": "Sections with failed embedding but no timestamp",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE s.embedding_status = 'failed'
              AND s.embedding_failed_at IS NULL
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "eligible_sections_missing_embeddings",
        "title": "Eligible sections still missing embeddings",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(s.embed, false) = true
              AND s.embedding IS NULL
            RETURN DISTINCT s.uid AS uid,
                   s.doc_id AS doc_id,
                   s.embedding_status AS embedding_status
            ORDER BY doc_id, uid
        """,
    },
]


def _log_with_level(level: str, message: str, *args) -> None:
    """
    Log a message using the requested severity level.
    """
    level = level.upper()

    if level == "ERROR":
        logger.error(message, *args)
    elif level == "WARNING":
        logger.warning(message, *args)
    else:
        logger.info(message, *args)


def _normalize_mode(mode: str) -> str:
    """
    Validate and normalize the requested sanity-check mode.
    """
    normalized = mode.lower().strip()
    if normalized not in ALLOWED_MODES:
        raise ValueError(
            f"Invalid sanity check mode: {mode!r}. Allowed values: {sorted(ALLOWED_MODES)}"
        )
    return normalized


def _build_count_query(query: str) -> str:
    """
    Wrap a check query so it returns only the total number of matching rows.
    """
    return f"""
        CALL {{
            {query.strip()}
        }}
        RETURN count(*) AS n
    """


def _build_sample_query(query: str) -> str:
    """
    Wrap a check query so it returns only a limited sample of matching rows.
    """
    return f"""
        CALL {{
            {query.strip()}
        }}
        RETURN *
        LIMIT $sample_limit
    """


def _run_check(tx, check: Dict[str, Any], sample_limit: int) -> Dict[str, Any]:
    """
    Execute a single check and return a structured result.

    Each check is run once for the exact count and, only when useful,
    once more to fetch a limited sample for logging.
    """
    count_query = check.get("count_query") or _build_count_query(check["query"])
    sample_query = check.get("sample_query") or _build_sample_query(check["query"])

    count_record = tx.run(count_query).single()
    count = int(count_record["n"]) if count_record is not None else 0

    is_summary = bool(check.get("is_summary", False))

    sample: List[Dict[str, Any]] = []
    if sample_limit > 0 and (is_summary or count > 0):
        sample_rows = list(tx.run(sample_query, sample_limit=sample_limit))
        sample = [dict(row) for row in sample_rows]

    return {
        "name": check["name"],
        "title": check["title"],
        "group": check["group"],
        "level": check["level"],
        "count": count,
        "sample": sample,
        "is_summary": is_summary,
    }


def run_sanity_checks(
    driver: Driver,
    mode: str = "full",
    sample_limit: int = 10,
    log_samples: bool = True,
) -> Dict[str, Any]:
    """
    Run graph sanity checks for the requested pipeline phase.

    Parameters:
        driver: active Neo4j driver
        mode: one of {"structure", "entities", "embeddings", "full"}
        sample_limit: maximum number of sample rows to keep/log per check
        log_samples: whether to log sample rows

    Returns:
        Structured summary dictionary with per-check results and aggregate counters.
    """
    mode = _normalize_mode(mode)

    if sample_limit < 0:
        raise ValueError("sample_limit must be >= 0")

    included_phases = PHASE_EXPANSION[mode]

    checks_to_run = [
        check for check in CHECKS
        if bool(check["phases"] & included_phases)
    ]

    results: List[Dict[str, Any]] = []

    with driver.session() as session:
        for check in checks_to_run:
            result = session.execute_read(_run_check, check, sample_limit)
            results.append(result)

    summary = {
        "mode": mode,
        "total_checks": len(results),
        "checks_with_issues": 0,
        "error_checks_with_issues": 0,
        "warning_checks_with_issues": 0,
        "info_checks_with_issues": 0,
        "results": results,
    }

    current_group = None

    for result in results:
        if result["group"] != current_group:
            current_group = result["group"]
            logger.info("Running %s checks", current_group)

        if result["is_summary"]:
            logger.info("%s", result["title"])
            if log_samples:
                for row in result["sample"]:
                    logger.info("  -> %s", row)
            continue

        if result["count"] > 0:
            summary["checks_with_issues"] += 1

            level = result["level"].upper()
            if level == "ERROR":
                summary["error_checks_with_issues"] += 1
            elif level == "WARNING":
                summary["warning_checks_with_issues"] += 1
            else:
                summary["info_checks_with_issues"] += 1

            _log_with_level(
                result["level"],
                "%s (%d issues)",
                result["title"],
                result["count"],
            )

            if log_samples:
                for row in result["sample"]:
                    _log_with_level(result["level"], "  -> %s", row)
        else:
            logger.info("%s: OK", result["title"])

    logger.info(
        "Sanity checks completed | mode=%s | total=%d | issue_checks=%d | error_checks=%d | warning_checks=%d | info_checks=%d",
        summary["mode"],
        summary["total_checks"],
        summary["checks_with_issues"],
        summary["error_checks_with_issues"],
        summary["warning_checks_with_issues"],
        summary["info_checks_with_issues"],
    )

    return summary