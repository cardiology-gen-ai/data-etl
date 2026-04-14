import logging
from typing import Dict, List, Any

from neo4j import Driver


logger = logging.getLogger(__name__)


CHECKS = [
    # -------------------------
    # DOCUMENT STRUCTURE
    # -------------------------
    {
        "name": "documents_without_sections",
        "title": "Documents without sections",
        "level": "ERROR",
        "query": """
            MATCH (d:Document)
            WHERE NOT (d)-[:HAS_SECTION]->(:Section)
            RETURN d.doc_id AS doc_id
        """,
    },
    {
        "name": "sections_linked_to_multiple_documents",
        "title": "Sections linked to multiple documents",
        "level": "ERROR",
        "query": """
            MATCH (d1:Document)-[:HAS_SECTION]->(s:Section)<-[:HAS_SECTION]-(d2:Document)
            WHERE d1 <> d2
            RETURN s.uid AS uid, d1.doc_id AS doc_1, d2.doc_id AS doc_2
        """,
    },
    {
        "name": "orphan_sections",
        "title": "Orphan sections (no document)",
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE NOT (:Document)-[:HAS_SECTION]->(s)
            RETURN s.uid AS uid
        """,
    },

    # -------------------------
    # SECTION IDENTITY
    # -------------------------
    {
        "name": "duplicate_section_uids",
        "title": "Duplicate section UID values",
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WITH s.uid AS uid, count(*) AS n
            WHERE n > 1
            RETURN uid, n
        """,
    },
    {
        "name": "uid_doc_id_mismatch",
        "title": "UID / doc_id mismatch",
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE NOT s.uid STARTS WITH s.doc_id + "::"
            RETURN s.uid AS uid, s.doc_id AS doc_id
        """,
    },

    # -------------------------
    # HIERARCHY
    # -------------------------
    {
        "name": "sections_with_multiple_parents",
        "title": "Sections with multiple parents",
        "level": "ERROR",
        "query": """
            MATCH (p:Section)-[:HAS_CHILD]->(c:Section)
            WITH c, count(p) AS parents
            WHERE parents > 1
            RETURN c.uid AS uid, parents
        """,
    },
    {
        "name": "cycles_in_has_child",
        "title": "Cycles in HAS_CHILD",
        "level": "ERROR",
        "query": """
            MATCH p=(s:Section)-[:HAS_CHILD*]->(s)
            RETURN s.uid AS uid
            LIMIT 50
        """,
    },
    {
        "name": "next_edges_crossing_documents",
        "title": "NEXT edges crossing documents",
        "level": "ERROR",
        "query": """
            MATCH (a:Section)-[:NEXT]->(b:Section)
            WHERE a.doc_id <> b.doc_id
            RETURN a.uid AS from_uid, b.uid AS to_uid
        """,
    },

    # -------------------------
    # SECTION CONTENT
    # -------------------------
    {
        "name": "sections_missing_text",
        "title": "Sections missing text",
        "level": "INFO",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(trim(s.text), '') = ''
            RETURN s.uid AS uid, s.title AS title
            LIMIT 20
        """,
    },
    {
        "name": "empty_leaf_sections",
        "title": "Empty leaf sections",
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE s.is_empty = true
              AND NOT (s)-[:HAS_CHILD]->(:Section)
            RETURN s.uid AS uid, s.title AS title
        """,
    },
    {
        "name": "non_empty_parent_sections",
        "title": "Non-empty parent sections",
        "level": "WARNING",
        "query": """
            MATCH (s:Section)-[:HAS_CHILD]->(:Section)
            WHERE coalesce(s.is_empty, false) = false
              AND s.text IS NOT NULL
              AND size(s.text) > 100
            RETURN s.uid AS uid, size(s.text) AS text_len
        """,
    },

    # -------------------------
    # CONCEPTS
    # -------------------------
    {
        "name": "orphan_concepts",
        "title": "Orphan concepts",
        "level": "INFO",
        "query": """
            MATCH (c:Concept)
            WHERE NOT (:Section)-[:MENTIONS]->(c)
            RETURN c.name AS name
        """,
    },
    {
        "name": "concepts_without_canonical_type",
        "title": "Concepts without canonical type",
        "level": "ERROR",
        "query": """
            MATCH (c:Concept)
            WHERE c.canonical_type IS NULL
            RETURN c.name AS name
        """,
    },
    {
        "name": "concepts_without_observed_types",
        "title": "Concepts without observed_types",
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE c.observed_types IS NULL OR size(c.observed_types) = 0
            RETURN c.name AS name, c.canonical_type AS canonical_type
        """,
    },
    {
        "name": "canonical_type_not_in_observed_types",
        "title": "Canonical type not present in observed_types",
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE c.canonical_type IS NOT NULL
              AND c.observed_types IS NOT NULL
              AND NOT c.canonical_type IN c.observed_types
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types
        """,
    },
    {
        "name": "ambiguous_concepts_needing_review",
        "title": "Ambiguous concepts needing review",
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE coalesce(c.needs_type_review, false) = true
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.type_support_pairs AS type_support_pairs,
                   c.type_resolution_status AS type_resolution_status
        """,
    },
    {
        "name": "concepts_missing_type_resolution_status",
        "title": "Concepts missing type resolution status",
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE c.type_resolution_status IS NULL
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types
        """,
    },
    {
        "name": "concepts_used_in_only_one_document",
        "title": "Concepts used in only one document",
        "level": "INFO",
        "query": """
            MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
            WITH c, collect(DISTINCT s.doc_id) AS docs
            WHERE size(docs) = 1
            RETURN c.name AS name, docs
        """,
    },
    {
        "name": "highly_overused_concepts",
        "title": "Highly overused concepts",
        "level": "WARNING",
        "query": """
            MATCH (s:Section)-[:MENTIONS]->(c:Concept)
            WITH c, count(s) AS n
            WHERE n > 30
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   n
            ORDER BY n DESC
        """,
    },

    # -------------------------
    # ENTITY EXTRACTION STATE
    # -------------------------
    {
        "name": "sections_missing_entity_extracted_flag",
        "title": "Sections missing entity_extracted flag",
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.entity_extracted IS NULL
            RETURN s.uid AS uid
        """,
    },
    {
        "name": "entity_extraction_status_summary",
        "title": "Entity extraction status summary",
        "level": "INFO",
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
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.entity_extraction_status = 'success'
              AND coalesce(s.entity_extracted, false) = false
            RETURN s.uid AS uid,
                   s.entity_extraction_status AS status,
                   s.entity_extracted AS entity_extracted
        """,
    },
    {
        "name": "entity_extracted_but_missing_status",
        "title": "Sections extracted but missing entity status",
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(s.entity_extracted, false) = true
              AND s.entity_extraction_status IS NULL
            RETURN s.uid AS uid
        """,
    },
    {
        "name": "entity_failed_without_timestamp",
        "title": "Sections with failed entity extraction but no timestamp",
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE s.entity_extraction_status = 'failed'
              AND s.entity_extraction_failed_at IS NULL
            RETURN s.uid AS uid
        """,
    },

    # -------------------------
    # EMBEDDING STATE
    # -------------------------
    {
        "name": "sections_missing_has_embedding_flag",
        "title": "Sections missing has_embedding flag",
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.has_embedding IS NULL
            RETURN s.uid AS uid
        """,
    },
    {
        "name": "embedding_status_summary",
        "title": "Embedding status summary",
        "level": "INFO",
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
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE (coalesce(s.has_embedding, false) = true AND s.embedding IS NULL)
               OR (coalesce(s.has_embedding, false) = false AND s.embedding IS NOT NULL)
            RETURN s.uid AS uid,
                   s.has_embedding AS has_embedding,
                   s.embedding IS NOT NULL AS has_embedding_vector
        """,
    },
    {
        "name": "embedding_metadata_inconsistencies",
        "title": "Embedding metadata inconsistencies",
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
        """,
    },
    {
        "name": "embedding_status_success_but_no_vector",
        "title": "Sections marked embedding success but with no vector",
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.embedding_status = 'success'
              AND (s.embedding IS NULL OR coalesce(s.has_embedding, false) = false)
            RETURN s.uid AS uid,
                   s.embedding_status AS status,
                   s.has_embedding AS has_embedding
        """,
    },
    {
        "name": "embedding_failed_without_timestamp",
        "title": "Sections with failed embedding but no timestamp",
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE s.embedding_status = 'failed'
              AND s.embedding_failed_at IS NULL
            RETURN s.uid AS uid
        """,
    },
    {
        "name": "eligible_sections_missing_embeddings",
        "title": "Eligible sections still missing embeddings",
        "level": "INFO",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(s.embed, false) = true
              AND s.embedding IS NULL
            RETURN s.uid AS uid,
                   s.doc_id AS doc_id,
                   s.embedding_status AS embedding_status
            LIMIT 20
        """,
    },
]


def _log_with_level(level: str, message: str, *args) -> None:
    """
    Log using the requested severity level.
    """
    level = level.upper()

    if level == "ERROR":
        logger.error(message, *args)
    elif level == "WARNING":
        logger.warning(message, *args)
    else:
        logger.info(message, *args)


def _run_check(tx, check: Dict[str, Any], sample_limit: int) -> Dict[str, Any]:
    """
    Execute a single check and return a structured result.
    """
    rows = list(tx.run(check["query"]))
    count = len(rows)
    sample = [dict(row) for row in rows[:sample_limit]]

    return {
        "name": check["name"],
        "title": check["title"],
        "level": check["level"],
        "count": count,
        "sample": sample,
    }


def run_sanity_checks(
    driver: Driver,
    sample_limit: int = 10,
    log_samples: bool = True,
) -> Dict[str, Any]:
    """
    Run all graph sanity checks.

    Parameters:
        driver: Neo4j driver
        sample_limit: maximum number of example rows to keep/log per check
        log_samples: whether to log sample offending rows

    Returns:
        A structured summary dictionary.
    """
    results: List[Dict[str, Any]] = []

    with driver.session() as session:
        for check in CHECKS:
            result = session.execute_read(_run_check, check, sample_limit)
            results.append(result)

    summary = {
        "total_checks": len(results),
        "checks_with_issues": 0,
        "error_checks_with_issues": 0,
        "warning_checks_with_issues": 0,
        "info_checks_with_issues": 0,
        "results": results,
    }

    for result in results:
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
        "Sanity checks completed | total=%d | issue_checks=%d | error_checks=%d | warning_checks=%d | info_checks=%d",
        summary["total_checks"],
        summary["checks_with_issues"],
        summary["error_checks_with_issues"],
        summary["warning_checks_with_issues"],
        summary["info_checks_with_issues"],
    )

    return summary