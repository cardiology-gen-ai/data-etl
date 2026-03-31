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
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE NOT (:Section)-[:MENTIONS]->(c)
            RETURN c.name AS name
        """,
    },
    {
        "name": "concepts_without_type",
        "title": "Concepts without type",
        "level": "ERROR",
        "query": """
            MATCH (c:Concept)
            WHERE c.type IS NULL
            RETURN c.name AS name
        """,
    },
    {
        "name": "ambiguous_concepts_needing_review",
        "title": "Ambiguous concepts needing review",
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE coalesce(c.needs_type_review, false) = true
            RETURN c.name AS name, c.type AS type, c.observed_types AS observed_types
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
            RETURN c.name AS name, c.type AS type, n
            ORDER BY n DESC
        """,
    },

    # -------------------------
    # ENTITY EXTRACTION STATE
    # -------------------------
    {
        "name": "sections_missing_entity_extracted_flag",
        "title": "Sections missing entity_extracted flag",
        "level": "INFO",
        "query": """
            MATCH (s:Section)
            WHERE s.entity_extracted IS NULL
            RETURN s.uid AS uid
        """,
    },

    # -------------------------
    # EMBEDDING STATE
    # -------------------------
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