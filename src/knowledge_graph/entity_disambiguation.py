import logging
from typing import Dict

from neo4j import Driver


logger = logging.getLogger(__name__)


def setup_disambiguation_schema(tx) -> None:
    """
    Optional indexes for later inspection/filtering.
    """
    tx.run(
        """
        CREATE INDEX concept_needs_type_review IF NOT EXISTS
        FOR (c:Concept)
        ON (c.needs_type_review)
        """
    )

    tx.run(
        """
        CREATE INDEX concept_type_resolution_status IF NOT EXISTS
        FOR (c:Concept)
        ON (c.type_resolution_status)
        """
    )


def reset_type_resolution_state(tx) -> None:
    """
    Recompute concept-level type state from current graph evidence.
    """
    tx.run(
        """
        MATCH (c:Concept)
        SET c.observed_types = [],
            c.type_support_pairs = [],
            c.needs_type_review = null,
            c.type_resolution_status = null,
            c.type_resolution_updated_at = datetime()
        """
    )


def resolve_types_by_section_support(tx) -> None:
    """
    Recompute concept type support from current Section-[:MENTIONS]->Concept relationships.

    Each section contributes at most one support per type for a given concept because
    relationship observed_types are already deduplicated at write time.
    """
    tx.run(
        """
        MATCH (c:Concept)<-[r:MENTIONS]-(:Section)
        UNWIND coalesce(r.observed_types, []) AS supported_type
        WITH c, supported_type, count(*) AS support_count
        ORDER BY c.name, support_count DESC, supported_type ASC
        WITH c, collect({type: supported_type, count: support_count}) AS support_rows
        WITH c,
             support_rows,
             [row IN support_rows | row.type] AS observed_types,
             [row IN support_rows | row.type + '=' + toString(row.count)] AS type_support_pairs,
             [row IN support_rows WHERE row.count = support_rows[0].count | row.type] AS top_types
        SET c.observed_types = observed_types,
            c.type_support_pairs = type_support_pairs,
            c.canonical_type =
                CASE
                    WHEN size(top_types) = 1 THEN top_types[0]
                    WHEN c.canonical_type IS NOT NULL AND c.canonical_type IN top_types THEN c.canonical_type
                    ELSE top_types[0]
                END,
            c.needs_type_review =
                CASE
                    WHEN size(observed_types) = 1 THEN false
                    WHEN size(top_types) = 1 THEN false
                    ELSE true
                END,
            c.type_resolution_status =
                CASE
                    WHEN size(observed_types) = 1 THEN 'resolved_single_supported_type'
                    WHEN size(top_types) = 1 THEN 'resolved_by_section_support'
                    ELSE 'ambiguous_tied_section_support'
                END
        """
    )


def mark_concepts_without_supported_types(tx) -> None:
    """
    Mark concepts for which no current section-level type support could be recomputed.
    This typically means stale/orphan concepts or malformed old graph state.
    """
    tx.run(
        """
        MATCH (c:Concept)
        WHERE c.observed_types IS NULL OR size(c.observed_types) = 0
        SET c.canonical_type = null,
            c.type_support_pairs = [],
            c.needs_type_review = true,
            c.type_resolution_status = 'no_supported_types_after_recompute'
        """
    )


def summarize_disambiguation(tx):
    result = tx.run(
        """
        MATCH (c:Concept)
        RETURN
            count(c) AS total_concepts,
            count(CASE WHEN c.needs_type_review = true THEN 1 END) AS concepts_needing_review,
            count(CASE WHEN c.type_resolution_status = 'resolved_single_supported_type' THEN 1 END) AS resolved_single_type,
            count(CASE WHEN c.type_resolution_status = 'resolved_by_section_support' THEN 1 END) AS resolved_by_support,
            count(CASE WHEN c.type_resolution_status = 'ambiguous_tied_section_support' THEN 1 END) AS ambiguous_tied_support,
            count(CASE WHEN c.type_resolution_status = 'no_supported_types_after_recompute' THEN 1 END) AS no_supported_types
        """
    )
    return result.single()


def disambiguate_concepts(driver: Driver) -> Dict[str, int]:
    """
    Recompute concept type resolution from current section-level support.

    Strategy:
    - reset concept-level type state
    - aggregate current support from Section-[:MENTIONS]->Concept relationships
    - resolve concepts with a single supported type
    - resolve multi-type concepts by majority support when there is a unique winner
    - flag concepts that remain tied across top-supported types
    - flag concepts with no current supported types
    """
    with driver.session() as session:
        session.execute_write(setup_disambiguation_schema)
        session.execute_write(reset_type_resolution_state)
        session.execute_write(resolve_types_by_section_support)
        session.execute_write(mark_concepts_without_supported_types)

        summary = session.execute_read(summarize_disambiguation)

    stats = {
        "total_concepts": summary["total_concepts"],
        "concepts_needing_review": summary["concepts_needing_review"],
        "resolved_single_type": summary["resolved_single_type"],
        "resolved_by_support": summary["resolved_by_support"],
        "ambiguous_tied_support": summary["ambiguous_tied_support"],
        "no_supported_types": summary["no_supported_types"],
    }

    logger.info(
        "Concept disambiguation completed | total=%d | resolved_single=%d | resolved_by_support=%d | ambiguous_tied=%d | no_supported_types=%d | needs_review=%d",
        stats["total_concepts"],
        stats["resolved_single_type"],
        stats["resolved_by_support"],
        stats["ambiguous_tied_support"],
        stats["no_supported_types"],
        stats["concepts_needing_review"],
    )

    return stats