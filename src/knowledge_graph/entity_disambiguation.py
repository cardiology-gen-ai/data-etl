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


def initialize_missing_observed_types(tx) -> None:
    """
    If a concept has a canonical type but no observed_types list yet,
    initialize observed_types from the canonical type.
    """
    tx.run(
        """
        MATCH (c:Concept)
        WHERE c.type IS NOT NULL
          AND (c.observed_types IS NULL OR size(c.observed_types) = 0)
        SET c.observed_types = [c.type]
        """
    )


def resolve_single_type_concepts(tx) -> None:
    """
    If a concept has exactly one observed type, make the canonical type agree with it.
    Mark it as not needing review.
    """
    tx.run(
        """
        MATCH (c:Concept)
        WHERE c.observed_types IS NOT NULL
          AND size(c.observed_types) = 1
        SET c.type = c.observed_types[0],
            c.needs_type_review = false,
            c.type_resolution_status = 'resolved_single_observed_type'
        """
    )


def mark_ambiguous_concepts(tx) -> None:
    """
    If a concept has multiple observed types, keep the current canonical type if possible
    and mark the concept as needing review.
    """
    tx.run(
        """
        MATCH (c:Concept)
        WHERE c.observed_types IS NOT NULL
          AND size(c.observed_types) > 1
        SET c.type =
            CASE
                WHEN c.type IS NOT NULL AND c.type IN c.observed_types THEN c.type
                ELSE c.observed_types[0]
            END,
            c.needs_type_review = true,
            c.type_resolution_status = 'ambiguous_multiple_observed_types'
        """
    )


def mark_missing_type_concepts(tx) -> None:
    """
    Mark concepts that still have no canonical type after the basic disambiguation pass.
    """
    tx.run(
        """
        MATCH (c:Concept)
        WHERE c.type IS NULL
        SET c.needs_type_review = true,
            c.type_resolution_status = 'missing_type_after_disambiguation'
        """
    )


def summarize_disambiguation(tx):
    result = tx.run(
        """
        MATCH (c:Concept)
        RETURN
            count(c) AS total_concepts,
            count(CASE WHEN c.needs_type_review = true THEN 1 END) AS concepts_needing_review,
            count(CASE WHEN c.type_resolution_status = 'resolved_single_observed_type' THEN 1 END) AS resolved_single_type,
            count(CASE WHEN c.type_resolution_status = 'ambiguous_multiple_observed_types' THEN 1 END) AS ambiguous_multi_type,
            count(CASE WHEN c.type_resolution_status = 'missing_type_after_disambiguation' THEN 1 END) AS missing_type
        """
    )
    return result.single()


def disambiguate_concepts(driver: Driver) -> Dict[str, int]:
    """
    Run a first-pass concept type disambiguation.

    This is intentionally simple:
    - resolves concepts with one observed type
    - flags concepts with multiple observed types
    - does not attempt majority voting, because counts are not stored yet
    """
    with driver.session() as session:
        session.execute_write(setup_disambiguation_schema)
        session.execute_write(initialize_missing_observed_types)
        session.execute_write(resolve_single_type_concepts)
        session.execute_write(mark_ambiguous_concepts)
        session.execute_write(mark_missing_type_concepts)

        summary = session.execute_read(summarize_disambiguation)

    stats = {
        "total_concepts": summary["total_concepts"],
        "concepts_needing_review": summary["concepts_needing_review"],
        "resolved_single_type": summary["resolved_single_type"],
        "ambiguous_multi_type": summary["ambiguous_multi_type"],
        "missing_type": summary["missing_type"],
    }

    logger.info(
        "Concept disambiguation completed | total=%d | resolved_single=%d | ambiguous=%d | missing_type=%d | needs_review=%d",
        stats["total_concepts"],
        stats["resolved_single_type"],
        stats["ambiguous_multi_type"],
        stats["missing_type"],
        stats["concepts_needing_review"],
    )

    return stats