"""
entity_disambiguation.py

Post-processing step for Concept nodes after entity extraction.

Goal:
- recompute concept-level type evidence from current Section-[:MENTIONS]->Concept links
- choose a canonical_type when the evidence is clear enough
- flag concepts that still need manual review
- optionally delete orphan concepts that are no longer referenced by any Section

Important note:
- We intentionally do NOT wipe canonical_type during reset.
  This allows a previous/manual canonical_type to be preserved in tie cases
  if it is still among the top-supported types.
- If a concept has tied type evidence and no existing defensible canonical_type,
  we set canonical_type = "ambiguous" rather than choosing an arbitrary type.
- Type values outside the current ALLOWED_TYPES schema are excluded from
  canonical resolution and preserved in invalid_observed_types for review.
- If a concept has no valid type evidence after recomputation, we set
  canonical_type = "no_supported_type".
- Therefore, after disambiguation, Concept.canonical_type can be either:
    - one of the allowed entity types;
    - "ambiguous";
    - "no_supported_type".
"""

import logging
from typing import Dict

from neo4j import Driver

from knowledge_graph.entity_schema import ALLOWED_TYPES


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

    tx.run(
        """
        CREATE INDEX concept_canonical_type IF NOT EXISTS
        FOR (c:Concept)
        ON (c.canonical_type)
        """
    )


def delete_orphan_concepts(tx) -> int:
    """
    Delete Concept nodes that are no longer referenced by any Section.

    This is useful after reruns where old MENTIONS relationships are replaced,
    or when failed/skipped sections clear stale mentions.
    """
    result = tx.run(
        """
        MATCH (c:Concept)
        WHERE NOT EXISTS {
            MATCH (:Section)-[:MENTIONS]->(c)
        }
        WITH collect(c) AS concepts_to_delete, count(c) AS deleted_count
        FOREACH (c IN concepts_to_delete | DETACH DELETE c)
        RETURN deleted_count
        """
    )

    record = result.single()
    return int(record["deleted_count"] or 0) if record is not None else 0


def reset_type_resolution_state(tx) -> None:
    """
    Reset recomputed type-resolution fields before rebuilding them from current evidence.

    We intentionally keep c.canonical_type untouched here so that, if the next
    recomputation ends in a tie, an existing/manual canonical_type can be kept
    provided it is still among the top-supported types.

    Concepts for which the old canonical_type is no longer supported will have it
    overwritten during resolution.
    """
    tx.run(
        """
        MATCH (c:Concept)
        SET c.observed_types = [],
            c.invalid_observed_types = [],
            c.type_support_pairs = [],
            c.needs_type_review = null,
            c.type_resolution_status = null,
            c.type_resolution_updated_at = datetime()
        """
    )



def collect_invalid_observed_types(tx, allowed_types) -> None:
    """
    Preserve schema-invalid type evidence for migration and review.

    Invalid values are excluded from canonical type resolution, but are stored
    on each Concept so stale or malformed graph state is visible rather than
    silently discarded.
    """
    tx.run(
        """
        MATCH (c:Concept)<-[r:MENTIONS]-(:Section)
        UNWIND coalesce(r.observed_types, []) AS observed_type
        WITH c, observed_type
        WHERE observed_type IS NOT NULL
          AND observed_type <> ""
          AND NOT (observed_type IN $allowed_types)
        WITH c, collect(DISTINCT observed_type) AS invalid_types
        SET c.invalid_observed_types = invalid_types,
            c.needs_type_review = true,
            c.type_resolution_updated_at = datetime()
        """,
        allowed_types=list(allowed_types),
    )


def resolve_types_by_section_support(tx, allowed_types) -> None:
    """
    Recompute concept type support from current Section-[:MENTIONS]->Concept relationships.

    Each section contributes at most one support per type for a given concept because
    relationship observed_types are expected to be deduplicated at write time.

    Resolution policy:
    - one observed type:
        canonical_type = that type
    - multiple observed types with a unique top-supported type:
        canonical_type = the unique winner
    - tied top-supported types:
        preserve existing canonical_type only if it is among the tied top types;
        otherwise canonical_type = "ambiguous"
    """
    tx.run(
        """
        MATCH (c:Concept)<-[r:MENTIONS]-(s:Section)
        UNWIND coalesce(r.observed_types, []) AS supported_type
        WITH c, s, supported_type
        WHERE supported_type IS NOT NULL
          AND supported_type <> ""
          AND supported_type IN $allowed_types

        WITH c, supported_type, count(DISTINCT s) AS support_count
        ORDER BY c.name, support_count DESC, supported_type ASC

        WITH c, collect({type: supported_type, count: support_count}) AS support_rows
        WITH
            c,
            support_rows,
            [row IN support_rows | row.type] AS observed_types,
            [row IN support_rows | row.type + '=' + toString(row.count)] AS type_support_pairs,
            [row IN support_rows WHERE row.count = support_rows[0].count | row.type] AS top_types

        SET c.observed_types = observed_types,
            c.type_support_pairs = type_support_pairs,
            c.canonical_type =
                CASE
                    WHEN size(observed_types) = 1 THEN observed_types[0]
                    WHEN size(top_types) = 1 THEN top_types[0]
                    WHEN c.canonical_type IS NOT NULL AND c.canonical_type IN top_types THEN c.canonical_type
                    ELSE 'ambiguous'
                END,
            c.needs_type_review =
                CASE
                    WHEN size(coalesce(c.invalid_observed_types, [])) > 0 THEN true
                    WHEN size(observed_types) = 1 THEN false
                    WHEN size(top_types) = 1 THEN false
                    ELSE true
                END,
            c.type_resolution_status =
                CASE
                    WHEN size(observed_types) = 1 THEN 'resolved_single_supported_type'
                    WHEN size(top_types) = 1 THEN 'resolved_by_section_support'
                    ELSE 'ambiguous_tied_section_support'
                END,
            c.type_resolution_updated_at = datetime()
        """,
        allowed_types=list(allowed_types),
    )


def mark_concepts_without_supported_types(tx) -> None:
    """
    Mark concepts for which no current section-level type support could be recomputed.

    After optional orphan deletion, this should mostly catch malformed old graph state,
    for example concepts still linked by MENTIONS relationships whose observed_types are
    missing or empty.
    """
    tx.run(
        """
        MATCH (c:Concept)
        WHERE c.observed_types IS NULL OR size(c.observed_types) = 0
        SET c.canonical_type = 'no_supported_type',
            c.type_support_pairs = [],
            c.needs_type_review = true,
            c.type_resolution_status = 'no_supported_types_after_recompute',
            c.type_resolution_updated_at = datetime()
        """
    )


def summarize_disambiguation(tx) -> Dict[str, int]:
    result = tx.run(
        """
        MATCH (c:Concept)
        RETURN
            count(c) AS total_concepts,
            count(CASE WHEN c.needs_type_review = true THEN 1 END) AS concepts_needing_review,
            count(CASE WHEN c.type_resolution_status = 'resolved_single_supported_type' THEN 1 END) AS resolved_single_type,
            count(CASE WHEN c.type_resolution_status = 'resolved_by_section_support' THEN 1 END) AS resolved_by_support,
            count(CASE WHEN c.type_resolution_status = 'ambiguous_tied_section_support' THEN 1 END) AS ambiguous_tied_support,
            count(CASE WHEN c.type_resolution_status = 'no_supported_types_after_recompute' THEN 1 END) AS no_supported_types,
            count(CASE WHEN c.canonical_type = 'ambiguous' THEN 1 END) AS concepts_with_ambiguous_canonical_type,
            count(CASE WHEN c.canonical_type = 'no_supported_type' THEN 1 END) AS concepts_with_no_supported_type,
            count(CASE WHEN size(coalesce(c.invalid_observed_types, [])) > 0 THEN 1 END) AS concepts_with_invalid_observed_types,
            sum(size(coalesce(c.invalid_observed_types, []))) AS invalid_observed_type_values
        """
    )

    record = result.single()

    if record is None:
        return {
            "total_concepts": 0,
            "concepts_needing_review": 0,
            "resolved_single_type": 0,
            "resolved_by_support": 0,
            "ambiguous_tied_support": 0,
            "no_supported_types": 0,
            "concepts_with_ambiguous_canonical_type": 0,
            "concepts_with_no_supported_type": 0,
            "concepts_with_invalid_observed_types": 0,
            "invalid_observed_type_values": 0,
        }

    return {
        "total_concepts": int(record["total_concepts"]),
        "concepts_needing_review": int(record["concepts_needing_review"]),
        "resolved_single_type": int(record["resolved_single_type"]),
        "resolved_by_support": int(record["resolved_by_support"]),
        "ambiguous_tied_support": int(record["ambiguous_tied_support"]),
        "no_supported_types": int(record["no_supported_types"]),
        "concepts_with_ambiguous_canonical_type": int(
            record["concepts_with_ambiguous_canonical_type"]
        ),
        "concepts_with_no_supported_type": int(
            record["concepts_with_no_supported_type"]
        ),
        "concepts_with_invalid_observed_types": int(
            record["concepts_with_invalid_observed_types"]
        ),
        "invalid_observed_type_values": int(
            record["invalid_observed_type_values"] or 0
        ),
    }


def disambiguate_concepts(
    driver: Driver,
    delete_orphans: bool = True,
) -> Dict[str, int]:
    """
    Recompute concept type resolution from current section-level support.

    Strategy:
    - optionally delete orphan concepts
    - reset recomputed concept-level type state
    - aggregate current support from Section-[:MENTIONS]->Concept relationships
    - resolve concepts with a single supported type
    - resolve multi-type concepts by majority support when there is a unique winner
    - preserve existing/manual canonical_type in tied cases only if still supported
    - mark unresolved tied concepts with canonical_type = "ambiguous"
    - mark malformed concepts with no current supported types as
      canonical_type = "no_supported_type"
    """
    with driver.session() as session:
        session.execute_write(setup_disambiguation_schema)

        deleted_orphan_concepts = 0
        if delete_orphans:
            deleted_orphan_concepts = session.execute_write(delete_orphan_concepts)

        allowed_types = sorted(ALLOWED_TYPES)

        session.execute_write(reset_type_resolution_state)
        session.execute_write(collect_invalid_observed_types, allowed_types)
        session.execute_write(resolve_types_by_section_support, allowed_types)
        session.execute_write(mark_concepts_without_supported_types)

        summary = session.execute_read(summarize_disambiguation)

    stats = {
        "total_concepts": summary["total_concepts"],
        "concepts_needing_review": summary["concepts_needing_review"],
        "resolved_single_type": summary["resolved_single_type"],
        "resolved_by_support": summary["resolved_by_support"],
        "ambiguous_tied_support": summary["ambiguous_tied_support"],
        "no_supported_types": summary["no_supported_types"],
        "concepts_with_ambiguous_canonical_type": summary[
            "concepts_with_ambiguous_canonical_type"
        ],
        "concepts_with_no_supported_type": summary[
            "concepts_with_no_supported_type"
        ],
        "concepts_with_invalid_observed_types": summary[
            "concepts_with_invalid_observed_types"
        ],
        "invalid_observed_type_values": summary[
            "invalid_observed_type_values"
        ],
        "deleted_orphan_concepts": deleted_orphan_concepts,
    }

    logger.info(
        "Concept disambiguation completed | total=%d | resolved_single=%d | resolved_by_support=%d | ambiguous_tied=%d | no_supported_types=%d | ambiguous_canonical=%d | no_supported_canonical=%d | concepts_with_invalid_types=%d | invalid_type_values=%d | deleted_orphans=%d | needs_review=%d",
        stats["total_concepts"],
        stats["resolved_single_type"],
        stats["resolved_by_support"],
        stats["ambiguous_tied_support"],
        stats["no_supported_types"],
        stats["concepts_with_ambiguous_canonical_type"],
        stats["concepts_with_no_supported_type"],
        stats["concepts_with_invalid_observed_types"],
        stats["invalid_observed_type_values"],
        stats["deleted_orphan_concepts"],
        stats["concepts_needing_review"],
    )

    return stats