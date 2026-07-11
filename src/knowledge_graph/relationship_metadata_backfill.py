"""
Dry-run and write backfill for managed relationship provenance metadata.
"""

import argparse
import json
import logging
from typing import Any, Sequence

from neo4j import Driver

from knowledge_graph.neo4j_utils import close_driver, get_neo4j_driver
from knowledge_graph.relationship_metadata import (
    build_mention_relationship_metadata,
    build_normalization_relationship_metadata,
    build_structural_relationship_metadata,
)
from knowledge_graph.umls_connections import UMLS_CONNECTION_RELATION_TYPES


logger = logging.getLogger(__name__)

COMMON_METADATA_FIELDS = [
    "relationship_family",
    "provenance",
    "provenance_source",
    "provenance_method",
]

STRUCTURAL_RELATIONSHIP_TYPES = ["HAS_SECTION", "HAS_CHILD", "NEXT"]
MENTION_RELATIONSHIP_TYPES = ["MENTIONS"]
NORMALIZATION_RELATIONSHIP_TYPES = ["SAME_AS", "POSSIBLY_SAME_AS"]
MANAGED_RELATIONSHIP_TYPES = sorted(
    set(
        STRUCTURAL_RELATIONSHIP_TYPES
        + MENTION_RELATIONSHIP_TYPES
        + NORMALIZATION_RELATIONSHIP_TYPES
        + list(UMLS_CONNECTION_RELATION_TYPES)
    )
)

DEFAULT_SAMPLE_LIMIT = 20


def _records_to_dicts(records: Any) -> list[dict[str, Any]]:
    return [dict(record) for record in records]


def _count(tx, query: str, **params: Any) -> int:
    record = tx.run(query, **params).single()
    return int(record["n"]) if record is not None else 0


def _count_by_key(tx, query: str, key: str, **params: Any) -> dict[str, int]:
    rows = tx.run(query, **params)
    return {
        str(row[key]): int(row["n"])
        for row in rows
    }


def _sample(tx, query: str, sample_limit: int, **params: Any) -> list[dict[str, Any]]:
    if sample_limit <= 0:
        return []
    return _records_to_dicts(tx.run(query, sample_limit=sample_limit, **params))


def _candidate_count_for_expected_metadata(
    tx,
    relationship_type: str,
    relationship_metadata: dict[str, object],
) -> int:
    query = f"""
        MATCH ()-[r:{relationship_type}]->()
        WITH r, $relationship_metadata AS expected_metadata
        WHERE any(key IN keys(expected_metadata)
                  WHERE r[key] IS NULL OR r[key] <> expected_metadata[key])
        RETURN count(r) AS n
    """
    return _count(tx, query, relationship_metadata=relationship_metadata)


def _write_expected_metadata(
    tx,
    relationship_type: str,
    relationship_metadata: dict[str, object],
) -> int:
    query = f"""
        MATCH ()-[r:{relationship_type}]->()
        WITH r, $relationship_metadata AS expected_metadata
        WHERE any(key IN keys(expected_metadata)
                  WHERE r[key] IS NULL OR r[key] <> expected_metadata[key])
        SET r += expected_metadata
        RETURN count(r) AS n
    """
    return _count(tx, query, relationship_metadata=relationship_metadata)


def _metadata_with_structural_doc_id_query(
    query_tail: str,
    match_pattern: str,
    source_doc_expression: str,
    target_doc_expression: str,
) -> str:
    return f"""
        MATCH {match_pattern}
        WITH r,
             coalesce(trim(toString({source_doc_expression})), '') AS source_doc_id,
             coalesce(trim(toString({target_doc_expression})), '') AS target_doc_id
        WITH r,
             CASE
                 WHEN source_doc_id <> '' AND target_doc_id <> ''
                      AND source_doc_id <> target_doc_id
                 THEN null
                 WHEN source_doc_id <> '' THEN source_doc_id
                 WHEN target_doc_id <> '' THEN target_doc_id
                 ELSE null
             END AS relationship_doc_id
        WITH r, $relationship_metadata AS relationship_metadata, relationship_doc_id
        WHERE any(key IN keys(relationship_metadata)
                  WHERE r[key] IS NULL OR r[key] <> relationship_metadata[key])
           OR (
                relationship_doc_id IS NOT NULL
                AND (r.doc_id IS NULL OR r.doc_id <> relationship_doc_id)
              )
        {query_tail}
    """


def _count_structural_update_candidates(tx, relationship_type: str) -> int:
    relationship_metadata = build_structural_relationship_metadata(relationship_type)
    if relationship_type == "HAS_SECTION":
        query = _metadata_with_structural_doc_id_query(
            "RETURN count(r) AS n",
            "(d:Document)-[r:HAS_SECTION]->(s:Section)",
            "d.doc_id",
            "s.doc_id",
        )
    elif relationship_type == "HAS_CHILD":
        query = _metadata_with_structural_doc_id_query(
            "RETURN count(r) AS n",
            "(source:Section)-[r:HAS_CHILD]->(target:Section)",
            "source.doc_id",
            "target.doc_id",
        )
    elif relationship_type == "NEXT":
        query = _metadata_with_structural_doc_id_query(
            "RETURN count(r) AS n",
            "(source:Section)-[r:NEXT]->(target:Section)",
            "source.doc_id",
            "target.doc_id",
        )
    else:
        raise ValueError(f"Unsupported structural relationship type: {relationship_type}")

    return _count(tx, query, relationship_metadata=relationship_metadata)


def _write_structural_metadata(tx, relationship_type: str) -> int:
    relationship_metadata = build_structural_relationship_metadata(relationship_type)
    if relationship_type == "HAS_SECTION":
        query = _metadata_with_structural_doc_id_query(
            """
            SET r += relationship_metadata
            FOREACH (_ IN CASE WHEN relationship_doc_id IS NULL THEN [] ELSE [1] END |
                SET r.doc_id = relationship_doc_id
            )
            RETURN count(r) AS n
            """,
            "(d:Document)-[r:HAS_SECTION]->(s:Section)",
            "d.doc_id",
            "s.doc_id",
        )
    elif relationship_type == "HAS_CHILD":
        query = _metadata_with_structural_doc_id_query(
            """
            SET r += relationship_metadata
            FOREACH (_ IN CASE WHEN relationship_doc_id IS NULL THEN [] ELSE [1] END |
                SET r.doc_id = relationship_doc_id
            )
            RETURN count(r) AS n
            """,
            "(source:Section)-[r:HAS_CHILD]->(target:Section)",
            "source.doc_id",
            "target.doc_id",
        )
    elif relationship_type == "NEXT":
        query = _metadata_with_structural_doc_id_query(
            """
            SET r += relationship_metadata
            FOREACH (_ IN CASE WHEN relationship_doc_id IS NULL THEN [] ELSE [1] END |
                SET r.doc_id = relationship_doc_id
            )
            RETURN count(r) AS n
            """,
            "(source:Section)-[r:NEXT]->(target:Section)",
            "source.doc_id",
            "target.doc_id",
        )
    else:
        raise ValueError(f"Unsupported structural relationship type: {relationship_type}")

    return _count(tx, query, relationship_metadata=relationship_metadata)


def _count_mentions_update_candidates(tx) -> int:
    relationship_metadata = build_mention_relationship_metadata()
    query = """
        MATCH (s:Section)-[r:MENTIONS]->(:Concept)
        WITH r,
             coalesce(trim(toString(s.doc_id)), '') AS section_doc_id
        WITH r, $relationship_metadata AS relationship_metadata, section_doc_id
        WHERE any(key IN keys(relationship_metadata)
                  WHERE r[key] IS NULL OR r[key] <> relationship_metadata[key])
           OR (
                section_doc_id <> ''
                AND (r.doc_id IS NULL OR r.doc_id <> section_doc_id)
              )
        RETURN count(r) AS n
    """
    return _count(tx, query, relationship_metadata=relationship_metadata)


def _write_mentions_metadata(tx) -> int:
    relationship_metadata = build_mention_relationship_metadata()
    query = """
        MATCH (s:Section)-[r:MENTIONS]->(:Concept)
        WITH r,
             coalesce(trim(toString(s.doc_id)), '') AS section_doc_id
        WITH r, $relationship_metadata AS relationship_metadata, section_doc_id
        WHERE any(key IN keys(relationship_metadata)
                  WHERE r[key] IS NULL OR r[key] <> relationship_metadata[key])
           OR (
                section_doc_id <> ''
                AND (r.doc_id IS NULL OR r.doc_id <> section_doc_id)
              )
        SET r += relationship_metadata
        FOREACH (_ IN CASE WHEN section_doc_id = '' THEN [] ELSE [1] END |
            SET r.doc_id = section_doc_id
        )
        RETURN count(r) AS n
    """
    return _count(tx, query, relationship_metadata=relationship_metadata)


def _count_ontology_update_candidates_by_type(tx) -> dict[str, int]:
    query = """
        MATCH ()-[r]->()
        WHERE type(r) IN $relationship_types
          AND r.source_vocabulary IS NOT NULL
          AND trim(toString(r.source_vocabulary)) <> ''
        WITH r, type(r) AS relationship_type,
             {
                 relationship_family: 'ontology',
                 provenance: 'umls_connections',
                 provenance_source: 'umls_metathesaurus',
                 provenance_method: 'umls_relations_api',
                 source_vocabulary: r.source_vocabulary
             } AS expected_metadata
        WHERE any(key IN keys(expected_metadata)
                  WHERE r[key] IS NULL OR r[key] <> expected_metadata[key])
        RETURN relationship_type, count(r) AS n
        ORDER BY relationship_type
    """
    return _count_by_key(
        tx,
        query,
        "relationship_type",
        relationship_types=UMLS_CONNECTION_RELATION_TYPES,
    )


def _write_ontology_metadata_by_type(tx) -> dict[str, int]:
    query = """
        MATCH ()-[r]->()
        WHERE type(r) IN $relationship_types
          AND r.source_vocabulary IS NOT NULL
          AND trim(toString(r.source_vocabulary)) <> ''
        WITH r, type(r) AS relationship_type,
             {
                 relationship_family: 'ontology',
                 provenance: 'umls_connections',
                 provenance_source: 'umls_metathesaurus',
                 provenance_method: 'umls_relations_api',
                 source_vocabulary: r.source_vocabulary
             } AS expected_metadata
        WHERE any(key IN keys(expected_metadata)
                  WHERE r[key] IS NULL OR r[key] <> expected_metadata[key])
        SET r += expected_metadata
        RETURN relationship_type, count(r) AS n
        ORDER BY relationship_type
    """
    return _count_by_key(
        tx,
        query,
        "relationship_type",
        relationship_types=UMLS_CONNECTION_RELATION_TYPES,
    )


def _collect_update_candidates(tx) -> dict[str, int]:
    updates: dict[str, int] = {}

    for relationship_type in STRUCTURAL_RELATIONSHIP_TYPES:
        updates[relationship_type] = _count_structural_update_candidates(
            tx,
            relationship_type,
        )

    updates["MENTIONS"] = _count_mentions_update_candidates(tx)

    for relationship_type in NORMALIZATION_RELATIONSHIP_TYPES:
        updates[relationship_type] = _candidate_count_for_expected_metadata(
            tx,
            relationship_type,
            build_normalization_relationship_metadata(relationship_type),
        )

    for relationship_type in UMLS_CONNECTION_RELATION_TYPES:
        updates[relationship_type] = 0
    updates.update(_count_ontology_update_candidates_by_type(tx))

    return updates


def _write_managed_metadata(tx) -> dict[str, int]:
    updates: dict[str, int] = {}

    for relationship_type in STRUCTURAL_RELATIONSHIP_TYPES:
        updates[relationship_type] = _write_structural_metadata(tx, relationship_type)

    updates["MENTIONS"] = _write_mentions_metadata(tx)

    for relationship_type in NORMALIZATION_RELATIONSHIP_TYPES:
        updates[relationship_type] = _write_expected_metadata(
            tx,
            relationship_type,
            build_normalization_relationship_metadata(relationship_type),
        )

    for relationship_type in UMLS_CONNECTION_RELATION_TYPES:
        updates[relationship_type] = 0
    updates.update(_write_ontology_metadata_by_type(tx))

    return updates


def _collect_doc_id_conflicts(tx, sample_limit: int) -> dict[str, Any]:
    count_query = """
        MATCH (d:Document)-[r:HAS_SECTION]->(s:Section)
        WITH r, d, s,
             coalesce(trim(toString(d.doc_id)), '') AS source_doc_id,
             coalesce(trim(toString(s.doc_id)), '') AS target_doc_id
        WHERE source_doc_id <> '' AND target_doc_id <> ''
          AND source_doc_id <> target_doc_id
        RETURN count(r) AS n
    """
    sample_query = """
        MATCH (d:Document)-[r:HAS_SECTION]->(s:Section)
        WITH r, d, s,
             coalesce(trim(toString(d.doc_id)), '') AS source_doc_id,
             coalesce(trim(toString(s.doc_id)), '') AS target_doc_id
        WHERE source_doc_id <> '' AND target_doc_id <> ''
          AND source_doc_id <> target_doc_id
        RETURN type(r) AS relationship_type,
               elementId(r) AS relationship_id,
               d.doc_id AS source_doc_id,
               s.doc_id AS target_doc_id
        ORDER BY relationship_id
        LIMIT $sample_limit
    """
    structural_count_query = """
        MATCH (source:Section)-[r]->(target:Section)
        WHERE type(r) IN ['HAS_CHILD', 'NEXT']
        WITH r, source, target,
             coalesce(trim(toString(source.doc_id)), '') AS source_doc_id,
             coalesce(trim(toString(target.doc_id)), '') AS target_doc_id
        WHERE source_doc_id <> '' AND target_doc_id <> ''
          AND source_doc_id <> target_doc_id
        RETURN count(r) AS n
    """
    structural_sample_query = """
        MATCH (source:Section)-[r]->(target:Section)
        WHERE type(r) IN ['HAS_CHILD', 'NEXT']
        WITH r, source, target,
             coalesce(trim(toString(source.doc_id)), '') AS source_doc_id,
             coalesce(trim(toString(target.doc_id)), '') AS target_doc_id
        WHERE source_doc_id <> '' AND target_doc_id <> ''
          AND source_doc_id <> target_doc_id
        RETURN type(r) AS relationship_type,
               elementId(r) AS relationship_id,
               source.uid AS source_uid,
               target.uid AS target_uid,
               source.doc_id AS source_doc_id,
               target.doc_id AS target_doc_id
        ORDER BY relationship_type, relationship_id
        LIMIT $sample_limit
    """
    has_section_count = _count(tx, count_query)
    structural_count = _count(tx, structural_count_query)
    return {
        "count": has_section_count + structural_count,
        "sample": _sample(tx, sample_query, sample_limit)
        + _sample(tx, structural_sample_query, sample_limit),
    }


def _collect_report(tx, sample_limit: int) -> dict[str, Any]:
    total_relationships = _count(
        tx,
        "MATCH ()-[r]->() RETURN count(r) AS n",
    )
    counts_by_relationship_type = _count_by_key(
        tx,
        """
        MATCH ()-[r]->()
        RETURN type(r) AS relationship_type, count(r) AS n
        ORDER BY n DESC, relationship_type ASC
        """,
        "relationship_type",
    )
    counts_by_relationship_family = _count_by_key(
        tx,
        """
        MATCH ()-[r]->()
        RETURN coalesce(r.relationship_family, 'UNSET') AS relationship_family,
               count(r) AS n
        ORDER BY n DESC, relationship_family ASC
        """,
        "relationship_family",
    )
    missing_common_metadata = _count(
        tx,
        """
        MATCH ()-[r]->()
        WHERE type(r) IN $relationship_types
          AND any(field IN $common_metadata_fields
                  WHERE r[field] IS NULL OR trim(toString(r[field])) = '')
        RETURN count(r) AS n
        """,
        relationship_types=MANAGED_RELATIONSHIP_TYPES,
        common_metadata_fields=COMMON_METADATA_FIELDS,
    )
    missing_common_metadata_sample = _sample(
        tx,
        """
        MATCH ()-[r]->()
        WHERE type(r) IN $relationship_types
          AND any(field IN $common_metadata_fields
                  WHERE r[field] IS NULL OR trim(toString(r[field])) = '')
        RETURN type(r) AS relationship_type,
               elementId(r) AS relationship_id,
               [field IN $common_metadata_fields
                WHERE r[field] IS NULL OR trim(toString(r[field])) = ''] AS missing_fields
        ORDER BY relationship_type, relationship_id
        LIMIT $sample_limit
        """,
        sample_limit,
        relationship_types=MANAGED_RELATIONSHIP_TYPES,
        common_metadata_fields=COMMON_METADATA_FIELDS,
    )
    unknown_or_unmanaged_relationship_types = _count_by_key(
        tx,
        """
        MATCH ()-[r]->()
        WHERE NOT type(r) IN $relationship_types
        RETURN type(r) AS relationship_type, count(r) AS n
        ORDER BY n DESC, relationship_type ASC
        """,
        "relationship_type",
        relationship_types=MANAGED_RELATIONSHIP_TYPES,
    )
    ontology_missing_source_vocabulary = _count(
        tx,
        """
        MATCH ()-[r]->()
        WHERE type(r) IN $relationship_types
          AND (r.source_vocabulary IS NULL OR trim(toString(r.source_vocabulary)) = '')
        RETURN count(r) AS n
        """,
        relationship_types=UMLS_CONNECTION_RELATION_TYPES,
    )
    ontology_missing_source_vocabulary_sample = _sample(
        tx,
        """
        MATCH ()-[r]->()
        WHERE type(r) IN $relationship_types
          AND (r.source_vocabulary IS NULL OR trim(toString(r.source_vocabulary)) = '')
        RETURN type(r) AS relationship_type,
               elementId(r) AS relationship_id,
               r.edge_key AS edge_key,
               r.source_cui AS source_cui,
               r.target_cui AS target_cui
        ORDER BY relationship_type, relationship_id
        LIMIT $sample_limit
        """,
        sample_limit,
        relationship_types=UMLS_CONNECTION_RELATION_TYPES,
    )
    updates_by_relationship_type = _collect_update_candidates(tx)

    return {
        "total_relationships": total_relationships,
        "counts_by_relationship_type": counts_by_relationship_type,
        "counts_by_relationship_family": counts_by_relationship_family,
        "relationships_missing_common_metadata": {
            "count": missing_common_metadata,
            "sample": missing_common_metadata_sample,
        },
        "relationships_that_would_be_updated": sum(
            updates_by_relationship_type.values()
        ),
        "updates_by_relationship_type": updates_by_relationship_type,
        "unknown_or_unmanaged_relationship_types": unknown_or_unmanaged_relationship_types,
        "ontology_relationships_missing_source_vocabulary": {
            "count": ontology_missing_source_vocabulary,
            "sample": ontology_missing_source_vocabulary_sample,
        },
        "relationships_with_conflicting_doc_ids": _collect_doc_id_conflicts(
            tx,
            sample_limit,
        ),
    }


def run_relationship_metadata_backfill(
    driver: Driver,
    write: bool = False,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """
    Report and optionally update managed relationship provenance metadata.
    """
    if sample_limit < 0:
        raise ValueError("sample_limit must be >= 0")

    with driver.session() as session:
        report = session.execute_read(_collect_report, sample_limit)

        if write:
            updated = session.execute_write(_write_managed_metadata)
            report["write"] = True
            report["relationships_updated"] = sum(updated.values())
            report["updated_by_relationship_type"] = updated
        else:
            report["write"] = False
            report["relationships_updated"] = 0
            report["updated_by_relationship_type"] = {}

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report and optionally backfill provenance metadata for managed "
            "knowledge-graph relationships. Defaults to read-only dry-run."
        )
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply idempotent metadata updates. Without this flag, no writes occur.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help=f"Maximum sample rows per diagnostic section (default: {DEFAULT_SAMPLE_LIMIT})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    driver = None
    try:
        driver = get_neo4j_driver()
        report = run_relationship_metadata_backfill(
            driver=driver,
            write=args.write,
            sample_limit=args.sample_limit,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        close_driver(driver)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMMON_METADATA_FIELDS",
    "MANAGED_RELATIONSHIP_TYPES",
    "run_relationship_metadata_backfill",
    "build_arg_parser",
    "main",
]
