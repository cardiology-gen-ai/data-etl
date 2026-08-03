"""
Section-view-aware Neo4j inspection script.

This module is safe to run immediately after the graph-loading phase. It does
not require Concept nodes, MENTIONS relationships, or embeddings to exist.
Optional entity/embedding state is reported only as a diagnostic summary.

Examples:
    PYTHONPATH=src python -m knowledge_graph.query_graph
    PYTHONPATH=src python -m knowledge_graph.query_graph --doc-id Cardiomyopathies_2023
    PYTHONPATH=src python -m knowledge_graph.query_graph \
        --doc-id Cardiomyopathies_2023 \
        --uid Cardiomyopathies_2023::7.1.4.1 \
        --limit 15 \
        --next-hops 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from knowledge_graph.neo4j_utils import close_driver, get_neo4j_driver


@dataclass(frozen=True)
class QuerySpec:
    title: str
    query: str
    params: Dict[str, Any] = field(default_factory=dict)


def _build_query_specs(
    *,
    doc_id: Optional[str],
    uid: Optional[str],
    limit: int,
    next_hops: int,
) -> List[QuerySpec]:
    """Build structure-safe diagnostics without hard-coded document identifiers."""
    common = {
        "doc_id": doc_id,
        "uid": uid,
        "limit": limit,
    }

    # Neo4j does not accept a parameter in the variable-length relationship
    # bound, so next_hops is validated before interpolation.
    next_query = f"""
        MATCH (d:Document)-[:HAS_SECTION]->(start:Section)
        WHERE ($doc_id IS NULL OR d.doc_id = $doc_id)
          AND start.section_view_role = 'retrieval'
          AND start.retrieval_order = 0
        OPTIONAL MATCH path = (start)-[:NEXT*0..{next_hops}]->(end:Section)
        WITH d, path
        ORDER BY d.doc_id, length(path) DESC
        WITH d, collect(path)[0] AS longest_path
        RETURN d.doc_id AS Document,
               [node IN nodes(longest_path) |
                    {{uid: node.uid,
                      section_id: node.section_id,
                      retrieval_order: node.retrieval_order,
                      title: node.title}}] AS Reading_Order
        ORDER BY Document
    """

    specs = [
        QuerySpec(
            "1. Node counts by label",
            """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label AS Node_Label, count(*) AS Quantity
            ORDER BY Node_Label
            """,
        ),
        QuerySpec(
            "2. Loaded documents and Section-view metadata",
            """
            MATCH (d:Document)
            WHERE ($doc_id IS NULL OR d.doc_id = $doc_id)
            RETURN d.doc_id AS Document,
                   d.retrieval_strategy AS Strategy,
                   d.aggregation_mode AS Aggregation_Mode,
                   d.retrieval_max_level AS Max_Level,
                   d.section_view_section_count AS Declared_Sections,
                   d.section_view_retrieval_count AS Declared_Retrieval,
                   d.section_view_structural_count AS Declared_Structural,
                   d.section_view_aggregated_count AS Declared_Aggregated,
                   d.section_view_file AS Section_View_File
            ORDER BY Document
            """,
            common,
        ),
        QuerySpec(
            "3. Actual Section counts by document and role",
            """
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
            WHERE ($doc_id IS NULL OR d.doc_id = $doc_id)
            RETURN d.doc_id AS Document,
                   count(s) AS Total_Sections,
                   count(CASE WHEN s.section_view_role = 'retrieval' THEN 1 END) AS Retrieval_Sections,
                   count(CASE WHEN s.section_view_role = 'structural' THEN 1 END) AS Structural_Sections,
                   count(CASE WHEN coalesce(s.is_aggregated, false) THEN 1 END) AS Aggregated_Sections,
                   count(CASE WHEN coalesce(s.embed, false) THEN 1 END) AS Embed_Eligible
            ORDER BY Document
            """,
            common,
        ),
        QuerySpec(
            "4. Relationship counts by type and provenance",
            """
            MATCH (a)-[r]->(b)
            WHERE ($doc_id IS NULL OR a.doc_id = $doc_id OR b.doc_id = $doc_id)
            RETURN type(r) AS Relationship,
                   coalesce(r.provenance, 'UNSET') AS Provenance,
                   coalesce(r.provenance_method, 'UNSET') AS Method,
                   count(*) AS Quantity
            ORDER BY Relationship, Provenance, Method
            """,
            common,
        ),
        QuerySpec(
            "5. Root Sections",
            """
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
            WHERE ($doc_id IS NULL OR d.doc_id = $doc_id)
              AND NOT EXISTS {
                  MATCH (:Section)-[:HAS_CHILD]->(s)
              }
            RETURN d.doc_id AS Document,
                   s.uid AS UID,
                   s.section_id AS Section_ID,
                   s.section_view_role AS Role,
                   s.title AS Title
            ORDER BY Document, s.section_view_order
            LIMIT $limit
            """,
            common,
        ),
        QuerySpec(
            "6. Sample hierarchy edges",
            """
            MATCH (parent:Section)-[:HAS_CHILD]->(child:Section)
            WHERE ($doc_id IS NULL OR parent.doc_id = $doc_id)
            RETURN parent.doc_id AS Document,
                   parent.uid AS Parent_UID,
                   parent.section_view_role AS Parent_Role,
                   child.uid AS Child_UID,
                   child.section_view_role AS Child_Role
            ORDER BY Document, child.section_view_order
            LIMIT $limit
            """,
            common,
        ),
        QuerySpec(
            f"7. Retrieval reading order from the first Section (up to {next_hops} NEXT hops)",
            next_query,
            common,
        ),
        QuerySpec(
            "8. Longest active retrieval Sections",
            """
            MATCH (s:Section)
            WHERE ($doc_id IS NULL OR s.doc_id = $doc_id)
              AND s.section_view_role = 'retrieval'
            RETURN s.doc_id AS Document,
                   s.uid AS UID,
                   s.section_id AS Section_ID,
                   s.title AS Title,
                   size(coalesce(s.text, '')) AS Text_Characters,
                   coalesce(s.is_aggregated, false) AS Is_Aggregated,
                   s.source_count AS Source_Count
            ORDER BY Text_Characters DESC, UID
            LIMIT $limit
            """,
            common,
        ),
        QuerySpec(
            "9. Aggregated retrieval Sections and provenance",
            """
            MATCH (s:Section)
            WHERE ($doc_id IS NULL OR s.doc_id = $doc_id)
              AND s.section_view_role = 'retrieval'
              AND coalesce(s.is_aggregated, false) = true
            RETURN s.doc_id AS Document,
                   s.uid AS UID,
                   s.section_id AS Section_ID,
                   s.title AS Title,
                   s.source_count AS Source_Count,
                   s.source_section_ids AS Source_Section_IDs,
                   s.represented_section_ids AS Represented_Section_IDs,
                   s.absorbed_section_ids AS Absorbed_Section_IDs,
                   size(coalesce(s.text, '')) AS Text_Characters
            ORDER BY Document, s.retrieval_order
            LIMIT $limit
            """,
            common,
        ),
        QuerySpec(
            "10. Structural Sections",
            """
            MATCH (s:Section)
            WHERE ($doc_id IS NULL OR s.doc_id = $doc_id)
              AND s.section_view_role = 'structural'
            RETURN s.doc_id AS Document,
                   s.uid AS UID,
                   s.section_id AS Section_ID,
                   s.title AS Title,
                   s.parent_section_id AS Parent_Section_ID,
                   size(coalesce(s.text, '')) AS Text_Characters,
                   s.embed AS Embed
            ORDER BY Document, s.section_view_order
            LIMIT $limit
            """,
            common,
        ),
        QuerySpec(
            "11. Structure invariant violations",
            """
            MATCH (s:Section)
            WHERE ($doc_id IS NULL OR s.doc_id = $doc_id)
              AND (
                    NOT (s.section_view_role IN ['retrieval', 'structural'])
                 OR (s.section_view_role = 'retrieval' AND (
                        coalesce(trim(s.text), '') = ''
                     OR coalesce(s.embed, false) = false
                     OR s.retrieval_order IS NULL
                 ))
                 OR (s.section_view_role = 'structural' AND (
                        coalesce(trim(s.text), '') <> ''
                     OR coalesce(s.embed, false) = true
                     OR s.retrieval_order IS NOT NULL
                 ))
              )
            RETURN s.doc_id AS Document,
                   s.uid AS UID,
                   s.section_view_role AS Role,
                   s.embed AS Embed,
                   s.retrieval_order AS Retrieval_Order,
                   size(coalesce(s.text, '')) AS Text_Characters
            ORDER BY Document, UID
            LIMIT $limit
            """,
            common,
        ),
        QuerySpec(
            "12. Orphan and cross-document relationship counts",
            """
            OPTIONAL MATCH (orphan:Section)
            WHERE ($doc_id IS NULL OR orphan.doc_id = $doc_id)
              AND NOT EXISTS {
                  MATCH (:Document)-[:HAS_SECTION]->(orphan)
              }
            WITH count(orphan) AS Orphan_Sections
            OPTIONAL MATCH (parent:Section)-[hc:HAS_CHILD]->(child:Section)
            WHERE ($doc_id IS NULL OR parent.doc_id = $doc_id)
              AND parent.doc_id <> child.doc_id
            WITH Orphan_Sections, count(hc) AS Cross_Document_HAS_CHILD
            OPTIONAL MATCH (a:Section)-[n:NEXT]->(b:Section)
            WHERE ($doc_id IS NULL OR a.doc_id = $doc_id)
              AND a.doc_id <> b.doc_id
            RETURN Orphan_Sections,
                   Cross_Document_HAS_CHILD,
                   count(n) AS Cross_Document_NEXT
            """,
            common,
        ),
        QuerySpec(
            "13. Optional processing state on retrieval Sections",
            """
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
            WHERE ($doc_id IS NULL OR d.doc_id = $doc_id)
              AND s.section_view_role = 'retrieval'
            RETURN d.doc_id AS Document,
                   count(s) AS Retrieval_Sections,
                   count(CASE WHEN coalesce(s.entity_extracted, false) THEN 1 END) AS Entity_Processed,
                   count(CASE WHEN coalesce(s.has_embedding, false) THEN 1 END) AS Embedded,
                   count(CASE WHEN coalesce(properties(s)['embedding_segment_count'], 0) > 1 THEN 1 END) AS Segmented_Embeddings
            ORDER BY Document
            """,
            common,
        ),
    ]

    if uid:
        specs.append(
            QuerySpec(
                "14. Requested Section detail",
                """
                MATCH (s:Section {uid: $uid})
                OPTIONAL MATCH (parent:Section)-[:HAS_CHILD]->(s)
                OPTIONAL MATCH (s)-[:HAS_CHILD]->(child:Section)
                WITH s,
                     parent,
                     collect(DISTINCT CASE
                         WHEN child IS NULL THEN null
                         ELSE {
                             uid: child.uid,
                             section_id: child.section_id,
                             role: child.section_view_role,
                             title: child.title
                         }
                     END) AS children
                RETURN s.uid AS UID,
                       s.doc_id AS Document,
                       s.section_id AS Section_ID,
                       s.title AS Title,
                       s.section_view_role AS Role,
                       s.retrieval_order AS Retrieval_Order,
                       s.section_view_order AS Section_View_Order,
                       coalesce(s.is_aggregated, false) AS Is_Aggregated,
                       s.source_section_ids AS Source_Section_IDs,
                       s.represented_section_ids AS Represented_Section_IDs,
                       s.absorbed_section_ids AS Absorbed_Section_IDs,
                       parent.uid AS Parent_UID,
                       children AS Children,
                       substring(coalesce(s.text, ''), 0, 500) AS Text_Preview
                """,
                common,
            )
        )

    return specs


def run_queries(
    *,
    doc_id: Optional[str] = None,
    uid: Optional[str] = None,
    limit: int = 10,
    next_hops: int = 5,
) -> None:
    """Connect to Neo4j, execute diagnostics, and print readable records."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if not 1 <= next_hops <= 20:
        raise ValueError("next_hops must be between 1 and 20")

    specs = _build_query_specs(
        doc_id=doc_id,
        uid=uid,
        limit=limit,
        next_hops=next_hops,
    )

    print("[INFO] Initializing connection via neo4j_utils...")
    if doc_id:
        print(f"[INFO] Document filter: {doc_id}")
    if uid:
        print(f"[INFO] Section detail: {uid}")

    driver = get_neo4j_driver(verify=True)

    try:
        with driver.session() as session:
            for spec in specs:
                print("\n" + "=" * 80)
                print(spec.title)
                print("-" * 80)

                records = list(session.run(spec.query, **spec.params))
                if not records:
                    print("No results found.")
                    continue

                for record in records:
                    print(dict(record))
    finally:
        close_driver(driver)
        print("\n[INFO] Query execution finished.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the Section-view-aware Neo4j graph structure."
    )
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Optional document id filter, for example Cardiomyopathies_2023.",
    )
    parser.add_argument(
        "--uid",
        default=None,
        help="Optional exact Section uid for the detailed Section query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum rows for sample queries (default: 10).",
    )
    parser.add_argument(
        "--next-hops",
        type=int,
        default=5,
        help="Maximum NEXT hops shown from retrieval_order=0 (1-20).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_queries(
        doc_id=args.doc_id,
        uid=args.uid,
        limit=args.limit,
        next_hops=args.next_hops,
    )


if __name__ == "__main__":
    main()
