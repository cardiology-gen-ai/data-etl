"""
query_graph.py

Lightweight Neo4j inspection script for the graph-loading phase.

Purpose:
- connect to the project's Neo4j instance
- run only structure-safe diagnostic queries
- print results in a simple readable format

Important:
This script is intended to run after the graph phase only.
It intentionally avoids queries that depend on later enrichment steps
such as embeddings, entity extraction, Concept nodes, or MENTIONS edges.
"""

from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver


QUERIES = {
    "1. Node Types Count": """
        MATCH (n)
        RETURN labels(n) AS Node_Type, count(*) AS Quantity
    """,
    "2. Loaded Documents": """
        MATCH (d:Document)
        RETURN d.doc_id AS Document_Name
    """,
    "3. Chunks per Document": """
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
        RETURN d.doc_id AS Document, count(s) AS Number_of_Chunks
    """,
    "4. Root Chapters (Sections directly linked to Document)": """
        MATCH (d:Document {doc_id: 'Cardiomyopathies_2023'})-[:HAS_SECTION]->(s:Section)
        WHERE NOT (:Section)-[:HAS_CHILD]->(s)
        RETURN s.uid AS Root_Chapter_ID, s.title AS Title
    """,
    "5. Sub-chapters of Chapter 3": """
        MATCH (s:Section {uid: 'Cardiomyopathies_2023::3'})-[:HAS_CHILD]->(sub:Section)
        RETURN sub.uid AS Sub_Chapter, sub.title AS Title
    """,
    "6. Sequential Reading (NEXT relation)": """
        MATCH path = (s1:Section)-[:NEXT*1..3]->(s2:Section)
        WHERE s1.uid = 'Cardiomyopathies_2023::1'
        RETURN [node IN nodes(path) | node.uid] AS Reading_Order
        LIMIT 1
    """,
    "7. Longest Sections (Calculated length)": """
        MATCH (s:Section)
        RETURN s.uid AS Section_ID, size(coalesce(s.text, '')) AS Calculated_Length
        ORDER BY Calculated_Length DESC
        LIMIT 5
    """,
    "8. Text Preview of a Specific Section": """
        MATCH (s:Section {uid: 'Cardiomyopathies_2023::4'})
        RETURN substring(coalesce(s.text, ''), 0, 200) + '...' AS Content_Preview
    """,
    "9. Leaf Nodes (Sections with no further children)": """
        MATCH (s:Section)
        WHERE NOT (s)-[:HAS_CHILD]->(:Section)
        RETURN s.uid AS Leaf_Section, s.title AS Title
        LIMIT 5
    """,
    "10. Real Orphan Check (No link from Doc or other Sections)": """
        MATCH (s:Section)
        WHERE NOT ()-[:HAS_SECTION|HAS_CHILD]->(s)
        RETURN count(s) AS Real_Orphans
    """,
    "11. Sections Missing Text": """
        MATCH (s:Section)
        WHERE coalesce(trim(s.text), '') = ''
        RETURN s.uid AS Section_ID, s.title AS Title
        LIMIT 10
    """,
    "12. Documents with Section Counts by Processing State": """
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
        RETURN d.doc_id AS Document,
               count(s) AS Total_Sections,
               count(CASE WHEN coalesce(s.has_embedding, false) = true THEN 1 END) AS Embedded_Sections,
               count(CASE WHEN coalesce(s.entity_extracted, false) = true THEN 1 END) AS Entity_Processed_Sections
        ORDER BY Document
    """,
}


def run_queries():
    """
    Connect to Neo4j using the project's utility driver,
    execute all defined graph-phase-safe queries,
    and print the results cleanly.
    """
    print("[INFO] Initializing connection via neo4j_utils...")

    driver = get_neo4j_driver(verify=True)

    print("=" * 60)

    try:
        with driver.session() as session:
            for title, cypher_query in QUERIES.items():
                print(f"\n---> {title}")

                result = session.run(cypher_query)
                records = list(result)

                if not records:
                    print("     No results found.")
                else:
                    for record in records:
                        print(f"     {dict(record)}")

                print("-" * 60)
    finally:
        close_driver(driver)
        print("[INFO] Queries execution finished.")


if __name__ == "__main__":
    run_queries()