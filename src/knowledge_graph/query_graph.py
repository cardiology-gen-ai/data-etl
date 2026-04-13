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

    "12. Embedding Status Summary": """
        MATCH (s:Section)
        RETURN coalesce(s.embedding_status, 'UNSET') AS Embedding_Status, count(*) AS Count
        ORDER BY Count DESC, Embedding_Status ASC
    """,

    "13. Entity Extraction Status Summary": """
        MATCH (s:Section)
        RETURN coalesce(s.entity_extraction_status, 'UNSET') AS Entity_Status, count(*) AS Count
        ORDER BY Count DESC, Entity_Status ASC
    """,

    "14. Eligible Sections Still Missing Embeddings": """
        MATCH (s:Section)
        WHERE coalesce(s.embed, false) = true
          AND s.embedding IS NULL
        RETURN s.uid AS Section_ID,
               s.doc_id AS Document,
               s.embedding_status AS Embedding_Status
        ORDER BY Section_ID
        LIMIT 20
    """,

    "15. Sections Not Yet Entity-Processed": """
        MATCH (s:Section)
        WHERE coalesce(s.entity_extracted, false) = false
        RETURN s.uid AS Section_ID,
               s.doc_id AS Document,
               s.entity_extraction_status AS Entity_Status
        ORDER BY Section_ID
        LIMIT 20
    """,

    "16. Sections Mentioning the Most Concepts": """
        MATCH (s:Section)-[:MENTIONS]->(c:Concept)
        RETURN s.uid AS Section_ID,
               s.title AS Title,
               count(c) AS Concept_Count
        ORDER BY Concept_Count DESC, Section_ID ASC
        LIMIT 10
    """,

    "17. Concepts Still Needing Review": """
        MATCH (c:Concept)
        WHERE c.needs_type_review = true
        RETURN c.name AS Concept,
               c.observed_types AS Observed_Types,
               c.type_support_pairs AS Type_Support,
               c.canonical_type AS Canonical_Type
        ORDER BY Concept ASC
        LIMIT 20
    """,

    "18. Documents with Section Counts by Processing State": """
        MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
        RETURN d.doc_id AS Document,
               count(s) AS Total_Sections,
               count(CASE WHEN coalesce(s.has_embedding, false) = true THEN 1 END) AS Embedded_Sections,
               count(CASE WHEN coalesce(s.entity_extracted, false) = true THEN 1 END) AS Entity_Processed_Sections
        ORDER BY Document
    """
}


def run_queries():
    """
    Connects to the Neo4j database using the project's utility driver,
    executes all defined queries, and prints the results cleanly.
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