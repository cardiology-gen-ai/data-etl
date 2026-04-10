import os
from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver

# ---------------------------------------------------------
# Dictionary containing our 10 Cypher exploration queries
# Updated to match graph_loader.py schema logic
# ---------------------------------------------------------
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
        RETURN s.uid AS Section_ID, size(s.text) AS Calculated_Length
        ORDER BY Calculated_Length DESC
        LIMIT 5
    """,
    "8. Text Preview of a Specific Section": """
        MATCH (s:Section {uid: 'Cardiomyopathies_2023::4'})
        RETURN substring(s.text, 0, 200) + '...' AS Content_Preview
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
    """
}

def run_queries():
    """
    Connects to the Neo4j database using the project's utility driver, 
    executes all defined queries, and prints the results cleanly.
    """
    print("[INFO] Initializing connection via neo4j_utils...")
    
    # Get driver using your utility function
    driver = get_neo4j_driver(verify=True)
    
    print("=" * 60)
    
    try:
        with driver.session() as session:
            for title, cypher_query in QUERIES.items():
                print(f"\n---> {title}")
                
                # Execute the query
                result = session.run(cypher_query)
                records = list(result)
                
                # Print results cleanly
                if not records:
                    print("     No results found.")
                else:
                    for record in records:
                        print(f"     {dict(record)}")
                
                print("-" * 60)
    finally:
        # Safely close the driver using your utility function
        close_driver(driver)
        print("[INFO] Queries execution finished.")

if __name__ == "__main__":
    run_queries()