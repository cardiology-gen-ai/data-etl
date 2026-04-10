import os
from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver

QUERIES = {
    "1. Top 10 Most Cited Concepts": """
        MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
        RETURN c.name AS Concept, count(s) AS Citations, c.canonical_type AS Type
        ORDER BY Citations DESC LIMIT 10
    """,
    
    "2. Ambiguous Concepts (Multi-type)": """
        MATCH (c:Concept)
        WHERE size(c.observed_types) > 1
        RETURN c.name AS Concept, c.observed_types AS Types, c.canonical_type AS Chosen_Canonical
    """,
    
    "3. Entities Distribution by Type": """
        MATCH (c:Concept)
        RETURN c.canonical_type AS Type, count(*) AS Count
        ORDER BY Count DESC
    """,
    
    "4. Sections with Most Entities (Density Check)": """
        MATCH (s:Section)-[:MENTIONS]->(c:Concept)
        RETURN s.uid AS Section, s.title AS Title, count(c) AS Entity_Count
        ORDER BY Entity_Count DESC LIMIT 5
    """,
    
    "5. Isolated Concepts (Mentions Check)": """
        MATCH (c:Concept)
        WHERE NOT (:Section)-[:MENTIONS]->(c)
        RETURN c.name AS Orphan_Concept
    """,

    "6. Cross-Document Concepts (Shared Knowledge)": """
        MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
        WITH c, collect(DISTINCT s.doc_id) AS docs
        WHERE size(docs) > 1
        RETURN c.name AS Shared_Concept, docs AS Found_In_Docs
    """
}

def run_viz():
    print("[INFO] Initializing connection for Entity Visualization...")
    driver = get_neo4j_driver(verify=True)
    
    try:
        with driver.session() as session:
            for title, cypher in QUERIES.items():
                print(f"\n---> {title}")
                result = session.run(cypher)
                records = list(result)
                if not records:
                    print("     No data found.")
                else:
                    for record in records:
                        print(f"     {dict(record)}")
                print("-" * 60)
    finally:
        close_driver(driver)
        print("\n[INFO] Visualization queries finished.")

if __name__ == "__main__":
    run_viz()