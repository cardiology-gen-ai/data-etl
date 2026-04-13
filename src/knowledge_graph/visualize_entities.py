from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver


QUERIES = {
    "1. Top 10 Most Supported Concepts": """
        MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
        RETURN
            c.name AS Concept,
            count(s) AS Supporting_Sections,
            c.canonical_type AS Type
        ORDER BY Supporting_Sections DESC, Concept ASC
        LIMIT 10
    """,

    "2. Concepts Still Needing Type Review": """
        MATCH (c:Concept)
        WHERE c.needs_type_review = true
        RETURN
            c.name AS Concept,
            c.observed_types AS Types,
            c.type_support_pairs AS Type_Support,
            c.canonical_type AS Chosen_Canonical,
            c.type_resolution_status AS Resolution_Status
        ORDER BY Concept ASC
    """,

    "3. Entities Distribution by Canonical Type": """
        MATCH (c:Concept)
        RETURN
            coalesce(c.canonical_type, 'UNRESOLVED') AS Type,
            count(*) AS Count
        ORDER BY Count DESC, Type ASC
    """,

    "4. Sections with Most Entities (Density Check)": """
        MATCH (s:Section)-[:MENTIONS]->(c:Concept)
        RETURN
            s.uid AS Section,
            s.title AS Title,
            count(c) AS Entity_Count
        ORDER BY Entity_Count DESC, Section ASC
        LIMIT 5
    """,

    "5. Ambiguous Concepts with Tied Support": """
    MATCH (c:Concept)
    WHERE c.type_resolution_status = 'ambiguous_tied_section_support'
    RETURN
        c.name AS Concept,
        c.observed_types AS Types,
        c.type_support_pairs AS Type_Support,
        c.canonical_type AS Chosen_Canonical
    ORDER BY Concept ASC
""",

    "6. Cross-Document Concepts (Shared Knowledge)": """
        MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
        WITH c, collect(DISTINCT s.doc_id) AS docs
        WHERE size(docs) > 1
        RETURN
            c.name AS Shared_Concept,
            docs AS Found_In_Docs
        ORDER BY Shared_Concept ASC
    """,

    "7. Type Resolution Status Summary": """
        MATCH (c:Concept)
        RETURN
            coalesce(c.type_resolution_status, 'UNSET') AS Resolution_Status,
            count(*) AS Count
        ORDER BY Count DESC, Resolution_Status ASC
    """,
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