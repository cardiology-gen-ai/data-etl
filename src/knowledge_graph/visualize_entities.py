from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver


QUERIES = {
    "1. Top 10 Most Supported Concepts": """
        MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
        RETURN
            c.name AS Concept,
            count(DISTINCT s) AS Supporting_Sections,
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
            count(DISTINCT c) AS Entity_Count
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

    "8. Failed Entity Extraction Sections by Length": """
        MATCH (s:Section)
        WHERE s.entity_extraction_status = 'failed'
        RETURN
            s.uid AS Section,
            s.title AS Title,
            size(coalesce(s.text, '')) AS Text_Length,
            s.entity_extraction_failed_at AS Failed_At
        ORDER BY Text_Length DESC, Section ASC
        LIMIT 20
    """,

    "9. Largest Sections Overall": """
        MATCH (s:Section)
        RETURN
            s.uid AS Section,
            s.title AS Title,
            size(coalesce(s.text, '')) AS Text_Length,
            coalesce(s.entity_extraction_status, 'UNSET') AS Status
        ORDER BY Text_Length DESC, Section ASC
        LIMIT 20
    """,

    "10. Failed Sections with Structural Children": """
        MATCH (s:Section)
        WHERE s.entity_extraction_status = 'failed'
        RETURN
            s.uid AS Section,
            s.title AS Title,
            size(coalesce(s.text, '')) AS Text_Length,
            EXISTS { (s)-[:HAS_CHILD]->(:Section) } AS Has_Children
        ORDER BY Text_Length DESC, Section ASC
        LIMIT 20
    """,

    "11. Entity Extraction Status vs Text Length": """
        MATCH (s:Section)
        RETURN
            coalesce(s.entity_extraction_status, 'UNSET') AS Status,
            count(*) AS Sections,
            avg(size(coalesce(s.text, ''))) AS Avg_Text_Length,
            max(size(coalesce(s.text, ''))) AS Max_Text_Length
        ORDER BY Sections DESC, Status ASC
    """,

    "12. Success Status but No Extracted Concepts": """
        MATCH (s:Section)
        WHERE s.entity_extraction_status = 'success'
          AND NOT (s)-[:MENTIONS]->(:Concept)
        RETURN
            s.uid AS Section,
            s.title AS Title,
            size(coalesce(s.text, '')) AS Text_Length
        ORDER BY Text_Length DESC, Section ASC
        LIMIT 30
    """,

    "13. Sections Still Unset for Entity Extraction": """
        MATCH (s:Section)
        WHERE s.entity_extraction_status IS NULL
           OR s.entity_extracted IS NULL
        RETURN
            s.uid AS Section,
            s.title AS Title,
            s.entity_extraction_status AS Status,
            s.entity_extracted AS Extracted_Flag,
            size(coalesce(s.text, '')) AS Text_Length
        ORDER BY Section ASC
        LIMIT 50
    """,

    "14. Sections Exceeding Emergency Single-Section Limit": """
        MATCH (s:Section)
        WHERE size(coalesce(s.text, '')) > 12000
        RETURN
            s.uid AS Section,
            s.title AS Title,
            size(coalesce(s.text, '')) AS Text_Length,
            coalesce(s.entity_extraction_status, 'UNSET') AS Status
        ORDER BY Text_Length DESC, Section ASC
        LIMIT 50
    """,

    "15. Duplicate MENTIONS Edges Per Section-Concept Pair": """
        MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
        WITH s, c, count(r) AS rel_count
        WHERE rel_count > 1
        RETURN
            s.uid AS Section,
            c.name AS Concept,
            rel_count AS Duplicate_Edges
        ORDER BY Duplicate_Edges DESC, Section ASC, Concept ASC
        LIMIT 50
    """,

    "16. Orphan Concepts with No Supporting Sections": """
        MATCH (c:Concept)
        WHERE NOT (:Section)-[:MENTIONS]->(c)
        RETURN
            c.name AS Concept,
            c.canonical_type AS Type,
            c.type_resolution_status AS Resolution_Status
        ORDER BY Concept ASC
        LIMIT 50
    """,

    "17. Concepts Missing Canonical or Observed Type Info": """
        MATCH (c:Concept)
        WHERE c.canonical_type IS NULL
           OR c.observed_types IS NULL
           OR size(c.observed_types) = 0
        RETURN
            c.name AS Concept,
            c.canonical_type AS Canonical_Type,
            c.observed_types AS Observed_Types,
            c.type_resolution_status AS Resolution_Status
        ORDER BY Concept ASC
        LIMIT 50
    """,

    "18. Potentially Overused Concepts with Document Spread": """
        MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
        WITH c, count(DISTINCT s) AS section_count, collect(DISTINCT s.doc_id) AS docs
        WHERE section_count > 20
        RETURN
            c.name AS Concept,
            c.canonical_type AS Type,
            section_count AS Supporting_Sections,
            size(docs) AS Document_Count,
            docs AS Documents
        ORDER BY Supporting_Sections DESC, Concept ASC
        LIMIT 30
    """,

    "19. Sections with Suspiciously High Entity Density": """
        MATCH (s:Section)-[:MENTIONS]->(c:Concept)
        WITH
            s,
            count(DISTINCT c) AS entity_count,
            size(coalesce(s.text, '')) AS text_len
        WHERE text_len > 0
        RETURN
            s.uid AS Section,
            s.title AS Title,
            text_len AS Text_Length,
            entity_count AS Entity_Count,
            round(1000.0 * entity_count / text_len, 3) AS Entities_Per_1000_Chars
        ORDER BY Entities_Per_1000_Chars DESC, Entity_Count DESC
        LIMIT 30
    """,

    "20. Failed Sections That Also Have Children": """
        MATCH (s:Section)
        WHERE s.entity_extraction_status = 'failed'
          AND EXISTS { (s)-[:HAS_CHILD]->(:Section) }
        RETURN
            s.uid AS Section,
            s.title AS Title,
            size(coalesce(s.text, '')) AS Text_Length
        ORDER BY Text_Length DESC, Section ASC
        LIMIT 30
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