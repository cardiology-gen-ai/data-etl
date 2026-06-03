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
            c.canonical_type AS Canonical_Type,
            c.type_resolution_status AS Resolution_Status
        ORDER BY Concept ASC
        LIMIT 100
    """,

    "3. Entity Distribution by Canonical Type": """
        MATCH (c:Concept)
        RETURN
            coalesce(c.canonical_type, 'UNRESOLVED') AS Type,
            count(*) AS Count
        ORDER BY Count DESC, Type ASC
    """,

    "4. Sections with Most Accepted Concepts": """
        MATCH (s:Section)-[:MENTIONS]->(c:Concept)
        RETURN
            s.uid AS Section,
            s.title AS Title,
            count(DISTINCT c) AS Entity_Count
        ORDER BY Entity_Count DESC, Section ASC
        LIMIT 10
    """,

    "5. Ambiguous Concepts with Tied Type Support": """
        MATCH (c:Concept)
        WHERE c.type_resolution_status = 'ambiguous_tied_section_support'
        RETURN
            c.name AS Concept,
            c.observed_types AS Types,
            c.type_support_pairs AS Type_Support,
            c.canonical_type AS Canonical_Type
        ORDER BY Concept ASC
        LIMIT 100
    """,

    "6. Cross-Document Concepts": """
        MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
        WITH c, collect(DISTINCT s.doc_id) AS docs
        WHERE size(docs) > 1
        RETURN
            c.name AS Shared_Concept,
            c.canonical_type AS Type,
            docs AS Found_In_Docs
        ORDER BY Shared_Concept ASC
        LIMIT 100
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
        WITH s, properties(s) AS section_props
        WHERE section_props['entity_extraction_status'] = 'failed'
        RETURN
            s.uid AS Section,
            s.title AS Title,
            size(coalesce(s.text, '')) AS Text_Length,
            section_props['entity_extraction_failed_at'] AS Failed_At
        ORDER BY Text_Length DESC, Section ASC
        LIMIT 20
    """,

    "9. Largest Sections Overall": """
        MATCH (s:Section)
        RETURN
            s.uid AS Section,
            s.title AS Title,
            size(coalesce(s.text, '')) AS Text_Length,
            coalesce(s.entity_extraction_status, 'UNSET') AS Entity_Status,
            coalesce(s.embedding_status, 'UNSET') AS Embedding_Status
        ORDER BY Text_Length DESC, Section ASC
        LIMIT 20
    """,

    "10. Entity Extraction Status vs Text Length": """
        MATCH (s:Section)
        RETURN
            coalesce(s.entity_extraction_status, 'UNSET') AS Status,
            count(*) AS Sections,
            round(avg(size(coalesce(s.text, ''))), 2) AS Avg_Text_Length,
            max(size(coalesce(s.text, ''))) AS Max_Text_Length
        ORDER BY Sections DESC, Status ASC
    """,

    "11. Successfully Processed Sections with Zero Accepted Concepts": """
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

    "12. Sections Still Unset for Entity Extraction": """
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

    "13. Sections with Body Text Longer Than 12000 Characters": """
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

    "14. Duplicate MENTIONS Edges Per Section-Concept Pair": """
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

    "15. Orphan Concepts with No Supporting Sections": """
        MATCH (c:Concept)
        WHERE NOT (:Section)-[:MENTIONS]->(c)
        RETURN
            c.name AS Concept,
            c.canonical_type AS Type,
            c.type_resolution_status AS Resolution_Status
        ORDER BY Concept ASC
        LIMIT 50
    """,

    "16. Concepts Missing Canonical or Observed Type Info": """
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

    "17. Potentially Overused Concepts with Document Spread": """
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

    "18. Sections with Suspiciously High Entity Density": """
        MATCH (s:Section)-[:MENTIONS]->(c:Concept)
        WITH
            s,
            count(DISTINCT c) AS entity_count,
            size(coalesce(s.text, '')) AS text_len
        WHERE text_len >= 500
        RETURN
            s.uid AS Section,
            s.title AS Title,
            text_len AS Text_Length,
            entity_count AS Entity_Count,
            round(1000.0 * entity_count / text_len, 3) AS Entities_Per_1000_Chars
        ORDER BY Entities_Per_1000_Chars DESC, Entity_Count DESC
        LIMIT 30
    """,

    "19. Failed Sections That Also Have Children": """
        MATCH (s:Section)
        WHERE s.entity_extraction_status = 'failed'
          AND EXISTS { MATCH (s)-[:HAS_CHILD]->(:Section) }
        RETURN
            s.uid AS Section,
            s.title AS Title,
            size(coalesce(s.text, '')) AS Text_Length
        ORDER BY Text_Length DESC, Section ASC
        LIMIT 30
    """,

    "20. Mention Support Method Summary": """
        MATCH (:Section)-[r:MENTIONS]->(:Concept)
        WITH properties(r) AS mention_props
        RETURN
            coalesce(mention_props['support_method'], 'UNSET') AS Support_Method,
            count(*) AS Count
        ORDER BY Count DESC, Support_Method ASC
    """,

    "21. Acronym-Supported Mentions": """
        MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
        WITH s, r, c, properties(r) AS mention_props
        WHERE mention_props['support_method'] = 'acronym'
        RETURN
            s.uid AS Section,
            s.title AS Title,
            c.name AS Concept,
            c.canonical_type AS Type,
            mention_props['acronym_short'] AS Acronym,
            mention_props['acronym_definition'] AS Acronym_Definition,
            mention_props['acronym_match_method'] AS Acronym_Match_Method,
            mention_props['expanded_from_acronym'] AS Expanded_From_Acronym,
            mention_props['raw_name'] AS Raw_Name
        ORDER BY Section ASC, Concept ASC
        LIMIT 50
    """,

    "22. Expanded Acronym Concepts": """
        MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
        WITH s, r, c, properties(r) AS mention_props
        WHERE coalesce(mention_props['expanded_from_acronym'], false) = true
        RETURN
            s.uid AS Section,
            s.title AS Title,
            mention_props['raw_name'] AS Raw_LLM_Name,
            c.name AS Written_Concept,
            c.canonical_type AS Type,
            mention_props['acronym_short'] AS Acronym,
            mention_props['acronym_definition'] AS Acronym_Definition,
            mention_props['acronym_match_method'] AS Acronym_Match_Method
        ORDER BY Section ASC, Written_Concept ASC
        LIMIT 50
    """,

    "23. Acronym-Supported Mentions Missing Metadata": """
        MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
        WITH s, r, c, properties(r) AS mention_props
        WHERE mention_props['support_method'] = 'acronym'
          AND (
                mention_props['acronym_short'] IS NULL
             OR mention_props['acronym_definition'] IS NULL
             OR mention_props['acronym_match_method'] IS NULL
          )
        RETURN
            s.uid AS Section,
            c.name AS Concept,
            mention_props['acronym_short'] AS Acronym,
            mention_props['acronym_definition'] AS Acronym_Definition,
            mention_props['acronym_match_method'] AS Acronym_Match_Method
        ORDER BY Section ASC, Concept ASC
        LIMIT 50
    """,

    "24. Mentions Missing Validation Metadata": """
        MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
        WITH s, r, c, properties(r) AS mention_props
        WHERE mention_props['support_method'] IS NULL
           OR mention_props['validation_reason'] IS NULL
        RETURN
            s.uid AS Section,
            c.name AS Concept,
            c.canonical_type AS Type,
            mention_props['support_method'] AS Support_Method,
            mention_props['validation_reason'] AS Validation_Reason
        ORDER BY Section ASC, Concept ASC
        LIMIT 50
    """,

    "25. Raw Acronym-Like Mentions Written Without Expansion": """
        MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
        WITH s, r, c, properties(r) AS mention_props
        WHERE mention_props['raw_name'] IS NOT NULL
          AND mention_props['raw_name'] =~ '^[A-Z0-9][A-Z0-9./+-]{1,}$'
          AND coalesce(mention_props['expanded_from_acronym'], false) = false
          AND c.name = toLower(mention_props['raw_name'])
        RETURN
            s.uid AS Section,
            c.name AS Concept,
            mention_props['raw_name'] AS Raw_Name,
            c.canonical_type AS Type,
            mention_props['support_method'] AS Support_Method,
            mention_props['validation_reason'] AS Validation_Reason
        ORDER BY Section ASC, Concept ASC
        LIMIT 50
    """,

    "26. UMLS Normalization Status Summary": """
        MATCH (c:Concept)
        WITH c, properties(c) AS concept_props
        WHERE concept_props['normalization_status'] IS NOT NULL
           OR concept_props['umls_cui'] IS NOT NULL
        RETURN
            coalesce(concept_props['normalization_status'], 'UNSET') AS Normalization_Status,
            count(*) AS Count
        ORDER BY Count DESC, Normalization_Status ASC
    """,

    "27. Top UMLS-Normalized Concepts": """
        MATCH (c:Concept)
        WITH c, properties(c) AS concept_props
        WHERE concept_props['umls_cui'] IS NOT NULL
        RETURN
            c.name AS Concept,
            c.canonical_type AS Type,
            concept_props['umls_cui'] AS CUI,
            concept_props['umls_canonical_name'] AS UMLS_Name,
            concept_props['umls_score'] AS Score,
            concept_props['normalization_method'] AS Method
        ORDER BY Score DESC, Concept ASC
        LIMIT 50
    """,

    "28. Low Confidence or Unmatched Concepts": """
        MATCH (c:Concept)
        WITH c, properties(c) AS concept_props
        WHERE concept_props['normalization_status'] IN ['umls_low_confidence', 'umls_no_match', 'failed']
        RETURN
            c.name AS Concept,
            c.canonical_type AS Type,
            concept_props['normalization_status'] AS Status,
            concept_props['umls_cui'] AS CUI,
            concept_props['umls_canonical_name'] AS UMLS_Name,
            concept_props['umls_score'] AS Score
        ORDER BY Status ASC, Concept ASC
        LIMIT 50
    """,

    "29. SAME_AS Duplicate Evidence": """
        MATCH (a:Concept)-[r]->(b:Concept)
        WITH a, r, b, properties(a) AS source_props, properties(r) AS rel_props, properties(b) AS target_props
        WHERE type(r) = 'SAME_AS'
        RETURN
            a.name AS Source,
            b.name AS Target,
            source_props['umls_cui'] AS Source_CUI,
            target_props['umls_cui'] AS Target_CUI,
            rel_props['method'] AS Method,
            rel_props['score'] AS Score,
            rel_props['status'] AS Status
        ORDER BY Source ASC, Target ASC
        LIMIT 50
    """,

    "30. POSSIBLY_SAME_AS Candidate Duplicates": """
        MATCH (a:Concept)-[r]->(b:Concept)
        WITH a, r, b, properties(r) AS rel_props
        WHERE type(r) = 'POSSIBLY_SAME_AS'
        RETURN
            a.name AS Source,
            b.name AS Target,
            a.canonical_type AS Source_Type,
            b.canonical_type AS Target_Type,
            rel_props['method'] AS Method,
            rel_props['score'] AS Score,
            rel_props['status'] AS Status
        ORDER BY Score DESC, Source ASC, Target ASC
        LIMIT 50
    """,

    "31. UMLS Metadata Problems": """
        MATCH (c:Concept)
        WITH c, properties(c) AS concept_props
        WHERE concept_props['normalization_status'] = 'umls_matched'
          AND (
                concept_props['umls_cui'] IS NULL
             OR concept_props['umls_canonical_name'] IS NULL
             OR concept_props['umls_score'] IS NULL
             OR concept_props['normalized_name'] IS NULL
             OR concept_props['normalized_at'] IS NULL
          )
        RETURN
            c.name AS Concept,
            c.canonical_type AS Type,
            concept_props['umls_cui'] AS CUI,
            concept_props['umls_canonical_name'] AS UMLS_Name,
            concept_props['umls_score'] AS Score,
            concept_props['normalization_status'] AS Status
        ORDER BY Concept ASC
        LIMIT 50
    """,
}


def run_viz() -> None:
    print("[INFO] Initializing connection for final KG diagnostic queries...")
    driver = get_neo4j_driver(verify=True)

    try:
        with driver.session() as session:
            for title, cypher in QUERIES.items():
                print(f"\n---> {title}")

                try:
                    result = session.run(cypher)
                    records = list(result)

                    if not records:
                        print("     No data found.")
                    else:
                        for record in records:
                            print(f"     {dict(record)}")

                except Exception as exc:
                    print(f"     [ERROR] Query failed: {exc}")

                print("-" * 60)

    finally:
        close_driver(driver)
        print("\n[INFO] Final KG diagnostic queries finished.")


if __name__ == "__main__":
    run_viz()
