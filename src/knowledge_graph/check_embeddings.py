import logging
from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_embeddings():
    print("[INFO] Checking embeddings in Neo4j database...")
    driver = get_neo4j_driver()

    try:
        with driver.session() as session:
            stats = session.run("""
                MATCH (s:Section)
                RETURN
                    count(s) AS total_sections,
                    count(CASE WHEN coalesce(s.embed, false) = true THEN 1 END) AS eligible_sections,
                    count(CASE WHEN s.embedding IS NOT NULL THEN 1 END) AS with_embedding,
                    count(CASE WHEN coalesce(s.has_embedding, false) = true THEN 1 END) AS has_embedding_flag
            """).single()

            total = stats["total_sections"]
            eligible = stats["eligible_sections"]
            with_emb = stats["with_embedding"]
            has_flag = stats["has_embedding_flag"]

            percent_total = (with_emb / total * 100) if total > 0 else 0
            percent_eligible = (with_emb / eligible * 100) if eligible > 0 else 0

            print("\n--- 1. General Statistics ---")
            print(f"Total sections in graph:          {total}")
            print(f"Eligible for embedding:           {eligible}")
            print(f"Sections with embedding:          {with_emb}")
            print(f"Sections with has_embedding=true: {has_flag}")
            print(f"Coverage over all sections:       {percent_total:.2f}%")
            print(f"Coverage over eligible sections:  {percent_eligible:.2f}%")

            status_rows = session.run("""
                MATCH (s:Section)
                RETURN
                    coalesce(s.embedding_status, 'UNSET') AS status,
                    count(s) AS count
                ORDER BY count DESC, status ASC
            """)

            print("\n--- 2. Embedding Status Summary ---")
            found_status = False
            for rec in status_rows:
                found_status = True
                print(f"{rec['status']}: {rec['count']}")
            if not found_status:
                print("No data found.")

            inconsistent_rows = session.run("""
                MATCH (s:Section)
                WHERE
                    (coalesce(s.has_embedding, false) = true AND s.embedding IS NULL)
                    OR
                    (s.embedding IS NOT NULL AND coalesce(s.has_embedding, false) = false)
                    OR
                    (s.embedding IS NOT NULL AND s.embedding_dim IS NOT NULL AND size(s.embedding) <> s.embedding_dim)
                    OR
                    (s.embedding IS NOT NULL AND s.embedding_model IS NULL)
                RETURN
                    s.uid AS uid,
                    s.doc_id AS doc_id,
                    s.section_id AS section_id,
                    s.has_embedding AS has_embedding,
                    s.embedding IS NOT NULL AS has_vector,
                    s.embedding_dim AS dim_property,
                    CASE WHEN s.embedding IS NOT NULL THEN size(s.embedding) ELSE null END AS actual_dim,
                    s.embedding_model AS embedding_model,
                    s.embedding_status AS embedding_status
                ORDER BY uid
                LIMIT 20
            """)

            print("\n--- 3. Inconsistencies (up to 20) ---")
            inconsistent_found = False
            for rec in inconsistent_rows:
                inconsistent_found = True
                print(dict(rec))
            if not inconsistent_found:
                print("[OK] No embedding inconsistencies found.")

            missing_eligible_rows = session.run("""
                MATCH (s:Section)
                WHERE coalesce(s.embed, false) = true
                  AND s.embedding IS NULL
                RETURN
                    s.uid AS uid,
                    s.doc_id AS doc_id,
                    s.section_id AS section_id,
                    s.embedding_status AS embedding_status
                ORDER BY uid
                LIMIT 20
            """)

            print("\n--- 4. Eligible Sections Still Missing Embeddings (up to 20) ---")
            missing_found = False
            for rec in missing_eligible_rows:
                missing_found = True
                print(dict(rec))
            if not missing_found:
                print("[OK] No eligible sections are currently missing embeddings.")

            if with_emb > 0:
                meta_res = session.run("""
                    MATCH (s:Section)
                    WHERE s.embedding IS NOT NULL
                    RETURN
                        s.embedding_model AS model,
                        s.embedding_dim AS dim_property,
                        size(s.embedding) AS actual_dim,
                        s.has_embedding AS has_flag,
                        s.embedding_updated_at IS NOT NULL AS has_timestamp,
                        s.embedding_status AS embedding_status
                    LIMIT 1
                """).single()

                print("\n--- 5. Metadata Verification (Quality Check) ---")
                print(f"Model used:                {meta_res['model']}")
                print(f"Flag 'has_embedding':      {meta_res['has_flag']}")
                print(f"Timestamp present:         {meta_res['has_timestamp']}")
                print(f"Embedding status:          {meta_res['embedding_status']}")
                print(f"Saved dimension:           {meta_res['dim_property']}")
                print(f"Actual array dimension:    {meta_res['actual_dim']}")

                if meta_res["dim_property"] != meta_res["actual_dim"]:
                    print("\n[WARNING] The embedding_dim property does not match the actual vector length.")
                else:
                    print("[OK] Vector dimension is consistent.")

                sample = session.run("""
                    MATCH (s:Section)
                    WHERE s.embedding IS NOT NULL
                    RETURN
                        s.uid AS uid,
                        s.embedding[0..3] AS vector_head
                    LIMIT 3
                """)

                print("\n--- 6. Sample Vectors (first 3 components) ---")
                for rec in sample:
                    print(f"UID: {rec['uid']} -> {rec['vector_head']}")
            else:
                print("\n[WARNING] No embeddings found in the database!")

            per_doc = session.run("""
                MATCH (s:Section)
                WITH s.doc_id AS doc_id,
                     count(s) AS total_sections,
                     count(CASE WHEN coalesce(s.embed, false) = true THEN 1 END) AS eligible_sections,
                     count(CASE WHEN s.embedding IS NOT NULL THEN 1 END) AS embedded_sections
                RETURN
                    doc_id,
                    total_sections,
                    eligible_sections,
                    embedded_sections
                ORDER BY doc_id
            """)

            print("\n--- 7. Per-Document Embedding Coverage ---")
            per_doc_found = False
            for rec in per_doc:
                per_doc_found = True
                print(dict(rec))
            if not per_doc_found:
                print("No data found.")

    except Exception as e:
        logger.error(f"Error during check: {e}")
    finally:
        close_driver(driver)


if __name__ == "__main__":
    check_embeddings()