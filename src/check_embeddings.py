import logging
from knowledge_graph.neo4j_utils import get_neo4j_driver, close_driver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_embeddings():
    print("[INFO] Checking embeddings in Neo4j database...")
    driver = get_neo4j_driver()
    
    try:
        with driver.session() as session:
            # General stats on embedding presence
            stats = session.run("""
                MATCH (s:Section)
                WITH count(s) AS total
                MATCH (s:Section) WHERE s.embedding IS NOT NULL
                RETURN total, count(s) AS with_embedding
            """).single()
            
            total = stats["total"]
            with_emb = stats["with_embedding"]
            percent = (with_emb / total * 100) if total > 0 else 0
            
            print(f"\n--- 1. General Statistics ---")
            print(f"Total sections in graph: {total}")
            print(f"Sections with embedding:    {with_emb}")
            print(f"Loading percentage:  {percent:.2f}%")

            if with_emb > 0:
                meta_res = session.run("""
                    MATCH (s:Section) WHERE s.embedding IS NOT NULL
                    RETURN 
                        s.embedding_model AS model,
                        s.embedding_dim AS dim_property,
                        size(s.embedding) AS actual_dim,
                        s.has_embedding AS has_flag,
                        s.embedding_updated_at IS NOT NULL AS has_timestamp
                    LIMIT 1
                """).single()
                
                print(f"\n--- 2. Metadata Verification (Quality Check) ---")
                print(f"Model used:       {meta_res['model']}")
                print(f"Flag 'has_embedding':     {meta_res['has_flag']}")
                print(f"Timestamp present:       {meta_res['has_timestamp']}")
                print(f"Saved dimension:       {meta_res['dim_property']}")
                print(f"Actual array dimension:   {meta_res['actual_dim']}")
                
                if meta_res['dim_property'] != meta_res['actual_dim']:
                    print("\n[WARNING] The embedding_dim property does not match the actual length of the vector!")
                else:
                    print("[OK] Vector dimension is consistent.")

                # Visual check of a few embeddings
                sample = session.run("""
                    MATCH (s:Section) WHERE s.embedding IS NOT NULL
                    RETURN s.uid AS uid, s.embedding[0..3] AS vector_head
                    LIMIT 3
                """)
                print(f"\n--- 3. Sample Vectors (first 3 components) ---")
                for rec in sample:
                    print(f"UID: {rec['uid']} -> {rec['vector_head']}")
            else:
                print("\n[WARNING] No embeddings found in the database!")

    except Exception as e:
        logger.error(f"Error during check: {e}")
    finally:
        close_driver(driver)

if __name__ == "__main__":
    check_embeddings()