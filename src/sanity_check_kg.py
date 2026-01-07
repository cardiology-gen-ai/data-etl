"""
sanity_check_kg.py

Runs structural and semantic sanity checks on a multi-document
Neo4j knowledge graph.

- Prints WARNINGS and ERRORS
- Does NOT modify the graph
"""

import os
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("kg_sanity")



# Neo4j connection
load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)



# Helper
def run_check(tx, title: str, query: str, level: str = "WARNING"):
    result = list(tx.run(query))
    if result:
        log_fn = logger.error if level == "ERROR" else logger.warning
        log_fn("%s (%d issues)", title, len(result))
        for row in result[:10]:
            log_fn("  → %s", dict(row))
    else:
        logger.info("%s: OK", title)

# Checks
CHECKS = [

    # DOCUMENT STRUCTURE

    ("Documents without sections",
     """
     MATCH (d:Document)
     WHERE NOT (d)-[:HAS_SECTION]->(:Section)
     RETURN d.doc_id
     """,
     "ERROR"),

    ("Sections linked to multiple documents",
     """
     MATCH (d1:Document)-[:HAS_SECTION]->(s:Section)<-[:HAS_SECTION]-(d2:Document)
     WHERE d1 <> d2
     RETURN s.uid, d1.doc_id, d2.doc_id
     """,
     "ERROR"),

    ("Orphan sections (no document)",
     """
     MATCH (s:Section)
     WHERE NOT (:Document)-[:HAS_SECTION]->(s)
     RETURN s.uid
     """,
     "ERROR"),

    # SECTION IDENTITY
    ("Duplicate section UID values",
     """
     MATCH (s:Section)
     WITH s.uid AS uid, count(*) AS n
     WHERE n > 1
     RETURN uid, n
     """,
     "ERROR"),

    ("UID / doc_id mismatch",
     """
     MATCH (s:Section)
     WHERE NOT s.uid STARTS WITH s.doc_id + "::"
     RETURN s.uid, s.doc_id
     """,
     "ERROR"),

    # HIERARCHY
    ("Sections with multiple parents",
     """
     MATCH (p:Section)-[:HAS_CHILD]->(c:Section)
     WITH c, count(p) AS parents
     WHERE parents > 1
     RETURN c.uid, parents
     """,
     "ERROR"),

    ("Cycles in HAS_CHILD",
     """
     MATCH p=(s:Section)-[:HAS_CHILD*]->(s)
     RETURN s.uid
     """,
     "ERROR"),

    ("NEXT edges crossing documents",
     """
     MATCH (a:Section)-[:NEXT]->(b:Section)
     WHERE a.doc_id <> b.doc_id
     RETURN a.uid, b.uid
     """,
     "ERROR"),

    # SECTION CONTENT
    ("Empty leaf sections",
     """
     MATCH (s:Section)
     WHERE s.is_empty = true
       AND NOT (s)-[:HAS_CHILD]->(:Section)
     RETURN s.uid, s.title
     """,
     "WARNING"),

    ("Non-empty parent sections",
     """
     MATCH (s:Section)-[:HAS_CHILD]->(:Section)
     WHERE s.is_empty = false AND size(s.text) > 100
     RETURN s.uid, size(s.text) AS text_len
     """,
     "WARNING"),

    # CONCEPTS
    ("Orphan concepts",
     """
     MATCH (c:Concept)
     WHERE NOT (:Section)-[:MENTIONS]->(c)
     RETURN c.name
     """,
     "WARNING"),

    ("Concepts without type",
     """
     MATCH (c:Concept)
     WHERE c.type IS NULL
     RETURN c.name
     """,
     "ERROR"),

    ("Concepts used in only one document",
     """
     MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
     WITH c, collect(DISTINCT s.doc_id) AS docs
     WHERE size(docs) = 1
     RETURN c.name, docs
     """,
     "INFO"),

    ("Highly overused concepts",
     """
     MATCH (s:Section)-[:MENTIONS]->(c:Concept)
     WITH c, count(s) AS n
     WHERE n > 30
     RETURN c.name, c.type, n
     ORDER BY n DESC
     """,
     "WARNING"),
]


def main():
    with driver.session() as session:
        for title, query, level in CHECKS:
            session.execute_read(run_check, title, query, level)

    logger.info("Sanity check completed")


if __name__ == "__main__":
    main()
