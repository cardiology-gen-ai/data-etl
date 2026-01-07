"""
build_graph.py

Builds a hierarchical knowledge graph in Neo4j from chunked guideline data.

Nodes:
  (:Document {doc_id})
  (:Section {section_id, title, level, text, is_empty, embed, page_start, page_end})

Relationships:
  (:Document)-[:HAS_SECTION]->(:Section)
  (:Section)-[:HAS_CHILD]->(:Section)
  (:Section)-[:NEXT]->(:Section)
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List

from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("build_graph")



NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise RuntimeError("Missing Neo4j credentials in environment variables")

logger.info("Connecting to Neo4j at %s", NEO4J_URI)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)

driver.verify_connectivity()
logger.info("Neo4j connectivity verified")


# Schema setup TODO: see if this schem is fine or if we need something more complex

def setup_schema(tx):
    """Create constraints for graph construction."""
    tx.run("""
        CREATE CONSTRAINT document_id IF NOT EXISTS
        FOR (d:Document)
        REQUIRE d.doc_id IS UNIQUE
    """)

    tx.run("""
        CREATE CONSTRAINT section_id IF NOT EXISTS
        FOR (s:Section)
        REQUIRE s.section_id IS UNIQUE
    """)



# Graph building logic
def create_document(tx, doc_id: str):
    tx.run(
        "MERGE (d:Document {doc_id: $doc_id})",
        doc_id=doc_id,
    )


def create_section(tx, section: Dict):
    tx.run(
        """
        MERGE (s:Section {section_id: $section_id})
        SET
            s.title = $title,
            s.level = $level,
            s.text = $text,
            s.is_empty = $is_empty,
            s.embed = $embed,
            s.page_start = $page_start,
            s.page_end = $page_end
        """,
        section_id=section["section_id"],
        title=section.get("section_title"),
        level=section.get("section_level"),
        text=section.get("text"),
        is_empty=section.get("is_empty"),
        embed=section.get("embed"),
        page_start=section.get("page_start"),
        page_end=section.get("page_end"),
    )


def link_document_section(tx, doc_id: str, section_id: str):
    tx.run(
        """
        MATCH (d:Document {doc_id: $doc_id})
        MATCH (s:Section {section_id: $section_id})
        MERGE (d)-[:HAS_SECTION]->(s)
        """,
        doc_id=doc_id,
        section_id=section_id,
    )


def link_parent_child(tx, parent_id: str, child_id: str):
    tx.run(
        """
        MATCH (p:Section {section_id: $parent_id})
        MATCH (c:Section {section_id: $child_id})
        MERGE (p)-[:HAS_CHILD]->(c)
        """,
        parent_id=parent_id,
        child_id=child_id,
    )


def link_next(tx, prev_id: str, next_id: str):
    tx.run(
        """
        MATCH (a:Section {section_id: $prev_id})
        MATCH (b:Section {section_id: $next_id})
        MERGE (a)-[:NEXT]->(b)
        """,
        prev_id=prev_id,
        next_id=next_id,
    )



# Main build function

def build_graph(chunks_path: str):
    chunks_path = Path(chunks_path)

    logger.info("Loading chunks from %s", chunks_path)

    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)

    chunks: List[Dict] = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("Chunk file is empty")

    logger.info("Loaded %d chunks", len(chunks))

    doc_id = chunks[0]["doc_id"]
    logger.info("Document ID: %s", doc_id)

    with driver.session() as session:
        logger.info("Setting up schema")
        session.execute_write(setup_schema)

        logger.info("Creating Document node")
        session.execute_write(create_document, doc_id)

        logger.info("Creating Section nodes")
        for chunk in chunks:
            session.execute_write(create_section, chunk)
            session.execute_write(
                link_document_section,
                doc_id,
                chunk["section_id"],
            )

        logger.info("Created %d Section nodes", len(chunks))

        logger.info("Creating hierarchy relationships")
        parent_links = 0
        next_links = 0

        for i, chunk in enumerate(chunks):
            parent_id = chunk.get("parent_section_id")
            if parent_id:
                session.execute_write(
                    link_parent_child,
                    parent_id,
                    chunk["section_id"],
                )
                parent_links += 1

            if i > 0:
                session.execute_write(
                    link_next,
                    chunks[i - 1]["section_id"],
                    chunk["section_id"],
                )
                next_links += 1

        logger.info("Created %d HAS_CHILD relationships", parent_links)
        logger.info("Created %d NEXT relationships", next_links)

    logger.info("Graph successfully built for document: %s", doc_id)


if __name__ == "__main__":
    build_graph("../test_data/chunks/Cardiomyopathies_2023_hier_chunks.json")
