"""
build_graph.py

Builds a hierarchical knowledge graph in Neo4j from chunked guideline data.

Nodes:
  (:Document {doc_id})
  (:Section {
      uid,            # globally unique: doc_id::section_id
      doc_id,
      section_id,
      title,
      level,
      text,
      is_empty,
      embed,
      page_start,
      page_end
  })

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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("build_graph")


# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise RuntimeError("Missing Neo4j credentials")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)

driver.verify_connectivity()
logger.info("Connected to Neo4j")

# Schema setup
def setup_schema(tx):
    tx.run("""
        CREATE CONSTRAINT document_id IF NOT EXISTS
        FOR (d:Document)
        REQUIRE d.doc_id IS UNIQUE
    """)

    tx.run("""
        CREATE CONSTRAINT section_uid IF NOT EXISTS
        FOR (s:Section)
        REQUIRE s.uid IS UNIQUE
    """)



# Create nodes
def create_document(tx, doc_id: str):
    tx.run(
        "MERGE (d:Document {doc_id: $doc_id})",
        doc_id=doc_id,
    )


def create_section(tx, section: Dict):
    uid = f"{section['doc_id']}::{section['section_id']}"

    tx.run(
        """
        MERGE (s:Section {uid: $uid})
        SET
            s.doc_id = $doc_id,
            s.section_id = $section_id,
            s.title = $title,
            s.level = $level,
            s.text = $text,
            s.is_empty = $is_empty,
            s.embed = $embed,
            s.page_start = $page_start,
            s.page_end = $page_end
        """,
        uid=uid,
        doc_id=section["doc_id"],
        section_id=section["section_id"],
        title=section.get("section_title"),
        level=section.get("section_level"),
        text=section.get("text"),
        is_empty=section.get("is_empty"),
        embed=section.get("embed"),
        page_start=section.get("page_start"),
        page_end=section.get("page_end"),
    )

# Relationships
def link_document_section(tx, doc_id: str, section_uid: str):
    tx.run(
        """
        MATCH (d:Document {doc_id: $doc_id})
        MATCH (s:Section {uid: $uid})
        MERGE (d)-[:HAS_SECTION]->(s)
        """,
        doc_id=doc_id,
        uid=section_uid,
    )


def link_parent_child(tx, parent_uid: str, child_uid: str):
    tx.run(
        """
        MATCH (p:Section {uid: $parent})
        MATCH (c:Section {uid: $child})
        MERGE (p)-[:HAS_CHILD]->(c)
        """,
        parent=parent_uid,
        child=child_uid,
    )


def link_next(tx, prev_uid: str, next_uid: str):
    tx.run(
        """
        MATCH (a:Section {uid: $prev})
        MATCH (b:Section {uid: $next})
        MERGE (a)-[:NEXT]->(b)
        """,
        prev=prev_uid,
        next=next_uid,
    )



# Build graph from chunk file

def build_graph_from_chunks(chunk_file: Path):
    logger.info("Loading chunks from %s", chunk_file)

    chunks: List[Dict] = json.loads(chunk_file.read_text(encoding="utf-8"))
    if not chunks:
        logger.warning("Empty chunk file: %s", chunk_file)
        return

    doc_id = chunks[0]["doc_id"]
    logger.info("Building graph for document: %s", doc_id)

    with driver.session() as session:
        session.execute_write(setup_schema)
        session.execute_write(create_document, doc_id)

        # Create sections
        for chunk in chunks:
            session.execute_write(create_section, chunk)
            uid = f"{doc_id}::{chunk['section_id']}"
            session.execute_write(link_document_section, doc_id, uid)

        # Hierarchy + NEXT
        for i, chunk in enumerate(chunks):
            child_uid = f"{doc_id}::{chunk['section_id']}"

            parent_id = chunk.get("parent_section_id")
            if parent_id:
                parent_uid = f"{doc_id}::{parent_id}"
                session.execute_write(
                    link_parent_child,
                    parent_uid,
                    child_uid,
                )

            if i > 0:
                prev_uid = f"{doc_id}::{chunks[i - 1]['section_id']}"
                session.execute_write(
                    link_next,
                    prev_uid,
                    child_uid,
                )

    logger.info("Graph built for document: %s", doc_id)



def main():
    chunks_dir = Path("../test_data/chunks")
    chunk_files = sorted(chunks_dir.glob("*_hier_chunks.json"))

    logger.info("Found %d chunk files", len(chunk_files))

    for chunk_file in chunk_files:
        build_graph_from_chunks(chunk_file)


if __name__ == "__main__":
    main()
