import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Iterable, Any, Optional

from neo4j import Driver


logger = logging.getLogger(__name__)

DEFAULT_MIN_TEXT_CHARS_TO_EMBED = int(os.getenv("MIN_TEXT_CHARS_TO_EMBED", "20"))


def chunked(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """
    Yield successive batches from a list.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def make_section_uid(doc_id: str, section_id: str) -> str:
    """
    Build the globally unique section UID.
    """
    return f"{doc_id}::{section_id}"


def infer_should_embed(
    section: Dict[str, Any],
    min_text_chars_to_embed: int = DEFAULT_MIN_TEXT_CHARS_TO_EMBED,
) -> bool:
    """
    Decide whether a section is eligible for embeddings.

    Rules:
    - If `embed` is explicitly provided, respect it.
    - Else, if `is_empty` is True, do not embed.
    - Else, embed only if there is enough text.
    """
    embed_flag = section.get("embed")
    if embed_flag is not None:
        return bool(embed_flag)

    if section.get("is_empty") is True:
        return False

    text = (section.get("text") or "").strip()
    return len(text) >= min_text_chars_to_embed


def normalize_section_record(
    section: Dict[str, Any],
    min_text_chars_to_embed: int = DEFAULT_MIN_TEXT_CHARS_TO_EMBED,
) -> Dict[str, Any]:
    """
    Normalize one raw chunk/section record into the structure used for Neo4j ingestion.
    """
    doc_id = section["doc_id"]
    section_id = section["section_id"]

    return {
        "uid": make_section_uid(doc_id, section_id),
        "doc_id": doc_id,
        "section_id": section_id,
        "printed_section_id": section.get("printed_section_id"),
        "title": section.get("section_title"),
        "level": section.get("section_level"),
        "text": section.get("text"),
        "is_empty": section.get("is_empty"),
        "embed": infer_should_embed(section, min_text_chars_to_embed),
        "page_start": section.get("page_start"),
        "page_end": section.get("page_end"),
        "parent_section_id": section.get("parent_section_id"),
        "part_index": section.get("part_index"),
        "part_count": section.get("part_count"),
        "quality_flags": section.get("quality_flags") or [],
        "boundary_source": section.get("boundary_source"),
    }


def setup_schema(tx) -> None:
    """
    Create constraints and indexes needed for graph loading.
    """
    tx.run(
        """
        CREATE CONSTRAINT document_id IF NOT EXISTS
        FOR (d:Document)
        REQUIRE d.doc_id IS UNIQUE
        """
    )

    tx.run(
        """
        CREATE CONSTRAINT section_uid IF NOT EXISTS
        FOR (s:Section)
        REQUIRE s.uid IS UNIQUE
        """
    )

    tx.run(
        """
        CREATE INDEX section_doc_id IF NOT EXISTS
        FOR (s:Section)
        ON (s.doc_id)
        """
    )


def create_document(tx, doc_id: str) -> None:
    """
    Create or reuse a Document node.
    """
    tx.run(
        """
        MERGE (d:Document {doc_id: $doc_id})
        """,
        doc_id=doc_id,
    )


def document_sections_exist(tx, doc_id: str) -> bool:
    result = tx.run(
        """
        MATCH (:Document {doc_id: $doc_id})-[:HAS_SECTION]->(s:Section {doc_id: $doc_id})
        RETURN count(s) > 0 AS has_sections
        """,
        doc_id=doc_id,
    )
    record = result.single()
    return bool(record["has_sections"]) if record is not None else False


def delete_existing_document_sections(tx, doc_id: str) -> None:
    """
    Remove all existing sections for one document before reloading them.
    """
    tx.run(
        """
        MATCH (:Document {doc_id: $doc_id})-[:HAS_SECTION]->(s:Section {doc_id: $doc_id})
        DETACH DELETE s
        """,
        doc_id=doc_id,
    )


def delete_orphan_concepts(tx) -> None:
    """
    Remove Concept nodes no longer mentioned by any section.
    Useful after deleting/reloading document sections.
    """
    tx.run(
        """
        MATCH (c:Concept)
        WHERE NOT (:Section)-[:MENTIONS]->(c)
        DETACH DELETE c
        """
    )


def create_sections_batch(tx, sections: List[Dict[str, Any]]) -> None:
    """
    Create Section nodes in batch.
    Embedding/entity-related fields are initialized but not populated yet.
    """
    tx.run(
        """
        UNWIND $sections AS section
        CREATE (s:Section {
            uid: section.uid,
            doc_id: section.doc_id,
            section_id: section.section_id,
            printed_section_id: section.printed_section_id,
            title: section.title,
            level: section.level,
            text: section.text,
            is_empty: section.is_empty,
            embed: section.embed,
            page_start: section.page_start,
            page_end: section.page_end,
            part_index: section.part_index,
            part_count: section.part_count,
            quality_flags: section.quality_flags,
            boundary_source: section.boundary_source,

            has_embedding: false,
            embedding: null,
            embedding_model: null,
            embedding_dim: null,
            embedding_updated_at: null,
            embedding_status: null,
            embedding_failed_at: null,

            entity_extracted: false,
            entity_extracted_at: null,
            entity_extraction_status: null,
            entity_extraction_failed_at: null
        })
        """,
        sections=sections,
    )


def link_document_sections_batch(tx, doc_id: str, section_uids: List[str]) -> None:
    """
    Link one Document node to many Section nodes.
    """
    tx.run(
        """
        MATCH (d:Document {doc_id: $doc_id})
        UNWIND $section_uids AS uid
        MATCH (s:Section {uid: uid})
        MERGE (d)-[:HAS_SECTION]->(s)
        """,
        doc_id=doc_id,
        section_uids=section_uids,
    )


def link_parent_child_batch(tx, pairs: List[Dict[str, str]]) -> None:
    """
    Create HAS_CHILD relationships in batch.
    Each pair is {"parent_uid": ..., "child_uid": ...}
    """
    if not pairs:
        return

    tx.run(
        """
        UNWIND $pairs AS pair
        MATCH (p:Section {uid: pair.parent_uid})
        MATCH (c:Section {uid: pair.child_uid})
        MERGE (p)-[:HAS_CHILD]->(c)
        """,
        pairs=pairs,
    )


def link_next_batch(tx, pairs: List[Dict[str, str]]) -> None:
    """
    Create NEXT relationships in batch.
    Each pair is {"prev_uid": ..., "next_uid": ...}
    """
    if not pairs:
        return

    tx.run(
        """
        UNWIND $pairs AS pair
        MATCH (a:Section {uid: pair.prev_uid})
        MATCH (b:Section {uid: pair.next_uid})
        MERGE (a)-[:NEXT]->(b)
        """,
        pairs=pairs,
    )


def load_chunks_from_file(chunk_file: Path) -> List[Dict[str, Any]]:
    """
    Load and validate chunk records from JSON file.
    """
    chunks = json.loads(chunk_file.read_text(encoding="utf-8"))

    if not isinstance(chunks, list):
        raise ValueError(f"Chunk file must contain a JSON list: {chunk_file}")

    return chunks


def build_graph_from_chunks(
    driver: Driver,
    chunk_file: Path,
    batch_size: int = 200,
    min_text_chars_to_embed: int = DEFAULT_MIN_TEXT_CHARS_TO_EMBED,
    replace_existing_document: bool = True,
) -> Optional[str]:
    """
    Build the structural graph for one document from a chunk JSON file.

    Parameters:
        driver: Neo4j driver
        chunk_file: JSON file containing hierarchical chunks for one document
        batch_size: number of rows/relationships written per batch
        min_text_chars_to_embed: minimum text length for section embedding
            eligibility when a section does not set `embed` explicitly
        replace_existing_document: if True, delete existing sections for this
            document before reloading them from the chunk file

    Returns:
        doc_id if ingestion succeeded, else None.
    """
    logger.info("Loading chunks from %s", chunk_file)

    chunks = load_chunks_from_file(chunk_file)
    if not chunks:
        logger.warning("Empty chunk file: %s", chunk_file)
        return None

    doc_ids = {chunk["doc_id"] for chunk in chunks}
    if len(doc_ids) != 1:
        raise ValueError(
            f"Chunk file {chunk_file} contains multiple doc_id values: {sorted(doc_ids)}"
        )

    doc_id = next(iter(doc_ids))
    logger.info(
        "Building graph structure for document: %s | replace_existing_document=%s",
        doc_id,
        replace_existing_document,
    )

    normalized_sections = [
        normalize_section_record(
            chunk,
            min_text_chars_to_embed=min_text_chars_to_embed,
        )
        for chunk in chunks
    ]
    section_uids = [section["uid"] for section in normalized_sections]

    parent_child_pairs = []
    next_pairs = []

    for i, section in enumerate(normalized_sections):
        parent_id = section.get("parent_section_id")
        if parent_id:
            parent_child_pairs.append(
                {
                    "parent_uid": make_section_uid(doc_id, parent_id),
                    "child_uid": section["uid"],
                }
            )

        if i > 0:
            next_pairs.append(
                {
                    "prev_uid": normalized_sections[i - 1]["uid"],
                    "next_uid": section["uid"],
                }
            )

    with driver.session() as session:
        session.execute_write(setup_schema)
        session.execute_write(create_document, doc_id)

        if replace_existing_document:
            session.execute_write(delete_existing_document_sections, doc_id)
            session.execute_write(delete_orphan_concepts)
        else:
            if session.execute_read(document_sections_exist, doc_id):
                raise ValueError(
                    f"Document {doc_id} already has sections in the graph. "
                    "Use replace_existing_document=True to reload it."
                )

        for batch in chunked(normalized_sections, batch_size):
            session.execute_write(create_sections_batch, batch)

        for uid_batch in chunked(
            [{"uid": uid} for uid in section_uids],
            batch_size,
        ):
            session.execute_write(
                link_document_sections_batch,
                doc_id,
                [row["uid"] for row in uid_batch],
            )

        for batch in chunked(parent_child_pairs, batch_size):
            session.execute_write(link_parent_child_batch, batch)

        for batch in chunked(next_pairs, batch_size):
            session.execute_write(link_next_batch, batch)

    logger.info(
        "Graph structure built for document: %s | sections=%d | parent_child=%d | next=%d",
        doc_id,
        len(normalized_sections),
        len(parent_child_pairs),
        len(next_pairs),
    )

    return doc_id
