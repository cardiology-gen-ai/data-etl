import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neo4j import Driver

from knowledge_graph.llm_utils import (
    embed_texts,
    get_embedding_model_name,
)


logger = logging.getLogger(__name__)


def chunked(
    items: List[Dict[str, Any]],
    batch_size: int,
) -> Iterable[List[Dict[str, Any]]]:
    """
    Yield successive batches from a list.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def build_embedding_text(
    row: Dict[str, Any],
    include_title: bool = True,
    include_body: bool = True,
) -> str:
    """
    Build the text to embed for a section.

    Default policy:
    - include title if present
    - include body text if present
    """
    parts = []

    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()

    if include_title and title:
        parts.append(f"Title: {title}")

    if include_body and body:
        parts.append(f"Body:\n{body}")

    return "\n\n".join(parts).strip()


def emergency_truncate(text: str, max_chars: Optional[int]) -> str:
    """
    Optional safety truncation for very long embedding inputs.

    Note:
    This is character-based, not token-based.
    """
    if max_chars is None:
        return text

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n...[truncated]"


def mark_sections_embedding_failed(tx, section_uids: List[str]) -> None:
    tx.run(
        """
        UNWIND $uids AS uid
        MATCH (s:Section {uid: uid})
        SET s.has_embedding = false,
            s.embedding_status = 'failed',
            s.embedding_failed_at = datetime()
        """,
        uids=section_uids,
    )


def mark_sections_embedding_skipped_empty(tx, section_uids: List[str]) -> None:
    tx.run(
        """
        UNWIND $uids AS uid
        MATCH (s:Section {uid: uid})
        SET s.has_embedding = false,
            s.embedding_status = 'skipped_empty',
            s.embedding_failed_at = null
        """,
        uids=section_uids,
    )


def fetch_sections_to_embed(
    driver: Driver,
    doc_id: Optional[str] = None,
    max_sections: Optional[int] = None,
    force_reembed: bool = False,
    include_title: bool = True,
    include_body: bool = True,
    max_chars_per_section: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch embed-eligible sections and build their embedding text.

    Returns:
        (rows_to_embed, skipped_empty_count)
    """
    with driver.session() as session:
        query = """
        MATCH (s:Section)
        WHERE ($doc_id IS NULL OR s.doc_id = $doc_id)
          AND coalesce(s.embed, false) = true
        """

        if not force_reembed:
            query += """
              AND coalesce(s.has_embedding, false) = false
              AND coalesce(s.embedding_status, '') <> 'skipped_empty'
            """

        query += """
        RETURN s.uid AS uid,
               s.doc_id AS doc_id,
               s.section_id AS section_id,
               s.title AS title,
               s.text AS text
        ORDER BY s.uid
        """

        if max_sections is not None:
            query += "\nLIMIT $max_sections"

        result = session.run(
            query,
            doc_id=doc_id,
            max_sections=max_sections,
        )

        rows: List[Dict[str, Any]] = []
        skipped_empty_uids: List[str] = []

        for record in result:
            row = {
                "uid": record["uid"],
                "doc_id": record["doc_id"],
                "section_id": record["section_id"],
                "title": record["title"],
                "text": record["text"],
            }

            embedding_text = build_embedding_text(
                row=row,
                include_title=include_title,
                include_body=include_body,
            )

            if not embedding_text.strip():
                skipped_empty_uids.append(row["uid"])
                logger.info(
                    "Skipping empty embedding text | doc=%s section=%s",
                    row["doc_id"],
                    row["section_id"],
                )
                continue

            row["embedding_text"] = emergency_truncate(
                embedding_text,
                max_chars=max_chars_per_section,
            )
            rows.append(row)

        if skipped_empty_uids:
            session.execute_write(
                mark_sections_embedding_skipped_empty,
                skipped_empty_uids,
            )

        return rows, len(skipped_empty_uids)


def request_embeddings(
    texts: List[str],
    batch_size: int,
) -> Optional[List[List[float]]]:
    """
    Request embeddings for a batch of texts from the local embedding backend.

    Returns:
        list of embedding vectors on success
        None on failure
    """
    if not texts:
        return []

    try:
        vectors = embed_texts(texts=texts, batch_size=batch_size)
    except Exception as e:
        logger.exception("Local embedding request failed: %s", e)
        return None

    if len(vectors) != len(texts):
        logger.error(
            "Embedding response size mismatch | expected=%d | received=%d",
            len(texts),
            len(vectors),
        )
        return None

    for i, vector in enumerate(vectors):
        if not isinstance(vector, list) or not vector:
            logger.error("Invalid embedding vector at index %d", i)
            return None

    return vectors


def write_embeddings_batch(
    tx,
    rows_with_embeddings: List[Dict[str, Any]],
    embedding_model: str,
) -> None:
    """
    Write embedding vectors back to Neo4j in one batch.
    """
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (s:Section {uid: row.uid})
        SET s.embedding = row.embedding,
            s.has_embedding = true,
            s.embedding_model = $embedding_model,
            s.embedding_dim = row.embedding_dim,
            s.embedding_updated_at = datetime(),
            s.embedding_status = 'success',
            s.embedding_failed_at = null
        """,
        rows=rows_with_embeddings,
        embedding_model=embedding_model,
    )


def add_embeddings_to_sections(
    driver: Driver,
    doc_id: Optional[str] = None,
    max_sections: Optional[int] = None,
    batch_size: int = 8,
    force_reembed: bool = False,
    include_title: bool = True,
    include_body: bool = True,
    max_chars_per_section: Optional[int] = 8000,
    allow_title_only: bool = False,
) -> Dict[str, int]:
    """
    Compute and store embeddings for eligible Section nodes.

    Parameters:
        driver: Neo4j driver
        doc_id: if provided, only embed sections from one document
        max_sections: optional cap on number of sections
        batch_size: number of sections per local embedding call
        force_reembed: if True, embed even sections that already have embeddings
        include_title: whether to include the title in the embedding text
        include_body: whether to include body text in the embedding text
        max_chars_per_section: optional safety truncation for very long inputs
        allow_title_only: if True, allow embedding text to be built from title only

    Returns:
        summary statistics
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if not include_title and not include_body:
        raise ValueError("At least one of include_title or include_body must be True")

    if include_title and not include_body and not allow_title_only:
        raise ValueError(
            "Title-only embeddings are disabled by default. "
            "Set include_body=True or pass allow_title_only=True."
        )

    rows, skipped_sections = fetch_sections_to_embed(
        driver=driver,
        doc_id=doc_id,
        max_sections=max_sections,
        force_reembed=force_reembed,
        include_title=include_title,
        include_body=include_body,
        max_chars_per_section=max_chars_per_section,
    )

    embedding_model = get_embedding_model_name()

    logger.info(
        "Preparing embeddings for %d sections%s | model=%s | batch_size=%d",
        len(rows),
        f" in document {doc_id}" if doc_id else "",
        embedding_model,
        batch_size,
    )

    stats = {
        "processed_sections": 0,
        "successful_sections": 0,
        "failed_sections": 0,
        "skipped_sections": skipped_sections,
        "written_embeddings": 0,
    }

    if not rows:
        return stats

    with driver.session() as session:
        for batch in chunked(rows, batch_size):
            texts = [row["embedding_text"] for row in batch]

            logger.info("Requesting embeddings for batch of %d sections", len(batch))
            vectors = request_embeddings(
                texts=texts,
                batch_size=len(batch),
            )

            stats["processed_sections"] += len(batch)

            if vectors is None:
                stats["failed_sections"] += len(batch)
                session.execute_write(
                    mark_sections_embedding_failed,
                    [row["uid"] for row in batch],
                )
                logger.warning(
                    "Skipping batch after embedding failure | batch_size=%d",
                    len(batch),
                )
                continue

            rows_with_embeddings = [
                {
                    "uid": row["uid"],
                    "embedding": vector,
                    "embedding_dim": len(vector),
                }
                for row, vector in zip(batch, vectors)
            ]

            session.execute_write(
                write_embeddings_batch,
                rows_with_embeddings,
                embedding_model,
            )

            stats["successful_sections"] += len(rows_with_embeddings)
            stats["written_embeddings"] += len(rows_with_embeddings)

            logger.info(
                "Stored embeddings for %d sections | dim=%d",
                len(rows_with_embeddings),
                rows_with_embeddings[0]["embedding_dim"] if rows_with_embeddings else 0,
            )

    logger.info(
        "Embedding completed | processed=%d | successful=%d | failed=%d | skipped=%d | written=%d",
        stats["processed_sections"],
        stats["successful_sections"],
        stats["failed_sections"],
        stats["skipped_sections"],
        stats["written_embeddings"],
    )

    return stats