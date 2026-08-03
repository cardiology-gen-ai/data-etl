"""
Section embedding pipeline for retrieval Section nodes.

Key guarantees:
- only active retrieval Sections marked ``embed=true`` are processed;
- structural Sections are never embedded or assigned embedding state;
- long Section text is segmented without truncation;
- segment vectors are combined into one Section vector with a deterministic,
  length-weighted mean;
- a Section is written only when every segment succeeded;
- stale vectors and pooling metadata are removed on failures/skips.

``max_chars_per_section`` is retained for backward compatibility with the
existing configuration, but now means *maximum characters per embedding
segment*. It no longer means "truncate the Section at this length".
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from neo4j import Driver

from knowledge_graph.llm_utils import (
    embed_texts,
    get_embedding_model_name,
)


logger = logging.getLogger(__name__)

RETRIEVAL_ROLE = "retrieval"
EMBEDDING_SOURCE = "retrieval_section_view"
SINGLE_SEGMENT_POOLING = "single_segment"
MULTI_SEGMENT_POOLING = "length_weighted_mean"


def chunked(
    items: Sequence[Dict[str, Any]],
    batch_size: int,
) -> Iterable[List[Dict[str, Any]]]:
    """Yield successive batches from a sequence."""
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def build_embedding_text(
    row: Dict[str, Any],
    include_title: bool = True,
    include_body: bool = True,
) -> str:
    """Build the complete, untruncated text represented by one Section."""
    parts: List[str] = []

    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()

    if include_title and title:
        parts.append(f"Title: {title}")
    if include_body and body:
        parts.append(f"Body:\n{body}")

    return "\n\n".join(parts).strip()


def _split_oversized_block(block: str, max_chars: int) -> List[str]:
    """
    Split one oversized block without dropping non-whitespace content.

    Prefer a whitespace boundary in the latter half of the current window;
    otherwise use a hard character boundary. Boundary whitespace is removed,
    but textual content is never truncated.
    """
    remaining = block.strip()
    pieces: List[str] = []

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        lower_bound = max(1, max_chars // 2)

        split_at = -1
        for match in re.finditer(r"\s+", window):
            if match.start() >= lower_bound:
                split_at = match.start()

        if split_at <= 0:
            split_at = max_chars

        piece = remaining[:split_at].strip()
        if not piece:
            piece = remaining[:max_chars]
            split_at = max_chars

        pieces.append(piece)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        pieces.append(remaining)

    return pieces


def split_text_for_embedding(
    text: str,
    max_chars: Optional[int],
) -> List[str]:
    """
    Split text into deterministic, non-empty segments without truncation.

    Paragraph boundaries are preferred. Oversized paragraphs are split at a
    nearby whitespace boundary and, only when necessary, at a hard character
    boundary.
    """
    text = (text or "").strip()
    if not text:
        return []

    if max_chars is None:
        return [text]
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1 or None")
    if len(text) <= max_chars:
        return [text]

    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    segments: List[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current:
            segments.append(current)
            current = ""

    for block in blocks:
        if len(block) > max_chars:
            flush_current()
            segments.extend(_split_oversized_block(block, max_chars))
            continue

        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            flush_current()
            current = block

    flush_current()

    if not segments:
        return _split_oversized_block(text, max_chars)

    if any(len(segment) > max_chars for segment in segments):
        raise AssertionError("Embedding segmentation produced an oversized segment")

    return segments


def build_embedding_segments(
    row: Dict[str, Any],
    *,
    include_title: bool,
    include_body: bool,
    max_segment_chars: Optional[int],
    allow_title_only: bool,
) -> List[Dict[str, Any]]:
    """
    Build one or more embedding inputs for a Section.

    For long body text, the title is repeated on every segment so each vector
    retains Section context. Segment pooling weights are based on the unique body
    content length, not on the repeated title prefix.
    """
    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()

    if include_body and not body and not allow_title_only:
        return []

    complete_text = build_embedding_text(
        row,
        include_title=include_title,
        include_body=include_body,
    )
    if not complete_text:
        return []

    if max_segment_chars is None or len(complete_text) <= max_segment_chars:
        return [
            {
                "text": complete_text,
                "weight_chars": max(1, len(body) if include_body and body else len(complete_text)),
            }
        ]

    if max_segment_chars < 1:
        raise ValueError("max_chars_per_section must be >= 1 or None")

    if include_body and body:
        title_prefix = f"Title: {title}\n\n" if include_title and title else ""
        body_prefix = "Body:\n"
        fixed_prefix = title_prefix + body_prefix
        body_budget = max_segment_chars - len(fixed_prefix)

        if body_budget < 1:
            raise ValueError(
                "max_chars_per_section is too small for the configured title/body prefix "
                f"for Section {row.get('uid')!r}"
            )

        body_segments = split_text_for_embedding(body, body_budget)
        return [
            {
                "text": f"{fixed_prefix}{segment}".strip(),
                "weight_chars": max(1, len(segment)),
            }
            for segment in body_segments
        ]

    # Title-only mode: preserve the full title through segmentation as well.
    return [
        {
            "text": segment,
            "weight_chars": max(1, len(segment)),
        }
        for segment in split_text_for_embedding(complete_text, max_segment_chars)
    ]


def mark_sections_embedding_failed(tx, section_uids: List[str]) -> None:
    """Mark Sections failed and clear all stale embedding state."""
    if not section_uids:
        return

    tx.run(
        """
        UNWIND $uids AS uid
        MATCH (s:Section {uid: uid})
        SET s.has_embedding = false,
            s.embedding_status = 'failed',
            s.embedding_failed_at = datetime()
        REMOVE s.embedding,
               s.embedding_model,
               s.embedding_provider,
               s.embedding_dim,
               s.embedding_updated_at,
               s.embedding_source,
               s.embedding_content_hash,
               s.embedding_input_chars,
               s.embedding_segment_count,
               s.embedding_max_segment_chars,
               s.embedding_pooling_method,
               s.embedding_was_segmented
        """,
        uids=sorted(set(section_uids)),
    )


def mark_sections_embedding_skipped_empty(tx, section_uids: List[str]) -> None:
    """Mark eligible retrieval Sections skipped because no valid input exists."""
    if not section_uids:
        return

    tx.run(
        """
        UNWIND $uids AS uid
        MATCH (s:Section {uid: uid})
        SET s.has_embedding = false,
            s.embedding_status = 'skipped_empty'
        REMOVE s.embedding,
               s.embedding_model,
               s.embedding_provider,
               s.embedding_dim,
               s.embedding_updated_at,
               s.embedding_failed_at,
               s.embedding_source,
               s.embedding_content_hash,
               s.embedding_input_chars,
               s.embedding_segment_count,
               s.embedding_max_segment_chars,
               s.embedding_pooling_method,
               s.embedding_was_segmented
        """,
        uids=sorted(set(section_uids)),
    )


def fetch_sections_to_embed(
    driver: Driver,
    doc_id: Optional[str] = None,
    max_sections: Optional[int] = None,
    force_reembed: bool = False,
    include_title: bool = True,
    include_body: bool = True,
    max_chars_per_section: Optional[int] = None,
    allow_title_only: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch active retrieval Sections and prepare complete embedding segments.

    ``max_chars_per_section`` is a segment-size limit. No Section content is
    discarded when this limit is exceeded.
    """
    with driver.session() as session:
        query = """
        MATCH (s:Section)
        WITH s, properties(s) AS section_props
        WHERE ($doc_id IS NULL OR s.doc_id = $doc_id)
          AND coalesce(s.section_view_role, '') = $retrieval_role
          AND coalesce(s.embed, false) = true
          AND coalesce(s.excluded, false) = false
        """

        if not force_reembed:
            query += """
              AND coalesce(s.has_embedding, false) = false
              AND coalesce(section_props['embedding_status'], '') <> 'skipped_empty'
            """

        query += """
        RETURN s.uid AS uid,
               s.doc_id AS doc_id,
               s.section_id AS section_id,
               s.title AS title,
               s.text AS text,
               s.retrieval_order AS retrieval_order,
               s.section_view_order AS section_view_order
        ORDER BY s.doc_id,
                 s.retrieval_order,
                 s.section_view_order,
                 s.uid
        """

        if max_sections is not None:
            query += "\nLIMIT $max_sections"

        result = session.run(
            query,
            doc_id=doc_id,
            retrieval_role=RETRIEVAL_ROLE,
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
                "retrieval_order": record["retrieval_order"],
                "section_view_order": record["section_view_order"],
            }

            segments = build_embedding_segments(
                row,
                include_title=include_title,
                include_body=include_body,
                max_segment_chars=max_chars_per_section,
                allow_title_only=allow_title_only,
            )

            if not segments:
                skipped_empty_uids.append(row["uid"])
                logger.info(
                    "Skipping empty embedding input | doc=%s section=%s",
                    row["doc_id"],
                    row["section_id"],
                )
                continue

            complete_text = build_embedding_text(
                row,
                include_title=include_title,
                include_body=include_body,
            )
            row["embedding_segments"] = segments
            row["embedding_input_chars"] = len(complete_text)
            row["embedding_content_hash"] = hashlib.sha256(
                complete_text.encode("utf-8")
            ).hexdigest()
            rows.append(row)

        if skipped_empty_uids:
            session.execute_write(
                mark_sections_embedding_skipped_empty,
                skipped_empty_uids,
            )

        return rows, len(skipped_empty_uids)


def build_embedding_units(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten Section segments into backend request units."""
    units: List[Dict[str, Any]] = []

    for row in rows:
        segments = list(row.get("embedding_segments") or [])
        segment_count = len(segments)

        for segment_index, segment in enumerate(segments):
            units.append(
                {
                    "unit_id": f"{row['uid']}::embedding_segment::{segment_index}",
                    "uid": row["uid"],
                    "doc_id": row["doc_id"],
                    "section_id": row["section_id"],
                    "segment_index": segment_index,
                    "segment_count": segment_count,
                    "text": segment["text"],
                    "weight_chars": int(segment["weight_chars"]),
                }
            )

    return units


def request_embeddings(
    texts: List[str],
    batch_size: int,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_dimensions: Optional[int] = None,
) -> Optional[List[List[float]]]:
    """Request one vector per supplied text; return None on any invalid response."""
    if not texts:
        return []

    try:
        vectors = embed_texts(
            texts=texts,
            batch_size=batch_size,
            provider=embedding_provider,
            model_name=embedding_model,
            dimensions=embedding_dimensions,
        )
    except Exception as exc:
        logger.exception("Embedding request failed: %s", exc)
        return None

    if len(vectors) != len(texts):
        logger.error(
            "Embedding response size mismatch | expected=%d | received=%d",
            len(texts),
            len(vectors),
        )
        return None

    expected_dim: Optional[int] = None
    for index, vector in enumerate(vectors):
        if not isinstance(vector, list) or not vector:
            logger.error("Invalid embedding vector at index %d", index)
            return None
        if expected_dim is None:
            expected_dim = len(vector)
        elif len(vector) != expected_dim:
            logger.error(
                "Embedding dimension mismatch inside response | expected=%d | received=%d | index=%d",
                expected_dim,
                len(vector),
                index,
            )
            return None
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in vector):
            logger.error("Non-finite embedding value at index %d", index)
            return None

    return vectors


def pool_segment_embeddings(
    vectors: Sequence[Sequence[float]],
    weights: Sequence[int],
) -> List[float]:
    """
    Combine all segment vectors for one Section.

    A single segment is returned unchanged. Multiple segments use a
    character-length-weighted arithmetic mean. No segment is silently omitted.
    """
    if not vectors:
        raise ValueError("Cannot pool an empty vector list")
    if len(vectors) != len(weights):
        raise ValueError("vectors and weights must have the same length")

    dimension = len(vectors[0])
    if dimension < 1:
        raise ValueError("Embedding vectors must be non-empty")
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("All segment vectors must have the same dimension")

    if len(vectors) == 1:
        return [float(value) for value in vectors[0]]

    normalized_weights = [max(1, int(weight)) for weight in weights]
    total_weight = float(sum(normalized_weights))
    pooled = [0.0] * dimension

    for vector, weight in zip(vectors, normalized_weights):
        for index, value in enumerate(vector):
            pooled[index] += float(value) * weight

    return [value / total_weight for value in pooled]


def write_embeddings_batch(
    tx,
    rows_with_embeddings: List[Dict[str, Any]],
    embedding_model: str,
    embedding_provider: Optional[str],
    max_segment_chars: Optional[int],
) -> None:
    """Write final Section vectors and auditable segmentation metadata."""
    if not rows_with_embeddings:
        return

    tx.run(
        """
        UNWIND $rows AS row
        MATCH (s:Section {uid: row.uid})
        WHERE coalesce(s.section_view_role, '') = $retrieval_role
          AND coalesce(s.embed, false) = true
          AND coalesce(s.excluded, false) = false
        SET s.embedding = row.embedding,
            s.has_embedding = true,
            s.embedding_model = $embedding_model,
            s.embedding_provider = $embedding_provider,
            s.embedding_dim = row.embedding_dim,
            s.embedding_updated_at = datetime(),
            s.embedding_status = 'success',
            s.embedding_source = $embedding_source,
            s.embedding_content_hash = row.embedding_content_hash,
            s.embedding_input_chars = row.embedding_input_chars,
            s.embedding_segment_count = row.embedding_segment_count,
            s.embedding_max_segment_chars = $max_segment_chars,
            s.embedding_pooling_method = row.embedding_pooling_method,
            s.embedding_was_segmented = row.embedding_segment_count > 1
        REMOVE s.embedding_failed_at
        """,
        rows=rows_with_embeddings,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider or "",
        max_segment_chars=max_segment_chars,
        embedding_source=EMBEDDING_SOURCE,
        retrieval_role=RETRIEVAL_ROLE,
    )


def add_embeddings_to_sections(
    driver: Driver,
    doc_id: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_dimensions: Optional[int] = None,
    max_sections: Optional[int] = None,
    batch_size: int = 8,
    force_reembed: bool = False,
    include_title: bool = True,
    include_body: bool = True,
    max_chars_per_section: Optional[int] = 8000,
    allow_title_only: bool = False,
) -> Dict[str, int]:
    """
    Compute one embedding per active retrieval Section.

    Long inputs are segmented at ``max_chars_per_section`` and all segment
    vectors are pooled. The function never truncates Section content.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_sections is not None and max_sections < 1:
        raise ValueError("max_sections must be >= 1 or None")
    if max_chars_per_section is not None and max_chars_per_section < 1:
        raise ValueError("max_chars_per_section must be >= 1 or None")
    if not include_title and not include_body:
        raise ValueError("At least one of include_title or include_body must be True")
    if include_title and not include_body and not allow_title_only:
        raise ValueError(
            "Title-only embeddings are disabled. Set allow_title_only=True or include_body=True."
        )

    rows, skipped_sections = fetch_sections_to_embed(
        driver=driver,
        doc_id=doc_id,
        max_sections=max_sections,
        force_reembed=force_reembed,
        include_title=include_title,
        include_body=include_body,
        max_chars_per_section=max_chars_per_section,
        allow_title_only=allow_title_only,
    )

    embedding_model = embedding_model or get_embedding_model_name()
    units = build_embedding_units(rows)
    segmented_sections = sum(
        1 for row in rows if len(row.get("embedding_segments") or []) > 1
    )

    logger.info(
        "Preparing embeddings for %d retrieval Sections (%d segments, %d segmented Sections)%s | "
        "provider=%s | model=%s | dimensions=%s | batch_size=%d | max_segment_chars=%s",
        len(rows),
        len(units),
        segmented_sections,
        f" in document {doc_id}" if doc_id else "",
        embedding_provider,
        embedding_model,
        embedding_dimensions,
        batch_size,
        max_chars_per_section,
    )

    stats = {
        "eligible_retrieval_sections": len(rows),
        "processed_sections": 0,
        "successful_sections": 0,
        "failed_sections": 0,
        "skipped_sections": skipped_sections,
        "written_embeddings": 0,
        "embedding_segments": len(units),
        "segmented_sections": segmented_sections,
        "failed_segments": 0,
    }

    if not rows:
        return stats

    vectors_by_uid: Dict[str, Dict[int, List[float]]] = defaultdict(dict)
    weights_by_uid: Dict[str, Dict[int, int]] = defaultdict(dict)
    failed_uids: set[str] = set()

    with driver.session() as session:
        for batch_number, batch in enumerate(chunked(units, batch_size), start=1):
            logger.info(
                "Requesting embedding batch %d | segments=%d | sections=%d",
                batch_number,
                len(batch),
                len({unit["uid"] for unit in batch}),
            )

            vectors = request_embeddings(
                texts=[unit["text"] for unit in batch],
                batch_size=len(batch),
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
            )

            if vectors is None:
                batch_failed_uids = {unit["uid"] for unit in batch}
                failed_uids.update(batch_failed_uids)
                stats["failed_segments"] += len(batch)
                logger.warning(
                    "Embedding batch failed; all affected Sections will be marked failed | sections=%s",
                    sorted(batch_failed_uids),
                )
                continue

            for unit, vector in zip(batch, vectors):
                uid = unit["uid"]
                vectors_by_uid[uid][unit["segment_index"]] = vector
                weights_by_uid[uid][unit["segment_index"]] = unit["weight_chars"]

        rows_with_embeddings: List[Dict[str, Any]] = []
        rows_by_uid = {row["uid"]: row for row in rows}

        for uid, row in rows_by_uid.items():
            expected_count = len(row["embedding_segments"])
            received = vectors_by_uid.get(uid, {})

            if uid in failed_uids or len(received) != expected_count:
                failed_uids.add(uid)
                continue

            ordered_vectors = [received[index] for index in range(expected_count)]
            ordered_weights = [weights_by_uid[uid][index] for index in range(expected_count)]

            try:
                pooled = pool_segment_embeddings(ordered_vectors, ordered_weights)
            except Exception as exc:
                logger.exception("Failed to pool Section embedding | uid=%s | error=%s", uid, exc)
                failed_uids.add(uid)
                continue

            rows_with_embeddings.append(
                {
                    "uid": uid,
                    "embedding": pooled,
                    "embedding_dim": len(pooled),
                    "embedding_content_hash": row["embedding_content_hash"],
                    "embedding_input_chars": row["embedding_input_chars"],
                    "embedding_segment_count": expected_count,
                    "embedding_pooling_method": (
                        SINGLE_SEGMENT_POOLING
                        if expected_count == 1
                        else MULTI_SEGMENT_POOLING
                    ),
                }
            )

        if failed_uids:
            session.execute_write(
                mark_sections_embedding_failed,
                sorted(failed_uids),
            )

        for write_batch in chunked(rows_with_embeddings, batch_size):
            session.execute_write(
                write_embeddings_batch,
                write_batch,
                embedding_model,
                embedding_provider,
                max_chars_per_section,
            )

    stats["processed_sections"] = len(rows)
    stats["failed_sections"] = len(failed_uids)
    stats["successful_sections"] = len(rows_with_embeddings)
    stats["written_embeddings"] = len(rows_with_embeddings)

    logger.info(
        "Embedding completed | eligible=%d | processed=%d | successful=%d | failed=%d | "
        "skipped=%d | written=%d | segments=%d | segmented_sections=%d | failed_segments=%d",
        stats["eligible_retrieval_sections"],
        stats["processed_sections"],
        stats["successful_sections"],
        stats["failed_sections"],
        stats["skipped_sections"],
        stats["written_embeddings"],
        stats["embedding_segments"],
        stats["segmented_sections"],
        stats["failed_segments"],
    )

    return stats
