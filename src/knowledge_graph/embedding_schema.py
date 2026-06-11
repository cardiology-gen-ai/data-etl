"""Neo4j schema helpers for Section embedding vector indexes."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from typing import Any

from neo4j import Driver


logger = logging.getLogger(__name__)

_INDEX_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUPPORTED_SIMILARITIES = {"cosine", "euclidean"}

_SECTION_LABEL = "Section"
_EMBEDDING_PROPERTY = "embedding"


def _validate_index_name(index_name: str) -> str:
    normalized = str(index_name).strip()

    if not normalized:
        raise ValueError("Vector index name must not be empty")

    if not _INDEX_NAME_RE.fullmatch(normalized):
        raise ValueError(
            "Invalid vector index name. Use only letters, numbers, and "
            "underscores, and do not start with a number."
        )

    return normalized


def _validate_similarity(similarity: str) -> str:
    normalized = str(similarity).strip().lower()

    if normalized not in _SUPPORTED_SIMILARITIES:
        raise ValueError(
            "Unsupported vector similarity function "
            f"{similarity!r}. Expected one of: "
            f"{sorted(_SUPPORTED_SIMILARITIES)}"
        )

    return normalized


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    return [str(item) for item in value if item is not None]


def _extract_index_config(index_record: Mapping[str, Any]) -> dict[str, Any]:
    options = index_record.get("options")

    if not isinstance(options, Mapping):
        return {}

    index_config = options.get("indexConfig")

    if not isinstance(index_config, Mapping):
        return {}

    return dict(index_config)


def _extract_index_dimensions(
    index_record: Mapping[str, Any],
) -> int | None:
    raw_value = _extract_index_config(index_record).get(
        "vector.dimensions"
    )

    if raw_value is None:
        return None

    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _extract_index_similarity(
    index_record: Mapping[str, Any],
) -> str | None:
    raw_value = _extract_index_config(index_record).get(
        "vector.similarity_function"
    )

    if raw_value is None:
        return None

    normalized = str(raw_value).strip().lower()
    return normalized or None


def _is_section_embedding_schema(
    index_record: Mapping[str, Any],
) -> bool:
    index_type = str(index_record.get("type") or "").upper()
    entity_type = str(index_record.get("entityType") or "").upper()
    labels = _normalize_string_list(index_record.get("labelsOrTypes"))
    properties = _normalize_string_list(index_record.get("properties"))

    return (
        index_type == "VECTOR"
        and entity_type == "NODE"
        and labels == [_SECTION_LABEL]
        and properties == [_EMBEDDING_PROPERTY]
    )


def _index_mismatches(
    index_record: Mapping[str, Any],
    *,
    dimensions: int,
    similarity: str,
) -> list[str]:
    mismatches: list[str] = []

    index_type = str(index_record.get("type") or "").upper()
    if index_type != "VECTOR":
        mismatches.append(
            f"type={index_type or None!r}, expected='VECTOR'"
        )

    entity_type = str(index_record.get("entityType") or "").upper()
    if entity_type != "NODE":
        mismatches.append(
            f"entityType={entity_type or None!r}, expected='NODE'"
        )

    labels = _normalize_string_list(index_record.get("labelsOrTypes"))
    if labels != [_SECTION_LABEL]:
        mismatches.append(
            f"labelsOrTypes={labels!r}, expected={[_SECTION_LABEL]!r}"
        )

    properties = _normalize_string_list(index_record.get("properties"))
    if properties != [_EMBEDDING_PROPERTY]:
        mismatches.append(
            f"properties={properties!r}, "
            f"expected={[_EMBEDDING_PROPERTY]!r}"
        )

    existing_dimensions = _extract_index_dimensions(index_record)
    if existing_dimensions != dimensions:
        mismatches.append(
            f"vector.dimensions={existing_dimensions!r}, "
            f"expected={dimensions!r}"
        )

    existing_similarity = _extract_index_similarity(index_record)
    if existing_similarity != similarity:
        mismatches.append(
            f"vector.similarity_function={existing_similarity!r}, "
            f"expected={similarity!r}"
        )

    return mismatches


def inspect_section_embeddings(driver: Driver) -> dict[str, Any]:
    """
    Inspect stored Section embeddings and infer the vector-index dimension.

    The function validates that:
    - at least one Section embedding exists
    - every stored embedding belongs to a retrievable Section
    - has_embedding and embedding are consistent
    - actual vector lengths are uniform
    - embedding_dim metadata matches actual vector lengths
    - embedding_model metadata is present and uniform
    """
    query = """
    MATCH (s:Section)
    WHERE s.embedding IS NOT NULL
       OR coalesce(s.has_embedding, false) = true
    WITH
        s,
        CASE
            WHEN s.embedding IS NULL THEN null
            ELSE size(s.embedding)
        END AS actual_dimension
    RETURN
        sum(
            CASE
                WHEN s.embedding IS NOT NULL THEN 1
                ELSE 0
            END
        ) AS embedded_sections,
        sum(
            CASE
                WHEN coalesce(s.has_embedding, false) = true THEN 1
                ELSE 0
            END
        ) AS has_embedding_sections,
        sum(
            CASE
                WHEN s.embedding IS NOT NULL
                 AND coalesce(s.has_embedding, false) = false
                THEN 1
                ELSE 0
            END
        ) AS stale_embedding_sections,
        sum(
            CASE
                WHEN s.embedding IS NULL
                 AND coalesce(s.has_embedding, false) = true
                THEN 1
                ELSE 0
            END
        ) AS missing_embedding_property_sections,
        sum(
            CASE
                WHEN s.embedding IS NOT NULL
                 AND coalesce(s.embed, false) = false
                THEN 1
                ELSE 0
            END
        ) AS non_retrievable_embedding_sections,
        sum(
            CASE
                WHEN s.embedding IS NOT NULL
                 AND s.embedding_dim IS NULL
                THEN 1
                ELSE 0
            END
        ) AS missing_embedding_dim_sections,
        sum(
            CASE
                WHEN s.embedding IS NOT NULL
                 AND coalesce(trim(toString(s.embedding_model)), '') = ''
                THEN 1
                ELSE 0
            END
        ) AS missing_embedding_model_sections,
        sum(
            CASE
                WHEN s.embedding IS NOT NULL
                 AND s.embedding_dim IS NOT NULL
                 AND s.embedding_dim <> actual_dimension
                THEN 1
                ELSE 0
            END
        ) AS embedding_dim_mismatch_sections,
        collect(DISTINCT actual_dimension) AS actual_dimensions,
        collect(
            DISTINCT CASE
                WHEN s.embedding IS NOT NULL THEN s.embedding_dim
                ELSE null
            END
        ) AS metadata_dimensions,
        collect(
            DISTINCT CASE
                WHEN s.embedding IS NOT NULL THEN s.embedding_model
                ELSE null
            END
        ) AS embedding_models
    """

    with driver.session() as session:
        record = session.run(query).single()

    if record is None:
        raise RuntimeError(
            "Unable to inspect Section embeddings: Neo4j returned no row"
        )

    data = record.data()

    embedded_sections = int(data.get("embedded_sections") or 0)
    has_embedding_sections = int(
        data.get("has_embedding_sections") or 0
    )
    stale_embedding_sections = int(
        data.get("stale_embedding_sections") or 0
    )
    missing_embedding_property_sections = int(
        data.get("missing_embedding_property_sections") or 0
    )
    non_retrievable_embedding_sections = int(
        data.get("non_retrievable_embedding_sections") or 0
    )
    missing_embedding_dim_sections = int(
        data.get("missing_embedding_dim_sections") or 0
    )
    missing_embedding_model_sections = int(
        data.get("missing_embedding_model_sections") or 0
    )
    embedding_dim_mismatch_sections = int(
        data.get("embedding_dim_mismatch_sections") or 0
    )

    actual_dimensions = sorted(
        {
            int(value)
            for value in (data.get("actual_dimensions") or [])
            if value is not None
        }
    )
    metadata_dimensions = sorted(
        {
            int(value)
            for value in (data.get("metadata_dimensions") or [])
            if value is not None
        }
    )
    embedding_models = sorted(
        {
            str(value).strip()
            for value in (data.get("embedding_models") or [])
            if value is not None and str(value).strip()
        }
    )

    if embedded_sections == 0:
        raise RuntimeError(
            "Cannot create the Section vector index because no stored "
            "Section embeddings were found"
        )

    consistency_errors: list[str] = []

    if stale_embedding_sections:
        consistency_errors.append(
            f"{stale_embedding_sections} Section nodes have an embedding "
            "but has_embedding=false"
        )

    if missing_embedding_property_sections:
        consistency_errors.append(
            f"{missing_embedding_property_sections} Section nodes have "
            "has_embedding=true but no embedding property"
        )

    if non_retrievable_embedding_sections:
        consistency_errors.append(
            f"{non_retrievable_embedding_sections} non-retrievable Section "
            "nodes still contain an embedding"
        )

    if missing_embedding_dim_sections:
        consistency_errors.append(
            f"{missing_embedding_dim_sections} embedded Section nodes are "
            "missing embedding_dim metadata"
        )

    if missing_embedding_model_sections:
        consistency_errors.append(
            f"{missing_embedding_model_sections} embedded Section nodes are "
            "missing embedding_model metadata"
        )

    if embedding_dim_mismatch_sections:
        consistency_errors.append(
            f"{embedding_dim_mismatch_sections} embedded Section nodes have "
            "embedding_dim metadata inconsistent with the actual vector length"
        )

    if len(actual_dimensions) != 1:
        consistency_errors.append(
            "Stored Section embeddings do not have one uniform actual "
            f"dimension: {actual_dimensions!r}"
        )

    if len(metadata_dimensions) != 1:
        consistency_errors.append(
            "Section embedding_dim metadata is not uniform: "
            f"{metadata_dimensions!r}"
        )

    if (
        len(actual_dimensions) == 1
        and len(metadata_dimensions) == 1
        and actual_dimensions[0] != metadata_dimensions[0]
    ):
        consistency_errors.append(
            "Actual and metadata embedding dimensions differ: "
            f"actual={actual_dimensions[0]}, "
            f"metadata={metadata_dimensions[0]}"
        )

    if len(embedding_models) != 1:
        consistency_errors.append(
            "Section embeddings do not use one uniform embedding model: "
            f"{embedding_models!r}"
        )

    if consistency_errors:
        raise RuntimeError(
            "Section embedding consistency validation failed:\n- "
            + "\n- ".join(consistency_errors)
        )

    return {
        "eligible_sections": embedded_sections,
        "has_embedding_sections": has_embedding_sections,
        "dimensions": actual_dimensions[0],
        "embedding_model": embedding_models[0],
        "actual_dimensions": actual_dimensions,
        "metadata_dimensions": metadata_dimensions,
        "embedding_models": embedding_models,
    }


def _list_indexes(driver: Driver) -> list[dict[str, Any]]:
    query = """
    SHOW INDEXES
    YIELD
        name,
        state,
        populationPercent,
        type,
        entityType,
        labelsOrTypes,
        properties,
        options,
        indexProvider
    RETURN
        name,
        state,
        populationPercent,
        type,
        entityType,
        labelsOrTypes,
        properties,
        options,
        indexProvider
    ORDER BY name
    """

    with driver.session() as session:
        return [record.data() for record in session.run(query)]


def _get_index_by_name(
    driver: Driver,
    index_name: str,
) -> dict[str, Any] | None:
    for index_record in _list_indexes(driver):
        if index_record.get("name") == index_name:
            return index_record

    return None


def _get_section_embedding_schema_indexes(
    driver: Driver,
) -> list[dict[str, Any]]:
    return [
        index_record
        for index_record in _list_indexes(driver)
        if _is_section_embedding_schema(index_record)
    ]


def _drop_index(driver: Driver, index_name: str) -> None:
    validated_name = _validate_index_name(index_name)
    query = f"DROP INDEX `{validated_name}` IF EXISTS"

    with driver.session() as session:
        session.run(query).consume()

    logger.info("Dropped Neo4j index: %s", validated_name)


def _create_section_vector_index(
    driver: Driver,
    *,
    index_name: str,
    dimensions: int,
    similarity: str,
) -> None:
    validated_name = _validate_index_name(index_name)
    validated_similarity = _validate_similarity(similarity)

    query = f"""
    CREATE VECTOR INDEX `{validated_name}` IF NOT EXISTS
    FOR (s:{_SECTION_LABEL})
    ON (s.{_EMBEDDING_PROPERTY})
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: $dimensions,
            `vector.similarity_function`: $similarity
        }}
    }}
    """

    with driver.session() as session:
        session.run(
            query,
            dimensions=int(dimensions),
            similarity=validated_similarity,
        ).consume()

    logger.info(
        "Requested Neo4j vector index creation | name=%s | "
        "dimensions=%d | similarity=%s",
        validated_name,
        dimensions,
        validated_similarity,
    )


def _wait_for_index_online(
    driver: Driver,
    index_name: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float = 0.5,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    deadline = time.monotonic() + timeout_seconds

    while True:
        index_record = _get_index_by_name(driver, index_name)

        if index_record is not None:
            state = str(index_record.get("state") or "").upper()

            if state == "ONLINE":
                return index_record

            if state == "FAILED":
                raise RuntimeError(
                    f"Neo4j vector index {index_name!r} entered FAILED state"
                )

        if time.monotonic() >= deadline:
            current_state = (
                index_record.get("state")
                if index_record is not None
                else None
            )
            raise TimeoutError(
                f"Timed out waiting for Neo4j vector index "
                f"{index_name!r} to become ONLINE; "
                f"current_state={current_state!r}"
            )

        time.sleep(poll_interval_seconds)


def _build_stats(
    *,
    requested_index_name: str,
    index_record: Mapping[str, Any],
    embedding_stats: Mapping[str, Any],
    similarity: str,
    created: bool,
    recreated: bool,
    reused_existing_schema_index: bool,
) -> dict[str, Any]:
    return {
        "requested_index_name": requested_index_name,
        "index_name": index_record.get("name"),
        "state": index_record.get("state"),
        "population_percent": index_record.get("populationPercent"),
        "index_provider": index_record.get("indexProvider"),
        "label": _SECTION_LABEL,
        "property": _EMBEDDING_PROPERTY,
        "dimensions": embedding_stats["dimensions"],
        "similarity": similarity,
        "embedding_model": embedding_stats["embedding_model"],
        "eligible_sections": embedding_stats["eligible_sections"],
        "created": created,
        "recreated": recreated,
        "reused_existing_schema_index": reused_existing_schema_index,
        "validated": True,
    }


def setup_section_vector_index(
    driver: Driver,
    index_name: str = "section_embedding_index",
    similarity: str = "cosine",
    recreate_if_mismatch: bool = False,
    wait_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """
    Create or validate the Neo4j vector index for Section.embedding.

    The function:
    1. validates all stored Section embeddings
    2. infers the vector dimension and embedding model from Neo4j
    3. validates an existing named index when present
    4. detects an equivalent schema index under a different name
    5. optionally recreates incompatible indexes
    6. waits for the resulting index to become ONLINE

    Returns a JSON-serializable statistics dictionary.
    """
    validated_name = _validate_index_name(index_name)
    validated_similarity = _validate_similarity(similarity)

    if wait_timeout_seconds <= 0:
        raise ValueError(
            "wait_timeout_seconds must be greater than zero"
        )

    embedding_stats = inspect_section_embeddings(driver)
    dimensions = int(embedding_stats["dimensions"])

    existing_named_index = _get_index_by_name(
        driver,
        validated_name,
    )

    created = False
    recreated = False
    reused_existing_schema_index = False

    if existing_named_index is not None:
        mismatches = _index_mismatches(
            existing_named_index,
            dimensions=dimensions,
            similarity=validated_similarity,
        )

        if not mismatches:
            online_index = _wait_for_index_online(
                driver,
                validated_name,
                timeout_seconds=wait_timeout_seconds,
            )

            logger.info(
                "Validated existing Section vector index: %s",
                validated_name,
            )

            return _build_stats(
                requested_index_name=validated_name,
                index_record=online_index,
                embedding_stats=embedding_stats,
                similarity=validated_similarity,
                created=False,
                recreated=False,
                reused_existing_schema_index=False,
            )

        if not recreate_if_mismatch:
            raise RuntimeError(
                f"Neo4j index {validated_name!r} exists but is incompatible:\n- "
                + "\n- ".join(mismatches)
                + "\nSet recreate_if_mismatch=true to replace it."
            )

        _drop_index(driver, validated_name)
        recreated = True

    schema_indexes = _get_section_embedding_schema_indexes(driver)

    if schema_indexes:
        matching_schema_indexes: list[dict[str, Any]] = []
        incompatible_schema_indexes: list[
            tuple[dict[str, Any], list[str]]
        ] = []

        for schema_index in schema_indexes:
            schema_index_name = str(schema_index.get("name") or "")

            if schema_index_name == validated_name:
                continue

            mismatches = _index_mismatches(
                schema_index,
                dimensions=dimensions,
                similarity=validated_similarity,
            )

            if mismatches:
                incompatible_schema_indexes.append(
                    (schema_index, mismatches)
                )
            else:
                matching_schema_indexes.append(schema_index)

        if matching_schema_indexes:
            if len(matching_schema_indexes) > 1:
                names = [
                    str(record.get("name"))
                    for record in matching_schema_indexes
                ]
                raise RuntimeError(
                    "Multiple compatible vector indexes already exist for "
                    f"Section.embedding: {names!r}"
                )

            existing_schema_index = matching_schema_indexes[0]
            existing_schema_name = str(
                existing_schema_index.get("name")
            )

            logger.warning(
                "A compatible Section.embedding vector index already exists "
                "under a different name; reusing %s instead of creating %s",
                existing_schema_name,
                validated_name,
            )

            online_index = _wait_for_index_online(
                driver,
                existing_schema_name,
                timeout_seconds=wait_timeout_seconds,
            )
            reused_existing_schema_index = True

            return _build_stats(
                requested_index_name=validated_name,
                index_record=online_index,
                embedding_stats=embedding_stats,
                similarity=validated_similarity,
                created=False,
                recreated=recreated,
                reused_existing_schema_index=True,
            )

        if incompatible_schema_indexes:
            details = []

            for schema_index, mismatches in incompatible_schema_indexes:
                details.append(
                    f"{schema_index.get('name')!r}: "
                    + "; ".join(mismatches)
                )

            if not recreate_if_mismatch:
                raise RuntimeError(
                    "An incompatible vector index already exists for "
                    "Section.embedding:\n- "
                    + "\n- ".join(details)
                    + "\nSet recreate_if_mismatch=true to replace it."
                )

            for schema_index, _ in incompatible_schema_indexes:
                schema_index_name = str(schema_index.get("name") or "")
                if schema_index_name:
                    _drop_index(driver, schema_index_name)

            recreated = True

    _create_section_vector_index(
        driver,
        index_name=validated_name,
        dimensions=dimensions,
        similarity=validated_similarity,
    )
    created = True

    online_index = _wait_for_index_online(
        driver,
        validated_name,
        timeout_seconds=wait_timeout_seconds,
    )

    return _build_stats(
        requested_index_name=validated_name,
        index_record=online_index,
        embedding_stats=embedding_stats,
        similarity=validated_similarity,
        created=created,
        recreated=recreated,
        reused_existing_schema_index=reused_existing_schema_index,
    )
