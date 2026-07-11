"""
Shared provenance metadata builders for managed knowledge-graph relationships.
"""


VALID_RELATIONSHIP_FAMILIES = {
    "structural",
    "mention",
    "normalization",
    "ontology",
}


STRUCTURAL_RELATIONSHIP_METADATA = {
    "HAS_SECTION": {
        "relationship_family": "structural",
        "provenance": "graph_loader",
        "provenance_source": "source_document",
        "provenance_method": "document_section_membership",
    },
    "HAS_CHILD": {
        "relationship_family": "structural",
        "provenance": "graph_loader",
        "provenance_source": "source_document",
        "provenance_method": "hierarchical_chunking",
    },
    "NEXT": {
        "relationship_family": "structural",
        "provenance": "graph_loader",
        "provenance_source": "source_document",
        "provenance_method": "sequential_section_order",
    },
}

NORMALIZATION_RELATIONSHIP_METADATA = {
    "SAME_AS": {
        "relationship_family": "normalization",
        "provenance": "umls_normalization",
        "provenance_source": "umls_metathesaurus",
        "provenance_method": "umls_cui",
    },
    "POSSIBLY_SAME_AS": {
        "relationship_family": "normalization",
        "provenance": "umls_normalization",
        "provenance_source": "local_matching",
        "provenance_method": "fuzzy_name",
    },
}

MENTION_RELATIONSHIP_METADATA = {
    "relationship_family": "mention",
    "provenance": "entity_extraction",
    "provenance_source": "source_document",
    "provenance_method": "llm_assisted_entity_extraction",
}

ONTOLOGY_RELATIONSHIP_METADATA = {
    "relationship_family": "ontology",
    "provenance": "umls_connections",
    "provenance_source": "umls_metathesaurus",
    "provenance_method": "umls_relations_api",
}


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _with_optional_doc_id(
    metadata: dict[str, object],
    doc_id: str | None,
) -> dict[str, object]:
    cleaned_doc_id = _clean_optional_text(doc_id)
    if cleaned_doc_id is not None:
        metadata["doc_id"] = cleaned_doc_id
    return metadata


def _assert_neo4j_scalar_values(metadata: dict[str, object]) -> None:
    scalar_types = (str, int, float, bool)
    for key, value in metadata.items():
        if value is None or isinstance(value, scalar_types):
            continue
        raise TypeError(
            f"Relationship metadata field {key!r} is not Neo4j-scalar compatible: "
            f"{type(value).__name__}"
        )


def _copy_metadata(metadata: dict[str, object]) -> dict[str, object]:
    copied = dict(metadata)
    _assert_neo4j_scalar_values(copied)
    return copied


def build_structural_relationship_metadata(
    relationship_type: str,
    doc_id: str | None = None,
) -> dict[str, object]:
    """
    Build provenance metadata for graph-loader structural relationships.
    """
    if relationship_type not in STRUCTURAL_RELATIONSHIP_METADATA:
        raise ValueError(
            f"Unsupported structural relationship type: {relationship_type!r}"
        )

    return _with_optional_doc_id(
        _copy_metadata(STRUCTURAL_RELATIONSHIP_METADATA[relationship_type]),
        doc_id,
    )


def build_mention_relationship_metadata(
    doc_id: str | None = None,
) -> dict[str, object]:
    """
    Build provenance metadata for Section-to-Concept MENTIONS relationships.
    """
    return _with_optional_doc_id(
        _copy_metadata(MENTION_RELATIONSHIP_METADATA),
        doc_id,
    )


def build_normalization_relationship_metadata(
    relationship_type: str,
) -> dict[str, object]:
    """
    Build provenance metadata for UMLS/local normalization relationships.
    """
    if relationship_type not in NORMALIZATION_RELATIONSHIP_METADATA:
        raise ValueError(
            f"Unsupported normalization relationship type: {relationship_type!r}"
        )

    return _copy_metadata(NORMALIZATION_RELATIONSHIP_METADATA[relationship_type])


def build_ontology_relationship_metadata(
    source_vocabulary: str,
) -> dict[str, object]:
    """
    Build provenance metadata for ontology-derived UMLS relationships.
    """
    cleaned_source_vocabulary = _clean_optional_text(source_vocabulary)
    if cleaned_source_vocabulary is None:
        raise ValueError("source_vocabulary is required for ontology relationships")

    metadata = _copy_metadata(ONTOLOGY_RELATIONSHIP_METADATA)
    metadata["source_vocabulary"] = cleaned_source_vocabulary
    _assert_neo4j_scalar_values(metadata)
    return metadata


def expected_metadata_for_relationship_type(
    relationship_type: str,
) -> dict[str, object] | None:
    """
    Return static expected metadata for managed types whose values are type-based.
    """
    if relationship_type in STRUCTURAL_RELATIONSHIP_METADATA:
        return build_structural_relationship_metadata(relationship_type)
    if relationship_type == "MENTIONS":
        return build_mention_relationship_metadata()
    if relationship_type in NORMALIZATION_RELATIONSHIP_METADATA:
        return build_normalization_relationship_metadata(relationship_type)
    return None


__all__ = [
    "VALID_RELATIONSHIP_FAMILIES",
    "STRUCTURAL_RELATIONSHIP_METADATA",
    "NORMALIZATION_RELATIONSHIP_METADATA",
    "MENTION_RELATIONSHIP_METADATA",
    "ONTOLOGY_RELATIONSHIP_METADATA",
    "build_structural_relationship_metadata",
    "build_mention_relationship_metadata",
    "build_normalization_relationship_metadata",
    "build_ontology_relationship_metadata",
    "expected_metadata_for_relationship_type",
]
