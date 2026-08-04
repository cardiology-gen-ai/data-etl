"""
Sanity checks for the knowledge-graph pipeline.

This module defines phase-aware Neo4j checks used to validate:
- graph structure after loading
- entity extraction outputs
- concept type-resolution outputs
- acronym-supported entity-validation metadata and expansion integrity
- invalid legacy entity types preserved during schema migration
- embedding outputs

Each check is tagged by phase so the pipeline can run only the
relevant validations for the current stage.

"""

import logging
from typing import Any, Dict, List, Set

from neo4j import Driver

from knowledge_graph.entity_schema import ALLOWED_TYPES
from knowledge_graph.umls_connections import (
    UMLS_CONNECTION_RELATION_TYPES,
    catalog_local_type_rule_rows,
    catalog_relationship_type_rows,
)


logger = logging.getLogger(__name__)


ALLOWED_MODES = {"structure", "entities", "embeddings", "full"}

PHASE_EXPANSION: Dict[str, Set[str]] = {
    "structure": {"structure"},
    "entities": {"structure", "entities"},
    "embeddings": {"structure", "embeddings"},
    "full": {"structure", "entities", "embeddings"},
}


SPECIAL_CANONICAL_TYPES = {
    "ambiguous",
    "no_supported_type",
}

VALID_CANONICAL_TYPES = sorted(ALLOWED_TYPES | SPECIAL_CANONICAL_TYPES)
VALID_ENTITY_TYPES = sorted(ALLOWED_TYPES)
UMLS_CATALOG_RELATIONSHIP_ROWS = catalog_relationship_type_rows()
UMLS_CATALOG_LOCAL_TYPE_RULES = catalog_local_type_rule_rows()
AUDIT_ONLY_UMLS_RELATIONSHIP_TYPES = sorted(
    row["relationship_type"]
    for row in UMLS_CATALOG_RELATIONSHIP_ROWS
    if not row["materialize_by_default"]
)
MATERIALIZABLE_UMLS_RELATIONSHIP_TYPES = sorted(
    row["relationship_type"]
    for row in UMLS_CATALOG_RELATIONSHIP_ROWS
    if row["materialize_by_default"]
)
COMMON_RELATIONSHIP_METADATA_FIELDS = [
    "relationship_family",
    "provenance",
    "provenance_source",
    "provenance_method",
]
MANAGED_RELATIONSHIP_TYPES = sorted(
    set(
        [
            "HAS_SECTION",
            "HAS_CHILD",
            "NEXT",
            "MENTIONS",
            "SAME_AS",
            "POSSIBLY_SAME_AS",
        ]
        + UMLS_CONNECTION_RELATION_TYPES
    )
)


CHECKS: List[Dict[str, Any]] = [
    {
        "name": "documents_missing_doc_id",
        "title": "Documents missing doc_id",
        "group": "Document structure",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (d:Document)
            WHERE d.doc_id IS NULL OR trim(d.doc_id) = ''
            RETURN elementId(d) AS node_id
            ORDER BY node_id
        """,
    },
    {
        "name": "documents_without_sections",
        "title": "Documents without sections",
        "group": "Document structure",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (d:Document)
            WHERE NOT EXISTS {
                MATCH (d)-[:HAS_SECTION]->(:Section)
            }
            RETURN d.doc_id AS doc_id
            ORDER BY doc_id
        """,
    },
    {
        "name": "sections_linked_to_multiple_documents",
        "title": "Sections linked to multiple documents",
        "group": "Document structure",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
            WITH s, collect(DISTINCT d.doc_id) AS docs
            WHERE size(docs) > 1
            RETURN s.uid AS uid, docs
            ORDER BY uid
        """,
    },
    {
        "name": "orphan_sections",
        "title": "Orphan sections with no document",
        "group": "Document structure",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE NOT EXISTS {
                MATCH (:Document)-[:HAS_SECTION]->(s)
            }
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "sections_missing_identity_fields",
        "title": "Sections missing identity fields",
        "group": "Section identity",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.uid IS NULL OR trim(s.uid) = ''
               OR s.doc_id IS NULL OR trim(s.doc_id) = ''
               OR s.section_id IS NULL OR trim(s.section_id) = ''
            RETURN s.uid AS uid,
                   s.doc_id AS doc_id,
                   s.section_id AS section_id,
                   s.title AS title
            ORDER BY doc_id, uid
        """,
    },
    {
        "name": "duplicate_section_uids",
        "title": "Duplicate section UID values",
        "group": "Section identity",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.uid IS NOT NULL
            WITH s.uid AS uid, count(*) AS n
            WHERE n > 1
            RETURN uid, n
            ORDER BY n DESC, uid
        """,
    },
    {
        "name": "uid_doc_id_mismatch",
        "title": "Section UID / doc_id mismatch",
        "group": "Section identity",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.uid IS NOT NULL
              AND s.doc_id IS NOT NULL
              AND NOT s.uid STARTS WITH s.doc_id + "::"
            RETURN s.uid AS uid, s.doc_id AS doc_id
            ORDER BY uid
        """,
    },
    {
        "name": "sections_with_multiple_parents",
        "title": "Sections with multiple parents",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (p:Section)-[:HAS_CHILD]->(c:Section)
            WITH c, count(DISTINCT p) AS parents
            WHERE parents > 1
            RETURN c.uid AS uid, parents
            ORDER BY parents DESC, uid
        """,
    },
    {
        "name": "has_child_edges_crossing_documents",
        "title": "HAS_CHILD edges crossing documents",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (p:Section)-[:HAS_CHILD]->(c:Section)
            WHERE p.doc_id <> c.doc_id
            RETURN p.uid AS parent_uid,
                   c.uid AS child_uid,
                   p.doc_id AS parent_doc_id,
                   c.doc_id AS child_doc_id
            ORDER BY parent_uid, child_uid
        """,
    },
    {
        "name": "cycles_in_has_child",
        "title": "Cycles in HAS_CHILD hierarchy",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE EXISTS {
                MATCH (s)-[:HAS_CHILD*1..]->(s)
            }
            RETURN DISTINCT s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "next_edges_crossing_documents",
        "title": "NEXT edges crossing documents",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (a:Section)-[:NEXT]->(b:Section)
            WHERE a.doc_id <> b.doc_id
            RETURN a.uid AS from_uid, b.uid AS to_uid
            ORDER BY from_uid, to_uid
        """,
    },
    {
        "name": "duplicate_next_edges_from_same_section",
        "title": "Sections with multiple outgoing NEXT edges",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "WARNING",
        "query": """
            MATCH (a:Section)-[:NEXT]->(b:Section)
            WITH a, count(DISTINCT b) AS outgoing_next
            WHERE outgoing_next > 1
            RETURN a.uid AS uid, outgoing_next
            ORDER BY outgoing_next DESC, uid
        """,
    },
    {
        "name": "duplicate_next_edges_to_same_section",
        "title": "Sections with multiple incoming NEXT edges",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "WARNING",
        "query": """
            MATCH (a:Section)-[:NEXT]->(b:Section)
            WITH b, count(DISTINCT a) AS incoming_next
            WHERE incoming_next > 1
            RETURN b.uid AS uid, incoming_next
            ORDER BY incoming_next DESC, uid
        """,
    },
    {
        "name": "sections_with_empty_body_text",
        "title": "Sections with empty body text",
        "group": "Section content",
        "phases": {"structure"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(trim(s.text), '') = ''
            RETURN s.uid AS uid, s.title AS title
            ORDER BY uid
        """,
    },
    {
        "name": "empty_leaf_sections",
        "title": "Empty leaf sections",
        "group": "Section content",
        "phases": {"structure"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(s.is_empty, false) = true
              AND NOT EXISTS {
                  MATCH (s)-[:HAS_CHILD]->(:Section)
              }
            RETURN s.uid AS uid, s.title AS title
            ORDER BY uid
        """,
    },
    {
        "name": "parent_sections_with_direct_body_text",
        "title": "Parent sections that also contain body text",
        "group": "Section content",
        "phases": {"structure"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)-[:HAS_CHILD]->(:Section)
            WHERE coalesce(s.is_empty, false) = false
              AND coalesce(trim(s.text), '') <> ''
              AND size(coalesce(s.text, '')) > 100
            RETURN DISTINCT s.uid AS uid,
                   s.title AS title,
                   size(coalesce(s.text, '')) AS text_len
            ORDER BY text_len DESC, uid
        """,
    },

    # ---------------------------------------------------------------------
    # Relationship provenance metadata checks
    # ---------------------------------------------------------------------
    {
        "name": "relationship_family_summary",
        "title": "Relationship counts by relationship_family",
        "group": "Relationship provenance",
        "phases": {"structure", "entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH ()-[r]->()
            RETURN coalesce(r.relationship_family, 'UNSET') AS relationship_family,
                   count(*) AS n
            ORDER BY n DESC, relationship_family ASC
        """,
    },
    {
        "name": "relationship_provenance_summary",
        "title": "Relationship counts by provenance",
        "group": "Relationship provenance",
        "phases": {"structure", "entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH ()-[r]->()
            RETURN coalesce(r.provenance, 'UNSET') AS provenance,
                   count(*) AS n
            ORDER BY n DESC, provenance ASC
        """,
    },
    {
        "name": "relationship_provenance_source_summary",
        "title": "Relationship counts by provenance_source",
        "group": "Relationship provenance",
        "phases": {"structure", "entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH ()-[r]->()
            RETURN coalesce(r.provenance_source, 'UNSET') AS provenance_source,
                   count(*) AS n
            ORDER BY n DESC, provenance_source ASC
        """,
    },
    {
        "name": "relationship_provenance_method_summary",
        "title": "Relationship counts by provenance_method",
        "group": "Relationship provenance",
        "phases": {"structure", "entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH ()-[r]->()
            RETURN coalesce(r.provenance_method, 'UNSET') AS provenance_method,
                   count(*) AS n
            ORDER BY n DESC, provenance_method ASC
        """,
    },
    {
        "name": "ontology_source_vocabulary_summary",
        "title": "Ontology relationship counts by source_vocabulary",
        "group": "Relationship provenance",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "params": {"relationship_types": UMLS_CONNECTION_RELATION_TYPES},
        "query": """
            MATCH ()-[r]->()
            WHERE type(r) IN $relationship_types
            RETURN coalesce(r.source_vocabulary, 'UNSET') AS source_vocabulary,
                   count(*) AS n
            ORDER BY n DESC, source_vocabulary ASC
        """,
    },
    {
        "name": "relationship_type_family_summary",
        "title": "Relationship type x relationship_family counts",
        "group": "Relationship provenance",
        "phases": {"structure", "entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH ()-[r]->()
            RETURN type(r) AS relationship_type,
                   coalesce(r.relationship_family, 'UNSET') AS relationship_family,
                   count(*) AS n
            ORDER BY relationship_type ASC, relationship_family ASC
        """,
    },
    {
        "name": "managed_relationships_missing_common_metadata",
        "title": "Managed relationships missing common provenance metadata",
        "group": "Relationship provenance",
        "phases": {"structure", "entities"},
        "level": "ERROR",
        "params": {
            "relationship_types": MANAGED_RELATIONSHIP_TYPES,
            "common_metadata_fields": COMMON_RELATIONSHIP_METADATA_FIELDS,
        },
        "query": """
            MATCH ()-[r]->()
            WHERE type(r) IN $relationship_types
              AND any(field IN $common_metadata_fields
                      WHERE r[field] IS NULL OR trim(toString(r[field])) = '')
            RETURN type(r) AS relationship_type,
                   elementId(r) AS relationship_id,
                   [field IN $common_metadata_fields
                    WHERE r[field] IS NULL OR trim(toString(r[field])) = ''] AS missing_fields
            ORDER BY relationship_type, relationship_id
        """,
    },
    {
        "name": "ontology_relationships_missing_source_vocabulary",
        "title": "Ontology relationships missing source_vocabulary",
        "group": "Relationship provenance",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {"relationship_types": UMLS_CONNECTION_RELATION_TYPES},
        "query": """
            MATCH ()-[r]->()
            WHERE type(r) IN $relationship_types
              AND (r.source_vocabulary IS NULL OR trim(toString(r.source_vocabulary)) = '')
            RETURN type(r) AS relationship_type,
                   elementId(r) AS relationship_id,
                   r.edge_key AS edge_key,
                   r.source_cui AS source_cui,
                   r.target_cui AS target_cui
            ORDER BY relationship_type, relationship_id
        """,
    },

    # ---------------------------------------------------------------------
    # Concept and type-resolution checks
    # ---------------------------------------------------------------------
    {
        "name": "orphan_concepts",
        "title": "Orphan concepts with no section mentions",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "INFO",
        "query": """
            MATCH (c:Concept)
            WHERE NOT EXISTS {
                MATCH (:Section)-[:MENTIONS]->(c)
            }
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "concept_type_resolution_status_summary",
        "title": "Concept type-resolution status summary",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH (c:Concept)
            RETURN coalesce(c.type_resolution_status, 'UNSET') AS status,
                   count(c) AS n
            ORDER BY n DESC, status ASC
        """,
    },
    {
        "name": "concept_canonical_type_summary",
        "title": "Concept canonical_type summary",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH (c:Concept)
            RETURN coalesce(c.canonical_type, 'UNSET') AS canonical_type,
                   count(c) AS n
            ORDER BY n DESC, canonical_type ASC
        """,
    },
    {
        "name": "concepts_still_pending_type_resolution",
        "title": "Concepts still pending type resolution",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "INFO",
        "query": """
            MATCH (c:Concept)
            WHERE c.type_resolution_status IS NULL
               OR c.type_resolution_status = 'pending'
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "concepts_with_unexpected_canonical_type",
        "title": "Concepts with unexpected canonical_type values",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {
            "valid_canonical_types": VALID_CANONICAL_TYPES,
        },
        "query": """
            MATCH (c:Concept)
            WHERE c.canonical_type IS NOT NULL
              AND NOT (c.canonical_type IN $valid_canonical_types)
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "concepts_without_observed_types",
        "title": "Concepts without observed_types",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE c.observed_types IS NULL OR size(c.observed_types) = 0
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "canonical_type_not_in_observed_types",
        "title": "Resolved canonical type not present in observed_types",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE c.canonical_type IS NOT NULL
              AND NOT (c.canonical_type IN ['ambiguous', 'no_supported_type'])
              AND c.observed_types IS NOT NULL
              AND NOT (c.canonical_type IN c.observed_types)
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.type_support_pairs AS type_support_pairs,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "resolved_concepts_without_valid_canonical_type",
        "title": "Resolved concepts without a valid canonical entity type",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {
            "allowed_entity_types": VALID_ENTITY_TYPES,
        },
        "query": """
            MATCH (c:Concept)
            WHERE c.type_resolution_status IN [
                'resolved_single_supported_type',
                'resolved_by_section_support'
            ]
              AND (
                    c.canonical_type IS NULL
                 OR NOT (c.canonical_type IN $allowed_entity_types)
              )
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.type_support_pairs AS type_support_pairs,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "concepts_marked_for_type_review",
        "title": "Concepts marked for type review",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE coalesce(c.needs_type_review, false) = true
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.type_support_pairs AS type_support_pairs,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "ambiguous_concepts_not_flagged_for_review",
        "title": "Ambiguous concepts not flagged for review",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (c:Concept)
            WHERE c.canonical_type = 'ambiguous'
              AND coalesce(c.needs_type_review, false) = false
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.type_support_pairs AS type_support_pairs,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "no_supported_type_concepts_not_flagged_for_review",
        "title": "No-supported-type concepts not flagged for review",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (c:Concept)
            WHERE c.canonical_type = 'no_supported_type'
              AND coalesce(c.needs_type_review, false) = false
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.type_support_pairs AS type_support_pairs,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "resolved_concepts_flagged_for_review",
        "title": "Resolved concepts still flagged for type review",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WHERE c.type_resolution_status IN [
                'resolved_single_supported_type',
                'resolved_by_section_support'
            ]
              AND coalesce(c.needs_type_review, false) = true
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.type_support_pairs AS type_support_pairs,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "concept_invalid_observed_type_summary",
        "title": "Concept invalid_observed_types summary",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH (c:Concept)
            UNWIND coalesce(c.invalid_observed_types, []) AS invalid_type
            RETURN invalid_type, count(c) AS n
            ORDER BY n DESC, invalid_type ASC
        """,
    },
    {
        "name": "concepts_with_invalid_types_not_flagged_for_review",
        "title": "Concepts with invalid observed types not flagged for review",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (c:Concept)
            WHERE size(coalesce(c.invalid_observed_types, [])) > 0
              AND coalesce(c.needs_type_review, false) = false
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types,
                   c.invalid_observed_types AS invalid_observed_types,
                   c.type_resolution_status AS type_resolution_status
            ORDER BY name
        """,
    },
    {
        "name": "allowed_types_stored_as_invalid_observed_types",
        "title": "Allowed entity types incorrectly stored as invalid_observed_types",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {
            "allowed_entity_types": VALID_ENTITY_TYPES,
        },
        "query": """
            MATCH (c:Concept)
            UNWIND coalesce(c.invalid_observed_types, []) AS invalid_type
            WITH c, invalid_type
            WHERE invalid_type IN $allowed_entity_types
            RETURN c.name AS name,
                   invalid_type,
                   c.invalid_observed_types AS invalid_observed_types
            ORDER BY name, invalid_type
        """,
    },
    {
        "name": "invalid_types_remaining_in_concept_observed_types",
        "title": "Invalid entity types remaining in Concept.observed_types",
        "group": "Concept type resolution",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {
            "allowed_entity_types": VALID_ENTITY_TYPES,
        },
        "query": """
            MATCH (c:Concept)
            UNWIND coalesce(c.observed_types, []) AS observed_type
            WITH c, observed_type
            WHERE observed_type IS NOT NULL
              AND observed_type <> ''
              AND NOT (observed_type IN $allowed_entity_types)
            RETURN c.name AS name,
                   observed_type,
                   c.observed_types AS observed_types,
                   c.invalid_observed_types AS invalid_observed_types
            ORDER BY name, observed_type
        """,
    },
    {
        "name": "document_specific_concepts",
        "title": "Document-specific concepts",
        "group": "Concept diagnostics",
        "phases": {"entities"},
        "level": "INFO",
        "query": """
            MATCH (c:Concept)<-[:MENTIONS]-(s:Section)
            WITH c, collect(DISTINCT s.doc_id) AS docs
            WHERE size(docs) = 1
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   docs
            ORDER BY name
        """,
    },
    {
        "name": "potentially_over_broad_high_frequency_concepts",
        "title": "Potentially over-broad high-frequency concepts",
        "group": "Concept diagnostics",
        "phases": {"entities"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)-[:MENTIONS]->(c:Concept)
            WITH c, count(DISTINCT s) AS n
            WHERE n > 30
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.type_resolution_status AS type_resolution_status,
                   n
            ORDER BY n DESC, name
        """,
    },
    {
        "name": "concept_normalization_status_summary",
        "title": "Concept normalization_status summary",
        "group": "UMLS normalization",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH (c:Concept)
            WITH c, properties(c) AS concept_props
            WHERE concept_props['normalization_status'] IS NOT NULL
               OR concept_props['umls_cui'] IS NOT NULL
            RETURN coalesce(concept_props['normalization_status'], 'UNSET') AS status,
                   count(c) AS n
            ORDER BY n DESC, status ASC
        """,
    },
    {
        "name": "umls_matched_concepts_missing_metadata",
        "title": "UMLS-matched concepts missing normalization metadata",
        "group": "UMLS normalization",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (c:Concept)
            WITH c, properties(c) AS concept_props
            WHERE concept_props['normalization_status'] = 'umls_matched'
              AND (
                    concept_props['umls_cui'] IS NULL
                 OR trim(concept_props['umls_cui']) = ''
                 OR concept_props['umls_canonical_name'] IS NULL
                 OR trim(concept_props['umls_canonical_name']) = ''
                 OR concept_props['umls_score'] IS NULL
                 OR concept_props['umls_linker_name'] IS NULL
                 OR concept_props['umls_model_name'] IS NULL
                 OR concept_props['normalized_name'] IS NULL
                 OR concept_props['normalized_at'] IS NULL
              )
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   concept_props['umls_cui'] AS umls_cui,
                   concept_props['umls_canonical_name'] AS umls_canonical_name,
                   concept_props['umls_score'] AS umls_score,
                   concept_props['normalization_status'] AS normalization_status
            ORDER BY name
        """,
    },
    {
        "name": "concepts_with_umls_cui_but_unmatched_status",
        "title": "Concepts with UMLS CUI but non-matched normalization status",
        "group": "UMLS normalization",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (c:Concept)
            WITH c, properties(c) AS concept_props
            WHERE concept_props['umls_cui'] IS NOT NULL
              AND trim(concept_props['umls_cui']) <> ''
              AND coalesce(concept_props['normalization_status'], '') <> 'umls_matched'
            RETURN c.name AS name,
                   c.canonical_type AS canonical_type,
                   concept_props['umls_cui'] AS umls_cui,
                   concept_props['normalization_status'] AS normalization_status,
                   concept_props['normalization_method'] AS normalization_method
            ORDER BY name
        """,
    },
    {
        "name": "same_as_edges_inconsistent_umls_cui",
        "title": "SAME_AS UMLS edges with missing or inconsistent CUIs",
        "group": "UMLS normalization",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (a:Concept)-[r]->(b:Concept)
            WITH a, r, b, properties(a) AS source_props, properties(r) AS rel_props, properties(b) AS target_props
            WHERE type(r) = 'SAME_AS'
              AND rel_props['method'] = 'umls_cui'
              AND (
                    a = b
                 OR source_props['umls_cui'] IS NULL
                 OR target_props['umls_cui'] IS NULL
                 OR source_props['umls_cui'] <> target_props['umls_cui']
              )
            RETURN a.name AS source_concept,
                   b.name AS target_concept,
                   source_props['umls_cui'] AS source_cui,
                   target_props['umls_cui'] AS target_cui,
                   rel_props['status'] AS status,
                   rel_props['score'] AS score
            ORDER BY source_concept, target_concept
        """,
    },
    {
        "name": "umls_connection_counts_by_type",
        "title": "Materialized UMLS connection counts by relationship type",
        "group": "UMLS connections",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "params": {"relationship_types": UMLS_CONNECTION_RELATION_TYPES},
        "query": """
            MATCH ()-[r]->()
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            RETURN type(r) AS relationship_type,
                   count(*) AS n
            ORDER BY relationship_type
        """,
    },
    {
        "name": "duplicate_umls_connection_edge_keys",
        "title": "Duplicate materialized UMLS connection edge_keys",
        "group": "UMLS connections",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {"relationship_types": UMLS_CONNECTION_RELATION_TYPES},
        "query": """
            MATCH ()-[r]->()
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH r.edge_key AS edge_key,
                 count(*) AS n,
                 collect(DISTINCT type(r)) AS relationship_types
            WHERE edge_key IS NULL OR trim(toString(edge_key)) = '' OR n > 1
            RETURN edge_key,
                   n,
                   relationship_types
            ORDER BY n DESC, edge_key
        """,
    },
    {
        "name": "umls_connection_cui_mismatches",
        "title": "Materialized UMLS connections with inconsistent endpoint CUIs",
        "group": "UMLS connections",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {"relationship_types": UMLS_CONNECTION_RELATION_TYPES},
        "query": """
            MATCH (source:Concept)-[r]->(target:Concept)
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH source, r, target,
                 properties(source) AS source_props,
                 properties(target) AS target_props,
                 properties(r) AS rel_props
            WHERE toUpper(coalesce(toString(source_props['umls_cui']), '')) <>
                  toUpper(coalesce(toString(rel_props['source_cui']), ''))
               OR toUpper(coalesce(toString(target_props['umls_cui']), '')) <>
                  toUpper(coalesce(toString(rel_props['target_cui']), ''))
            RETURN type(r) AS relationship_type,
                   rel_props['edge_key'] AS edge_key,
                   source.name AS source_concept,
                   target.name AS target_concept,
                   source_props['umls_cui'] AS source_node_cui,
                   rel_props['source_cui'] AS relationship_source_cui,
                   target_props['umls_cui'] AS target_node_cui,
                   rel_props['target_cui'] AS relationship_target_cui
            ORDER BY relationship_type, edge_key
        """,
    },
    {
        "name": "umls_connections_missing_compatibility_metadata",
        "title": "Materialized UMLS connections missing compatibility metadata",
        "group": "UMLS connections",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {"relationship_types": UMLS_CONNECTION_RELATION_TYPES},
        "query": """
            MATCH (source:Concept)-[r]->(target:Concept)
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH source, r, target, properties(r) AS rel_props
            WHERE (
                   rel_props['compatibility_status'] IS NULL
                OR rel_props['compatibility_reason'] IS NULL
                OR rel_props['local_type_compatible'] IS NULL
                OR rel_props['local_type_compatibility_reason'] IS NULL
                OR rel_props['relation_family'] IS NULL
                OR rel_props['materialize_by_default'] IS NULL
                OR rel_props['materialization_decision'] IS NULL
                OR rel_props['materialization_decision_reason'] IS NULL
              )
            RETURN type(r) AS relationship_type,
                   rel_props['edge_key'] AS edge_key,
                   rel_props['relation_name'] AS relation_name,
                   source.name AS source_concept,
                   target.name AS target_concept,
                   rel_props['compatibility_status'] AS compatibility_status,
                   rel_props['compatibility_reason'] AS compatibility_reason,
                   rel_props['local_type_compatible'] AS local_type_compatible,
                   rel_props['local_type_compatibility_reason'] AS local_type_compatibility_reason,
                   rel_props['relation_family'] AS relation_family,
                   rel_props['materialize_by_default'] AS materialize_by_default,
                   rel_props['materialization_decision'] AS materialization_decision,
                   rel_props['materialization_decision_reason'] AS materialization_decision_reason
            ORDER BY relationship_type, edge_key
        """,
    },
    {
        "name": "safe_only_umls_connections_policy_violations",
        "title": "Materialized UMLS connections violating safe_only policy",
        "group": "UMLS connections",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {"relationship_types": MATERIALIZABLE_UMLS_RELATIONSHIP_TYPES},
        "query": """
            MATCH (source:Concept)-[r]->(target:Concept)
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH source, r, target, properties(r) AS rel_props
            WHERE coalesce(toString(rel_props['materialization_mode']), '') <> 'safe_only'
               OR coalesce(rel_props['materialization_decision'], false) <> true
               OR coalesce(toString(rel_props['compatibility_status']), '') <> 'compatible'
               OR coalesce(rel_props['local_type_compatible'], false) <> true
               OR coalesce(rel_props['review_needed'], true) <> false
               OR NOT (coalesce(toString(rel_props['traversal_policy']), '') IN ['safe', 'hierarchy'])
            RETURN type(r) AS relationship_type,
                   rel_props['edge_key'] AS edge_key,
                   rel_props['relation_name'] AS relation_name,
                   source.name AS source_concept,
                   source.canonical_type AS source_canonical_type,
                   target.name AS target_concept,
                   target.canonical_type AS target_canonical_type,
                   rel_props['materialization_mode'] AS materialization_mode,
                   rel_props['materialization_decision'] AS materialization_decision,
                   rel_props['compatibility_status'] AS compatibility_status,
                   rel_props['local_type_compatible'] AS local_type_compatible,
                   rel_props['review_needed'] AS review_needed,
                   rel_props['traversal_policy'] AS traversal_policy
            ORDER BY relationship_type, edge_key
        """,
    },
    {
        "name": "umls_connection_exact_type_rule_violations",
        "title": "Materialized UMLS connections violating exact catalog type rules",
        "group": "UMLS connections",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {"local_type_rules": UMLS_CATALOG_LOCAL_TYPE_RULES},
        "query": """
            UNWIND $local_type_rules AS rule
            MATCH (source:Concept)-[r]->(target:Concept)
            WHERE type(r) = rule.relationship_type
              AND r.provenance = 'umls_connections'
            WITH rule, source, r, target,
                 coalesce(toString(source.canonical_type), '') AS source_type,
                 coalesce(toString(target.canonical_type), '') AS target_type
            WHERE NOT (source_type IN rule.source_types)
               OR NOT (target_type IN rule.target_types)
            RETURN type(r) AS relationship_type,
                   r.edge_key AS edge_key,
                   rule.relation_name AS relation_name,
                   source.name AS source_concept,
                   source_type AS source_canonical_type,
                   rule.source_types AS allowed_source_types,
                   target.name AS target_concept,
                   target_type AS target_canonical_type,
                   rule.target_types AS allowed_target_types
            ORDER BY relationship_type, edge_key
        """,
    },
    {
        "name": "umls_connection_counts_by_compatibility_status",
        "title": "Materialized UMLS connection counts by compatibility status",
        "group": "UMLS connections",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "params": {"relationship_types": UMLS_CONNECTION_RELATION_TYPES},
        "query": """
            MATCH ()-[r]->()
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH properties(r) AS rel_props
            RETURN coalesce(toString(rel_props['compatibility_status']), '(missing)') AS compatibility_status,
                   count(*) AS n
            ORDER BY compatibility_status
        """,
    },
    {
        "name": "umls_connection_counts_by_traversal_policy",
        "title": "Materialized UMLS connection counts by traversal policy",
        "group": "UMLS connections",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "params": {"relationship_types": UMLS_CONNECTION_RELATION_TYPES},
        "query": """
            MATCH ()-[r]->()
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH properties(r) AS rel_props
            RETURN coalesce(toString(rel_props['traversal_policy']), '(missing)') AS traversal_policy,
                   count(*) AS n
            ORDER BY traversal_policy
        """,
    },
    {
        "name": "review_needed_materialized_umls_connections",
        "title": "Materialized UMLS connections marked review-needed",
        "group": "UMLS connections",
        "phases": {"entities"},
        "level": "WARNING",
        "params": {"relationship_types": UMLS_CONNECTION_RELATION_TYPES},
        "query": """
            MATCH (source:Concept)-[r]->(target:Concept)
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH source, r, target, properties(r) AS rel_props
            WHERE (
                   coalesce(rel_props['review_needed'], false) = true
                OR coalesce(toString(rel_props['traversal_policy']), '') IN [
                    'hierarchy_review',
                    'reverse_review',
                    'review',
                    'type_review'
                ]
              )
            RETURN type(r) AS relationship_type,
                   rel_props['edge_key'] AS edge_key,
                   rel_props['relation_name'] AS relation_name,
                   source.name AS source_concept,
                   target.name AS target_concept,
                   rel_props['compatibility_status'] AS compatibility_status,
                   rel_props['compatibility_reason'] AS compatibility_reason,
                   rel_props['traversal_policy'] AS traversal_policy,
                   rel_props['review_needed'] AS review_needed
            ORDER BY relationship_type, edge_key
        """,
    },
    {
        "name": "audit_only_umls_candidates_materialized",
        "title": "Audit-only UMLS candidate relations materialized in Neo4j",
        "group": "UMLS connections",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {"relationship_types": AUDIT_ONLY_UMLS_RELATIONSHIP_TYPES},
        "query": """
            MATCH (source:Concept)-[r]->(target:Concept)
            WHERE type(r) IN $relationship_types
              AND r.provenance = 'umls_connections'
            WITH source, r, target, properties(r) AS rel_props
            RETURN type(r) AS relationship_type,
                   rel_props['edge_key'] AS edge_key,
                   rel_props['relation_name'] AS relation_name,
                   source.name AS source_concept,
                   target.name AS target_concept,
                   rel_props['materialization_mode'] AS materialization_mode,
                   rel_props['materialization_decision_reason'] AS materialization_decision_reason
            ORDER BY relationship_type, edge_key
        """,
    },
    {
        "name": "possibly_same_as_self_edges",
        "title": "POSSIBLY_SAME_AS self edges",
        "group": "UMLS normalization",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (a:Concept)-[r]->(b:Concept)
            WITH a, r, b, properties(r) AS rel_props
            WHERE type(r) = 'POSSIBLY_SAME_AS'
              AND a = b
            RETURN a.name AS concept,
                   rel_props['method'] AS method,
                   rel_props['status'] AS status,
                   rel_props['score'] AS score
            ORDER BY concept
        """,
    },
    {
        "name": "short_fuzzy_duplicate_candidates_without_evidence",
        "title": "Short fuzzy duplicate candidates without UMLS/acronym evidence",
        "group": "UMLS normalization",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (a:Concept)-[r]->(b:Concept)
            WITH a, r, b, properties(a) AS source_props, properties(r) AS rel_props, properties(b) AS target_props
            WHERE type(r) = 'POSSIBLY_SAME_AS'
              AND rel_props['method'] = 'fuzzy_name'
              AND size(a.name) <= 3
              AND size(b.name) <= 3
              AND (
                    source_props['umls_cui'] IS NULL
                 OR target_props['umls_cui'] IS NULL
                 OR source_props['umls_cui'] <> target_props['umls_cui']
              )
            RETURN a.name AS source_concept,
                   b.name AS target_concept,
                   source_props['umls_cui'] AS source_cui,
                   target_props['umls_cui'] AS target_cui,
                   rel_props['score'] AS score,
                   rel_props['status'] AS status
            ORDER BY score DESC, source_concept, target_concept
        """,
    },

    # ---------------------------------------------------------------------
    # Entity extraction state checks
    # ---------------------------------------------------------------------
    {
        "name": "sections_missing_entity_extracted_flag",
        "title": "Sections missing entity_extracted flag",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.entity_extracted IS NULL
            RETURN s.uid AS uid,
                   s.doc_id AS doc_id,
                   s.title AS title
            ORDER BY doc_id, uid
        """,
    },
    {
        "name": "entity_extraction_status_summary",
        "title": "Entity extraction status summary",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH (s:Section)
            RETURN coalesce(s.entity_extraction_status, 'UNSET') AS status,
                   count(s) AS n
            ORDER BY n DESC, status ASC
        """,
    },
    {
        "name": "entity_status_success_but_not_extracted",
        "title": "Sections marked entity success but not extracted",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.entity_extraction_status = 'success'
              AND coalesce(s.entity_extracted, false) = false
            RETURN s.uid AS uid,
                   s.entity_extraction_status AS status,
                   s.entity_extracted AS entity_extracted
            ORDER BY uid
        """,
    },
    {
        "name": "entity_extracted_but_missing_status",
        "title": "Sections extracted but missing entity status",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(s.entity_extracted, false) = true
              AND s.entity_extraction_status IS NULL
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "entity_failed_without_timestamp",
        "title": "Sections with failed entity extraction but no timestamp",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WITH s, properties(s) AS section_props
            WHERE s.entity_extraction_status = 'failed'
              AND section_props['entity_extraction_failed_at'] IS NULL
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "skipped_empty_sections_with_mentions",
        "title": "Skipped-empty sections that still have entity mentions",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)-[:MENTIONS]->(c:Concept)
            WHERE s.entity_extraction_status = 'skipped_empty'
            RETURN s.uid AS uid,
                   s.title AS title,
                   c.name AS concept
            ORDER BY uid, concept
        """,
    },
    {
        "name": "failed_sections_with_mentions",
        "title": "Failed entity-extraction sections that still have mentions",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)-[:MENTIONS]->(c:Concept)
            WHERE s.entity_extraction_status = 'failed'
            RETURN s.uid AS uid,
                   s.title AS title,
                   c.name AS concept
            ORDER BY uid, concept
        """,
    },

    # ---------------------------------------------------------------------
    # Mention relationship checks
    # ---------------------------------------------------------------------
    {
        "name": "mention_support_method_summary",
        "title": "MENTIONS support_method summary",
        "group": "Mention relationships",
        "phases": {"entities"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH (:Section)-[r:MENTIONS]->(:Concept)
            RETURN coalesce(r.support_method, 'UNSET') AS support_method,
                   count(*) AS n
            ORDER BY n DESC, support_method ASC
        """,
    },
    {
        "name": "mentions_without_observed_types",
        "title": "MENTIONS relationships without observed_types",
        "group": "Mention relationships",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            WHERE r.observed_types IS NULL OR size(r.observed_types) = 0
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   r.observed_types AS observed_types
            ORDER BY section_uid, concept
        """,
    },
    {
        "name": "mentions_with_unexpected_observed_type",
        "title": "MENTIONS relationships with unexpected observed type values",
        "group": "Mention relationships",
        "phases": {"entities"},
        "level": "ERROR",
        "params": {
            "allowed_entity_types": VALID_ENTITY_TYPES,
        },
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            UNWIND coalesce(r.observed_types, []) AS observed_type
            WITH s, r, c, observed_type
            WHERE observed_type IS NOT NULL
              AND observed_type <> ''
              AND NOT (observed_type IN $allowed_entity_types)
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   observed_type,
                   r.observed_types AS observed_types
            ORDER BY section_uid, concept, observed_type
        """,
    },
    {
        "name": "mentions_missing_validation_metadata",
        "title": "MENTIONS relationships missing validation metadata",
        "group": "Mention relationships",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            WHERE r.support_method IS NULL
               OR r.validation_reason IS NULL
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   r.support_method AS support_method,
                   r.validation_reason AS validation_reason
            ORDER BY section_uid, concept
        """,
    },
    {
        "name": "mentions_with_multiple_observed_types",
        "title": "MENTIONS relationships with multiple observed types",
        "group": "Mention relationships",
        "phases": {"entities"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            WHERE r.observed_types IS NOT NULL
              AND size(r.observed_types) > 1
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   r.observed_types AS observed_types
            ORDER BY section_uid, concept
        """,
    },

    # ---------------------------------------------------------------------
    # Acronym validation checks
    # ---------------------------------------------------------------------
    {
        "name": "acronym_supported_mentions_missing_metadata",
        "title": "Acronym-supported mentions missing acronym metadata",
        "group": "Acronym validation",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            WITH s, r, c, properties(r) AS mention_props
            WHERE mention_props['support_method'] = 'acronym'
              AND (
                    mention_props['acronym_short'] IS NULL
                 OR trim(mention_props['acronym_short']) = ''
                 OR mention_props['acronym_definition'] IS NULL
                 OR trim(mention_props['acronym_definition']) = ''
                 OR mention_props['acronym_match_method'] IS NULL
                 OR trim(mention_props['acronym_match_method']) = ''
              )
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   mention_props['acronym_short'] AS acronym_short,
                   mention_props['acronym_definition'] AS acronym_definition,
                   mention_props['acronym_match_method'] AS acronym_match_method
            ORDER BY section_uid, concept
        """,
    },
    {
        "name": "expanded_acronym_mentions_missing_raw_name",
        "title": "Expanded acronym mentions missing raw_name",
        "group": "Acronym validation",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            WITH s, r, c, properties(r) AS mention_props
            WHERE coalesce(mention_props['expanded_from_acronym'], false) = true
              AND (
                    mention_props['raw_name'] IS NULL
                 OR trim(mention_props['raw_name']) = ''
              )
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   mention_props['raw_name'] AS raw_name,
                   mention_props['acronym_short'] AS acronym_short,
                   mention_props['acronym_definition'] AS acronym_definition
            ORDER BY section_uid, concept
        """,
    },
    {
        "name": "expanded_acronym_mentions_without_acronym_support",
        "title": "Expanded acronym mentions without acronym support metadata",
        "group": "Acronym validation",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            WITH s, r, c, properties(r) AS mention_props
            WHERE coalesce(mention_props['expanded_from_acronym'], false) = true
              AND mention_props['support_method'] <> 'acronym'
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   mention_props['support_method'] AS support_method,
                   mention_props['raw_name'] AS raw_name,
                   mention_props['acronym_short'] AS acronym_short
            ORDER BY section_uid, concept
        """,
    },

    {
        "name": "acronym_supported_mentions_short_not_in_section",
        "title": "Acronym-supported mentions whose short form is absent from the Section",
        "group": "Acronym validation",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            WITH s, r, c, properties(r) AS mention_props,
                 toLower(
                     coalesce(s.title, '') + ' ' + coalesce(s.text, '')
                 ) AS section_source
            WHERE mention_props['support_method'] = 'acronym'
              AND mention_props['acronym_short'] IS NOT NULL
              AND trim(toString(mention_props['acronym_short'])) <> ''
              AND NOT section_source CONTAINS
                  toLower(trim(toString(mention_props['acronym_short'])))
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   mention_props['acronym_short'] AS acronym_short,
                   mention_props['acronym_definition'] AS acronym_definition,
                   mention_props['acronym_match_method'] AS acronym_match_method
            ORDER BY section_uid, concept
        """,
    },
    {
        "name": "expanded_acronym_mentions_still_named_as_short_form",
        "title": "Expanded acronym mentions whose Concept name is still the short form",
        "group": "Acronym validation",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            WITH s, r, c, properties(r) AS mention_props
            WHERE coalesce(mention_props['expanded_from_acronym'], false) = true
              AND mention_props['acronym_short'] IS NOT NULL
              AND toLower(trim(c.name)) =
                  toLower(trim(toString(mention_props['acronym_short'])))
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   mention_props['raw_name'] AS raw_name,
                   mention_props['acronym_short'] AS acronym_short,
                   mention_props['acronym_definition'] AS acronym_definition
            ORDER BY section_uid, concept
        """,
    },
    {
        "name": "expanded_acronym_raw_name_mismatches_short_form",
        "title": "Expanded acronym mentions whose raw_name differs from acronym_short",
        "group": "Acronym validation",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            WITH s, r, c, properties(r) AS mention_props
            WHERE coalesce(mention_props['expanded_from_acronym'], false) = true
              AND mention_props['raw_name'] IS NOT NULL
              AND mention_props['acronym_short'] IS NOT NULL
              AND toUpper(trim(toString(mention_props['raw_name']))) <>
                  toUpper(trim(toString(mention_props['acronym_short'])))
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   mention_props['raw_name'] AS raw_name,
                   mention_props['acronym_short'] AS acronym_short,
                   mention_props['acronym_definition'] AS acronym_definition
            ORDER BY section_uid, concept
        """,
    },
    {
        "name": "likely_unexpanded_uppercase_acronym_mentions",
        "title": "Likely uppercase acronym mentions not expanded through acronym validation",
        "group": "Acronym validation",
        "phases": {"entities"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)-[r:MENTIONS]->(c:Concept)
            WITH s, r, c, properties(r) AS mention_props
            WHERE mention_props['raw_name'] IS NOT NULL
              AND trim(toString(mention_props['raw_name'])) =~
                  '^[A-Z][A-Z0-9-]{1,7}$'
              AND toLower(trim(c.name)) =
                  toLower(trim(toString(mention_props['raw_name'])))
              AND coalesce(mention_props['expanded_from_acronym'], false) = false
              AND coalesce(toString(mention_props['support_method']), '') <>
                  'acronym'
            RETURN s.uid AS section_uid,
                   c.name AS concept,
                   mention_props['raw_name'] AS raw_name,
                   mention_props['support_method'] AS support_method,
                   mention_props['validation_reason'] AS validation_reason
            ORDER BY section_uid, concept
        """,
    },

    # ---------------------------------------------------------------------
    # Embedding checks
    # ---------------------------------------------------------------------
    {
        "name": "sections_missing_has_embedding_flag",
        "title": "Sections missing has_embedding flag",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.has_embedding IS NULL
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "embedding_status_summary",
        "title": "Embedding status summary",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "INFO",
        "is_summary": True,
        "query": """
            MATCH (s:Section)
            WITH s, properties(s) AS section_props
            RETURN coalesce(section_props['embedding_status'], 'UNSET') AS status,
                   count(s) AS n
            ORDER BY n DESC, status ASC
        """,
    },
    {
        "name": "embedding_flag_inconsistencies",
        "title": "Embedding flag inconsistencies",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WITH s, properties(s) AS section_props
            WHERE (
                    coalesce(section_props['has_embedding'], false) = true
                    AND section_props['embedding'] IS NULL
                  )
               OR (
                    coalesce(section_props['has_embedding'], false) = false
                    AND section_props['embedding'] IS NOT NULL
                  )
            RETURN s.uid AS uid,
                   section_props['has_embedding'] AS has_embedding,
                   section_props['embedding'] IS NOT NULL AS has_embedding_vector
            ORDER BY uid
        """,
    },
    {
        "name": "embedding_metadata_inconsistencies",
        "title": "Embedding metadata inconsistencies",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WITH s, properties(s) AS section_props
            WHERE section_props['embedding'] IS NOT NULL
              AND (
                    section_props['embedding_dim'] IS NULL
                 OR section_props['embedding_model'] IS NULL
                 OR section_props['embedding_updated_at'] IS NULL
                 OR size(section_props['embedding']) <> section_props['embedding_dim']
              )
            RETURN s.uid AS uid,
                   section_props['embedding_dim'] AS embedding_dim,
                   size(section_props['embedding']) AS actual_dim,
                   section_props['embedding_model'] AS embedding_model,
                   section_props['embedding_updated_at'] AS embedding_updated_at
            ORDER BY uid
        """,
    },
    {
        "name": "embedding_status_success_but_no_vector",
        "title": "Sections marked embedding success but with no vector",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WITH s, properties(s) AS section_props
            WHERE section_props['embedding_status'] = 'success'
              AND (
                    section_props['embedding'] IS NULL
                 OR coalesce(section_props['has_embedding'], false) = false
              )
            RETURN s.uid AS uid,
                   section_props['embedding_status'] AS status,
                   section_props['has_embedding'] AS has_embedding
            ORDER BY uid
        """,
    },
    {
        "name": "embedding_failed_without_timestamp",
        "title": "Sections with failed embedding but no timestamp",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "WARNING",
        "query": """
            MATCH (s:Section)
            WITH s, properties(s) AS section_props
            WHERE section_props['embedding_status'] = 'failed'
              AND section_props['embedding_failed_at'] IS NULL
            RETURN s.uid AS uid
            ORDER BY uid
        """,
    },
    {
        "name": "eligible_sections_missing_embeddings",
        "title": "Eligible sections still missing embeddings",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)
            WITH s, properties(s) AS section_props
            WHERE coalesce(s.embed, false) = true
              AND section_props['embedding'] IS NULL
            RETURN DISTINCT s.uid AS uid,
                   s.doc_id AS doc_id,
                   section_props['embedding_status'] AS embedding_status
            ORDER BY doc_id, uid
        """,
    },
]



# -------------------------------------------------------------------------
# Retrieval Section-view checks and role-aware overrides
# -------------------------------------------------------------------------

SECTION_VIEW_CHECKS: List[Dict[str, Any]] = [
    {
        "name": "documents_missing_section_view_metadata",
        "title": "Documents missing active Section-view metadata",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "params": {
            "required_fields": [
                "retrieval_strategy",
                "aggregation_mode",
                "section_view_schema_version",
                "section_view_file",
                "section_view_section_count",
                "section_view_retrieval_count",
                "section_view_structural_count",
                "section_view_aggregated_count",
                "section_view_source_section_count",
            ]
        },
        "query": """
            MATCH (d:Document)
            WITH d, properties(d) AS props
            WITH d,
                 [field IN $required_fields
                  WHERE props[field] IS NULL] AS missing_fields
            WHERE size(missing_fields) > 0
            RETURN d.doc_id AS doc_id, missing_fields
            ORDER BY doc_id
        """,
    },
    {
        "name": "document_section_view_count_mismatches",
        "title": "Document Section-view metadata counts do not match loaded nodes",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
            WITH d,
                 count(s) AS actual_sections,
                 sum(CASE
                     WHEN s.section_view_role = 'retrieval' THEN 1
                     ELSE 0
                 END) AS actual_retrieval,
                 sum(CASE
                     WHEN s.section_view_role = 'structural' THEN 1
                     ELSE 0
                 END) AS actual_structural,
                 sum(CASE
                     WHEN coalesce(s.is_aggregated, false) THEN 1
                     ELSE 0
                 END) AS actual_aggregated,
                 sum(CASE
                     WHEN s.section_view_role = 'retrieval'
                     THEN coalesce(s.source_count, 0)
                     ELSE 0
                 END) AS actual_source_sections
            WHERE coalesce(d.section_view_section_count, -1) <> actual_sections
               OR coalesce(d.section_view_retrieval_count, -1) <> actual_retrieval
               OR coalesce(d.section_view_structural_count, -1) <> actual_structural
               OR coalesce(d.section_view_aggregated_count, -1) <> actual_aggregated
               OR coalesce(d.section_view_source_section_count, -1)
                  <> actual_source_sections
            RETURN d.doc_id AS doc_id,
                   d.section_view_section_count AS expected_sections,
                   actual_sections,
                   d.section_view_retrieval_count AS expected_retrieval,
                   actual_retrieval,
                   d.section_view_structural_count AS expected_structural,
                   actual_structural,
                   d.section_view_aggregated_count AS expected_aggregated,
                   actual_aggregated,
                   d.section_view_source_section_count AS expected_source_sections,
                   actual_source_sections
            ORDER BY doc_id
        """,
    },
    {
        "name": "sections_with_invalid_section_view_role",
        "title": "Sections with invalid or missing section_view_role",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.section_view_role IS NULL
               OR NOT (s.section_view_role IN ['retrieval', 'structural'])
            RETURN s.uid AS uid,
                   s.doc_id AS doc_id,
                   s.section_id AS section_id,
                   s.section_view_role AS section_view_role
            ORDER BY doc_id, uid
        """,
    },
    {
        "name": "section_strategy_mismatches_document",
        "title": "Section strategy metadata differs from its Document",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
            WHERE coalesce(toString(s.retrieval_strategy), '') <>
                  coalesce(toString(d.retrieval_strategy), '')
               OR coalesce(toString(s.aggregation_mode), '') <>
                  coalesce(toString(d.aggregation_mode), '')
               OR coalesce(toString(s.aggregation_max_level), '') <>
                  coalesce(toString(d.retrieval_max_level), '')
               OR coalesce(toString(s.section_view_schema_version), '') <>
                  coalesce(toString(d.section_view_schema_version), '')
            RETURN d.doc_id AS doc_id,
                   s.uid AS uid,
                   d.retrieval_strategy AS document_strategy,
                   s.retrieval_strategy AS section_strategy,
                   d.retrieval_max_level AS document_max_level,
                   s.aggregation_max_level AS section_max_level,
                   d.aggregation_mode AS document_aggregation_mode,
                   s.aggregation_mode AS section_aggregation_mode
            ORDER BY doc_id, uid
        """,
    },
    {
        "name": "retrieval_section_invariant_violations",
        "title": "Retrieval Sections violating content or provenance invariants",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.section_view_role = 'retrieval'
              AND (
                    coalesce(trim(s.text), '') = ''
                 OR coalesce(s.embed, false) = false
                 OR coalesce(s.is_empty, true) = true
                 OR coalesce(s.excluded, false) = true
                 OR s.retrieval_order IS NULL
                 OR s.section_view_order IS NULL
                 OR s.retrieval_unit_id IS NULL
                 OR s.content_owner_section_id IS NULL
                 OR s.content_owner_section_id <> s.section_id
                 OR s.source_section_ids IS NULL
                 OR size(s.source_section_ids) = 0
                 OR s.source_count IS NULL
                 OR s.source_count <> size(coalesce(s.source_section_ids, []))
                 OR size(coalesce(s.source_chunk_ids, [])) <>
                    size(coalesce(s.source_section_ids, []))
                 OR s.represented_section_count IS NULL
                 OR s.represented_section_count <>
                    size(coalesce(s.represented_section_ids, []))
                 OR NOT (s.section_id IN coalesce(s.represented_section_ids, []))
                 OR NOT all(
                        source_id IN coalesce(s.source_section_ids, [])
                        WHERE source_id IN coalesce(s.represented_section_ids, [])
                    )
              )
            RETURN s.uid AS uid,
                   s.section_id AS section_id,
                   s.embed AS embed,
                   s.is_empty AS is_empty,
                   s.excluded AS excluded,
                   s.retrieval_order AS retrieval_order,
                   s.source_count AS source_count,
                   size(coalesce(s.source_section_ids, [])) AS source_id_count,
                   s.represented_section_count AS represented_section_count,
                   size(coalesce(s.represented_section_ids, []))
                       AS represented_id_count
            ORDER BY s.doc_id, s.retrieval_order, uid
        """,
    },
    {
        "name": "structural_section_invariant_violations",
        "title": "Structural Sections violating hierarchy-only invariants",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.section_view_role = 'structural'
              AND (
                    coalesce(trim(s.text), '') <> ''
                 OR coalesce(s.embed, true) = true
                 OR coalesce(s.is_empty, false) = false
                 OR coalesce(s.excluded, false) = true
                 OR coalesce(s.is_aggregated, false) = true
                 OR s.retrieval_order IS NOT NULL
                 OR s.retrieval_unit_id IS NOT NULL
                 OR s.content_owner_section_id IS NOT NULL
                 OR size(coalesce(s.source_section_ids, [])) > 0
                 OR size(coalesce(s.source_chunk_ids, [])) > 0
                 OR size(coalesce(s.represented_section_ids, [])) > 0
                 OR size(coalesce(s.absorbed_section_ids, [])) > 0
                 OR size(coalesce(s.absorbed_source_section_ids, [])) > 0
              )
            RETURN s.uid AS uid,
                   s.section_id AS section_id,
                   size(coalesce(s.text, '')) AS text_chars,
                   s.embed AS embed,
                   s.is_empty AS is_empty,
                   s.is_aggregated AS is_aggregated,
                   s.retrieval_order AS retrieval_order
            ORDER BY s.doc_id, s.section_view_order, uid
        """,
    },
    {
        "name": "section_view_order_sequence_mismatches",
        "title": "Section-view order is not unique and contiguous",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
            WITH d,
                 count(s) AS section_count,
                 count(DISTINCT s.section_view_order) AS distinct_orders,
                 min(s.section_view_order) AS min_order,
                 max(s.section_view_order) AS max_order
            WHERE distinct_orders <> section_count
               OR min_order <> 0
               OR max_order <> section_count - 1
            RETURN d.doc_id AS doc_id,
                   section_count,
                   distinct_orders,
                   min_order,
                   max_order
            ORDER BY doc_id
        """,
    },
    {
        "name": "retrieval_order_sequence_mismatches",
        "title": "Retrieval order is not unique and contiguous",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (d:Document)-[:HAS_SECTION]->(s:Section)
            WHERE s.section_view_role = 'retrieval'
            WITH d,
                 count(s) AS retrieval_count,
                 count(DISTINCT s.retrieval_order) AS distinct_orders,
                 min(s.retrieval_order) AS min_order,
                 max(s.retrieval_order) AS max_order
            WHERE distinct_orders <> retrieval_count
               OR min_order <> 0
               OR max_order <> retrieval_count - 1
            RETURN d.doc_id AS doc_id,
                   retrieval_count,
                   distinct_orders,
                   min_order,
                   max_order
            ORDER BY doc_id
        """,
    },
    {
        "name": "parent_section_relationship_mismatches",
        "title": "parent_section_id values without the matching HAS_CHILD edge",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (child:Section)
            WHERE child.parent_section_id IS NOT NULL
            OPTIONAL MATCH
                (parent:Section {
                    doc_id: child.doc_id,
                    section_id: child.parent_section_id
                })-[r:HAS_CHILD]->(child)
            WITH child, parent, r
            WHERE parent IS NULL OR r IS NULL
            RETURN child.uid AS child_uid,
                   child.parent_section_id AS expected_parent_section_id
            ORDER BY child.doc_id, child.section_view_order
        """,
    },
    {
        "name": "has_child_relationship_property_mismatches",
        "title": "HAS_CHILD edges inconsistent with child parent_section_id",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (parent:Section)-[:HAS_CHILD]->(child:Section)
            WHERE parent.doc_id <> child.doc_id
               OR child.parent_section_id IS NULL
               OR child.parent_section_id <> parent.section_id
            RETURN parent.uid AS parent_uid,
                   child.uid AS child_uid,
                   child.parent_section_id AS expected_parent_section_id
            ORDER BY parent_uid, child_uid
        """,
    },
    {
        "name": "next_role_or_order_mismatches",
        "title": "NEXT edges inconsistent with retrieval roles or order",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (a:Section)-[:NEXT]->(b:Section)
            WHERE a.section_view_role <> 'retrieval'
               OR b.section_view_role <> 'retrieval'
               OR a.doc_id <> b.doc_id
               OR a.retrieval_order IS NULL
               OR b.retrieval_order IS NULL
               OR b.retrieval_order <> a.retrieval_order + 1
            RETURN a.uid AS from_uid,
                   a.section_view_role AS from_role,
                   a.retrieval_order AS from_order,
                   b.uid AS to_uid,
                   b.section_view_role AS to_role,
                   b.retrieval_order AS to_order
            ORDER BY from_uid, to_uid
        """,
    },
    {
        "name": "next_count_mismatches",
        "title": "NEXT relationship count differs from retrieval_count - 1",
        "group": "Hierarchy",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:HAS_SECTION]->(s:Section)
            WITH d,
                 sum(CASE
                     WHEN s.section_view_role = 'retrieval' THEN 1
                     ELSE 0
                 END) AS retrieval_count
            OPTIONAL MATCH
                (:Section {doc_id: d.doc_id})
                -[r:NEXT]->
                (:Section {doc_id: d.doc_id})
            WITH d, retrieval_count, count(r) AS next_count
            WHERE next_count <>
                  CASE
                      WHEN retrieval_count > 0 THEN retrieval_count - 1
                      ELSE 0
                  END
            RETURN d.doc_id AS doc_id,
                   retrieval_count,
                   next_count,
                   CASE
                       WHEN retrieval_count > 0 THEN retrieval_count - 1
                       ELSE 0
                   END AS expected_next_count
            ORDER BY doc_id
        """,
    },
    {
        "name": "aggregated_section_provenance_violations",
        "title": "Aggregated retrieval Sections with invalid provenance",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE coalesce(s.is_aggregated, false) = true
              AND (
                    s.section_view_role <> 'retrieval'
                 OR size(coalesce(s.absorbed_section_ids, [])) = 0
                 OR size(coalesce(s.source_section_ids, [])) = 0
                 OR NOT all(
                        absorbed_id IN coalesce(s.absorbed_section_ids, [])
                        WHERE absorbed_id IN
                              coalesce(s.represented_section_ids, [])
                    )
                 OR NOT all(
                        absorbed_source_id IN
                            coalesce(s.absorbed_source_section_ids, [])
                        WHERE absorbed_source_id IN
                              coalesce(s.source_section_ids, [])
                    )
              )
            RETURN s.uid AS uid,
                   s.section_id AS section_id,
                   s.section_view_role AS role,
                   s.source_section_ids AS source_section_ids,
                   s.represented_section_ids AS represented_section_ids,
                   s.absorbed_section_ids AS absorbed_section_ids,
                   s.absorbed_source_section_ids AS
                       absorbed_source_section_ids
            ORDER BY s.doc_id, s.section_view_order
        """,
    },
    {
        "name": "absorbed_sections_still_materialized",
        "title": "Absorbed Section IDs still present as active nodes",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (owner:Section)
            WHERE coalesce(owner.is_aggregated, false) = true
            UNWIND coalesce(owner.absorbed_section_ids, []) AS absorbed_id
            MATCH (absorbed:Section {
                doc_id: owner.doc_id,
                section_id: absorbed_id
            })
            RETURN owner.uid AS owner_uid,
                   absorbed.uid AS absorbed_uid,
                   absorbed_id
            ORDER BY owner_uid, absorbed_id
        """,
    },
    {
        "name": "nonaggregated_retrieval_provenance_violations",
        "title": "Non-aggregated retrieval Sections with expanded provenance",
        "group": "Retrieval Section view",
        "phases": {"structure"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.section_view_role = 'retrieval'
              AND coalesce(s.is_aggregated, false) = false
              AND (
                    coalesce(s.source_section_ids, []) <> [s.section_id]
                 OR coalesce(s.represented_section_ids, []) <> [s.section_id]
                 OR size(coalesce(s.absorbed_section_ids, [])) > 0
                 OR size(coalesce(s.absorbed_source_section_ids, [])) > 0
              )
            RETURN s.uid AS uid,
                   s.section_id AS section_id,
                   s.source_section_ids AS source_section_ids,
                   s.represented_section_ids AS represented_section_ids,
                   s.absorbed_section_ids AS absorbed_section_ids,
                   s.absorbed_source_section_ids AS
                       absorbed_source_section_ids
            ORDER BY s.doc_id, s.section_view_order
        """,
    },
    {
        "name": "structural_sections_with_entity_state_or_mentions",
        "title": "Structural Sections carrying entity state or MENTIONS",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WHERE s.section_view_role = 'structural'
            OPTIONAL MATCH (s)-[r:MENTIONS]->(:Concept)
            WITH s, count(r) AS mention_count
            WHERE mention_count > 0
               OR coalesce(s.entity_extracted, false) = true
               OR s.entity_extraction_status IS NOT NULL
               OR s.entity_extracted_at IS NOT NULL
               OR s.entity_extraction_failed_at IS NOT NULL
            RETURN s.uid AS uid,
                   mention_count,
                   s.entity_extracted AS entity_extracted,
                   s.entity_extraction_status AS entity_extraction_status
            ORDER BY s.doc_id, s.section_view_order
        """,
    },
    {
        "name": "retrieval_sections_without_entity_terminal_state",
        "title": "Retrieval Sections without a terminal entity-extraction state",
        "group": "Entity extraction state",
        "phases": {"entities"},
        "level": "INFO",
        "query": """
            MATCH (s:Section)
            WHERE s.section_view_role = 'retrieval'
              AND (
                    s.entity_extraction_status IS NULL
                 OR NOT (
                        s.entity_extraction_status IN [
                            'success',
                            'failed',
                            'skipped_empty'
                        ]
                    )
              )
            RETURN s.uid AS uid,
                   s.doc_id AS doc_id,
                   s.entity_extracted AS entity_extracted,
                   s.entity_extraction_status AS entity_extraction_status
            ORDER BY s.doc_id, s.retrieval_order
        """,
    },
    {
        "name": "structural_sections_with_embedding_state",
        "title": "Structural Sections carrying embedding vectors or status",
        "group": "Embedding state",
        "phases": {"embeddings"},
        "level": "ERROR",
        "query": """
            MATCH (s:Section)
            WITH s, properties(s) AS props
            WHERE s.section_view_role = 'structural'
              AND (
                    coalesce(props['has_embedding'], false) = true
                 OR props['embedding'] IS NOT NULL
                 OR props['embedding_status'] IS NOT NULL
                 OR props['embedding_model'] IS NOT NULL
                 OR props['embedding_dim'] IS NOT NULL
                 OR props['embedding_updated_at'] IS NOT NULL
                 OR props['embedding_failed_at'] IS NOT NULL
              )
            RETURN s.uid AS uid,
                   props['has_embedding'] AS has_embedding,
                   props['embedding_status'] AS embedding_status,
                   props['embedding_model'] AS embedding_model
            ORDER BY s.doc_id, s.section_view_order
        """,
    },
]


def _apply_section_view_overrides(
    checks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Make legacy checks role-aware without duplicating the large historical list.
    """
    overrides: Dict[str, Dict[str, Any]] = {
        "sections_with_empty_body_text": {
            "title": "Retrieval Sections with empty body text",
            "level": "ERROR",
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                  AND coalesce(trim(s.text), '') = ''
                RETURN s.uid AS uid, s.title AS title
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
        "empty_leaf_sections": {
            "title": "Structural leaf Sections",
            "level": "WARNING",
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'structural'
                  AND NOT EXISTS {
                      MATCH (s)-[:HAS_CHILD]->(:Section)
                  }
                RETURN s.uid AS uid, s.title AS title
                ORDER BY s.doc_id, s.section_view_order
            """,
        },
        "sections_missing_entity_extracted_flag": {
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                  AND s.entity_extracted IS NULL
                RETURN s.uid AS uid,
                       s.doc_id AS doc_id,
                       s.title AS title
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
        "entity_extraction_status_summary": {
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                RETURN coalesce(
                           s.entity_extraction_status,
                           'UNSET'
                       ) AS status,
                       count(s) AS n
                ORDER BY n DESC, status ASC
            """,
        },
        "entity_status_success_but_not_extracted": {
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                  AND s.entity_extraction_status = 'success'
                  AND coalesce(s.entity_extracted, false) = false
                RETURN s.uid AS uid,
                       s.entity_extraction_status AS status,
                       s.entity_extracted AS entity_extracted
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
        "entity_extracted_but_missing_status": {
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                  AND coalesce(s.entity_extracted, false) = true
                  AND s.entity_extraction_status IS NULL
                RETURN s.uid AS uid
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
        "entity_failed_without_timestamp": {
            "query": """
                MATCH (s:Section)
                WITH s, properties(s) AS section_props
                WHERE s.section_view_role = 'retrieval'
                  AND s.entity_extraction_status = 'failed'
                  AND section_props['entity_extraction_failed_at'] IS NULL
                RETURN s.uid AS uid
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
        "sections_missing_has_embedding_flag": {
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                  AND s.has_embedding IS NULL
                RETURN s.uid AS uid
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
        "embedding_status_summary": {
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                WITH s, properties(s) AS section_props
                RETURN coalesce(
                           section_props['embedding_status'],
                           'UNSET'
                       ) AS status,
                       count(s) AS n
                ORDER BY n DESC, status ASC
            """,
        },
        "embedding_flag_inconsistencies": {
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                WITH s, properties(s) AS section_props
                WHERE (
                        coalesce(
                            section_props['has_embedding'],
                            false
                        ) = true
                        AND section_props['embedding'] IS NULL
                      )
                   OR (
                        coalesce(
                            section_props['has_embedding'],
                            false
                        ) = false
                        AND section_props['embedding'] IS NOT NULL
                      )
                RETURN s.uid AS uid,
                       section_props['has_embedding'] AS has_embedding,
                       section_props['embedding'] IS NOT NULL
                           AS has_embedding_vector
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
        "embedding_metadata_inconsistencies": {
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                WITH s, properties(s) AS section_props
                WHERE section_props['embedding'] IS NOT NULL
                  AND (
                        section_props['embedding_dim'] IS NULL
                     OR section_props['embedding_model'] IS NULL
                     OR section_props['embedding_updated_at'] IS NULL
                     OR size(section_props['embedding']) <>
                        section_props['embedding_dim']
                  )
                RETURN s.uid AS uid,
                       section_props['embedding_dim'] AS embedding_dim,
                       size(section_props['embedding']) AS actual_dim,
                       section_props['embedding_model'] AS embedding_model,
                       section_props['embedding_updated_at']
                           AS embedding_updated_at
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
        "embedding_status_success_but_no_vector": {
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                WITH s, properties(s) AS section_props
                WHERE section_props['embedding_status'] = 'success'
                  AND (
                        section_props['embedding'] IS NULL
                     OR coalesce(
                            section_props['has_embedding'],
                            false
                        ) = false
                  )
                RETURN s.uid AS uid,
                       section_props['embedding_status'] AS status,
                       section_props['has_embedding'] AS has_embedding
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
        "embedding_failed_without_timestamp": {
            "query": """
                MATCH (s:Section)
                WHERE s.section_view_role = 'retrieval'
                WITH s, properties(s) AS section_props
                WHERE section_props['embedding_status'] = 'failed'
                  AND section_props['embedding_failed_at'] IS NULL
                RETURN s.uid AS uid
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
        "eligible_sections_missing_embeddings": {
            "query": """
                MATCH (s:Section)
                WITH s, properties(s) AS section_props
                WHERE s.section_view_role = 'retrieval'
                  AND coalesce(s.embed, false) = true
                  AND section_props['embedding'] IS NULL
                RETURN DISTINCT s.uid AS uid,
                       s.doc_id AS doc_id,
                       section_props['embedding_status']
                           AS embedding_status
                ORDER BY s.doc_id, s.retrieval_order
            """,
        },
    }

    seen_names = {check["name"] for check in checks}
    missing = sorted(set(overrides) - seen_names)
    if missing:
        raise RuntimeError(
            "Sanity-check override targets are missing: "
            + ", ".join(missing)
        )

    for check in checks:
        override = overrides.get(check["name"])
        if override:
            check.update(override)

    return checks


CHECKS = SECTION_VIEW_CHECKS + _apply_section_view_overrides(CHECKS)


def _log_with_level(level: str, message: str, *args) -> None:
    """
    Log a message using the requested severity level.
    """
    level = level.upper()

    if level == "ERROR":
        logger.error(message, *args)
    elif level == "WARNING":
        logger.warning(message, *args)
    else:
        logger.info(message, *args)


def _normalize_mode(mode: str) -> str:
    """
    Validate and normalize the requested sanity-check mode.
    """
    normalized = mode.lower().strip()
    if normalized not in ALLOWED_MODES:
        raise ValueError(
            f"Invalid sanity check mode: {mode!r}. Allowed values: {sorted(ALLOWED_MODES)}"
        )
    return normalized


def _build_count_query(query: str) -> str:
    """
    Wrap a check query so it returns only the total number of matching rows.
    """
    return f"""
        CALL () {{
            {query.strip()}
        }}
        RETURN count(*) AS n
    """


def _build_sample_query(query: str) -> str:
    """
    Wrap a check query so it returns only a limited sample of matching rows.
    """
    return f"""
        CALL () {{
            {query.strip()}
        }}
        RETURN *
        LIMIT $sample_limit
    """


def _run_check(tx, check: Dict[str, Any], sample_limit: int) -> Dict[str, Any]:
    """
    Execute a single check and return a structured result.

    Each check is run once for the exact count and, only when useful,
    once more to fetch a limited sample for logging.
    """
    query_params = dict(check.get("params") or {})

    count_query = check.get("count_query") or _build_count_query(check["query"])
    sample_query = check.get("sample_query") or _build_sample_query(check["query"])

    count_record = tx.run(count_query, **query_params).single()
    count = int(count_record["n"]) if count_record is not None else 0

    is_summary = bool(check.get("is_summary", False))

    sample: List[Dict[str, Any]] = []
    if sample_limit > 0 and (is_summary or count > 0):
        sample_params = dict(query_params)
        sample_params["sample_limit"] = sample_limit

        sample_rows = list(tx.run(sample_query, **sample_params))
        sample = [dict(row) for row in sample_rows]

    return {
        "name": check["name"],
        "title": check["title"],
        "group": check["group"],
        "level": check["level"],
        "count": count,
        "sample": sample,
        "is_summary": is_summary,
    }


def run_sanity_checks(
    driver: Driver,
    mode: str = "full",
    sample_limit: int = 10,
    log_samples: bool = True,
) -> Dict[str, Any]:
    """
    Run graph sanity checks for the requested pipeline phase.

    Parameters:
        driver: active Neo4j driver
        mode: one of {"structure", "entities", "embeddings", "full"}
        sample_limit: maximum number of sample rows to keep/log per check
        log_samples: whether to log sample rows

    Returns:
        Structured summary dictionary with per-check results and aggregate counters.
    """
    mode = _normalize_mode(mode)

    if sample_limit < 0:
        raise ValueError("sample_limit must be >= 0")

    included_phases = PHASE_EXPANSION[mode]

    checks_to_run = [
        check for check in CHECKS
        if bool(check["phases"] & included_phases)
    ]

    results: List[Dict[str, Any]] = []

    with driver.session() as session:
        for check in checks_to_run:
            result = session.execute_read(_run_check, check, sample_limit)
            results.append(result)

    summary = {
        "mode": mode,
        "total_checks": len(results),
        "checks_with_issues": 0,
        "error_checks_with_issues": 0,
        "warning_checks_with_issues": 0,
        "info_checks_with_issues": 0,
        "results": results,
    }

    current_group = None

    for result in results:
        if result["group"] != current_group:
            current_group = result["group"]
            logger.info("Running %s checks", current_group)

        if result["is_summary"]:
            logger.info("%s", result["title"])
            if log_samples:
                for row in result["sample"]:
                    logger.info("  -> %s", row)
            continue

        if result["count"] > 0:
            summary["checks_with_issues"] += 1

            level = result["level"].upper()
            if level == "ERROR":
                summary["error_checks_with_issues"] += 1
            elif level == "WARNING":
                summary["warning_checks_with_issues"] += 1
            else:
                summary["info_checks_with_issues"] += 1

            _log_with_level(
                result["level"],
                "%s (%d issues)",
                result["title"],
                result["count"],
            )

            if log_samples:
                for row in result["sample"]:
                    _log_with_level(result["level"], "  -> %s", row)
        else:
            logger.info("%s: OK", result["title"])

    logger.info(
        "Sanity checks completed | mode=%s | total=%d | issue_checks=%d | error_checks=%d | warning_checks=%d | info_checks=%d",
        summary["mode"],
        summary["total_checks"],
        summary["checks_with_issues"],
        summary["error_checks_with_issues"],
        summary["warning_checks_with_issues"],
        summary["info_checks_with_issues"],
    )

    return summary