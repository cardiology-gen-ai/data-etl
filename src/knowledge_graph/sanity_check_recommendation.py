from typing import Any, Dict, List


RECOMMENDATION_CHECKS: List[Dict[str, Any]] = [

    # ----- presence / basic counts -----

    {
        "name": "recommendation_nodes_present",
        "title": "Recommendation nodes present",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "INFO",
        "query": """
            MATCH (r:Recommendation)
            RETURN count(r) AS recommendation_count
        """,
    },

    {
        "name": "recommendation_modality_distribution",
        "title": "Recommendation modality distribution",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "INFO",
        "query": """
            MATCH (r:Recommendation)
            RETURN coalesce(r.modality, 'UNSET') AS modality, count(*) AS c
            ORDER BY c DESC
        """,
    },

    {
        "name": "recommendation_class_distribution",
        "title": "Recommendation Class distribution",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "INFO",
        "query": """
            MATCH (r:Recommendation)
            RETURN coalesce(r.class, 'UNSET') AS class, count(*) AS c
            ORDER BY c DESC
        """,
    },

    # ----- consistency / red flags -----

    {
        "name": "recommendation_missing_class_and_level",
        "title": "Recommendations missing both Class and Level",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "WARNING",
        "query": """
            MATCH (r:Recommendation)
            WHERE r.class IS NULL AND r.level IS NULL
            RETURN r.uid AS uid, r.doc_id AS doc_id, r.source_text AS source_text
            ORDER BY doc_id, uid
        """,
    },

    {
        "name": "recommendation_modality_class_mismatch",
        "title": "Recommendations whose LLM modality disagrees with parsed Class",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "WARNING",
        "query": """
            MATCH (r:Recommendation)
            WHERE 'modality_mismatch' IN coalesce(r.validation_flags, [])
            RETURN r.uid AS uid, r.doc_id AS doc_id,
                   r.class AS class, r.modality AS modality,
                   r.source_text AS source_text
            ORDER BY doc_id, uid
        """,
    },

    {
        "name": "recommendation_non_verbatim_spans",
        "title": "Recommendations with at least one non-verbatim entity span",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "WARNING",
        "query": """
            MATCH (r:Recommendation)
            WHERE any(f IN coalesce(r.validation_flags, [])
                     WHERE f STARTS WITH 'non_verbatim_')
            RETURN r.uid AS uid, r.doc_id AS doc_id,
                   r.validation_flags AS validation_flags
            ORDER BY doc_id, uid
        """,
    },

    {
        "name": "recommendation_orphan_no_concepts",
        "title": "Recommendations with no role edges to any Concept",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "WARNING",
        "query": """
            MATCH (r:Recommendation)
            WHERE NOT EXISTS {
                MATCH (r)-[]->(:Concept)
            }
            RETURN r.uid AS uid, r.doc_id AS doc_id, r.source_text AS source_text
            ORDER BY doc_id, uid
        """,
    },

    {
        "name": "recommendation_without_containing_section",
        "title": "Recommendations not attached to any Section",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "WARNING",
        "query": """
            MATCH (r:Recommendation)
            WHERE NOT EXISTS {
                MATCH (:Section)-[:CONTAINS_RECOMMENDATION]->(r)
            }
            RETURN r.uid AS uid, r.doc_id AS doc_id,
                   r.container_id AS container_id
            ORDER BY doc_id, uid
        """,
    },

    # ----- concept-side checks reachable via Recommendation edges -----

    {
        "name": "recommendation_concept_ambiguous_canonical_type",
        "title": "Concepts cited by recommendations with ambiguous canonical_type",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "WARNING",
        "query": """
            MATCH (r:Recommendation)-[]->(c:Concept)
            WHERE c.canonical_type IN ['ambiguous', 'no_supported_type']
            RETURN DISTINCT c.name AS name,
                   c.canonical_type AS canonical_type,
                   c.observed_types AS observed_types
            ORDER BY name
        """,
    },

    {
        "name": "recommendation_concept_without_umls_match",
        "title": "Concepts cited by recommendations not linked to UMLS",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "INFO",
        "query": """
            MATCH (r:Recommendation)-[]->(c:Concept)
            WHERE c.cui IS NULL
            RETURN DISTINCT c.name AS name, c.canonical_type AS canonical_type
            ORDER BY name
        """,
    },

    # ----- role / hint shape -----

    {
        "name": "recommendation_role_edge_distribution",
        "title": "Distribution of role-typed edges out of Recommendations",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "INFO",
        "query": """
            MATCH (r:Recommendation)-[e]->(:Concept)
            RETURN type(e) AS rel_type,
                   coalesce(e.role, '(implicit)') AS role,
                   count(*) AS c
            ORDER BY c DESC
        """,
    },

    {
        "name": "recommendation_role_semgroup_hint_mismatch",
        "title": "Role edges whose expected SemGroups do not match the linked Concept's UMLS SemGroups",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "WARNING",
        "query": """
            MATCH (r:Recommendation)-[e]->(c:Concept)
            WHERE e.expected_semgroups IS NOT NULL
              AND size(e.expected_semgroups) > 0
              AND c.umls_semantic_groups IS NOT NULL
              AND none(sg IN c.umls_semantic_groups
                       WHERE sg IN e.expected_semgroups)
            RETURN r.uid AS recommendation_uid,
                   type(e) AS rel_type,
                   e.role AS role,
                   c.name AS concept_name,
                   e.expected_semgroups AS expected_semgroups,
                   c.umls_semantic_groups AS actual_umls_semantic_groups
            ORDER BY recommendation_uid
        """,
    },

    # ----- section linking (recommendation -> section) -----

    {
        "name": "recommendation_section_link_strategy_distribution",
        "title": "Recommendation->Section link strategies (from SectionLinkingManager)",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "INFO",
        "query": """
            MATCH (:Section)-[e:CONTAINS_RECOMMENDATION]->(:Recommendation)
            RETURN coalesce(e.match_strategy, '(heuristic)') AS strategy,
                   count(*) AS c
            ORDER BY c DESC
        """,
    },

    {
        "name": "recommendation_section_link_title_fallback",
        "title": "Recommendations linked to a section only via title-fallback",
        "group": "Recommendations",
        "phases": {"recommendations"},
        "level": "WARNING",
        "query": """
            MATCH (:Section)-[e:CONTAINS_RECOMMENDATION]->(r:Recommendation)
            WHERE e.match_strategy = 'title_fallback'
            RETURN r.uid AS recommendation_uid,
                   e.target_section_title AS target_section_title
            ORDER BY recommendation_uid
        """,
    },
]