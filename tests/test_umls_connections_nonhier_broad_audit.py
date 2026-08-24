from knowledge_graph.umls_connections import (
    AUDIT_ALL_SNOMED_RELATION_NAMES,
    BROAD_NONHIER_AUDIT_RELATION_NAMES,
    BROAD_NONHIER_EXTRA_RELATION_NAMES,
    RELATION_NAMES_BY_PROFILE,
    raw_relation_names_for_canonical_names,
)


def test_nonhier_broad_profile_excludes_hierarchy():
    profile = RELATION_NAMES_BY_PROFILE["nonhier_broad_audit"]
    assert "isa" not in profile
    raw = set(raw_relation_names_for_canonical_names(sorted(profile)))
    assert "isa" not in raw
    assert "inverse_isa" not in raw


def test_nonhier_broad_profile_superset_of_known_nonhier_relations():
    profile = RELATION_NAMES_BY_PROFILE["nonhier_broad_audit"]
    known_nonhier = set(AUDIT_ALL_SNOMED_RELATION_NAMES) - {"isa"}
    assert known_nonhier <= set(profile)


def test_nonhier_broad_profile_contains_new_audit_labels():
    profile = RELATION_NAMES_BY_PROFILE["nonhier_broad_audit"]
    for name in {
        "associated_with",
        "component_of",
        "has_component",
        "has_finding_context",
        "has_intent",
        "has_onset",
        "occurs_before",
        "occurs_after",
        "part_of",
        "uses_substance",
    }:
        assert name in profile
        assert name in BROAD_NONHIER_EXTRA_RELATION_NAMES


def test_nonhier_broad_profile_omits_equivalence_mapping_and_history_labels():
    profile = RELATION_NAMES_BY_PROFILE["nonhier_broad_audit"]
    for name in {
        "same_as",
        "possibly_equivalent_to",
        "partially_equivalent_to",
        "mapped_from",
        "mapped_to",
        "replaced_by",
        "replaces",
        "was_a",
        "inverse_was_a",
    }:
        assert name not in profile
