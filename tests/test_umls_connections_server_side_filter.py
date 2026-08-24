from knowledge_graph import umls_connections as uc


def test_isa_server_filter_requests_forward_and_inverse_raw_labels():
    assert uc.raw_relation_names_for_canonical_names(["isa"]) == (
        "inverse_isa",
        "isa",
    )


def test_excluding_canonical_isa_removes_both_raw_isa_directions():
    include_names, exclude_names, profile = uc.resolve_relation_name_filters(
        relation_profile="explore_known",
        exclude_relation_names=["isa"],
    )
    assert profile == "explore_known"
    assert "isa" in exclude_names
    assert "isa" not in include_names

    raw_names = set(uc.raw_relation_names_for_canonical_names(sorted(include_names)))
    assert "isa" not in raw_names
    assert "inverse_isa" not in raw_names
    assert raw_names


def test_relation_page_params_pushes_rela_filter_to_umls_api():
    client = object.__new__(uc.UMLSRelationsClient)
    client.page_size = 200

    params = client.relation_page_params(
        cui="C0000001",
        source_vocab="SNOMEDCT_US",
        page_number=1,
        include_additional_relation_labels=["isa", "inverse_isa"],
    )

    assert params["sabs"] == "SNOMEDCT_US"
    assert params["includeAdditionalRelationLabels"] == "inverse_isa,isa"


def test_relation_cache_keys_are_scoped_by_server_side_filter():
    client = object.__new__(uc.UMLSRelationsClient)
    client.version = "2026AA"

    isa_key = client.relation_negative_cache_key(
        "C0000001",
        "SNOMEDCT_US",
        include_additional_relation_labels=["isa", "inverse_isa"],
    )
    morphology_key = client.relation_negative_cache_key(
        "C0000001",
        "SNOMEDCT_US",
        include_additional_relation_labels=["has_associated_morphology"],
    )

    assert isa_key != morphology_key
    assert isa_key["version"] == "2026AA"
    assert isa_key["include_additional_relation_labels"] == ["inverse_isa", "isa"]
