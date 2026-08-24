import json
from pathlib import Path

import pytest

from knowledge_graph.umls_frozen_materialization import (
    FrozenMaterializationContract,
    _scope_cuis,
    _validate_unique_payload_rows,
    build_frozen_materialization_rows,
)


def contract(tmp_path: Path):
    manifest = {
        "direct": {
            "policy_sha256": "d" * 64,
            "manifest_sha256": "a" * 64,
            "pair_evidence_sha256": "b" * 64,
            "pair_evidence_path": "direct/pair_evidence.jsonl",
        },
        "bridge": {
            "policy_sha256": "e" * 64,
            "manifest_sha256": "c" * 64,
            "pair_evidence_sha256": "f" * 64,
            "pair_evidence_path": "bridge/pair_evidence.jsonl",
        },
    }
    direct = {
        "local_cui_a": "C0000001",
        "local_cui_b": "C0000002",
        "overall_tier": "MEDIUM",
        "direct_id": "D1",
        "sources": ["NCI"],
        "projection_ambiguity": False,
        "source_evidence": [
            {
                "relation_names": ["enzyme_metabolizes_chemical_or_drug"],
                "relation_families": [{"relation_family": "drug_metabolism"}],
            }
        ],
    }
    bridge = {
        "local_cui_a": "C0000002",
        "local_cui_b": "C0000003",
        "overall_tier": "STRONG",
        "retained_for_retrieval": True,
        "bridge_id": "B1",
        "bridge_score_v1_1": 0.75,
        "score_top_k": 5,
        "retained_external_hub_count": 1,
        "all_external_hub_count": 2,
        "retrieval_sources": ["SNOMEDCT_US"],
        "profile_scores_v1_1": {
            "balanced": {"score": 0.75, "top_k": 5, "sources": ["SNOMEDCT_US"]}
        },
        "external_path_evidence": [
            {
                "retained_for_retrieval": True,
                "relation_pair_evidence": [
                    {"tier": "STRONG", "policy_rule_id": "specific_rule"}
                ],
            }
        ],
    }
    return FrozenMaterializationContract(
        manifest_path=tmp_path / "manifest.json",
        manifest_sha256="1" * 64,
        manifest=manifest,
        scope_path=tmp_path / "scope.json",
        scope_cuis=("C0000001", "C0000002", "C0000003"),
        document_ids=("D1", "D2"),
        direct_payload_path=tmp_path / "direct.jsonl",
        bridge_payload_path=tmp_path / "bridge.jsonl",
        direct_rows=(direct,),
        bridge_rows=(bridge,),
    )


def test_build_rows_are_cui_pair_level_and_keep_frozen_provenance(tmp_path):
    direct, bridge = build_frozen_materialization_rows(contract(tmp_path))
    assert direct[0]["cui_a"] == "C0000001"
    assert direct[0]["cui_b"] == "C0000002"
    assert direct[0]["edge_key"] == "UMLS_DIRECT::C0000001::C0000002"
    assert direct[0]["properties"]["tier"] == "MEDIUM"
    assert direct[0]["properties"]["relation_families"] == ["drug_metabolism"]
    assert direct[0]["properties"]["policy_sha256"] == "d" * 64

    assert bridge[0]["edge_key"] == "UMLS_BRIDGE::C0000002::C0000003"
    assert bridge[0]["properties"]["score"] == pytest.approx(0.75)
    assert bridge[0]["properties"]["policy_rule_ids"] == ["specific_rule"]
    assert bridge[0]["properties"]["profile"] == "balanced"


def test_payload_validation_rejects_weak_edges():
    rows = [
        {
            "local_cui_a": "C0000001",
            "local_cui_b": "C0000002",
            "overall_tier": "WEAK",
        }
    ]
    with pytest.raises(ValueError, match="non-materializable tier"):
        _validate_unique_payload_rows(rows, kind="DIRECT")


def test_payload_validation_rejects_noncanonical_pair_direction():
    rows = [
        {
            "local_cui_a": "C0000002",
            "local_cui_b": "C0000001",
            "overall_tier": "MEDIUM",
        }
    ]
    with pytest.raises(ValueError, match="canonical lexicographic order"):
        _validate_unique_payload_rows(rows, kind="DIRECT")


def test_payload_validation_rejects_duplicate_pairs():
    rows = [
        {"local_cui_a": "C0000001", "local_cui_b": "C0000002", "overall_tier": "MEDIUM"},
        {"local_cui_a": "C0000001", "local_cui_b": "C0000002", "overall_tier": "STRONG"},
    ]
    with pytest.raises(ValueError, match="Duplicate DIRECT pair"):
        _validate_unique_payload_rows(rows, kind="DIRECT")


def test_artifact_workflow_dispatches_materialize_frozen(monkeypatch, tmp_path):
    import knowledge_graph.umls_frozen_materialization as frozen
    from knowledge_graph.umls_relation_artifacts import run_umls_relation_artifact_workflow

    seen = {}

    def fake_materialize(driver, **kwargs):
        seen["driver"] = driver
        seen.update(kwargs)
        return {"neo4j_writes": True, "status": "MATERIALIZED_PASS"}

    monkeypatch.setattr(frozen, "run_frozen_relation_materialization", fake_materialize)
    driver = object()
    result = run_umls_relation_artifact_workflow(
        driver,
        project_root=tmp_path,
        work_root=tmp_path / "work",
        action="materialize_frozen",
        materialization_freeze_config={"manifest_path": "freeze.json"},
        write_neo4j=True,
        replace_existing_connections=True,
    )
    assert result["neo4j_writes"] is True
    assert result["materialization"]["status"] == "MATERIALIZED_PASS"
    assert seen["driver"] is driver
    assert seen["write_neo4j"] is True
    assert seen["replace_existing_connections"] is True

def test_scope_cuis_accepts_current_local_scope_record_schema():
    scope = {
        "schema_version": "local_umls_scope_v1",
        "unique_cui_count": 3,
        "cuis": [
            {"cui": "C0001206", "names": ["alpha"]},
            {"cui": "C0001403", "names": ["beta"]},
            {"cui": "C1142644", "names": ["CYP3A4"]},
        ],
    }
    assert _scope_cuis(scope) == (
        "C0001206",
        "C0001403",
        "C1142644",
    )


def test_scope_cuis_keeps_legacy_string_schema_compatible():
    scope = {
        "unique_cui_count": 2,
        "cuis": ["C0001206", "c1142644"],
    }
    assert _scope_cuis(scope) == ("C0001206", "C1142644")


def test_scope_cuis_rejects_record_without_cui():
    scope = {
        "unique_cui_count": 1,
        "cuis": [{"names": ["missing id"]}],
    }
    with pytest.raises(ValueError, match="missing required 'cui'"):
        _scope_cuis(scope)

