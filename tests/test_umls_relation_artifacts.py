from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from knowledge_graph.umls_relation_artifacts import (
    _final_bridge_audit_summary,
    _relation_cache_payload,
    _relation_cache_status,
    _relation_negative_cache_payload,
    _require_delta_discovery_complete,
    _ExternalLabelTransientError,
    _redact_api_key,
    _validate_delta_source_summary,
    build_delta_discovery_plan,
    build_current_bridge_evidence_root,
    build_external_label_plan,
    compare_local_umls_scopes,
    CurrentFinalBuildPaths,
    GeneralizedBridgeBuildPaths,
    DeltaDiscoveryPaths,
    HistoricalRegressionPaths,
    resolve_external_labels,
    run_umls_relation_artifact_workflow,
)


def cache_key(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LocalUmlsScopeTests(unittest.TestCase):
    def test_scope_delta_uses_cui_membership_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "previous.json"
            current = root / "current.json"
            previous.write_text(
                json.dumps({"cuis": [{"cui": "C1"}, {"cui": "C2"}, {"cui": "C4"}]}),
                encoding="utf-8",
            )
            current.write_text(
                json.dumps({"cuis": [{"cui": "C1"}, {"cui": "C2"}, {"cui": "C3"}]}),
                encoding="utf-8",
            )

            delta = compare_local_umls_scopes(current, previous)

            self.assertEqual(delta["current_cui_count"], 3)
            self.assertEqual(delta["previous_cui_count"], 3)
            self.assertEqual(delta["shared_cui_count"], 2)
            self.assertEqual(delta["new_cui_count"], 1)
            self.assertEqual(delta["retired_cui_count"], 1)
            self.assertEqual(delta["new_cuis"], ["C3"])
            self.assertEqual(delta["retired_cuis"], ["C4"])

    def test_relation_cache_status_mirrors_frozen_cache_keys(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            positive_payload = _relation_cache_payload(
                cui="C123",
                source="NCI",
                umls_version="2026AA",
                max_relations_per_cui=1000,
            )
            positive = root / "relations" / f"{cache_key(positive_payload)}.json"
            positive.parent.mkdir(parents=True)
            positive.write_text(json.dumps({"records": []}), encoding="utf-8")
            self.assertEqual(
                _relation_cache_status(
                    root,
                    cui="C123",
                    source="NCI",
                    umls_version="2026AA",
                    max_relations_per_cui=1000,
                ),
                "positive",
            )

            positive.unlink()
            negative_payload = _relation_negative_cache_payload(
                cui="C123", source="NCI", umls_version="2026AA"
            )
            negative = (
                root
                / "relations_negative"
                / f"{cache_key(negative_payload)}.json"
            )
            negative.parent.mkdir(parents=True)
            negative.write_text(
                json.dumps({"status": "source_vocab_relations_absent"}),
                encoding="utf-8",
            )
            self.assertEqual(
                _relation_cache_status(
                    root,
                    cui="C123",
                    source="NCI",
                    umls_version="2026AA",
                    max_relations_per_cui=1000,
                ),
                "negative",
            )

            negative.unlink()
            self.assertEqual(
                _relation_cache_status(
                    root,
                    cui="C123",
                    source="NCI",
                    umls_version="2026AA",
                    max_relations_per_cui=1000,
                ),
                "missing",
            )


    def test_delta_plan_reuses_historical_cache_and_plans_only_new_cui(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "current.json"
            previous = root / "previous.json"
            current.write_text(
                json.dumps(
                    {
                        "cuis": [
                            {"cui": "C1", "document_ids": ["CM"]},
                            {"cui": "C2", "document_ids": ["CO"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            previous.write_text(
                json.dumps({"cuis": [{"cui": "C1"}]}), encoding="utf-8"
            )

            profile = root / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "umls_version": "2026AA",
                        "sources": {"NCI": {"enabled": True}},
                    }
                ),
                encoding="utf-8",
            )

            direct_census = root / "direct_census"
            direct_census.mkdir()
            profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
            (direct_census / "manifest.json").write_text(
                json.dumps({"source_profile_sha256": profile_hash}),
                encoding="utf-8",
            )

            bridge_root = root / "bridge"
            (bridge_root / "NCI").mkdir(parents=True)
            (bridge_root / "NCI" / "summary.json").write_text(
                json.dumps(
                    {
                        "umls_version": "2026AA",
                        "inputs": {
                            "max_relations_per_cui": 1000,
                            "source_profile_sha256": profile_hash,
                        },
                    }
                ),
                encoding="utf-8",
            )

            regression = root / "regression.json"
            previous_hash = hashlib.sha256(previous.read_bytes()).hexdigest()
            regression.write_text(
                json.dumps(
                    {
                        "pass": True,
                        "safety": {"umls_api_calls": False, "neo4j_writes": False},
                        "bridge": {"historical_scope_sha256": previous_hash},
                    }
                ),
                encoding="utf-8",
            )

            cache = root / "cache"
            payload = _relation_cache_payload(
                cui="C1",
                source="NCI",
                umls_version="2026AA",
                max_relations_per_cui=1000,
            )
            cached = cache / "relations" / f"{cache_key(payload)}.json"
            cached.parent.mkdir(parents=True)
            cached.write_text(json.dumps({"records": []}), encoding="utf-8")

            bridge_script = root / "bridge_census.py"
            bridge_script.write_text("# test\n", encoding="utf-8")
            dummy = root / "dummy"
            dummy.mkdir()
            dummy_file = root / "dummy.json"
            dummy_file.write_text("{}", encoding="utf-8")

            historical = HistoricalRegressionPaths(
                direct_census_dir=direct_census,
                direct_policy=dummy_file,
                expected_direct_artifact_dir=dummy,
                bridge_root=bridge_root,
                historical_scope=previous,
                bridge_policy=dummy_file,
                expected_bridge_artifact_dir=dummy,
                external_label_map=None,
                direct_builder=bridge_script,
                bridge_builder=bridge_script,
            )
            discovery = DeltaDiscoveryPaths(
                source_profile=profile,
                relation_cache_dir=cache,
                bridge_census_script=bridge_script,
            )
            out = root / "plan.json"
            plan = build_delta_discovery_plan(
                current_scope_path=current,
                previous_scope_path=previous,
                regression_report_path=regression,
                historical_paths=historical,
                discovery_paths=discovery,
                sources=["NCI"],
                umls_version="2026AA",
                max_relations_per_cui=1000,
                output_path=out,
            )
            self.assertTrue(plan["ready_for_discovery"])
            self.assertEqual(plan["scope"]["new_cuis"], ["C2"])
            self.assertEqual(
                plan["relation_cache"]["new_top_level_relation_fetches_estimated"],
                1,
            )
            self.assertEqual(
                plan["relation_cache"]["coverage_by_source"]["NCI"][
                    "historical_missing_or_invalid_count"
                ],
                0,
            )

    def test_delta_completion_uses_per_source_summary_for_legacy_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            discovery = root / "delta_discovery_v1"
            source_dir = discovery / "NCI"
            source_dir.mkdir(parents=True)

            scope_path = root / "scope.json"
            scope_path.write_text(
                json.dumps({"cuis": [{"cui": "C1"}, {"cui": "C2"}]}),
                encoding="utf-8",
            )
            scope_sha = hashlib.sha256(scope_path.read_bytes()).hexdigest()

            summary = {
                "source_vocabulary": "NCI",
                "local_universe_count": 2,
                "processed_local_cui_count": 1,
                "umls_version": "2026AA",
                "fetch_failure_count": 0,
                "client_stats": {"api_errors": 0},
                "safety": {
                    "neo4j_writes": False,
                    "second_hop_requests": False,
                },
                "inputs": {
                    "local_universe_sha256": scope_sha,
                    "source_profile_sha256": "profilehash",
                    "max_relations_per_cui": 1000,
                },
            }
            (source_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            # Legacy Phase-B v1 aggregate entry deliberately omits
            # processed_local_cui_count and fetch_failure_count.
            manifest = {
                "schema_version": "umls_relation_delta_discovery_v1",
                "scope": {
                    "current_cui_count": 2,
                    "new_cui_count": 1,
                    "current_scope_sha256": scope_sha,
                },
                "frozen_parameters": {
                    "source_profile_sha256": "profilehash",
                    "umls_version": "2026AA",
                    "max_relations_per_cui": 1000,
                },
                "source_results": {
                    "NCI": {
                        "summary_path": str(source_dir / "summary.json"),
                        "raw_relation_record_count": 123,
                    }
                },
            }
            manifest_path = discovery / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = _require_delta_discovery_complete(
                manifest_path=manifest_path,
                current_scope_path=scope_path,
                sources=["NCI"],
            )
            self.assertEqual(result["scope"]["new_cui_count"], 1)

    def test_delta_completion_rejects_incomplete_per_source_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            discovery = root / "delta_discovery_v1"
            source_dir = discovery / "NCI"
            source_dir.mkdir(parents=True)

            scope_path = root / "scope.json"
            scope_path.write_text(
                json.dumps({"cuis": [{"cui": "C1"}, {"cui": "C2"}]}),
                encoding="utf-8",
            )
            scope_sha = hashlib.sha256(scope_path.read_bytes()).hexdigest()

            summary = {
                "source_vocabulary": "NCI",
                "local_universe_count": 2,
                "processed_local_cui_count": 0,
                "umls_version": "2026AA",
                "fetch_failure_count": 0,
                "client_stats": {"api_errors": 0},
                "safety": {
                    "neo4j_writes": False,
                    "second_hop_requests": False,
                },
                "inputs": {
                    "local_universe_sha256": scope_sha,
                    "source_profile_sha256": "profilehash",
                    "max_relations_per_cui": 1000,
                },
            }
            (source_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            manifest = {
                "schema_version": "umls_relation_delta_discovery_v1",
                "scope": {
                    "current_cui_count": 2,
                    "new_cui_count": 1,
                    "current_scope_sha256": scope_sha,
                },
                "frozen_parameters": {
                    "source_profile_sha256": "profilehash",
                    "umls_version": "2026AA",
                    "max_relations_per_cui": 1000,
                },
                "source_results": {"NCI": {"summary_path": str(source_dir / "summary.json")}},
            }
            manifest_path = discovery / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "processed_local_cui_count"
            ):
                _require_delta_discovery_complete(
                    manifest_path=manifest_path,
                    current_scope_path=scope_path,
                    sources=["NCI"],
                )

    def test_delta_source_summary_validation_is_strict(self):
        summary = {
            "source_vocabulary": "OMIM",
            "local_universe_count": 1007,
            "processed_local_cui_count": 427,
            "umls_version": "2026AA",
            "fetch_failure_count": 0,
            "client_stats": {"api_errors": 0},
            "safety": {"neo4j_writes": False, "second_hop_requests": False},
            "inputs": {
                "local_universe_sha256": "scopehash",
                "source_profile_sha256": "profilehash",
                "max_relations_per_cui": 1000,
            },
        }
        self.assertEqual(
            _validate_delta_source_summary(
                summary,
                source="OMIM",
                expected_scope_count=1007,
                expected_delta_count=427,
                current_scope_sha256="scopehash",
                source_profile_sha256="profilehash",
                umls_version="2026AA",
                max_relations_per_cui=1000,
            ),
            [],
        )
        summary["fetch_failure_count"] = 1
        self.assertIn(
            "fetch_failure_count",
            _validate_delta_source_summary(
                summary,
                source="OMIM",
                expected_scope_count=1007,
                expected_delta_count=427,
                current_scope_sha256="scopehash",
                source_profile_sha256="profilehash",
                umls_version="2026AA",
                max_relations_per_cui=1000,
            ),
        )


    def test_current_bridge_evidence_reclassifies_external_to_local(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            scope = root / "scope.json"
            scope.write_text(
                json.dumps(
                    {
                        "cuis": [
                            {"cui": "C1"},
                            {"cui": "C2"},
                            {"cui": "C3"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            historical = root / "historical"
            delta = root / "delta"
            for parent in (historical, delta):
                (parent / "NCI").mkdir(parents=True)

            base_row = {
                "relation_name": "related_to",
                "local_endpoint_role": "subject",
                "relation_labels": ["RO"],
                "root_sources": ["NCI"],
                "relation_ids": ["R1"],
                "raw_rows": 1,
                "max_counterpart_fanout": 1,
                "raw_subject_identifier_kinds": ["aui"],
                "raw_object_identifier_kinds": ["aui"],
            }
            (historical / "NCI" / "collapsed_first_hop_assertions.json").write_text(
                json.dumps(
                    [
                        {
                            **base_row,
                            "local_cui": "C1",
                            "local_source_cui": "C1",
                            "external_cui": "C2",
                        },
                        {
                            **base_row,
                            "local_cui": "C9",
                            "local_source_cui": "C9",
                            "external_cui": "C8",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            (delta / "NCI" / "collapsed_first_hop_assertions.json").write_text(
                json.dumps(
                    [
                        {
                            **base_row,
                            "local_cui": "C3",
                            "local_source_cui": "C3",
                            "external_cui": "C8",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            out = root / "merged"
            manifest = build_current_bridge_evidence_root(
                current_scope_path=scope,
                historical_bridge_root=historical,
                delta_discovery_root=delta,
                sources=["NCI"],
                output_root=out,
            )
            rows = json.loads(
                (out / "NCI" / "collapsed_first_hop_assertions.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["local_cui"], "C3")
            self.assertEqual(rows[0]["external_cui"], "C8")
            source = manifest["source_reports"]["NCI"]
            self.assertEqual(source["promoted_to_local_assertion_count"], 1)
            self.assertEqual(source["retired_seed_assertion_count"], 1)

    def test_external_label_plan_reuses_historical_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            retained = root / "retained.txt"
            retained.write_text("C1\nC2\nC3\n", encoding="utf-8")
            labels = root / "labels.json"
            labels.write_text(
                json.dumps(
                    {
                        "labels": {"C1": {"name": "one"}},
                        "failures": {"C2": {"reason": "old failure"}},
                    }
                ),
                encoding="utf-8",
            )
            plan = build_external_label_plan(
                retained_external_cuis_path=retained,
                historical_label_map_path=labels,
                umls_version="2026AA",
                output_dir=root / "plan",
            )
            self.assertEqual(plan["retained_external_cui_count"], 3)
            self.assertEqual(plan["already_resolved_from_historical_map_count"], 1)
            self.assertEqual(plan["to_resolve_count"], 2)
            self.assertEqual(plan["retry_historical_failure_count"], 1)
            self.assertEqual(plan["completely_new_to_resolve_count"], 1)

    def test_external_label_resolution_reuses_history_and_resume_checkpoint(self):
        class FakeClient:
            def __init__(self):
                self.calls = []
                self.stats = {"api_requests": 1, "cache_hits": 0}

            def get_concept_label(self, cui):
                self.calls.append(cui)
                return {
                    "name": "three",
                    "semantic_types": [
                        {"name": "Disease or Syndrome", "uri": "T047"}
                    ],
                    "class_type": "Concept",
                    "ui": cui,
                }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            retained = root / "retained.txt"
            retained.write_text("C1\nC2\nC3\n", encoding="utf-8")
            historical = root / "historical.json"
            historical.write_text(
                json.dumps(
                    {
                        "schema_version": "external_cui_labels_v1",
                        "umls_version": "2026AA",
                        "labels": {"C1": {"name": "one"}},
                        "failures": {"C2": "old failure"},
                    }
                ),
                encoding="utf-8",
            )
            plan_dir = root / "plan"
            build_external_label_plan(
                retained_external_cuis_path=retained,
                historical_label_map_path=historical,
                umls_version="2026AA",
                output_dir=plan_dir,
            )

            output = root / "labels"
            output.mkdir()
            (output / "external_cui_labels_v1.json").write_text(
                json.dumps(
                    {
                        "schema_version": "external_cui_labels_v1",
                        "umls_version": "2026AA",
                        "labels": {
                            "C1": {"name": "one"},
                            "C2": {
                                "name": "two",
                                "semantic_types": [],
                                "class_type": "Concept",
                                "ui": "C2",
                            },
                        },
                        "failures": {},
                    }
                ),
                encoding="utf-8",
            )

            fake = FakeClient()
            manifest = resolve_external_labels(
                label_plan_path=plan_dir / "external_label_plan_v1.json",
                retained_external_cuis_path=retained,
                historical_label_map_path=historical,
                output_dir=output,
                umls_version="2026AA",
                cache_dir=root / "cache",
                client=fake,
            )

            self.assertEqual(fake.calls, ["C3"])
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["counts"]["historical_reused_count"], 1)
            self.assertEqual(manifest["counts"]["resumed_reused_count"], 1)
            self.assertEqual(manifest["counts"]["attempted_this_run_count"], 1)
            self.assertEqual(manifest["counts"]["resolved_total_count"], 3)
            self.assertEqual(manifest["counts"]["unresolved_count"], 0)

            labels = json.loads(
                (output / "external_cui_labels_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(labels["requested_cui_count"], 3)
            self.assertEqual(labels["resolved_cui_count"], 3)
            self.assertEqual(set(labels["labels"]), {"C1", "C2", "C3"})

    def test_external_label_resolution_checkpoints_on_sustained_transient_failure(self):
        class FailingClient:
            stats = {"api_requests": 3, "api_errors": 3}

            def get_concept_label(self, cui):
                raise _ExternalLabelTransientError("temporary DNS failure")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            retained = root / "retained.txt"
            retained.write_text("C1\nC2\n", encoding="utf-8")
            historical = root / "historical.json"
            historical.write_text(
                json.dumps(
                    {
                        "schema_version": "external_cui_labels_v1",
                        "umls_version": "2026AA",
                        "labels": {},
                        "failures": {},
                    }
                ),
                encoding="utf-8",
            )
            plan_dir = root / "plan"
            build_external_label_plan(
                retained_external_cuis_path=retained,
                historical_label_map_path=historical,
                umls_version="2026AA",
                output_dir=plan_dir,
            )
            output = root / "labels"

            with self.assertRaisesRegex(RuntimeError, "sustained transient"):
                resolve_external_labels(
                    label_plan_path=plan_dir / "external_label_plan_v1.json",
                    retained_external_cuis_path=retained,
                    historical_label_map_path=historical,
                    output_dir=output,
                    umls_version="2026AA",
                    cache_dir=root / "cache",
                    max_consecutive_transient_failures=1,
                    client=FailingClient(),
                )

            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["complete"])
            self.assertEqual(manifest["counts"]["unresolved_count"], 2)
            self.assertTrue((output / "external_cui_labels_v1.json").exists())

    def test_current_final_build_is_artifact_only_and_preserves_candidate_universe(self):
        class ExplodingDriver:
            def __getattribute__(self, name):
                raise AssertionError(f"Neo4j driver accessed during C3: {name}")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out_root = work / "umls_relation_artifacts"
            current = out_root / "current_build_v1"
            scope = out_root / "inputs" / "local_umls_scope_v1.json"
            scope.parent.mkdir(parents=True)
            scope.write_text(
                json.dumps(
                    {
                        "schema_version": "local_umls_scope_v1",
                        "scope_name": "A+B",
                        "document_ids": ["A", "B"],
                        "cuis": [
                            {"cui": "C1", "document_ids": ["A"]},
                            {"cui": "C2", "document_ids": ["B"]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            scope_sha = hashlib.sha256(scope.read_bytes()).hexdigest()

            (current / "bridge_evidence").mkdir(parents=True)
            (current / "bridge_evidence" / "manifest.json").write_text(
                "{}", encoding="utf-8"
            )
            (current / "direct_artifact").mkdir(parents=True)
            (current / "direct_artifact" / "manifest.json").write_text(
                json.dumps({"pair_count": 1}), encoding="utf-8"
            )
            pre = current / "bridge_prelabel_artifact"
            pre.mkdir(parents=True)
            (pre / "retained_external_cuis.txt").write_text(
                "C9\n", encoding="utf-8"
            )
            (pre / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ontology_bridge_artifact_v1",
                        "local_scope_sha256": scope_sha,
                        "all_distinct_local_pair_count": 1,
                        "retained_distinct_local_pair_count": 1,
                        "retained_local_cui_count": 2,
                        "retained_external_cui_count": 1,
                        "overall_tier_counts": {"MEDIUM": 1},
                    }
                ),
                encoding="utf-8",
            )
            (current / "prelabel_build_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": "umls_relation_current_prelabel_v1",
                        "scope_name": "A+B",
                        "document_ids": ["A", "B"],
                        "current_scope_sha256": scope_sha,
                    }
                ),
                encoding="utf-8",
            )

            labels_dir = current / "external_labels"
            labels_dir.mkdir(parents=True)
            label_map = labels_dir / "external_cui_labels_v1.json"
            label_map.write_text(
                json.dumps(
                    {
                        "schema_version": "external_cui_labels_v1",
                        "umls_version": "2026AA",
                        "labels": {
                            "C9": {
                                "ui": "C9",
                                "name": "hub",
                                "class_type": "Concept",
                                "semantic_types": [
                                    {"name": "Finding", "uri": "T033"}
                                ],
                            }
                        },
                        "failures": {},
                    }
                ),
                encoding="utf-8",
            )
            label_sha = hashlib.sha256(label_map.read_bytes()).hexdigest()
            retained_sha = hashlib.sha256(
                (pre / "retained_external_cuis.txt").read_bytes()
            ).hexdigest()
            (labels_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "umls_relation_external_label_resolution_v1",
                        "umls_version": "2026AA",
                        "complete": True,
                        "counts": {
                            "retained_external_cui_count": 1,
                            "resolved_total_count": 1,
                            "unresolved_count": 0,
                            "failure_record_count": 0,
                        },
                        "inputs": {
                            "retained_external_cuis_sha256": retained_sha,
                        },
                        "outputs": {
                            "external_label_map_sha256": label_sha,
                        },
                        "safety": {
                            "neo4j_reads": False,
                            "neo4j_writes": False,
                            "retrieval_metrics_used": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            policy = root / "policy.json"
            policy.write_text("{}\n", encoding="utf-8")
            builder = root / "fake_builder.py"
            builder.write_text('import argparse, csv, hashlib, json\nfrom pathlib import Path\nap=argparse.ArgumentParser()\nap.add_argument("--bridge-root")\nap.add_argument("--local-scope", required=True)\nap.add_argument("--policy", required=True)\nap.add_argument("--external-label-map", required=True)\nap.add_argument("--output-dir", required=True)\nap.add_argument("--sources", nargs="*")\na=ap.parse_args()\nout=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)\nsha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()\nmanifest={\n "schema_version":"ontology_bridge_artifact_v1_1",\n "local_scope_sha256":sha(a.local_scope),\n "policy_sha256":sha(a.policy),\n "external_label_map_sha256":sha(a.external_label_map),\n "sources_requested":a.sources,\n "all_distinct_local_pair_count":1,\n "retained_distinct_local_pair_count":1,\n "retained_local_cui_count":2,\n "retained_external_cui_count":1,\n "overall_tier_counts":{"MEDIUM":1},\n "retrieval_profiles":{},\n "safety":{"umls_api_calls":False,"neo4j_writes":False,"second_hop_requests":False,"retrieval_metrics_used":False,"benchmark_tuned":False},\n}\n(out/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")\nwith (out/"pair_policy_audit.csv").open("w", newline="", encoding="utf-8") as f:\n w=csv.DictWriter(f, fieldnames=["local_cui_a","local_cui_b","overall_tier","retained_for_retrieval","policy_rule_counts"]); w.writeheader(); w.writerow({"local_cui_a":"C1","local_cui_b":"C2","overall_tier":"MEDIUM","retained_for_retrieval":"True","policy_rule_counts":json.dumps({"nci_default":1})})\nrow={"local_cui_a":"C1","local_cui_b":"C2","external_path_evidence":[{"retained_for_retrieval":True,"source_vocabulary":"NCI","external_cui":"C9","external_preferred_name":"hub","external_semantic_type_names":["finding"],"external_hub_degree":2,"relation_pair_evidence":[{"tier":"MEDIUM","policy_rule_id":"nci_default"}]}]}\n(out/"pair_evidence.jsonl").write_text(json.dumps(row)+"\\n", encoding="utf-8")\n', encoding="utf-8")

            result = run_umls_relation_artifact_workflow(
                ExplodingDriver(),
                project_root=root,
                work_root=work,
                action="build_current_final",
                current_final_build_paths=CurrentFinalBuildPaths(
                    bridge_final_builder=builder,
                    bridge_final_policy=policy,
                ),
                sources=["NCI"],
                umls_version="2026AA",
            )
            final = result["current_final"]
            self.assertEqual(
                final["final"]["all_distinct_local_pair_count"], 1
            )
            self.assertEqual(
                final["audit"]["default_only_retained"]["count"], 1
            )
            self.assertFalse(final["safety"]["neo4j_reads"])
            self.assertTrue((current / "current_final_report.json").exists())


    def test_default_only_audit_ignores_nonretaining_rules(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact"
            artifact.mkdir()
            scope = root / "scope.json"
            scope.write_text(
                json.dumps(
                    {
                        "cuis": [
                            {"cui": "C1", "document_ids": ["A"]},
                            {"cui": "C2", "document_ids": ["B"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with (artifact / "pair_policy_audit.csv").open(
                "w", newline="", encoding="utf-8"
            ) as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "local_cui_a",
                        "local_cui_b",
                        "overall_tier",
                        "retained_for_retrieval",
                        "policy_rule_counts",
                    ],
                )
                w.writeheader()
                w.writerow({
                    "local_cui_a": "C1",
                    "local_cui_b": "C2",
                    "overall_tier": "MEDIUM",
                    "retained_for_retrieval": "True",
                    "policy_rule_counts": json.dumps({
                        "lnc_default": 1,
                        "lnc_relation_outside_allowed_set": 1,
                    }),
                })
            row = {
                "local_cui_a": "C1",
                "local_cui_b": "C2",
                "retained_for_retrieval": True,
                "external_path_evidence": [{
                    "retained_for_retrieval": True,
                    "source_vocabulary": "LNC",
                    "external_cui": "C9",
                    "external_hub_degree": 2,
                    "relation_pair_evidence": [
                        {"tier": "MEDIUM", "policy_rule_id": "lnc_default"},
                        {"tier": "REJECT", "policy_rule_id": "lnc_relation_outside_allowed_set"},
                    ],
                }],
            }
            (artifact / "pair_evidence.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            audit = _final_bridge_audit_summary(
                final_artifact_dir=artifact,
                current_scope_path=scope,
            )
            self.assertEqual(audit["default_only_retained"]["count"], 1)

    def test_generalized_v2_build_preserves_frozen_c3_and_candidate_universe(self):
        class ExplodingDriver:
            def __getattribute__(self, name):
                raise AssertionError(f"Neo4j driver accessed during v2 build: {name}")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            out_root = work / "umls_relation_artifacts"
            current = out_root / "current_build_v1"
            scope = out_root / "inputs" / "local_umls_scope_v1.json"
            scope.parent.mkdir(parents=True)
            scope.write_text(
                json.dumps({
                    "schema_version": "local_umls_scope_v1",
                    "scope_name": "A+B",
                    "document_ids": ["A", "B"],
                    "cuis": [
                        {"cui": "C1", "document_ids": ["A"]},
                        {"cui": "C2", "document_ids": ["B"]},
                    ],
                }),
                encoding="utf-8",
            )
            scope_sha = hashlib.sha256(scope.read_bytes()).hexdigest()

            evidence = current / "bridge_evidence"
            evidence.mkdir(parents=True)
            (evidence / "manifest.json").write_text("{}", encoding="utf-8")
            (current / "direct_artifact").mkdir(parents=True)
            (current / "direct_artifact" / "manifest.json").write_text(
                json.dumps({"pair_count": 1}), encoding="utf-8"
            )
            (current / "bridge_prelabel_artifact").mkdir(parents=True)
            (current / "bridge_prelabel_artifact" / "manifest.json").write_text(
                json.dumps({"all_distinct_local_pair_count": 1}), encoding="utf-8"
            )
            labels_dir = current / "external_labels"
            labels_dir.mkdir(parents=True)
            (labels_dir / "manifest.json").write_text("{}", encoding="utf-8")
            label_map = labels_dir / "external_cui_labels_v1.json"
            label_map.write_text(json.dumps({"labels": {"C9": {"name": "hub"}}}), encoding="utf-8")
            label_sha = hashlib.sha256(label_map.read_bytes()).hexdigest()

            frozen_policy = root / "frozen_policy.json"
            frozen_policy.write_text('{"name":"v1.1"}\n', encoding="utf-8")
            frozen_policy_sha = hashlib.sha256(frozen_policy.read_bytes()).hexdigest()
            frozen_dir = current / "bridge_final_artifact"
            frozen_dir.mkdir(parents=True)
            frozen_manifest = frozen_dir / "manifest.json"
            frozen_manifest.write_text(
                json.dumps({
                    "schema_version": "ontology_bridge_artifact_v1_1",
                    "local_scope_sha256": scope_sha,
                    "external_label_map_sha256": label_sha,
                    "all_distinct_local_pair_count": 1,
                    "retained_distinct_local_pair_count": 1,
                    "retained_local_cui_count": 2,
                    "retained_external_cui_count": 1,
                    "overall_tier_counts": {"MEDIUM": 1},
                    "retrieval_profiles": {},
                }),
                encoding="utf-8",
            )
            frozen_manifest_sha = hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()
            (current / "current_final_report.json").write_text(
                json.dumps({
                    "schema_version": "umls_relation_current_final_v1",
                    "scope_name": "A+B",
                    "document_ids": ["A", "B"],
                    "inputs": {"bridge_final_policy_sha256": frozen_policy_sha},
                    "audit": {
                        "default_only_retained": {
                            "count": 1, "retained_pair_count": 1, "fraction": 1.0,
                            "counts_by_membership": {}, "fraction_by_membership": {}
                        }
                    },
                }),
                encoding="utf-8",
            )

            v2_policy = root / "v2_policy.json"
            v2_policy.write_text(
                json.dumps({
                    "design_constraints": {
                        "retrieval_metrics_used": False,
                        "benchmark_tuned": False,
                    }
                }) + "\n",
                encoding="utf-8",
            )
            builder = root / "fake_builder.py"
            builder.write_text(
                'import argparse,csv,hashlib,json\nfrom pathlib import Path\n'
                'ap=argparse.ArgumentParser(); ap.add_argument("--bridge-root"); ap.add_argument("--local-scope"); ap.add_argument("--policy"); ap.add_argument("--external-label-map"); ap.add_argument("--output-dir"); ap.add_argument("--sources",nargs="*"); a=ap.parse_args()\n'
                'out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()\n'
                'm={"schema_version":"ontology_bridge_artifact_v1_1","local_scope_sha256":sha(a.local_scope),"policy_sha256":sha(a.policy),"external_label_map_sha256":sha(a.external_label_map),"sources_requested":a.sources,"all_distinct_local_pair_count":1,"retained_distinct_local_pair_count":1,"retained_local_cui_count":2,"retained_external_cui_count":1,"overall_tier_counts":{"MEDIUM":1},"retrieval_profiles":{},"safety":{"umls_api_calls":False,"neo4j_writes":False,"second_hop_requests":False,"retrieval_metrics_used":False,"benchmark_tuned":False}}; (out/"manifest.json").write_text(json.dumps(m))\n'
                'with (out/"pair_policy_audit.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=["local_cui_a","local_cui_b","overall_tier","retained_for_retrieval","policy_rule_counts"]); w.writeheader(); w.writerow({"local_cui_a":"C1","local_cui_b":"C2","overall_tier":"MEDIUM","retained_for_retrieval":"True","policy_rule_counts":json.dumps({"explicit_medium":1})})\n'
                'row={"local_cui_a":"C1","local_cui_b":"C2","external_path_evidence":[{"retained_for_retrieval":True,"source_vocabulary":"NCI","external_cui":"C9","external_preferred_name":"hub","external_semantic_type_names":["finding"],"external_hub_degree":2,"relation_pair_evidence":[{"tier":"MEDIUM","policy_rule_id":"explicit_medium"}]}]}; (out/"pair_evidence.jsonl").write_text(json.dumps(row)+"\\n")\n',
                encoding="utf-8",
            )

            result = run_umls_relation_artifact_workflow(
                ExplodingDriver(),
                project_root=root,
                work_root=work,
                action="build_current_generalized",
                generalized_bridge_build_paths=GeneralizedBridgeBuildPaths(
                    bridge_builder=builder,
                    bridge_policy=v2_policy,
                    frozen_v1_1_policy_sha256=frozen_policy_sha,
                    frozen_c3_manifest_sha256=frozen_manifest_sha,
                ),
                sources=["NCI"],
                umls_version="2026AA",
            )
            report = result["current_generalized_v2"]
            self.assertEqual(report["frozen_v1_1"]["all_distinct_local_pair_count"], 1)
            self.assertEqual(report["generalized_v2"]["all_distinct_local_pair_count"], 1)
            self.assertFalse(report["safety"]["neo4j_reads"])
            self.assertFalse(report["safety"]["retrieval_metrics_used"])
            self.assertTrue(
                (current / "current_generalized_v2_report.json").exists()
            )

    def test_external_label_error_redaction_does_not_persist_api_key(self):
        secret = "very-secret-key"
        message = (
            "GET https://uts-ws.nlm.nih.gov/rest/content/2026AA/CUI/C1"
            f"?apiKey={secret} failed"
        )
        redacted = _redact_api_key(message, secret)
        self.assertNotIn(secret, redacted)
        self.assertIn("apiKey=<redacted>", redacted)


if __name__ == "__main__":
    unittest.main()
