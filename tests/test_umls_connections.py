import importlib.util
import io
import os
import re
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def stub_optional_module(name, **attrs):
    try:
        spec = importlib.util.find_spec(name)
    except ValueError:
        spec = None
    if spec is not None:
        return

    module = types.ModuleType(name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    sys.modules[name] = module


class UnusedSession:
    pass


stub_optional_module(
    "neo4j",
    Driver=object,
    GraphDatabase=types.SimpleNamespace(driver=lambda *args, **kwargs: None),
)
stub_optional_module("dotenv", load_dotenv=lambda *args, **kwargs: False)
stub_optional_module(
    "rapidfuzz",
    fuzz=types.SimpleNamespace(
        token_sort_ratio=lambda left, right: 0,
        token_set_ratio=lambda left, right: 0,
    ),
)
stub_optional_module("requests", Session=UnusedSession)

from knowledge_graph import umls_connections as conn


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        if not self.responses:
            raise AssertionError("Unexpected UMLS API call")
        return self.responses.pop(0)


def relation_payload():
    return {
        "result": [
            {
                "ui": "R1",
                "relationLabel": "RO",
                "additionalRelationLabel": "has_finding_site",
                "rootSource": "SNOMEDCT_US",
                "relatedId": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0000002",
            }
        ]
    }


def concept(
    concept_id,
    name,
    canonical_type,
    cui,
    score,
):
    return conn.LocalConcept(
        concept_id=concept_id,
        name=name,
        canonical_type=canonical_type,
        umls_cui=cui,
        umls_canonical_name=name,
        umls_semantic_types=(),
        umls_score=score,
        observed_types=(),
        type_support_pairs=(),
        type_resolution_status="single_supported_type",
        needs_type_review=False,
    )


class FakeNeo4jResult:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None


class FakeNeo4jTx:
    def __init__(self, store):
        self.store = store

    def run(self, query, **params):
        if "RETURN deleted_count" in query:
            relationship_types = set(params.get("relationship_types") or [])
            doc_id = params.get("doc_id")
            keys_to_delete = [
                key
                for key, value in self.store.items()
                if key[2] in relationship_types
                and value.get("params", {}).get("provenance") == "umls_connections"
                and value.get("params", {}).get("doc_id") == doc_id
            ]
            for key in keys_to_delete:
                del self.store[key]
            return FakeNeo4jResult([{"deleted_count": len(keys_to_delete)}])

        if "RETURN elementId(r) AS relationship_id" in query:
            relationship_type = re.search(
                r":(UMLS_[A-Z_]+) \{edge_key",
                query,
            ).group(1)
            key = (
                params["source_concept_id"],
                params["target_concept_id"],
                relationship_type,
                params["edge_key"],
            )
            if key not in self.store:
                self.store[key] = {
                    "relationship_id": f"rel-{len(self.store) + 1}",
                    "params": dict(params),
                }
            else:
                self.store[key]["params"] = dict(params)
            relationship_id = self.store[key]["relationship_id"]
            return FakeNeo4jResult([{"relationship_id": relationship_id}])

        if "collect(DISTINCT relationship_type)" in query:
            return FakeNeo4jResult([])

        if "relationship_source_cui" in query:
            return FakeNeo4jResult([])

        if "strict_policy_violations" in query or "materialization_mode" in query:
            return FakeNeo4jResult([])

        if (
            "RETURN type(r) AS relationship_type, count(r) AS n" in query
            or "RETURN type(r) AS relationship_type, count(*) AS n" in query
        ):
            counts = {}
            for _source_id, _target_id, relationship_type, _edge_key in self.store:
                counts[relationship_type] = counts.get(relationship_type, 0) + 1
            return FakeNeo4jResult(
                [
                    {"relationship_type": relationship_type, "n": count}
                    for relationship_type, count in sorted(counts.items())
                ]
            )

        raise AssertionError(f"Unexpected query: {query}")


class FakeNeo4jSession:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_write(self, fn, *args):
        return fn(FakeNeo4jTx(self.store), *args)

    def execute_read(self, fn, *args):
        return fn(FakeNeo4jTx(self.store), *args)


class FakeNeo4jDriver:
    def __init__(self):
        self.store = {}

    def session(self):
        return FakeNeo4jSession(self.store)


class UMLSConnectionsProfileTests(unittest.TestCase):
    def test_relation_sets_and_mappings_are_exact(self):
        expected_core = {
            "isa",
            "has_finding_site",
            "has_associated_morphology",
            "has_procedure_site",
            "has_direct_procedure_site",
        }
        expected_extension = {
            "has_definitional_manifestation",
            "uses_device",
            "has_direct_device",
            "has_measured_component",
        }
        expected_new_mappings = {
            "has_definitional_manifestation": "UMLS_HAS_DEFINITIONAL_MANIFESTATION",
            "uses_device": "UMLS_USES_DEVICE",
            "has_direct_device": "UMLS_HAS_DIRECT_DEVICE",
            "has_measured_component": "UMLS_HAS_MEASURED_COMPONENT",
        }
        expected_audit_only = {
            "has_method",
            "has_causative_agent",
            "has_pathological_process",
            "has_associated_finding",
            "has_associated_etiologic_finding",
            "has_active_ingredient",
            "has_precise_active_ingredient",
            "interprets",
            "has_interpretation",
            "has_clinical_course",
            "due_to",
        }
        expected_experimental_families = {
            "has_finding_site",
            "has_procedure_site",
            "has_direct_procedure_site",
            "has_indirect_procedure_site",
            "has_device_intended_site",
            "has_associated_morphology",
            "has_direct_morphology",
            "has_indirect_morphology",
            "has_procedure_morphology",
            "has_causative_agent",
            "due_to",
            "has_pathological_process",
            "has_focus",
            "uses_device",
            "has_direct_device",
            "has_indirect_device",
            "has_procedure_device",
            "has_method",
            "has_finding_method",
            "interprets",
            "has_interpretation",
            "has_measured_component",
            "has_associated_finding",
            "has_definitional_manifestation",
            "has_associated_etiologic_finding",
            "has_active_ingredient",
            "has_precise_active_ingredient",
        }

        self.assertEqual(conn.CORE_SNOMED_RELATION_NAMES, expected_core)
        self.assertEqual(
            set(conn.RELATION_NAMES_BY_PROFILE),
            {
                "isa_only",
                "semantic_seed",
                "seed_core",
                "first_extension",
                "seed_expanded",
                "site_relations",
                "morphology_relations",
                "causal_relations",
                "focus_relations",
                "device_relations",
                "method_measurement_relations",
                "association_relations",
                "manifestation_relations",
                "etiology_extension_relations",
                "drug_composition_relations",
                "explore_known",
                "discover",
                "core",
                "expanded",
                "balanced_core",
                "audit_all",
                "nonhier_broad_audit",
            },
        )
        self.assertEqual(
            conn.FIRST_SNOMED_EXTENSION_RELATION_NAMES,
            expected_extension,
        )
        self.assertEqual(
            conn.EXPANDED_SNOMED_RELATION_NAMES,
            expected_core | expected_extension,
        )
        self.assertIs(conn.STRONG_RELATION_NAMES, conn.EXPANDED_SNOMED_RELATION_NAMES)
        for relation_name, relationship_type in expected_new_mappings.items():
            self.assertEqual(conn.RELATION_TYPE_BY_NAME[relation_name], relationship_type)
            self.assertIn(relationship_type, conn.UMLS_CONNECTION_RELATION_TYPES)
        self.assertEqual(conn.AUDIT_ONLY_RELATION_NAMES, expected_audit_only)
        self.assertEqual(
            conn.EXPERIMENTAL_FAMILY_RELATION_NAMES,
            expected_experimental_families,
        )
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["audit_all"],
            expected_core
            | expected_extension
            | expected_audit_only
            | expected_experimental_families,
        )
        for relation_name in expected_audit_only:
            self.assertIn(relation_name, conn.RELATION_SPECS)
            self.assertFalse(conn.RELATION_SPECS[relation_name].materialize_by_default)
            self.assertTrue(
                conn.RELATION_SPECS[relation_name].relationship_type.startswith("UMLS_")
            )
        self.assertEqual(conn.RELATION_SPECS["isa"].family, "hierarchy_seed")
        for relation_name in expected_core - {"isa"}:
            self.assertEqual(
                conn.RELATION_SPECS[relation_name].family,
                "semantic_seed",
            )
        self.assertEqual(conn.APPROVED_MATERIALIZATION_RELATION_NAMES, frozenset())
        self.assertTrue(
            all(
                not spec.materialize_by_default
                for spec in conn.RELATION_SPECS.values()
            )
        )

        for raw_inverse in {
            "inverse_isa",
            "finding_site_of",
            "associated_morphology_of",
            "procedure_site_of",
            "direct_procedure_site_of",
            "definitional_manifestation_of",
            "device_used_by",
            "direct_device_of",
            "measured_component_of",
            "indirect_procedure_site_of",
            "device_intended_site_of",
            "direct_morphology_of",
            "indirect_morphology_of",
            "procedure_morphology_of",
            "focus_of",
            "finding_method_of",
            "indirect_device_of",
            "procedure_device_of",
            "associated_etiologic_finding_of",
            "active_ingredient_of",
            "precise_active_ingredient_of",
        }:
            self.assertNotIn(raw_inverse, conn.RELATION_TYPE_BY_NAME)

    def test_relation_profile_filter_resolution(self):
        include, exclude, profile = conn.resolve_relation_name_filters(
            relation_profile="core"
        )
        self.assertEqual(include, conn.CORE_SNOMED_RELATION_NAMES)
        self.assertEqual(exclude, set())
        self.assertEqual(profile, "core")

        include, _exclude, profile = conn.resolve_relation_name_filters(
            relation_profile="first_extension",
            include_relation_names=["isa"],
            exclude_relation_names=["uses_device"],
        )
        self.assertEqual(
            include,
            (conn.FIRST_SNOMED_EXTENSION_RELATION_NAMES - {"uses_device"})
            | {"isa"},
        )
        self.assertEqual(profile, "first_extension")

        include, _exclude, profile = conn.resolve_relation_name_filters(
            relation_profile="expanded"
        )
        self.assertEqual(include, conn.EXPANDED_SNOMED_RELATION_NAMES)
        self.assertEqual(profile, "expanded")

        include, exclude, profile = conn.resolve_relation_name_filters(
            relation_profile="discover"
        )
        self.assertEqual(include, set())
        self.assertEqual(exclude, set())
        self.assertEqual(profile, "discover")

    def test_strong_relations_only_is_expanded_profile(self):
        include, _exclude, profile = conn.resolve_relation_name_filters(
            strong_relations_only=True
        )

        self.assertEqual(include, conn.EXPANDED_SNOMED_RELATION_NAMES)
        self.assertEqual(profile, "expanded")

        include, _exclude, profile = conn.resolve_relation_name_filters(
            strong_relations_only=True,
            relation_profile="expanded",
        )
        self.assertEqual(include, conn.EXPANDED_SNOMED_RELATION_NAMES)
        self.assertEqual(profile, "expanded")

        with self.assertRaises(ValueError):
            conn.resolve_relation_name_filters(
                strong_relations_only=True,
                relation_profile="core",
            )

    def test_arg_parser_accepts_relation_profiles(self):
        parser = conn.build_arg_parser()

        for profile in (
            "isa_only",
            "semantic_seed",
            "seed_core",
            "first_extension",
            "seed_expanded",
            "association_relations",
            "manifestation_relations",
            "etiology_extension_relations",
            "drug_composition_relations",
            "explore_known",
            "discover",
            "core",
            "expanded",
            "balanced_core",
            "audit_all",
        ):
            with self.subTest(profile=profile):
                args = parser.parse_args(
                    ["--doc-id", "doc-a", "--relation-profile", profile]
                )
                self.assertEqual(args.relation_profile, profile)
        args = parser.parse_args(
            [
                "--doc-id",
                "doc-a",
                "--relation-profile",
                "audit_all",
                "--materialization-mode",
                "none",
            ]
        )
        self.assertEqual(args.materialization_mode, "none")

        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parser.parse_args(["--doc-id", "doc-a", "--relation-profile", "legacy_core"])

    def test_exploratory_profiles_cannot_write_relations(self):
        with self.assertRaises(ValueError):
            conn.run_umls_connections(
                doc_id="doc-a",
                relation_profile="audit_all",
                write_neo4j=True,
                materialization_mode="safe_only",
                driver=FakeNeo4jDriver(),
            )


class UMLSConnectionsExperimentalProfilesV4Tests(unittest.TestCase):
    def test_canonical_direction_policy_is_explicit_and_retrieval_agnostic(self):
        self.assertEqual(
            conn.CANONICAL_DIRECTION_POLICY,
            "snomed_attribute_domain_to_attribute_value",
        )
        self.assertIn("inverse labels are swapped once", conn.CANONICAL_DIRECTION_DESCRIPTION)
        self.assertIn("retrieval traversal direction", conn.CANONICAL_DIRECTION_DESCRIPTION)

    def test_retrieval_family_profiles_are_exact_and_separate_from_isa(self):
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["site_relations"],
            conn.SITE_RELATION_NAMES,
        )
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["morphology_relations"],
            conn.MORPHOLOGY_RELATION_NAMES,
        )
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["causal_relations"],
            conn.CAUSAL_RELATION_NAMES,
        )
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["focus_relations"],
            conn.FOCUS_RELATION_NAMES,
        )
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["device_relations"],
            conn.DEVICE_RELATION_NAMES,
        )
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["method_measurement_relations"],
            conn.METHOD_MEASUREMENT_RELATION_NAMES,
        )
        for profile in (
            "site_relations",
            "morphology_relations",
            "causal_relations",
            "focus_relations",
            "device_relations",
            "method_measurement_relations",
        ):
            with self.subTest(profile=profile):
                self.assertNotIn(
                    "isa",
                    conn.RELATION_NAMES_BY_PROFILE[profile],
                )

    def test_selected_inverse_pairs_canonicalize_once_to_domain_value_direction(self):
        cases = {
            "indirect_procedure_site_of": "has_indirect_procedure_site",
            "device_intended_site_of": "has_device_intended_site",
            "direct_morphology_of": "has_direct_morphology",
            "indirect_morphology_of": "has_indirect_morphology",
            "procedure_morphology_of": "has_procedure_morphology",
            "focus_of": "has_focus",
            "finding_method_of": "has_finding_method",
            "indirect_device_of": "has_indirect_device",
            "procedure_device_of": "has_procedure_device",
        }
        for raw_inverse, canonical_name in cases.items():
            with self.subTest(raw_inverse=raw_inverse):
                relation_name, source_cuis, target_cuis, swapped = (
                    conn.canonicalize_resolved_relation(
                        raw_inverse,
                        ["C0000002"],  # raw subject: attribute value
                        ["C0000001"],  # raw object: attribute domain concept
                    )
                )
                self.assertEqual(relation_name, canonical_name)
                self.assertEqual(source_cuis, ["C0000001"])
                self.assertEqual(target_cuis, ["C0000002"])
                self.assertTrue(swapped)
                self.assertNotIn(raw_inverse, conn.RELATION_TYPE_BY_NAME)

    def test_new_site_and_focus_type_rules_follow_canonical_direction(self):
        self.assertEqual(
            conn.evaluate_relation_compatibility(
                "has_indirect_procedure_site",
                "procedure_or_intervention",
                "anatomical_structure",
            ).status,
            "compatible",
        )
        self.assertEqual(
            conn.evaluate_relation_compatibility(
                "has_device_intended_site",
                "device",
                "anatomical_structure",
            ).status,
            "compatible",
        )
        self.assertEqual(
            conn.evaluate_relation_compatibility(
                "has_focus",
                "procedure_or_intervention",
                "disease",
            ).status,
            "compatible",
        )
        self.assertEqual(
            conn.evaluate_relation_compatibility(
                "has_finding_method",
                "clinical_finding",
                "diagnostic_test",
            ).status,
            "compatible",
        )

    def test_method_relations_without_clean_local_range_are_review_only(self):
        for relation_name in (
            "has_method",
            "interprets",
            "has_interpretation",
            "has_pathological_process",
        ):
            with self.subTest(relation_name=relation_name):
                self.assertEqual(
                    conn.RELATION_SPECS[relation_name].validation_mode,
                    "review",
                )
                self.assertEqual(
                    conn.evaluate_relation_compatibility(
                        relation_name,
                        "clinical_finding",
                        "diagnostic_test",
                    ).status,
                    "review",
                )

    def test_no_experimental_family_relation_is_approved_for_materialization(self):
        self.assertEqual(conn.APPROVED_MATERIALIZATION_RELATION_NAMES, frozenset())
        for relation_name in conn.EXPERIMENTAL_FAMILY_RELATION_NAMES:
            with self.subTest(relation_name=relation_name):
                self.assertIn(relation_name, conn.RELATION_SPECS)
                self.assertFalse(
                    conn.RELATION_SPECS[relation_name].materialize_by_default
                )



class UMLSConnectionsLocalEndpointV6Tests(unittest.TestCase):
    def test_new_profiles_are_review_only_and_canonicalized(self):
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["association_relations"],
            frozenset({"has_associated_finding"}),
        )
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["manifestation_relations"],
            frozenset({"has_definitional_manifestation"}),
        )
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["etiology_extension_relations"],
            frozenset({"has_associated_etiologic_finding"}),
        )
        self.assertEqual(
            conn.RELATION_NAMES_BY_PROFILE["drug_composition_relations"],
            frozenset({"has_active_ingredient", "has_precise_active_ingredient"}),
        )
        for relation_name in (
            "has_associated_finding",
            "has_definitional_manifestation",
            "has_associated_etiologic_finding",
            "has_active_ingredient",
            "has_precise_active_ingredient",
        ):
            with self.subTest(relation_name=relation_name):
                self.assertEqual(
                    conn.RELATION_SPECS[relation_name].default_traversal_policy,
                    "review",
                )
                self.assertFalse(
                    conn.RELATION_SPECS[relation_name].materialize_by_default
                )

        self.assertEqual(
            conn.canonicalize_raw_relation_name("associated_etiologic_finding_of"),
            ("has_associated_etiologic_finding", True),
        )
        self.assertEqual(
            conn.canonicalize_raw_relation_name("active_ingredient_of"),
            ("has_active_ingredient", True),
        )
        self.assertEqual(
            conn.canonicalize_raw_relation_name("precise_active_ingredient_of"),
            ("has_precise_active_ingredient", True),
        )

    def test_search_source_code_parser_and_completeness(self):
        payload = {
            "result": {
                "recCount": 2,
                "results": [
                    {"ui": "111", "rootSource": "SNOMEDCT_US"},
                    {"ui": "222", "rootSource": "SNOMEDCT_US"},
                ],
            }
        }
        self.assertEqual(
            conn.parse_search_source_codes(payload, root_source="SNOMEDCT_US"),
            ["111", "222"],
        )
        self.assertEqual(conn.parse_search_record_count(payload), 2)

    def test_local_source_ui_index_shortcuts_complete_nonlocal_endpoints(self):
        index = conn.LocalSourceUiIndex(
            source_vocab="SNOMEDCT_US",
            source_ui_to_cuis={"111": ("C0000001",)},
            queried_local_cuis=("C0000001",),
            complete_local_cuis=("C0000001",),
            incomplete_local_cuis=(),
            source_code_counts_by_cui={"C0000001": 1},
            reported_record_counts_by_cui={"C0000001": 1},
            ambiguous_source_uis=(),
        )
        client = types.SimpleNamespace(lookup_cuis_for_source_ui=lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run")))
        record = {"rootSource": "SNOMEDCT_US"}
        local, error = conn.resolve_identifier_cuis(
            client,
            "https://uts-ws.nlm.nih.gov/rest/content/current/source/SNOMEDCT_US/111",
            record=record,
            source_vocab="SNOMEDCT_US",
            missing_reason="missing",
            unresolved_reason="unresolved",
            local_source_ui_index=index,
        )
        self.assertEqual(local, ["C0000001"])
        self.assertIsNone(error)
        external, error = conn.resolve_identifier_cuis(
            client,
            "https://uts-ws.nlm.nih.gov/rest/content/current/source/SNOMEDCT_US/999",
            record=record,
            source_vocab="SNOMEDCT_US",
            missing_reason="missing",
            unresolved_reason="unresolved",
            local_source_ui_index=index,
        )
        self.assertEqual(external, [])
        self.assertEqual(error, "source_ui_not_in_complete_local_index")
        self.assertEqual(index.hits, 1)
        self.assertEqual(index.external_shortcuts, 1)

    def test_incomplete_local_index_falls_back_to_legacy_lookup(self):
        index = conn.LocalSourceUiIndex(
            source_vocab="SNOMEDCT_US",
            source_ui_to_cuis={},
            queried_local_cuis=("C0000001",),
            complete_local_cuis=(),
            incomplete_local_cuis=("C0000001",),
            source_code_counts_by_cui={"C0000001": 200},
            reported_record_counts_by_cui={"C0000001": 250},
            ambiguous_source_uis=(),
        )
        client = types.SimpleNamespace(
            lookup_cuis_for_source_ui=lambda **kwargs: ["C0000001"]
        )
        cuis, error = conn.resolve_identifier_cuis(
            client,
            "https://uts-ws.nlm.nih.gov/rest/content/current/source/SNOMEDCT_US/999",
            record={"rootSource": "SNOMEDCT_US"},
            source_vocab="SNOMEDCT_US",
            missing_reason="missing",
            unresolved_reason="unresolved",
            local_source_ui_index=index,
        )
        self.assertEqual(cuis, ["C0000001"])
        self.assertIsNone(error)
        self.assertEqual(index.fallback_lookups, 1)

    def test_ambiguous_local_source_ui_index_fails_closed(self):
        index = conn.LocalSourceUiIndex(
            source_vocab="SNOMEDCT_US",
            source_ui_to_cuis={"111": ("C0000001", "C0000002")},
            queried_local_cuis=("C0000001", "C0000002"),
            complete_local_cuis=("C0000001", "C0000002"),
            incomplete_local_cuis=(),
            source_code_counts_by_cui={"C0000001": 1, "C0000002": 1},
            reported_record_counts_by_cui={"C0000001": 1, "C0000002": 1},
            ambiguous_source_uis=("111",),
        )
        client = types.SimpleNamespace(
            lookup_cuis_for_source_ui=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("ambiguous indexed sourceUi must fail closed")
            )
        )
        cuis, error = conn.resolve_identifier_cuis(
            client,
            "https://uts-ws.nlm.nih.gov/rest/content/current/source/SNOMEDCT_US/111",
            record={"rootSource": "SNOMEDCT_US"},
            source_vocab="SNOMEDCT_US",
            missing_reason="missing",
            unresolved_reason="unresolved",
            local_source_ui_index=index,
        )
        self.assertEqual(cuis, [])
        self.assertEqual(error, "ambiguous_source_ui_in_local_index")
        self.assertEqual(index.ambiguous_hits, 1)

    def test_build_local_source_ui_index_marks_incomplete_cui(self):
        class FakeClient:
            def lookup_source_codes_for_cui(self, cui, root_source):
                if cui == "C0000001":
                    return ["111"], 1, True
                return [f"X{i}" for i in range(200)], 250, False

        index = conn.build_local_source_ui_index(
            FakeClient(),
            local_cuis=["C0000001", "C0000002"],
            source_vocab="SNOMEDCT_US",
        )
        self.assertFalse(index.globally_complete)
        self.assertEqual(index.complete_local_cuis, ("C0000001",))
        self.assertEqual(index.incomplete_local_cuis, ("C0000002",))
        self.assertEqual(index.lookup("111"), ("C0000001",))

    def test_parser_accepts_local_source_ui_index_flag(self):
        args = conn.build_arg_parser().parse_args(
            ["--doc-id", "doc-a", "--use-local-source-ui-index"]
        )
        self.assertTrue(args.use_local_source_ui_index)

    def test_manifestation_is_no_longer_implicitly_safe(self):
        self.assertEqual(
            conn.traversal_policy_for_relation(
                "has_definitional_manifestation",
                ["disease"],
                ["clinical_finding"],
            ),
            "review",
        )


class UMLSConnectionsTypeCompatibilityTests(unittest.TestCase):
    def test_materialization_modes_are_strict(self):
        self.assertEqual(conn.DEFAULT_MATERIALIZATION_MODE, "none")
        self.assertEqual(conn.MATERIALIZATION_MODE_CHOICES, ("none", "safe_only"))
        self.assertEqual(conn.normalize_materialization_mode(None), "none")
        self.assertEqual(conn.normalize_materialization_mode("safe_only"), "safe_only")

        for obsolete_mode in ("legacy", "approved"):
            with self.subTest(mode=obsolete_mode):
                with self.assertRaisesRegex(ValueError, "none and safe_only"):
                    conn.normalize_materialization_mode(obsolete_mode)

    def test_none_mode_rejects_every_edge(self):
        decision, reason = conn.should_materialize_relation(
            {
                "relation_name": "has_finding_site",
                "relationship_type": "UMLS_HAS_FINDING_SITE",
                "source_representative": {"concept_id": "s"},
                "target_representative": {"concept_id": "t"},
                "compatibility_status": "compatible",
                "local_type_compatible": True,
                "traversal_policy": "safe",
                "review_needed": False,
                "materialize_by_default": True,
            }
        )

        self.assertFalse(decision)
        self.assertEqual(reason, "materialization_mode_none")

    def test_configured_compatible_type_pairs_are_accepted(self):
        accepted = [
            ("has_definitional_manifestation", "disease", "clinical_finding"),
            ("uses_device", "diagnostic_test", "device"),
            ("has_direct_device", "procedure_or_intervention", "device"),
            ("has_measured_component", "diagnostic_test", "biomarker"),
            ("has_causative_agent", "disease", "microorganism_or_pathogen"),
            ("has_causative_agent", "clinical_finding", "exposure_or_lifestyle_factor"),
        ]

        for relation_name, source_type, target_type in accepted:
            with self.subTest(
                relation_name=relation_name,
                source_type=source_type,
                target_type=target_type,
            ):
                compatible, reason = conn.evaluate_local_type_compatibility(
                    relation_name,
                    source_type,
                    target_type,
                )
                self.assertTrue(compatible)
                self.assertEqual(reason, "compatible")

    def test_configured_incompatible_type_pairs_are_rejected(self):
        rejected = [
            ("uses_device", "disease", "device"),
            ("has_direct_device", "biomarker", "device"),
            ("has_measured_component", "diagnostic_test", "disease"),
            ("device_used_by", "device", "disease"),
        ]

        for relation_name, source_type, target_type in rejected:
            with self.subTest(
                relation_name=relation_name,
                source_type=source_type,
                target_type=target_type,
            ):
                compatible, reason = conn.evaluate_local_type_compatibility(
                    relation_name,
                    source_type,
                    target_type,
                )
                self.assertFalse(compatible)
                self.assertNotEqual(reason, "compatible")

    def test_core_relation_uses_catalog_compatibility(self):
        compatible, reason = conn.evaluate_local_type_compatibility(
            "has_finding_site",
            "disease",
            "anatomical_structure",
        )

        self.assertTrue(compatible)
        self.assertEqual(reason, "compatible")

        result = conn.evaluate_relation_compatibility(
            "has_finding_site",
            "disease",
            "device",
        )
        self.assertEqual(result.status, "incompatible")
        self.assertEqual(result.reason, "target_type_not_allowed")

    def test_unknown_relation_is_unsupported_for_local_type_wrapper(self):
        compatible, reason = conn.evaluate_local_type_compatibility(
            "mapped_to",
            "disease",
            "disease",
        )

        self.assertIsNone(compatible)
        self.assertEqual(reason, "relation_not_in_catalog")

    def test_empty_types_are_incompatible_when_rule_exists(self):
        compatible, reason = conn.evaluate_local_type_compatibility(
            "uses_device",
            "",
            "device",
        )

        self.assertFalse(compatible)
        self.assertEqual(reason, "missing_source_type")

    def test_hierarchy_compatibility_uses_broad_families(self):
        same_type = conn.evaluate_relation_compatibility(
            "isa",
            "disease",
            "disease",
        )
        self.assertEqual(same_type.status, "compatible")
        self.assertEqual(same_type.reason, "same_canonical_type")

        disease_family = conn.evaluate_relation_compatibility(
            "isa",
            "disease",
            "clinical_finding",
        )
        self.assertEqual(disease_family.status, "compatible_broad")

        diagnostic_cross_type = conn.evaluate_relation_compatibility(
            "isa",
            "diagnostic_test",
            "procedure_or_intervention",
        )
        self.assertEqual(diagnostic_cross_type.status, "review")

        cross_family = conn.evaluate_relation_compatibility(
            "isa",
            "disease",
            "diagnostic_test",
        )
        self.assertEqual(cross_family.status, "review")
        self.assertEqual(
            conn.traversal_policy_for_relation(
                "isa",
                ["disease"],
                ["clinical_finding"],
            ),
            "hierarchy_review",
        )


class UMLSConnectionsTraversalPolicyTests(unittest.TestCase):
    def test_compatible_forward_first_extension_relations_are_safe(self):
        for relation_name, source_type, target_type in [
            ("uses_device", "diagnostic_test", "device"),
            ("has_direct_device", "procedure_or_intervention", "device"),
            ("has_measured_component", "diagnostic_test", "biomarker"),
        ]:
            with self.subTest(relation_name=relation_name):
                self.assertEqual(
                    conn.traversal_policy_for_relation(
                        relation_name,
                        [source_type],
                        [target_type],
                    ),
                    "safe",
                )

    def test_compatible_broad_semantic_relation_requires_type_review(self):
        self.assertEqual(
            conn.traversal_policy_for_relation(
                "has_associated_morphology",
                ["disease"],
                ["disease"],
                compatibility_status="compatible_broad",
            ),
            "type_review",
        )

    def test_multi_type_cui_forces_review_independent_of_representative(self):
        self.assertEqual(
            conn.traversal_policy_for_relation(
                "isa",
                ["disease"],
                ["clinical_finding", "disease"],
                compatibility_status="compatible",
            ),
            "hierarchy_review",
        )
        self.assertEqual(
            conn.traversal_policy_for_relation(
                "has_associated_morphology",
                ["disease"],
                ["clinical_finding", "disease"],
                compatibility_status="compatible",
            ),
            "type_review",
        )

    def test_inverse_raw_labels_are_not_graph_relations(self):
        for raw_inverse in [
            "definitional_manifestation_of",
            "device_used_by",
            "direct_device_of",
            "measured_component_of",
        ]:
            with self.subTest(raw_inverse=raw_inverse):
                canonical_name, swapped = conn.canonicalize_raw_relation_name(
                    raw_inverse
                )
                self.assertTrue(swapped)
                self.assertIn(canonical_name, conn.RELATION_SPECS)
                self.assertNotIn(raw_inverse, conn.RELATION_TYPE_BY_NAME)

    def test_incompatible_configured_relation_is_type_review(self):
        self.assertEqual(
            conn.traversal_policy_for_relation(
                "uses_device",
                ["disease"],
                ["device"],
            ),
            "type_review",
        )
        self.assertTrue(conn.review_needed_for_policy("type_review"))



class UMLSConnectionsSchemaAndDirectionTests(unittest.TestCase):
    def test_relation_catalog_uses_only_entity_schema_v21_types(self):
        from knowledge_graph.entity_schema import ALLOWED_TYPES, ENTITY_SCHEMA_VERSION

        self.assertEqual(ENTITY_SCHEMA_VERSION, "2.1")
        obsolete = {
            "risk_factor",
            "complication_or_comorbidity",
            "imaging_modality",
        }

        for relation_name, spec in conn.RELATION_SPECS.items():
            referenced = (
                set(spec.source_types)
                | set(spec.target_types)
                | set(spec.broad_source_types)
                | set(spec.broad_target_types)
            )
            self.assertTrue(
                referenced.issubset(ALLOWED_TYPES),
                (relation_name, sorted(referenced - set(ALLOWED_TYPES))),
            )
            self.assertTrue(referenced.isdisjoint(obsolete))

    def test_inverse_isa_is_canonicalized_child_to_parent(self):
        relation_name, source_cuis, target_cuis, swapped = (
            conn.canonicalize_resolved_relation(
                "inverse_isa",
                ["C0000001"],  # raw subject: parent
                ["C0000002"],  # raw object: child
            )
        )

        self.assertEqual(relation_name, "isa")
        self.assertEqual(source_cuis, ["C0000002"])
        self.assertEqual(target_cuis, ["C0000001"])
        self.assertTrue(swapped)

    def test_forward_finding_site_keeps_subject_object_direction(self):
        relation_name, source_cuis, target_cuis, swapped = (
            conn.canonicalize_resolved_relation(
                "has_finding_site",
                ["C0000001"],
                ["C0000002"],
            )
        )

        self.assertEqual(relation_name, "has_finding_site")
        self.assertEqual(source_cuis, ["C0000001"])
        self.assertEqual(target_cuis, ["C0000002"])
        self.assertFalse(swapped)

    def test_inverse_finding_site_is_reoriented_to_forward_relation(self):
        relation_name, source_cuis, target_cuis, swapped = (
            conn.canonicalize_resolved_relation(
                "finding_site_of",
                ["C0000002"],  # anatomy
                ["C0000001"],  # disease/finding
            )
        )

        self.assertEqual(relation_name, "has_finding_site")
        self.assertEqual(source_cuis, ["C0000001"])
        self.assertEqual(target_cuis, ["C0000002"])
        self.assertTrue(swapped)

    def test_missing_related_from_id_is_not_directionally_guessed(self):
        class NoLookupClient:
            def lookup_cuis_for_source_ui(self, *args, **kwargs):
                raise AssertionError("sourceUi lookup should not be needed")

        subject_cuis, object_cuis, error = conn.resolve_relation_endpoint_cuis(
            NoLookupClient(),
            {
                "relatedId": (
                    "https://uts-ws.nlm.nih.gov/rest/content/current/"
                    "CUI/C0000002"
                )
            },
            "SNOMEDCT_US",
        )

        self.assertEqual(subject_cuis, [])
        self.assertEqual(object_cuis, [])
        self.assertEqual(error, "missing_related_from_id")

    def test_forward_and_inverse_raw_evidence_collapse_to_one_edge(self):
        representatives = conn.select_representative_concepts(
            [
                concept("s1", "heart failure", "disease", "C0000001", 0.99),
                concept(
                    "t1",
                    "heart",
                    "anatomical_structure",
                    "C0000002",
                    0.98,
                ),
            ]
        )
        collapsed = conn.build_collapsed_connections(
            edges=[
                {
                    "source_name": "heart failure",
                    "source_type": "disease",
                    "source_cui": "C0000001",
                    "target_name": "heart",
                    "target_type": "anatomical_structure",
                    "target_cui": "C0000002",
                    "umls_relation_label": "RO",
                    "umls_additional_relation_label": "has_finding_site",
                    "umls_raw_additional_relation_label": "has_finding_site",
                    "umls_relation_ui": "R-forward",
                },
                {
                    "source_name": "heart failure",
                    "source_type": "disease",
                    "source_cui": "C0000001",
                    "target_name": "heart",
                    "target_type": "anatomical_structure",
                    "target_cui": "C0000002",
                    "umls_relation_label": "RO",
                    "umls_additional_relation_label": "has_finding_site",
                    "umls_raw_additional_relation_label": "finding_site_of",
                    "umls_relation_ui": "R-inverse",
                },
            ],
            doc_id="doc-a",
            source_vocab="SNOMEDCT_US",
            umls_version="current",
            representatives_by_cui=representatives,
            materialization_mode="safe_only",
        )

        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["relation_name"], "has_finding_site")
        self.assertEqual(
            collapsed[0]["raw_relation_names"],
            ["finding_site_of", "has_finding_site"],
        )
        self.assertEqual(
            collapsed[0]["relation_ids"],
            ["R-forward", "R-inverse"],
        )
        self.assertFalse(collapsed[0]["materialization_decision"])
        self.assertEqual(
            collapsed[0]["materialization_decision_reason"],
            "not_materialize_by_default",
        )


class UMLSConnectionsExplorationSafetyV2Tests(unittest.TestCase):
    def test_discover_profile_keeps_unclassified_relation_review_only(self):
        representatives = conn.select_representative_concepts(
            [
                concept("s1", "source", "disease", "C7000001", 0.99),
                concept("t1", "target", "disease", "C7000002", 0.99),
            ]
        )
        collapsed = conn.build_collapsed_connections(
            edges=[
                {
                    "source_name": "source",
                    "source_type": "disease",
                    "source_cui": "C7000001",
                    "target_name": "target",
                    "target_type": "disease",
                    "target_cui": "C7000002",
                    "umls_relation_label": "RO",
                    "umls_additional_relation_label": "part_of",
                    "umls_relation_ui": "R-part-of",
                }
            ],
            doc_id="doc-a",
            source_vocab="SNOMEDCT_US",
            umls_version="2026AA",
            representatives_by_cui=representatives,
            materialization_mode="safe_only",
        )

        self.assertEqual(len(collapsed), 1)
        edge = collapsed[0]
        self.assertEqual(edge["relation_name"], "part_of")
        self.assertFalse(edge["relation_catalogued"])
        self.assertTrue(edge["exploratory_unclassified"])
        self.assertIsNone(edge["relationship_type"])
        self.assertEqual(edge["traversal_policy"], "type_review")
        self.assertTrue(edge["review_needed"])
        self.assertFalse(edge["materialization_decision"])
        self.assertEqual(
            edge["materialization_decision_reason"],
            "unsupported_relation",
        )

    def test_multi_type_cui_is_explicitly_marked_and_never_safe(self):
        representatives = conn.select_representative_concepts(
            [
                concept("s1", "atrial fibrillation", "disease", "C8000001", 1.0),
                concept("t1", "atrial arrhythmia", "clinical_finding", "C8000002", 1.0),
            ]
        )
        collapsed = conn.build_collapsed_connections(
            edges=[
                {
                    "source_name": "atrial fibrillation",
                    "source_type": "disease",
                    "source_cui": "C8000001",
                    "target_name": "atrial arrhythmia",
                    "target_type": "clinical_finding",
                    "target_cui": "C8000002",
                    "umls_relation_label": "CHD",
                    "umls_additional_relation_label": "isa",
                    "umls_relation_ui": "R1",
                },
                {
                    "source_name": "atrial fibrillation",
                    "source_type": "disease",
                    "source_cui": "C8000001",
                    "target_name": "atrial arrhythmias",
                    "target_type": "disease",
                    "target_cui": "C8000002",
                    "umls_relation_label": "CHD",
                    "umls_additional_relation_label": "isa",
                    "umls_relation_ui": "R2",
                },
            ],
            doc_id="doc-a",
            source_vocab="SNOMEDCT_US",
            umls_version="2026AA",
            representatives_by_cui=representatives,
            materialization_mode="safe_only",
        )

        edge = collapsed[0]
        self.assertTrue(edge["has_local_type_ambiguity"])
        self.assertEqual(edge["source_type_count"], 1)
        self.assertEqual(edge["target_type_count"], 2)
        self.assertEqual(edge["traversal_policy"], "hierarchy_review")
        self.assertTrue(edge["review_needed"])
        self.assertFalse(edge["materialization_decision"])
        self.assertEqual(
            edge["materialization_decision_reason"],
            "local_type_ambiguity",
        )

    def test_no_relation_is_approved_for_materialization_during_exploration(self):
        self.assertEqual(conn.APPROVED_MATERIALIZATION_RELATION_NAMES, frozenset())
        for relation_name, spec in conn.RELATION_SPECS.items():
            with self.subTest(relation_name=relation_name):
                self.assertFalse(spec.materialize_by_default)

class UMLSConnectionsPolicyV5Tests(unittest.TestCase):
    def test_isa_same_type_is_review_gated_even_when_compatible(self):
        compatibility = conn.evaluate_relation_compatibility(
            "isa",
            "disease",
            "disease",
        )
        self.assertEqual(compatibility.status, "compatible")
        self.assertEqual(compatibility.reason, "same_canonical_type")
        policy = conn.traversal_policy_for_relation(
            "isa",
            ["disease"],
            ["disease"],
        )
        self.assertEqual(policy, "hierarchy_review")
        self.assertTrue(conn.review_needed_for_policy(policy))

    def test_isa_spec_declares_review_gated_default(self):
        self.assertEqual(
            conn.RELATION_SPECS["isa"].default_traversal_policy,
            "hierarchy_review",
        )

    def test_causal_profile_excludes_associated_finding_but_audit_retains_it(self):
        self.assertNotIn("has_associated_finding", conn.CAUSAL_RELATION_NAMES)
        self.assertNotIn(
            "has_associated_finding",
            conn.RELATION_NAMES_BY_PROFILE["causal_relations"],
        )
        self.assertIn("has_associated_finding", conn.AUDIT_ONLY_RELATION_NAMES)
        self.assertIn(
            "has_associated_finding",
            conn.RELATION_NAMES_BY_PROFILE["audit_all"],
        )

    def test_clinical_outcome_finding_site_is_broad_and_review_gated(self):
        compatibility = conn.evaluate_relation_compatibility(
            "has_finding_site",
            "clinical_outcome",
            "anatomical_structure",
        )
        self.assertEqual(compatibility.status, "compatible_broad")
        self.assertEqual(compatibility.reason, "broad_type_match")
        policy = conn.traversal_policy_for_relation(
            "has_finding_site",
            ["clinical_outcome"],
            ["anatomical_structure"],
        )
        self.assertEqual(policy, "type_review")
        self.assertTrue(conn.review_needed_for_policy(policy))

    def test_associated_finding_allows_clinical_outcome_only_as_broad_review(self):
        compatibility = conn.evaluate_relation_compatibility(
            "has_associated_finding",
            "clinical_finding",
            "clinical_outcome",
        )
        self.assertEqual(compatibility.status, "compatible_broad")
        self.assertEqual(compatibility.reason, "broad_type_match")

        policy = conn.traversal_policy_for_relation(
            "has_associated_finding",
            ["clinical_finding"],
            ["clinical_outcome"],
        )
        self.assertEqual(policy, "type_review")
        self.assertTrue(conn.review_needed_for_policy(policy))

        reverse_compatibility = conn.evaluate_relation_compatibility(
            "has_associated_finding",
            "clinical_outcome",
            "clinical_finding",
        )
        self.assertEqual(reverse_compatibility.status, "incompatible")
        self.assertEqual(reverse_compatibility.reason, "source_type_not_allowed")


class UMLSConnectionsSanityCheckTests(unittest.TestCase):
    def test_sanity_checks_import_umls_relationship_types_dynamically(self):
        from knowledge_graph import sanity_checks

        new_types = {
            conn.RELATION_TYPE_BY_NAME[relation_name]
            for relation_name in conn.RELATION_NAMES_BY_PROFILE["audit_all"]
        }

        self.assertTrue(new_types.issubset(sanity_checks.UMLS_CONNECTION_RELATION_TYPES))
        self.assertTrue(new_types.issubset(sanity_checks.MANAGED_RELATIONSHIP_TYPES))

        check_names = {check["name"] for check in sanity_checks.CHECKS}
        self.assertIn(
            "umls_connections_missing_compatibility_metadata",
            check_names,
        )
        self.assertIn(
            "umls_connection_exact_type_rule_violations",
            check_names,
        )
        self.assertIn("safe_only_umls_connections_policy_violations", check_names)
        self.assertIn("audit_only_umls_candidates_materialized", check_names)
        self.assertIn(
            "materialized_umls_connections_with_local_type_ambiguity",
            check_names,
        )
        self.assertIn("umls_connection_counts_by_compatibility_status", check_names)
        self.assertIn("umls_connections_noncanonical_relation_names", check_names)
        self.assertIn(
            "umls_connections_entity_schema_version_mismatch",
            check_names,
        )
        review_check = next(
            check
            for check in sanity_checks.CHECKS
            if check["name"] == "review_needed_materialized_umls_connections"
        )
        self.assertEqual(review_check["level"], "WARNING")
        self.assertEqual(
            sanity_checks.UMLS_CATALOG_LOCAL_TYPE_RULES,
            conn.catalog_local_type_rule_rows(),
        )


class UMLSConnectionsNegativeCacheTests(unittest.TestCase):
    def make_client(self, cache_dir, responses, ignore_negative_cache=False):
        return conn.UMLSRelationsClient(
            cache_dir=cache_dir,
            timeout=1,
            rate_limit_per_second=0,
            version="current",
            page_size=200,
            ignore_negative_cache=ignore_negative_cache,
            session=FakeSession(responses),
        )

    def test_source_vocab_404_on_valid_cui_is_stable_empty_not_failure(self):
        cui = "C1414172"
        source_vocab = "SNOMEDCT_US"

        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {"UMLS_API_KEY": "secret-test-key"},
            clear=False,
        ):
            cache_dir = Path(tmp_dir)
            client = self.make_client(
                cache_dir,
                [
                    FakeResponse(404),
                    FakeResponse(200, {"result": {"name": "DSP gene"}}),
                ],
            )

            result = client.get_relations(cui, source_vocab)
            self.assertEqual(result.status, "source_vocab_relations_absent")
            self.assertFalse(result.from_negative_cache)
            self.assertEqual(result.records, [])
            self.assertEqual(client.stats["api_errors"], 0)
            self.assertEqual(client.stats["source_vocab_absence_checks"], 1)
            self.assertEqual(client.stats["source_vocab_absence_confirmed"], 1)

            cached_client = self.make_client(cache_dir, [])
            cached = cached_client.get_relations(cui, source_vocab)
            self.assertEqual(cached.status, "source_vocab_relations_absent")
            self.assertTrue(cached.from_negative_cache)
            self.assertEqual(cached_client.session.calls, [])

    def test_unfiltered_repeated_404_keeps_legacy_unavailable_guard(self):
        cui = "C0000001"
        source_vocab = ""

        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {"UMLS_API_KEY": "secret-test-key"},
            clear=False,
        ):
            cache_dir = Path(tmp_dir)
            first_client = self.make_client(
                cache_dir,
                [FakeResponse(404), FakeResponse(404), FakeResponse(404)],
            )

            for expected_failure_count in (1, 2):
                with self.assertRaises(conn.UMLSAPIError) as raised:
                    first_client.get_relations(cui, source_vocab)
                self.assertEqual(getattr(raised.exception, "status_code", None), 404)
                marker = first_client.get_cached_payload(
                    "relations_negative",
                    first_client.relation_negative_cache_key(cui, source_vocab),
                    count_hit=False,
                )
                self.assertEqual(marker["failure_count"], expected_failure_count)
                self.assertEqual(marker["status"], "relations_404_retryable")

            unavailable = first_client.get_relations(cui, source_vocab)
            self.assertEqual(unavailable.status, "relations_unavailable")
            self.assertFalse(unavailable.from_negative_cache)

            cached_client = self.make_client(cache_dir, [])
            cached_unavailable = cached_client.get_relations(cui, source_vocab)
            self.assertEqual(cached_unavailable.status, "relations_unavailable")
            self.assertTrue(cached_unavailable.from_negative_cache)
            self.assertEqual(cached_client.session.calls, [])



    def test_forced_success_clears_stale_relations_negative_marker(self):
        cui = "C0000001"
        source_vocab = ""

        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {"UMLS_API_KEY": "secret-test-key"},
            clear=False,
        ):
            cache_dir = Path(tmp_dir)
            first_client = self.make_client(
                cache_dir,
                [
                    FakeResponse(404),
                    FakeResponse(404),
                    FakeResponse(404),
                ],
            )

            for expected_failure_count in (1, 2):
                with self.assertRaises(conn.UMLSAPIError) as raised:
                    first_client.get_relations(cui, source_vocab)
                self.assertEqual(getattr(raised.exception, "status_code", None), 404)

                marker = first_client.get_cached_payload(
                    "relations_negative",
                    first_client.relation_negative_cache_key(cui, source_vocab),
                    count_hit=False,
                )
                self.assertEqual(marker["failure_count"], expected_failure_count)
                self.assertEqual(marker["status"], "relations_404_retryable")

            unavailable = first_client.get_relations(cui, source_vocab)
            self.assertEqual(unavailable.status, "relations_unavailable")
            self.assertFalse(unavailable.from_negative_cache)
            self.assertEqual(unavailable.records, [])

            cached_client = self.make_client(cache_dir, [])
            cached_unavailable = cached_client.get_relations(cui, source_vocab)
            self.assertEqual(cached_unavailable.status, "relations_unavailable")
            self.assertTrue(cached_unavailable.from_negative_cache)
            self.assertEqual(cached_client.session.calls, [])

            forced_client = self.make_client(
                cache_dir,
                [FakeResponse(200, relation_payload())],
                ignore_negative_cache=True,
            )
            fetched = forced_client.get_relations(cui, source_vocab)
            self.assertEqual(fetched.status, "processed")
            self.assertEqual(len(fetched.records), 1)
            self.assertEqual(len(forced_client.session.calls), 1)

            stale_marker = forced_client.get_cached_payload(
                "relations_negative",
                forced_client.relation_negative_cache_key(cui, source_vocab),
                count_hit=False,
            )
            self.assertIsNone(stale_marker)

            next_client = self.make_client(cache_dir, [])
            cached_positive = next_client.get_relations(cui, source_vocab)
            self.assertEqual(cached_positive.status, "processed")
            self.assertFalse(cached_positive.from_negative_cache)
            self.assertEqual(len(cached_positive.records), 1)
            self.assertEqual(next_client.session.calls, [])


class UMLSConnectionsMaterializationTests(unittest.TestCase):
    def test_representative_prefers_descriptive_name_after_score_tie(self):
        concepts = [
            concept("e2", "AF", "condition", "C0004238", 0.95),
            concept("e1", "atrial fibrillation", "condition", "C0004238", 0.95),
        ]

        representatives = conn.select_representative_concepts(concepts)

        self.assertEqual(representatives["C0004238"].concept_id, "e1")

    def test_collapsed_connections_are_machine_readable_and_whitelisted(self):
        representatives = conn.select_representative_concepts(
            [
                concept("s1", "heart failure", "disease", "C0018801", 0.99),
                concept(
                    "t1",
                    "heart structure",
                    "anatomical_structure",
                    "C0018787",
                    0.98,
                ),
            ]
        )
        edges = [
            {
                "doc_id": "doc-a",
                "source_concept_id": "s1",
                "source_name": "heart failure",
                "source_type": "disease",
                "source_cui": "C0018801",
                "target_concept_id": "t1",
                "target_name": "heart structure",
                "target_type": "anatomical_structure",
                "target_cui": "C0018787",
                "umls_relation_label": "RO",
                "umls_additional_relation_label": "has_finding_site",
                "umls_relation_ui": "R1",
            },
            {
                "doc_id": "doc-a",
                "source_concept_id": "s1",
                "source_name": "heart failure",
                "source_type": "disease",
                "source_cui": "C0018801",
                "target_concept_id": "t1",
                "target_name": "heart structure",
                "target_type": "anatomical_structure",
                "target_cui": "C0018787",
                "umls_relation_label": "RO",
                "umls_additional_relation_label": "has_finding_site",
                "umls_relation_ui": "R2",
            },
        ]

        collapsed = conn.build_collapsed_connections(
            edges=edges,
            doc_id="doc-a",
            source_vocab="SNOMEDCT_US",
            umls_version="current",
            representatives_by_cui=representatives,
            materialization_mode="safe_only",
        )

        self.assertEqual(len(collapsed), 1)
        edge = collapsed[0]
        self.assertEqual(edge["relationship_type"], "UMLS_HAS_FINDING_SITE")
        self.assertEqual(edge["traversal_policy"], "safe")
        self.assertFalse(edge["review_needed"])
        self.assertEqual(edge["raw_rows"], 2)
        self.assertEqual(edge["relation_ids"], ["R1", "R2"])
        self.assertEqual(edge["source_representative"]["concept_id"], "s1")
        self.assertTrue(edge["local_type_compatible"])
        self.assertEqual(edge["local_type_compatibility_reason"], "compatible")
        self.assertEqual(edge["compatibility_status"], "compatible")
        self.assertEqual(edge["compatibility_reason"], "compatible")
        self.assertEqual(edge["relation_family"], "semantic_seed")
        self.assertFalse(edge["materialize_by_default"])
        self.assertFalse(edge["materialization_decision"])
        self.assertEqual(edge["materialization_decision_reason"], "not_materialize_by_default")
        self.assertTrue(edge["relation_catalogued"])
        self.assertFalse(edge["has_local_type_ambiguity"])
        self.assertEqual(edge["representative_source_type"], "disease")
        self.assertEqual(edge["representative_target_type"], "anatomical_structure")
        self.assertEqual(edge["relationship_family"], "ontology")
        self.assertEqual(edge["provenance"], "umls_connections")
        self.assertEqual(edge["provenance_source"], "umls_metathesaurus")
        self.assertEqual(edge["provenance_method"], "umls_relations_api")
        self.assertEqual(edge["source_vocabulary"], "SNOMEDCT_US")
        self.assertIn("edge_key", edge)

        stats = conn.build_collapsed_connection_statistics(collapsed)
        self.assertEqual(stats["counts_by_compatibility_status"], {"compatible": 1})
        self.assertEqual(stats["counts_by_materialize_by_default"], {"false": 1})
        self.assertEqual(stats["counts_by_materialization_decision"], {"false": 1})
        self.assertEqual(
            stats["examples_by_relation_and_compatibility_status"][0][
                "relation_name"
            ],
            "has_finding_site",
        )

    def test_compatible_first_extension_edge_remains_exploratory_and_unmaterialized(self):
        representatives = conn.select_representative_concepts(
            [
                concept("s1", "cardiac MRI", "diagnostic_test", "C1000001", 0.99),
                concept("t1", "MRI scanner", "device", "C1000002", 0.98),
            ]
        )
        collapsed = conn.build_collapsed_connections(
            edges=[
                {
                    "source_name": "cardiac MRI",
                    "source_type": "diagnostic_test",
                    "source_cui": "C1000001",
                    "target_name": "MRI scanner",
                    "target_type": "device",
                    "target_cui": "C1000002",
                    "umls_relation_label": "RO",
                    "umls_additional_relation_label": "uses_device",
                    "umls_relation_ui": "R-device",
                }
            ],
            doc_id="doc-a",
            source_vocab="SNOMEDCT_US",
            umls_version="current",
            representatives_by_cui=representatives,
        )

        edge = collapsed[0]
        self.assertEqual(edge["relationship_type"], "UMLS_USES_DEVICE")
        self.assertTrue(edge["local_type_compatible"])
        self.assertEqual(edge["local_type_compatibility_reason"], "compatible")
        self.assertEqual(edge["representative_source_type"], "diagnostic_test")
        self.assertEqual(edge["representative_target_type"], "device")
        self.assertEqual(edge["traversal_policy"], "safe")
        self.assertFalse(edge["review_needed"])
        self.assertFalse(edge["materialize_by_default"])
        self.assertEqual(edge["relationship_family"], "ontology")
        self.assertEqual(edge["provenance"], "umls_connections")
        self.assertEqual(edge["provenance_source"], "umls_metathesaurus")
        self.assertEqual(edge["provenance_method"], "umls_relations_api")

        driver = FakeNeo4jDriver()
        report = conn.materialize_collapsed_connections(
            driver,
            collapsed,
            materialization_mode="safe_only",
        )

        self.assertEqual(report["relationships_written"], 0)
        self.assertEqual(len(driver.store), 0)
        self.assertEqual(report["skipped_not_materialize_by_default"], 1)
        self.assertEqual(report["compatible_first_extension_connections"], 1)
        self.assertEqual(report["incompatible_first_extension_connections"], 0)
        self.assertEqual(
            report["relationships"][0]["materialization_status"],
            "skipped_not_materialize_by_default",
        )

    def test_incompatible_first_extension_edge_is_reported_but_not_materialized(self):
        representatives = conn.select_representative_concepts(
            [
                concept("s1", "heart failure", "disease", "C2000001", 0.99),
                concept("t1", "MRI scanner", "device", "C2000002", 0.98),
            ]
        )
        collapsed = conn.build_collapsed_connections(
            edges=[
                {
                    "source_name": "heart failure",
                    "source_type": "disease",
                    "source_cui": "C2000001",
                    "target_name": "MRI scanner",
                    "target_type": "device",
                    "target_cui": "C2000002",
                    "umls_relation_label": "RO",
                    "umls_additional_relation_label": "uses_device",
                    "umls_relation_ui": "R-incompatible",
                }
            ],
            doc_id="doc-a",
            source_vocab="SNOMEDCT_US",
            umls_version="current",
            representatives_by_cui=representatives,
        )

        edge = collapsed[0]
        self.assertFalse(edge["local_type_compatible"])
        self.assertEqual(
            edge["local_type_compatibility_reason"],
            "source_type_not_allowed",
        )
        self.assertEqual(edge["traversal_policy"], "type_review")
        self.assertTrue(edge["review_needed"])

        driver = FakeNeo4jDriver()
        report = conn.materialize_collapsed_connections(
            driver,
            collapsed,
            materialization_mode="safe_only",
        )

        self.assertEqual(len(driver.store), 0)
        self.assertEqual(report["relationships_written"], 0)
        self.assertEqual(report["skipped_incompatible_local_types"], 1)
        self.assertEqual(report["incompatible_first_extension_connections"], 1)
        self.assertEqual(
            report["relationships"][0]["materialization_status"],
            "skipped_compatibility_status_incompatible",
        )

    def test_audit_all_none_mode_performs_no_writes(self):
        representatives = conn.select_representative_concepts(
            [
                concept("s1", "myocarditis", "disease", "C3000001", 0.99),
                concept("t1", "viral exposure", "exposure_or_lifestyle_factor", "C3000002", 0.98),
            ]
        )
        collapsed = conn.build_collapsed_connections(
            edges=[
                {
                    "source_name": "myocarditis",
                    "source_type": "disease",
                    "source_cui": "C3000001",
                    "target_name": "viral exposure",
                    "target_type": "exposure_or_lifestyle_factor",
                    "target_cui": "C3000002",
                    "umls_relation_label": "RO",
                    "umls_additional_relation_label": "has_causative_agent",
                    "umls_relation_ui": "R-candidate",
                }
            ],
            doc_id="doc-a",
            source_vocab="SNOMEDCT_US",
            umls_version="current",
            representatives_by_cui=representatives,
            materialization_mode="none",
        )

        edge = collapsed[0]
        self.assertEqual(edge["relation_family"], "audit_candidate")
        self.assertFalse(edge["materialize_by_default"])
        self.assertFalse(edge["materialization_decision"])
        self.assertEqual(
            edge["materialization_decision_reason"],
            "materialization_mode_none",
        )

        driver = FakeNeo4jDriver()
        report = conn.materialize_collapsed_connections(
            driver,
            collapsed,
            materialization_mode="none",
        )

        self.assertEqual(len(driver.store), 0)
        self.assertFalse(report["write_neo4j"])
        self.assertEqual(report["relationships_written"], 0)
        self.assertEqual(
            report["skipped_by_materialization_reason"],
            {"materialization_mode_none": 1},
        )

    def test_review_only_candidate_is_not_materialized_in_safe_only_mode(self):
        representatives = conn.select_representative_concepts(
            [
                concept("s1", "myocarditis", "disease", "C3100001", 0.99),
                concept("t1", "viral exposure", "exposure_or_lifestyle_factor", "C3100002", 0.98),
            ]
        )
        collapsed = conn.build_collapsed_connections(
            edges=[
                {
                    "source_name": "myocarditis",
                    "source_type": "disease",
                    "source_cui": "C3100001",
                    "target_name": "viral exposure",
                    "target_type": "exposure_or_lifestyle_factor",
                    "target_cui": "C3100002",
                    "umls_relation_label": "RO",
                    "umls_additional_relation_label": "has_causative_agent",
                    "umls_relation_ui": "R-review-only",
                }
            ],
            doc_id="doc-a",
            source_vocab="SNOMEDCT_US",
            umls_version="current",
            representatives_by_cui=representatives,
        )

        self.assertEqual(collapsed[0]["compatibility_status"], "compatible")
        self.assertFalse(collapsed[0]["materialize_by_default"])

        driver = FakeNeo4jDriver()
        report = conn.materialize_collapsed_connections(
            driver,
            collapsed,
            materialization_mode="safe_only",
        )

        self.assertEqual(len(driver.store), 0)
        self.assertEqual(report["relationships_written"], 0)
        self.assertEqual(report["skipped_not_materialize_by_default"], 0)
        self.assertEqual(report["skipped_incompatible_local_types"], 1)
        self.assertEqual(
            report["relationships"][0]["materialization_status"],
            "skipped_traversal_policy_review",
        )

    def test_safe_only_rejects_unapproved_safe_and_hierarchy_edges(self):
        safe_edge = {
            "relation_name": "has_finding_site",
            "relationship_type": "UMLS_HAS_FINDING_SITE",
            "source_representative": {"concept_id": "s"},
            "target_representative": {"concept_id": "t"},
            "compatibility_status": "compatible",
            "local_type_compatible": True,
            "traversal_policy": "safe",
            "review_needed": False,
            "materialize_by_default": True,
        }
        hierarchy_edge = {
            **safe_edge,
            "relation_name": "isa",
            "relationship_type": "UMLS_ISA",
            "traversal_policy": "hierarchy",
        }

        self.assertEqual(
            conn.should_materialize_relation(safe_edge, "safe_only"),
            (False, "not_materialize_by_default"),
        )
        self.assertEqual(
            conn.should_materialize_relation(hierarchy_edge, "safe_only"),
            (False, "not_materialize_by_default"),
        )

    def test_safe_only_rejects_non_strict_candidates(self):
        base = {
            "relation_name": "has_finding_site",
            "relationship_type": "UMLS_HAS_FINDING_SITE",
            "source_representative": {"concept_id": "s"},
            "target_representative": {"concept_id": "t"},
            "compatibility_status": "compatible",
            "local_type_compatible": True,
            "traversal_policy": "safe",
            "review_needed": False,
            "materialize_by_default": True,
        }
        cases = [
            ({**base, "compatibility_status": "compatible_broad"}, "compatibility_status_compatible_broad"),
            ({**base, "review_needed": True}, "review_needed"),
            ({**base, "traversal_policy": "reverse_review"}, "traversal_policy_reverse_review"),
            ({**base, "source_representative": None}, "missing_representative"),
            (
                {
                    **base,
                    "relation_name": "has_causative_agent",
                    "relationship_type": "UMLS_HAS_CAUSATIVE_AGENT",
                },
                "not_materialize_by_default",
            ),
            ({**base, "local_type_compatible": False}, "local_type_not_compatible"),
        ]

        for edge, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                decision, reason = conn.should_materialize_relation(edge, "safe_only")
                self.assertFalse(decision)
                self.assertEqual(reason, expected_reason)

    def test_repeated_exploration_materialization_attempts_write_nothing(self):
        representatives = conn.select_representative_concepts(
            [
                concept("s1", "heart failure", "disease", "C0018801", 0.99),
                concept(
                    "t1",
                    "heart structure",
                    "anatomical_structure",
                    "C0018787",
                    0.98,
                ),
            ]
        )
        collapsed = conn.build_collapsed_connections(
            edges=[
                {
                    "source_name": "heart failure",
                    "source_type": "disease",
                    "source_cui": "C0018801",
                    "target_name": "heart structure",
                    "target_type": "anatomical_structure",
                    "target_cui": "C0018787",
                    "umls_relation_label": "RO",
                    "umls_additional_relation_label": "has_finding_site",
                    "umls_relation_ui": "R1",
                }
            ],
            doc_id="doc-a",
            source_vocab="SNOMEDCT_US",
            umls_version="current",
            representatives_by_cui=representatives,
        )
        driver = FakeNeo4jDriver()

        first_report = conn.materialize_collapsed_connections(
            driver,
            collapsed,
            materialization_mode="safe_only",
        )
        second_report = conn.materialize_collapsed_connections(
            driver,
            collapsed,
            materialization_mode="safe_only",
        )

        self.assertEqual(len(driver.store), 0)
        self.assertEqual(first_report["relationships_written"], 0)
        self.assertEqual(second_report["relationships_written"], 0)
        self.assertEqual(first_report["skipped_not_materialize_by_default"], 1)
        self.assertEqual(second_report["skipped_not_materialize_by_default"], 1)
        self.assertEqual(second_report["counts_by_relationship_type"], {})


class UMLSConnectionsCUISelectionTests(unittest.TestCase):
    def test_empty_include_selects_all_with_skip_and_max_last(self):
        concepts = [
            concept("c1", "one", "disease", "C0000001", 0.9),
            concept("c2", "two", "disease", "C0000002", 0.9),
            concept("c3", "three", "disease", "C0000003", 0.9),
        ]
        by_cui = conn.concepts_by_cui(concepts)

        selected, skipped = conn.select_source_cuis(
            local_cuis=set(by_cui),
            include_cuis=[],
            skip_cuis=["c0000002"],
            max_cuis=1,
            by_cui=by_cui,
        )

        self.assertEqual(selected, ["C0000001"])
        self.assertEqual(
            [(item["cui"], item["reason"]) for item in skipped],
            [("C0000002", "requested_skip"), ("C0000003", "max_cuis_limit")],
        )

    def test_include_is_applied_before_skip_and_reports_missing_requests(self):
        concepts = [
            concept("c1", "one", "disease", "C0000001", 0.9),
            concept("c3", "three", "disease", "C0000003", 0.9),
        ]
        by_cui = conn.concepts_by_cui(concepts)

        selected, skipped = conn.select_source_cuis(
            local_cuis=set(by_cui),
            include_cuis=["c0000003", "C9999999"],
            skip_cuis=["C0000003", "C8888888"],
            max_cuis=None,
            by_cui=by_cui,
        )

        self.assertEqual(selected, [])
        self.assertEqual(
            [(item["cui"], item["reason"]) for item in skipped],
            [
                ("C9999999", "requested_include_not_in_local_cuis"),
                ("C8888888", "requested_skip_not_in_local_cuis"),
                ("C0000003", "requested_skip"),
            ],
        )


class UMLSConnectionsReplacementTests(unittest.TestCase):
    def test_replacement_configuration_is_strict(self):
        with self.assertRaisesRegex(ValueError, "write_neo4j=true"):
            conn.run_umls_connections(
                doc_id="doc-a",
                write_neo4j=False,
                replace_existing_connections=True,
                materialization_mode="safe_only",
                driver=FakeNeo4jDriver(),
            )

        with self.assertRaisesRegex(ValueError, "materialization is disabled"):
            conn.run_umls_connections(
                doc_id="doc-a",
                write_neo4j=True,
                replace_existing_connections=True,
                materialization_mode="none",
                driver=FakeNeo4jDriver(),
            )

    def test_cleanup_is_scoped_to_doc_provenance_and_managed_types(self):
        driver = FakeNeo4jDriver()
        driver.store[("s", "t", "UMLS_HAS_FINDING_SITE", "old")] = {
            "relationship_id": "old",
            "params": {"doc_id": "doc-a", "provenance": "umls_connections"},
        }
        driver.store[("s", "t", "UMLS_USES_DEVICE", "other-doc")] = {
            "relationship_id": "other-doc",
            "params": {"doc_id": "doc-b", "provenance": "umls_connections"},
        }
        driver.store[("s", "t", "UMLS_HAS_FINDING_SITE", "manual")] = {
            "relationship_id": "manual",
            "params": {"doc_id": "doc-a", "provenance": "manual"},
        }
        driver.store[("s", "t", "SAME_AS", "same")] = {
            "relationship_id": "same",
            "params": {"doc_id": "doc-a", "provenance": "umls_connections"},
        }

        deleted = conn.delete_existing_umls_connections_for_doc(
            FakeNeo4jTx(driver.store),
            "doc-a",
        )

        self.assertEqual(deleted, 1)
        self.assertNotIn(("s", "t", "UMLS_HAS_FINDING_SITE", "old"), driver.store)
        self.assertIn(("s", "t", "UMLS_USES_DEVICE", "other-doc"), driver.store)
        self.assertIn(("s", "t", "UMLS_HAS_FINDING_SITE", "manual"), driver.store)
        self.assertIn(("s", "t", "SAME_AS", "same"), driver.store)

    def test_replacement_refuses_deletion_when_nothing_is_approved(self):
        representatives = conn.select_representative_concepts(
            [
                concept("s1", "heart failure", "disease", "C0018801", 0.99),
                concept("t1", "heart structure", "anatomical_structure", "C0018787", 0.98),
            ]
        )
        collapsed = conn.build_collapsed_connections(
            edges=[
                {
                    "source_name": "heart failure",
                    "source_type": "disease",
                    "source_cui": "C0018801",
                    "target_name": "heart structure",
                    "target_type": "anatomical_structure",
                    "target_cui": "C0018787",
                    "umls_relation_label": "RO",
                    "umls_additional_relation_label": "has_finding_site",
                    "umls_relation_ui": "R1",
                }
            ],
            doc_id="doc-a",
            source_vocab="SNOMEDCT_US",
            umls_version="current",
            representatives_by_cui=representatives,
            materialization_mode="safe_only",
        )
        driver = FakeNeo4jDriver()
        driver.store[("old-s", "old-t", "UMLS_HAS_FINDING_SITE", "old")] = {
            "relationship_id": "old",
            "params": {"doc_id": "doc-a", "provenance": "umls_connections"},
        }

        with self.assertRaisesRegex(ValueError, "no collapsed relation"):
            conn.materialize_collapsed_connections(
                driver,
                collapsed,
                materialization_mode="safe_only",
                doc_id="doc-a",
                replace_existing_connections=True,
            )

        self.assertIn(
            ("old-s", "old-t", "UMLS_HAS_FINDING_SITE", "old"),
            driver.store,
        )


class UMLSConnectionsGraphStatisticsTests(unittest.TestCase):
    def test_graph_statistics_cover_degree_and_components(self):
        edges = [
            {
                "source_cui": "C0000001",
                "target_cui": "C0000002",
                "materialization_decision": True,
            },
            {
                "source_cui": "C0000002",
                "target_cui": "C0000003",
                "materialization_decision": False,
            },
        ]

        stats = conn.compute_cui_graph_statistics(
            edges,
            eligible_cuis=["C0000001", "C0000002", "C0000003", "C0000004"],
            selected_source_cuis=["C0000001", "C0000002"],
        )

        self.assertEqual(stats["unique_connected_cuis"], 3)
        self.assertEqual(stats["isolated_eligible_local_cui_count"], 1)
        self.assertEqual(stats["cui_coverage_ratio"], 0.75)
        self.assertEqual(stats["directed_edges"], 2)
        self.assertEqual(stats["unique_undirected_cui_pairs"], 2)
        self.assertEqual(stats["minimum_degree"], 0)
        self.assertEqual(stats["median_degree"], 1.0)
        self.assertEqual(stats["mean_degree"], 1.0)
        self.assertEqual(stats["p95_degree"], 2)
        self.assertEqual(stats["maximum_degree"], 2)
        self.assertEqual(stats["weakly_connected_components"], 2)
        self.assertEqual(stats["largest_component_size"], 3)

        collapsed_stats = conn.build_collapsed_connection_statistics(
            edges,
            eligible_cuis=["C0000001", "C0000002", "C0000003", "C0000004"],
            selected_source_cuis=["C0000001", "C0000002"],
        )
        self.assertEqual(
            collapsed_stats["candidate_graph_statistics"]["unique_connected_cuis"],
            3,
        )
        self.assertEqual(
            collapsed_stats["materializable_graph_statistics"]["unique_connected_cuis"],
            2,
        )


class UMLSConnectionsRelationCensusTests(unittest.TestCase):
    def test_relation_census_reports_source_coverage_and_hub_concentration(self):
        stats = {
            "eligible_local_cui_count": 4,
            "observed_canonical_relation_names": {
                "isa": 4,
                "has_focus": 2,
            },
            "_observed_canonical_relation_source_cui_counts": {
                "isa": {"C1": 3, "C2": 1},
                "has_focus": {"C2": 1, "C3": 1},
            },
            "_observed_raw_names_by_canonical_relation": {
                "isa": {"isa": 2, "inverse_isa": 2},
                "has_focus": {"has_focus": 2},
            },
            "_observed_relation_examples": {
                "isa": [{"query_cui": "C1", "raw_relation_name": "isa"}],
            },
        }

        census = conn.build_relation_census_statistics(stats)

        self.assertEqual(census["relation_bearing_source_cui_count"], 3)
        self.assertAlmostEqual(census["relation_bearing_source_cui_coverage"], 0.75)
        isa = census["relations"][0]
        self.assertEqual(isa["relation_name"], "isa")
        self.assertEqual(isa["distinct_source_cuis"], 2)
        self.assertAlmostEqual(isa["coverage_over_eligible_local_cuis"], 0.5)
        self.assertAlmostEqual(isa["coverage_over_relation_bearing_cuis"], 2 / 3)
        self.assertEqual(isa["max_rows_from_one_cui"], 3)
        self.assertAlmostEqual(isa["top_source_cui_share"], 0.75)
        self.assertEqual(isa["p50_rows_per_source_cui"], 2.0)
        self.assertEqual(isa["p95_rows_per_source_cui"], 3)
        self.assertEqual(
            isa["top_source_cuis"],
            [
                {"cui": "C1", "relation_records": 3},
                {"cui": "C2", "relation_records": 1},
            ],
        )
        self.assertEqual(isa["examples"][0]["query_cui"], "C1")

    def test_record_observed_relation_census_keeps_compact_examples(self):
        stats = {}
        for index in range(8):
            conn.record_observed_relation_census(
                stats,
                source_cui=f"C{index}",
                raw_relation_name="focus_of",
                canonical_relation_name="focus_of",
                record={
                    "relationLabel": "RO",
                    "additionalRelationLabel": "focus_of",
                    "rootSource": "SNOMEDCT_US",
                    "relatedFromId": f"from-{index}",
                    "relatedId": f"to-{index}",
                },
            )

        self.assertEqual(
            len(stats["_observed_relation_examples"]["focus_of"]),
            conn.RELATION_CENSUS_EXAMPLE_LIMIT,
        )
        self.assertEqual(
            len(stats["_observed_canonical_relation_source_cui_counts"]["focus_of"]),
            8,
        )

    def test_inverse_pair_candidates_identify_unmerged_and_known_pairs(self):
        stats = {
            "observed_raw_relation_names": {
                "isa": 10,
                "inverse_isa": 20,
                "has_finding_site": 3,
                "finding_site_of": 7,
                "has_focus": 1,
                "focus_of": 9,
            }
        }

        pairs = conn.build_observed_inverse_pair_candidates(stats)
        by_pair = {
            (row["forward_relation_name"], row["inverse_relation_name"]): row
            for row in pairs
        }

        self.assertTrue(by_pair[("isa", "inverse_isa")]["canonicalized_together"])
        self.assertEqual(
            by_pair[("has_finding_site", "finding_site_of")][
                "canonical_relation_name"
            ],
            "has_finding_site",
        )
        self.assertTrue(
            by_pair[("has_focus", "focus_of")]["canonicalized_together"]
        )
        self.assertEqual(
            by_pair[("has_focus", "focus_of")]["canonical_relation_name"],
            "has_focus",
        )

    def test_public_relation_stats_hides_private_accumulators(self):
        stats = {
            "eligible_local_cui_count": 1,
            "observed_raw_relation_names": {"isa": 1},
            "observed_canonical_relation_names": {"isa": 1},
            "_observed_canonical_relation_source_cui_counts": {
                "isa": {"C1": 1}
            },
            "_observed_raw_names_by_canonical_relation": {"isa": {"isa": 1}},
            "_observed_relation_examples": {},
        }

        public = conn.public_relation_stats(stats)

        self.assertNotIn("_observed_canonical_relation_source_cui_counts", public)
        self.assertIn("relation_census", public)
        self.assertIn("observed_inverse_pair_candidates", public)



class UMLSConnectionsRunNameTests(unittest.TestCase):
    def test_explicit_run_name_is_sanitized(self):
        self.assertEqual(
            conn.resolve_run_name(run_name="core safe/only!"),
            "core_safe_only",
        )

    def test_default_run_name_is_deterministic(self):
        self.assertEqual(
            conn.resolve_run_name(
                relation_profile="core",
                materialization_mode="safe_only",
                max_relations_per_cui=500,
                max_source_ui_lookups_per_cui=100,
            ),
            "core__safe_only__r500__ui100",
        )
        self.assertNotEqual(
            conn.resolve_run_name(
                relation_profile="core",
                materialization_mode="safe_only",
                max_relations_per_cui=500,
                max_source_ui_lookups_per_cui=100,
            ),
            conn.resolve_run_name(
                relation_profile="expanded",
                materialization_mode="safe_only",
                max_relations_per_cui=500,
                max_source_ui_lookups_per_cui=100,
            ),
        )

    def test_output_paths_share_resolved_run_directory(self):
        run_dir = conn.resolve_run_output_dir(Path("/tmp/umls"), "core__safe_only__r500__ui100")
        csv_path, summary_path = conn.output_paths("doc-a", run_dir)
        collapsed_path = conn.collapsed_connections_path("doc-a", run_dir)

        self.assertEqual(csv_path.parent, run_dir)
        self.assertEqual(summary_path.parent, run_dir)
        self.assertEqual(collapsed_path.parent, run_dir)


if __name__ == "__main__":
    unittest.main()
