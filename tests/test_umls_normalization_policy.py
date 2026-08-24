"""Offline regression tests for the UMLS normalization safety-v1 patch.

These tests must not call the UMLS API and must not connect to Neo4j.
Run from the repository root with:

    PYTHONPATH=src python -m unittest \
      tests.test_umls_normalization_policy -v
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from knowledge_graph.umls_normalization import (
    FUZZY_ONLY_BACKEND,
    SEMANTIC_COMPATIBLE,
    SEMANTIC_CONTEXTUAL_MISMATCH,
    SEMANTIC_INCOMPATIBLE,
    SEMANTIC_TYPES_MISSING,
    SEMANTIC_UNKNOWN_LOCAL_TYPE,
    ConceptRecord,
    UMLSAPIClient,
    UMLSMatch,
    build_aliases_for_concept,
    build_existing_umls_match,
    build_review_record,
    classify_semantic_compatibility,
    is_confident_umls_match,
    normalize_concepts_with_umls,
    semantic_types_are_compatible,
    setup_normalization_schema,
)


class FakeSession:
    """Minimal Neo4j session double that records reads and writes."""

    def __init__(self, concepts=None):
        self.concepts = list(concepts or [])
        self.read_calls = []
        self.write_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_read(self, function, *args, **kwargs):
        self.read_calls.append((function, args, kwargs))
        return list(self.concepts)

    def execute_write(self, function, *args, **kwargs):
        self.write_calls.append((function, args, kwargs))
        return None


class FakeDriver:
    def __init__(self, session):
        self.fake_session = session
        self.session_calls = 0

    def session(self):
        self.session_calls += 1
        return self.fake_session


class SemanticCompatibilityTests(unittest.TestCase):
    def test_risk_factor_disease_is_contextual_mismatch(self):
        semantic_types = ["Disease or Syndrome"]
        self.assertEqual(
            classify_semantic_compatibility("risk_factor", semantic_types),
            SEMANTIC_CONTEXTUAL_MISMATCH,
        )
        self.assertIsNone(
            semantic_types_are_compatible("risk_factor", semantic_types)
        )

    def test_care_strategy_drug_is_contextual_mismatch(self):
        semantic_types = ["Pharmacologic Substance"]
        self.assertEqual(
            classify_semantic_compatibility("care_strategy", semantic_types),
            SEMANTIC_CONTEXTUAL_MISMATCH,
        )
        self.assertIsNone(
            semantic_types_are_compatible("care_strategy", semantic_types)
        )

    def test_clinical_outcome_population_is_contextual_mismatch(self):
        semantic_types = ["Population Group"]
        self.assertEqual(
            classify_semantic_compatibility("clinical_outcome", semantic_types),
            SEMANTIC_CONTEXTUAL_MISMATCH,
        )

    def test_explicit_compatibility_remains_true(self):
        self.assertEqual(
            classify_semantic_compatibility(
                "disease", ["Disease or Syndrome"]
            ),
            SEMANTIC_COMPATIBLE,
        )
        self.assertTrue(
            semantic_types_are_compatible(
                "disease", ["Disease or Syndrome"]
            )
        )

    def test_intrinsic_incompatibility_remains_false(self):
        self.assertEqual(
            classify_semantic_compatibility(
                "disease", ["Pharmacologic Substance"]
            ),
            SEMANTIC_INCOMPATIBLE,
        )
        self.assertFalse(
            semantic_types_are_compatible(
                "disease", ["Pharmacologic Substance"]
            )
        )

    def test_missing_semantic_types_are_explicitly_classified(self):
        self.assertEqual(
            classify_semantic_compatibility("disease", []),
            SEMANTIC_TYPES_MISSING,
        )
        self.assertIsNone(semantic_types_are_compatible("disease", []))

    def test_unknown_local_type_is_explicitly_classified(self):
        self.assertEqual(
            classify_semantic_compatibility(
                "not_a_controlled_type", ["Finding"]
            ),
            SEMANTIC_UNKNOWN_LOCAL_TYPE,
        )
        self.assertIsNone(
            semantic_types_are_compatible(
                "not_a_controlled_type", ["Finding"]
            )
        )


class CandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def client_with_payload(payload):
        client = object.__new__(UMLSAPIClient)
        client.request_search = lambda alias, search_type: payload
        return client

    def test_contextual_candidate_survives_candidate_generation(self):
        client = self.client_with_payload(
            {
                "result": {
                    "results": [
                        {
                            "ui": "C0011849",
                            "name": "Diabetes",
                            "semanticTypes": ["Disease or Syndrome"],
                        }
                    ]
                }
            }
        )

        match = client.search_alias(
            alias="diabetes",
            search_type="exact",
            canonical_type="risk_factor",
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.cui, "C0011849")
        self.assertEqual(
            match.semantic_compatibility,
            SEMANTIC_CONTEXTUAL_MISMATCH,
        )
        self.assertIsNone(match.type_compatible)
        self.assertEqual(match.score, 0.98)

    def test_strongly_incompatible_intrinsic_candidate_is_rejected(self):
        client = self.client_with_payload(
            {
                "result": {
                    "results": [
                        {
                            "ui": "C9999999",
                            "name": "Cardiomyopathy",
                            "semanticTypes": ["Pharmacologic Substance"],
                        }
                    ]
                }
            }
        )

        match = client.search_alias(
            alias="cardiomyopathy",
            search_type="exact",
            canonical_type="disease",
        )
        self.assertIsNone(match)

    def test_contextual_exact_candidate_can_pass_exact_threshold(self):
        match = UMLSMatch(
            alias="diabetes",
            cui="C0011849",
            canonical_name="Diabetes",
            definition=None,
            aliases=[],
            score=0.98,
            semantic_types=["Disease or Syndrome"],
            search_type="exact",
            type_compatible=None,
            semantic_compatibility=SEMANTIC_CONTEXTUAL_MISMATCH,
        )
        self.assertTrue(
            is_confident_umls_match(
                match,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )

    def test_strong_incompatibility_never_auto_accepts(self):
        match = UMLSMatch(
            alias="cardiomyopathy",
            cui="C9999999",
            canonical_name="Cardiomyopathy",
            definition=None,
            aliases=[],
            score=1.0,
            semantic_types=["Pharmacologic Substance"],
            search_type="exact",
            type_compatible=False,
            semantic_compatibility=SEMANTIC_INCOMPATIBLE,
        )
        self.assertFalse(
            is_confident_umls_match(
                match,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )

    def test_words_search_remains_review_only(self):
        match = UMLSMatch(
            alias="diabetes",
            cui="C0011849",
            canonical_name="Diabetes",
            definition=None,
            aliases=[],
            score=1.0,
            semantic_types=["Disease or Syndrome"],
            search_type="words",
            type_compatible=None,
            semantic_compatibility=SEMANTIC_CONTEXTUAL_MISMATCH,
        )
        self.assertFalse(
            is_confident_umls_match(
                match,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )


class AcronymAliasTests(unittest.TestCase):
    def test_document_expansion_precedes_short_form(self):
        concept = ConceptRecord(
            concept_id="concept-1",
            name="CMR",
            canonical_type="imaging_modality",
            doc_ids=["DocA"],
        )
        aliases = build_aliases_for_concept(
            concept,
            {"DocA": {"CMR": "cardiac magnetic resonance"}},
        )
        self.assertEqual(aliases[0], "cardiac magnetic resonance")
        self.assertIn("cmr", aliases)
        self.assertLess(
            aliases.index("cardiac magnetic resonance"),
            aliases.index("cmr"),
        )

    def test_expansion_is_not_imported_from_unrelated_document(self):
        concept = ConceptRecord(
            concept_id="concept-1",
            name="CMR",
            canonical_type="imaging_modality",
            doc_ids=["DocA"],
        )
        aliases = build_aliases_for_concept(
            concept,
            {
                "DocA": {"CMR": "cardiac magnetic resonance"},
                "DocB": {"CMR": "unrelated document expansion"},
            },
        )
        self.assertIn("cardiac magnetic resonance", aliases)
        self.assertNotIn("unrelated document expansion", aliases)


class ReviewRecordTests(unittest.TestCase):
    def test_review_record_distinguishes_contextual_mismatch(self):
        concept = ConceptRecord(
            concept_id="concept-1",
            name="diabetes",
            canonical_type="risk_factor",
            best_match=UMLSMatch(
                alias="diabetes",
                cui="C0011849",
                canonical_name="Diabetes",
                definition=None,
                aliases=[],
                score=0.98,
                semantic_types=["Disease or Syndrome"],
                search_type="exact",
                type_compatible=None,
                semantic_compatibility=SEMANTIC_CONTEXTUAL_MISMATCH,
            ),
        )
        record = build_review_record(
            concept=concept,
            run_id="test-run",
            doc_id="DocA",
            model_name="UMLS REST API",
            linker_name="umls_api",
            backend="umls_api",
            threshold=0.85,
            exact_threshold=0.75,
            fuzzy_threshold=90,
        )
        self.assertIsNone(record["umls_type_compatible"])
        self.assertEqual(
            record["umls_semantic_compatibility"],
            SEMANTIC_CONTEXTUAL_MISMATCH,
        )

    def test_existing_mapping_receives_auditable_classification(self):
        concept = ConceptRecord(
            concept_id="concept-1",
            name="diabetes",
            canonical_type="risk_factor",
            properties={
                "umls_cui": "C0011849",
                "umls_canonical_name": "Diabetes",
                "umls_semantic_types": ["Disease or Syndrome"],
                "umls_score": 1.0,
            },
        )
        match = build_existing_umls_match(concept)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertIsNone(match.type_compatible)
        self.assertEqual(
            match.semantic_compatibility,
            SEMANTIC_CONTEXTUAL_MISMATCH,
        )


class DryRunSafetyTests(unittest.TestCase):
    def test_dry_run_with_no_concepts_performs_no_schema_write(self):
        session = FakeSession(concepts=[])
        driver = FakeDriver(session)

        stats = normalize_concepts_with_umls(
            driver=driver,
            backend=FUZZY_ONLY_BACKEND,
            dry_run=True,
            export_review=False,
            create_same_as_edges=True,
            create_fuzzy_candidate_edges=True,
        )

        self.assertEqual(stats["concepts_seen"], 0)
        self.assertEqual(session.write_calls, [])
        self.assertEqual(len(session.read_calls), 1)

    def test_real_run_still_creates_normalization_schema(self):
        session = FakeSession(concepts=[])
        driver = FakeDriver(session)

        stats = normalize_concepts_with_umls(
            driver=driver,
            backend=FUZZY_ONLY_BACKEND,
            dry_run=False,
            export_review=False,
        )

        self.assertEqual(stats["concepts_seen"], 0)
        self.assertEqual(len(session.write_calls), 1)
        self.assertIs(session.write_calls[0][0], setup_normalization_schema)

    def test_dry_run_error_path_performs_no_write(self):
        concept = ConceptRecord(
            concept_id="concept-1",
            name="diabetes",
            canonical_type="risk_factor",
            doc_ids=["DocA"],
        )
        session = FakeSession(concepts=[concept])
        driver = FakeDriver(session)

        with (
            patch(
                "knowledge_graph.umls_normalization.build_aliases_for_concept",
                side_effect=RuntimeError("synthetic offline failure"),
            ),
            patch("knowledge_graph.umls_normalization.logger.exception"),
        ):
            stats = normalize_concepts_with_umls(
                driver=driver,
                backend=FUZZY_ONLY_BACKEND,
                dry_run=True,
                export_review=False,
                create_same_as_edges=True,
                create_fuzzy_candidate_edges=True,
            )

        self.assertEqual(stats["concepts_failed"], 1)
        self.assertEqual(session.write_calls, [])


if __name__ == "__main__":
    unittest.main()
