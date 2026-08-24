"""Offline regressions for UMLS normalization eligibility-v1.

The suite must not call the UMLS API and must not connect to Neo4j.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from knowledge_graph.umls_normalization import (
    FUZZY_ONLY_BACKEND,
    REVIEW_REQUIRED_STATUS,
    SEMANTIC_COMPATIBLE,
    SEMANTIC_CONTEXTUAL_MISMATCH,
    TYPE_REVIEW_REQUIRED_METHOD,
    TYPE_REVIEW_REQUIRED_REASON,
    UMLS_API_BACKEND,
    ConceptRecord,
    build_review_record,
    classify_semantic_compatibility,
    concept_requires_type_review,
    create_duplicate_evidence,
    fetch_concepts_for_normalization,
    normalize_concepts_with_umls,
    semantic_types_are_compatible,
    setup_normalization_schema,
)


class FakeSession:
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

    def session(self):
        return self.fake_session


class FakeTx:
    def __init__(self, rows):
        self.rows = rows
        self.query = None
        self.parameters = None

    def run(self, query, **parameters):
        self.query = query
        self.parameters = parameters
        return list(self.rows)


class SemanticPolicySynchronizationTests(unittest.TestCase):
    def test_population_group_has_explicit_policy(self):
        self.assertEqual(
            classify_semantic_compatibility(
                "population_or_patient_group",
                ["Population Group"],
            ),
            SEMANTIC_COMPATIBLE,
        )
        self.assertTrue(
            semantic_types_are_compatible(
                "population_or_patient_group",
                ["Patient or Disabled Group"],
            )
        )

    def test_microorganism_has_explicit_policy(self):
        self.assertEqual(
            classify_semantic_compatibility(
                "microorganism_or_pathogen",
                ["Virus"],
            ),
            SEMANTIC_COMPATIBLE,
        )

    def test_exposure_has_explicit_direct_policy(self):
        self.assertEqual(
            classify_semantic_compatibility(
                "exposure_or_lifestyle_factor",
                ["Individual Behavior"],
            ),
            SEMANTIC_COMPATIBLE,
        )

    def test_exposure_substance_mismatch_is_contextual_not_veto(self):
        self.assertEqual(
            classify_semantic_compatibility(
                "exposure_or_lifestyle_factor",
                ["Pharmacologic Substance"],
            ),
            SEMANTIC_CONTEXTUAL_MISMATCH,
        )
        self.assertIsNone(
            semantic_types_are_compatible(
                "exposure_or_lifestyle_factor",
                ["Pharmacologic Substance"],
            )
        )


class EligibilityClassificationTests(unittest.TestCase):
    def test_explicit_review_flag_blocks_normalization(self):
        concept = ConceptRecord(
            concept_id="c1",
            name="electrophysiological study",
            canonical_type="procedure_or_intervention",
            needs_type_review=True,
        )
        self.assertTrue(concept_requires_type_review(concept))

    def test_ambiguous_type_blocks_even_without_flag(self):
        concept = ConceptRecord(
            concept_id="c1",
            name="anaemia",
            canonical_type="ambiguous",
            needs_type_review=False,
        )
        self.assertTrue(concept_requires_type_review(concept))

    def test_resolved_concept_remains_eligible(self):
        concept = ConceptRecord(
            concept_id="c1",
            name="cardiomyopathy",
            canonical_type="disease",
            needs_type_review=False,
        )
        self.assertFalse(concept_requires_type_review(concept))


class FetchAndReviewMetadataTests(unittest.TestCase):
    def test_fetch_preserves_type_resolution_metadata(self):
        tx = FakeTx(
            [
                {
                    "concept_id": "c1",
                    "name": "electrophysiological study",
                    "canonical_type": "procedure_or_intervention",
                    "needs_type_review": True,
                    "type_resolution_status": "ambiguous_tied_section_support",
                    "observed_types": [
                        "diagnostic_test",
                        "procedure_or_intervention",
                    ],
                    "invalid_observed_types": [],
                    "type_support_pairs": [
                        "diagnostic_test=1",
                        "procedure_or_intervention=1",
                    ],
                    "umls_cui": None,
                    "umls_canonical_name": None,
                    "umls_definition": None,
                    "umls_aliases": None,
                    "umls_score": None,
                    "umls_semantic_types": None,
                    "umls_linker_name": None,
                    "umls_model_name": None,
                    "normalized_name": None,
                    "normalization_status": None,
                    "normalization_method": None,
                    "doc_ids": ["Cardiomyopathies_2023"],
                    "acronym_rows": [],
                }
            ]
        )

        records = fetch_concepts_for_normalization(
            tx,
            "Cardiomyopathies_2023",
        )

        self.assertEqual(len(records), 1)
        concept = records[0]
        self.assertTrue(concept.needs_type_review)
        self.assertEqual(
            concept.type_resolution_status,
            "ambiguous_tied_section_support",
        )
        self.assertEqual(
            concept.observed_types,
            ["diagnostic_test", "procedure_or_intervention"],
        )
        self.assertIn("c.needs_type_review", tx.query)
        self.assertEqual(
            tx.parameters,
            {"doc_id": "Cardiomyopathies_2023"},
        )

    def test_review_record_exports_eligibility_metadata(self):
        concept = ConceptRecord(
            concept_id="c1",
            name="electrophysiological study",
            canonical_type="procedure_or_intervention",
            needs_type_review=True,
            type_resolution_status="ambiguous_tied_section_support",
            observed_types=[
                "diagnostic_test",
                "procedure_or_intervention",
            ],
            invalid_observed_types=[],
            type_support_pairs=[
                "diagnostic_test=1",
                "procedure_or_intervention=1",
            ],
            normalization_status=REVIEW_REQUIRED_STATUS,
            normalization_method=TYPE_REVIEW_REQUIRED_METHOD,
            reason=TYPE_REVIEW_REQUIRED_REASON,
        )

        record = build_review_record(
            concept=concept,
            run_id="test",
            doc_id="Cardiomyopathies_2023",
            model_name="UMLS REST API",
            linker_name="umls_api",
            backend=UMLS_API_BACKEND,
            threshold=0.85,
            exact_threshold=0.75,
            fuzzy_threshold=90,
        )

        self.assertFalse(record["normalization_eligible"])
        self.assertTrue(record["needs_type_review"])
        self.assertEqual(
            record["type_resolution_status"],
            "ambiguous_tied_section_support",
        )
        self.assertEqual(record["observed_types"], concept.observed_types)
        self.assertEqual(record["normalization_status"], REVIEW_REQUIRED_STATUS)
        self.assertEqual(record["reason"], TYPE_REVIEW_REQUIRED_REASON)


class NormalizationGateTests(unittest.TestCase):
    def test_review_required_dry_run_does_not_initialize_api_or_write(self):
        concept = ConceptRecord(
            concept_id="c1",
            name="anaemia",
            canonical_type="ambiguous",
            needs_type_review=True,
            doc_ids=["Cardiomyopathies_2023"],
        )
        session = FakeSession([concept])
        driver = FakeDriver(session)

        with patch(
            "knowledge_graph.umls_normalization.UMLSAPIClient",
            side_effect=AssertionError("API client must not be initialized"),
        ):
            stats = normalize_concepts_with_umls(
                driver=driver,
                doc_id="Cardiomyopathies_2023",
                backend=UMLS_API_BACKEND,
                dry_run=True,
                export_review=False,
                force=True,
            )

        self.assertEqual(stats["concepts_seen"], 1)
        self.assertEqual(stats["concepts_review_required"], 1)
        self.assertEqual(session.write_calls, [])
        self.assertEqual(concept.normalization_status, REVIEW_REQUIRED_STATUS)
        self.assertEqual(concept.normalization_method, TYPE_REVIEW_REQUIRED_METHOD)
        self.assertEqual(concept.reason, TYPE_REVIEW_REQUIRED_REASON)
        self.assertIsNone(concept.best_match)

    def test_review_required_real_run_only_creates_schema(self):
        concept = ConceptRecord(
            concept_id="c1",
            name="electrophysiological study",
            canonical_type="procedure_or_intervention",
            needs_type_review=True,
        )
        session = FakeSession([concept])
        driver = FakeDriver(session)

        with patch(
            "knowledge_graph.umls_normalization.UMLSAPIClient",
            side_effect=AssertionError("API client must not be initialized"),
        ):
            stats = normalize_concepts_with_umls(
                driver=driver,
                backend=UMLS_API_BACKEND,
                dry_run=False,
                export_review=False,
                force=True,
            )

        self.assertEqual(stats["concepts_review_required"], 1)
        self.assertEqual(len(session.write_calls), 1)
        self.assertIs(session.write_calls[0][0], setup_normalization_schema)

    def test_review_required_count_is_separate(self):
        concepts = [
            ConceptRecord(
                concept_id=f"c{i}",
                name=f"ambiguous concept {i}",
                canonical_type="ambiguous",
                needs_type_review=True,
            )
            for i in range(23)
        ]
        session = FakeSession(concepts)
        driver = FakeDriver(session)

        stats = normalize_concepts_with_umls(
            driver=driver,
            backend=FUZZY_ONLY_BACKEND,
            dry_run=True,
            export_review=False,
        )

        self.assertEqual(stats["concepts_seen"], 23)
        self.assertEqual(stats["concepts_review_required"], 23)
        self.assertEqual(stats["concepts_normalized"], 0)
        self.assertEqual(stats["concepts_no_match"], 0)
        self.assertEqual(stats["concepts_low_confidence"], 0)

    def test_review_required_concepts_are_excluded_from_duplicate_evidence(self):
        review_concept = ConceptRecord(
            concept_id="c1",
            name="endomyocardial biopsy",
            canonical_type="ambiguous",
            needs_type_review=True,
            normalization_status=REVIEW_REQUIRED_STATUS,
        )
        eligible_concept = ConceptRecord(
            concept_id="c2",
            name="endomyocardial biopsies",
            canonical_type="diagnostic_test",
        )

        with (
            patch(
                "knowledge_graph.umls_normalization.compute_same_cui_pairs",
                return_value=[],
            ) as same_mock,
            patch(
                "knowledge_graph.umls_normalization.compute_fuzzy_pairs",
                return_value=[],
            ) as fuzzy_mock,
        ):
            create_duplicate_evidence(
                driver=object(),
                concepts=[review_concept, eligible_concept],
                fuzzy_threshold=90,
                normalized_at="2026-08-06T00:00:00+00:00",
                dry_run=True,
            )

        same_concepts = same_mock.call_args.args[0]
        fuzzy_concepts = fuzzy_mock.call_args.kwargs["concepts"]
        self.assertEqual(same_concepts, [eligible_concept])
        self.assertEqual(fuzzy_concepts, [eligible_concept])


if __name__ == "__main__":
    unittest.main()
