"""Offline regressions for candidate-trace v2 and ranking-policy v3.

These tests must not call the UMLS API and must not connect to Neo4j.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from knowledge_graph.umls_normalization import (
    SEMANTIC_INCOMPATIBLE,
    UMLS_API_BACKEND,
    ConceptRecord,
    UMLSAPIClient,
    UMLSMatch,
    build_alias_provenance_for_concept,
    build_aliases_for_concept,
    build_review_record,
    normalize_concepts_with_umls,
    select_best_umls_api_match_with_trace,
)


class FakeSession:
    def __init__(self, concepts=None):
        self.concepts = list(concepts or [])
        self.write_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_read(self, function, *args, **kwargs):
        return list(self.concepts)

    def execute_write(self, function, *args, **kwargs):
        self.write_calls.append((function, args, kwargs))
        return None


class FakeDriver:
    def __init__(self, concepts=None):
        self.fake_session = FakeSession(concepts)

    def session(self):
        return self.fake_session


class PayloadClient:
    """UMLSAPIClient instance without constructor/network side effects."""

    @staticmethod
    def create(payloads):
        client = object.__new__(UMLSAPIClient)
        client.request_search = lambda alias, search_type: payloads[search_type]
        return client


class CandidateTraceCollectionTests(unittest.TestCase):
    def test_incompatible_candidate_remains_visible_but_not_selectable(self):
        client = PayloadClient.create(
            {
                "exact": {
                    "result": {
                        "results": [
                            {
                                "ui": "C_BAD",
                                "name": "Cardiomyopathy",
                                "semanticTypes": ["Pharmacologic Substance"],
                            },
                            {
                                "ui": "C_GOOD",
                                "name": "Cardiomyopathy",
                                "semanticTypes": ["Disease or Syndrome"],
                            },
                        ]
                    }
                }
            }
        )

        match, trace = client.search_alias_candidates(
            alias="cardiomyopathy",
            search_type="exact",
            canonical_type="disease",
            trace_limit=3,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.cui, "C_GOOD")
        self.assertEqual(len(trace), 2)
        self.assertFalse(trace[0]["selection_eligible"])
        self.assertEqual(
            trace[0]["semantic_compatibility"],
            SEMANTIC_INCOMPATIBLE,
        )
        self.assertEqual(
            trace[0]["exclusion_reason"],
            "strong_semantic_incompatibility",
        )
        self.assertTrue(trace[1]["selected_for_search_strategy"])
        self.assertFalse(trace[1]["selected_for_alias"])

    def test_selected_candidate_is_retained_outside_api_rank_window(self):
        client = PayloadClient.create(
            {
                "exact": {
                    "result": {
                        "results": [
                            {
                                "ui": "C1",
                                "name": "Unrelated procedure one",
                                "semanticTypes": ["Diagnostic Procedure"],
                            },
                            {
                                "ui": "C2",
                                "name": "Unrelated procedure two",
                                "semanticTypes": ["Diagnostic Procedure"],
                            },
                            {
                                "ui": "C3",
                                "name": "Unrelated procedure three",
                                "semanticTypes": ["Diagnostic Procedure"],
                            },
                            {
                                "ui": "C4",
                                "name": "Genetic testing",
                                "semanticTypes": ["Diagnostic Procedure"],
                            },
                        ]
                    }
                }
            }
        )

        match, trace = client.search_alias_candidates(
            alias="genetic testing",
            search_type="exact",
            canonical_type="diagnostic_test",
            trace_limit=3,
        )

        self.assertEqual(match.cui, "C4")
        self.assertEqual(len(trace), 4)
        selected = [
            row
            for row in trace
            if row["selected_for_search_strategy"]
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["api_rank"], 4)
        self.assertFalse(selected[0]["selected_for_alias"])
        self.assertEqual(
            selected[0]["retained_reason"],
            "selected_for_search_strategy_outside_trace_limit",
        )

    def test_trace_exposes_raw_and_adjusted_scores(self):
        client = PayloadClient.create(
            {
                "exact": {
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
            }
        )

        match, trace = client.search_alias_candidates(
            alias="diabetes",
            search_type="exact",
            canonical_type="risk_factor",
            trace_limit=3,
        )

        self.assertEqual(match.score, 0.98)
        self.assertEqual(trace[0]["lexical_score"], 1.0)
        self.assertEqual(trace[0]["adjusted_score"], 0.98)
        self.assertIsNone(trace[0]["type_compatible"])


class CrossStrategyTraceTests(unittest.TestCase):
    def test_all_three_search_strategies_are_traced(self):
        payloads = {
            search_type: {
                "result": {
                    "results": [
                        {
                            "ui": f"C_{search_type}",
                            "name": "Cardiac magnetic resonance",
                            "semanticTypes": ["Diagnostic Procedure"],
                        }
                    ]
                }
            }
            for search_type in ("exact", "normalizedString", "words")
        }
        client = PayloadClient.create(payloads)

        match, trace = select_best_umls_api_match_with_trace(
            aliases=["cardiac magnetic resonance"],
            client=client,
            canonical_type="diagnostic_test",
            alias_provenance=[
                {
                    "alias": "cardiac magnetic resonance",
                    "alias_index": 0,
                    "alias_sources": ["document_acronym_expansion"],
                    "alias_doc_ids": ["Cardiomyopathies_2023"],
                    "acronym_shorts": ["CMR"],
                }
            ],
            trace_limit=3,
        )

        self.assertEqual(match.search_type, "exact")
        self.assertEqual(
            {row["search_type"] for row in trace},
            {"exact", "normalizedString", "words"},
        )
        strategy_selected = [
            row
            for row in trace
            if row["selected_for_search_strategy"]
        ]
        self.assertEqual(len(strategy_selected), 3)

        alias_selected = [
            row
            for row in trace
            if row["selected_for_alias"]
        ]
        self.assertEqual(len(alias_selected), 1)
        self.assertEqual(alias_selected[0]["search_type"], "exact")

        final = [row for row in trace if row["selected_final"]]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["search_type"], "exact")
        self.assertEqual(final[0]["acronym_shorts"], ["CMR"])
        self.assertEqual(
            final[0]["alias_sources"],
            ["document_acronym_expansion"],
        )

    def test_alias_order_still_controls_exact_tie(self):
        class FakeTraceClient:
            def search_alias_candidates(
                self, alias, search_type, canonical_type=None, trace_limit=3
            ):
                match = UMLSMatch(
                    alias=alias,
                    cui=f"C::{alias}::{search_type}",
                    canonical_name=alias,
                    definition=None,
                    aliases=[],
                    score=1.0,
                    semantic_types=["Diagnostic Procedure"],
                    search_type=search_type,
                    type_compatible=True,
                    semantic_compatibility="compatible",
                    api_rank=1,
                )
                return match, [
                    {
                        "query_alias": alias,
                        "search_type": search_type,
                        "api_rank": 1,
                        "cui": match.cui,
                        "canonical_name": alias,
                        "semantic_types": match.semantic_types,
                        "lexical_score": 1.0,
                        "adjusted_score": 1.0,
                        "semantic_compatibility": "compatible",
                        "type_compatible": True,
                        "selection_eligible": True,
                        "exclusion_reason": None,
                        "selected_for_search_strategy": True,
                        "selected_for_alias": False,
                        "selected_final": False,
                        "retained_reason": "top_api_rank",
                    }
                ]

        match, _ = select_best_umls_api_match_with_trace(
            aliases=["cardiac magnetic resonance", "cmr"],
            client=FakeTraceClient(),
            canonical_type="diagnostic_test",
            trace_limit=3,
        )
        self.assertEqual(match.alias, "cardiac magnetic resonance")
        self.assertEqual(match.search_type, "exact")


class AliasProvenanceTests(unittest.TestCase):
    def test_document_acronym_provenance_is_exported(self):
        concept = ConceptRecord(
            concept_id="c1",
            name="cmr",
            canonical_type="diagnostic_test",
            doc_ids=["Cardiomyopathies_2023"],
        )
        acronyms = {
            "Cardiomyopathies_2023": {
                "CMR": "cardiac magnetic resonance",
            }
        }
        aliases = build_aliases_for_concept(concept, acronyms)
        provenance = build_alias_provenance_for_concept(
            concept,
            acronyms,
            aliases,
        )

        self.assertEqual(aliases[0], "cardiac magnetic resonance")
        self.assertIn(
            "document_acronym_expansion",
            provenance[0]["alias_sources"],
        )
        self.assertEqual(provenance[0]["acronym_shorts"], ["CMR"])
        concept_name_row = next(
            row for row in provenance if row["alias"] == "cmr"
        )
        self.assertIn("concept_name", concept_name_row["alias_sources"])


class ReviewAndNormalizationIntegrationTests(unittest.TestCase):
    def test_review_record_contains_versioned_candidate_trace(self):
        concept = ConceptRecord(
            concept_id="c1",
            name="diabetes",
            canonical_type="risk_factor",
            candidate_trace=[{"cui": "C0011849", "selected_final": True}],
            alias_provenance=[
                {
                    "alias": "diabetes",
                    "alias_index": 0,
                    "alias_sources": ["concept_name"],
                    "alias_doc_ids": [],
                    "acronym_shorts": [],
                }
            ],
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
        self.assertEqual(
            record["candidate_trace_version"],
            "umls_candidate_trace_v2",
        )
        self.assertEqual(
            record["ranking_policy_version"],
            "umls_candidate_quality_v8_acronym_provenance",
        )
        self.assertEqual(record["candidate_trace"], concept.candidate_trace)
        self.assertEqual(record["alias_provenance"], concept.alias_provenance)

    def test_concept_id_filter_disambiguates_same_normalized_name(self):
        concepts = [
            ConceptRecord(
                concept_id="gene-id",
                name="DMD",
                canonical_type="genetic_factor",
                doc_ids=["Cardiomyopathies_2023"],
            ),
            ConceptRecord(
                concept_id="disease-id",
                name="dmd",
                canonical_type="disease",
                doc_ids=["Cardiomyopathies_2023"],
            ),
        ]
        driver = FakeDriver(concepts)

        class FakeAPIClient:
            def __init__(self, *args, **kwargs):
                self.stats = {
                    "api_cache_hits": 0,
                    "api_cache_misses": 0,
                    "api_requests": 0,
                    "api_retries": 0,
                    "api_errors": 0,
                }

            def search_alias_candidates(
                self,
                alias,
                search_type,
                canonical_type=None,
                trace_limit=3,
            ):
                match = UMLSMatch(
                    alias=alias,
                    cui=f"C::{canonical_type}",
                    canonical_name=alias,
                    definition=None,
                    aliases=[],
                    score=1.0,
                    semantic_types=["Finding"],
                    search_type=search_type,
                    type_compatible=True,
                    semantic_compatibility="compatible",
                    api_rank=1,
                )
                return match, [
                    {
                        "query_alias": alias,
                        "search_type": search_type,
                        "api_rank": 1,
                        "cui": match.cui,
                        "canonical_name": alias,
                        "semantic_types": ["Finding"],
                        "lexical_score": 1.0,
                        "adjusted_score": 1.0,
                        "selection_score": 1.0,
                        "semantic_compatibility": "compatible",
                        "type_compatible": True,
                        "selection_eligible": True,
                        "exclusion_reason": None,
                        "selected_for_search_strategy": True,
                        "selected_for_alias": False,
                        "selected_final": False,
                        "retained_reason": "top_api_rank",
                    }
                ]

        with patch.dict(os.environ, {"UMLS_API_KEY": "test"}), patch(
            "knowledge_graph.umls_normalization.UMLSAPIClient",
            FakeAPIClient,
        ):
            stats = normalize_concepts_with_umls(
                driver=driver,
                doc_id="Cardiomyopathies_2023",
                backend=UMLS_API_BACKEND,
                dry_run=True,
                export_review=False,
                collect_candidate_trace=True,
                concept_ids=["gene-id"],
                max_candidates=3,
            )

        self.assertEqual(stats["concepts_seen"], 1)
        self.assertTrue(concepts[0].candidate_trace)
        self.assertEqual(concepts[1].candidate_trace, [])
        self.assertEqual(driver.fake_session.write_calls, [])

    def test_concept_name_filter_limits_api_audit_scope(self):
        concepts = [
            ConceptRecord(
                concept_id="c1",
                name="diabetes",
                canonical_type="risk_factor",
                doc_ids=["Cardiomyopathies_2023"],
            ),
            ConceptRecord(
                concept_id="c2",
                name="obesity",
                canonical_type="risk_factor",
                doc_ids=["Cardiomyopathies_2023"],
            ),
        ]
        driver = FakeDriver(concepts)

        class FakeAPIClient:
            def __init__(self, *args, **kwargs):
                self.stats = {
                    "api_cache_hits": 0,
                    "api_cache_misses": 0,
                    "api_requests": 0,
                    "api_retries": 0,
                    "api_errors": 0,
                }

            def search_alias_candidates(
                self, alias, search_type, canonical_type=None, trace_limit=3
            ):
                match = UMLSMatch(
                    alias=alias,
                    cui="C_TEST",
                    canonical_name=alias,
                    definition=None,
                    aliases=[],
                    score=1.0,
                    semantic_types=["Disease or Syndrome"],
                    search_type=search_type,
                    type_compatible=None,
                    semantic_compatibility="contextual_mismatch",
                    api_rank=1,
                )
                return match, [
                    {
                        "query_alias": alias,
                        "search_type": search_type,
                        "api_rank": 1,
                        "cui": "C_TEST",
                        "canonical_name": alias,
                        "semantic_types": ["Disease or Syndrome"],
                        "lexical_score": 1.0,
                        "adjusted_score": 0.98,
                        "semantic_compatibility": "contextual_mismatch",
                        "type_compatible": None,
                        "selection_eligible": True,
                        "exclusion_reason": None,
                        "selected_for_search_strategy": True,
                        "selected_for_alias": False,
                        "selected_final": False,
                        "retained_reason": "top_api_rank",
                    }
                ]

        with patch.dict(os.environ, {"UMLS_API_KEY": "test"}), patch(
            "knowledge_graph.umls_normalization.UMLSAPIClient",
            FakeAPIClient,
        ):
            stats = normalize_concepts_with_umls(
                driver=driver,
                doc_id="Cardiomyopathies_2023",
                backend=UMLS_API_BACKEND,
                dry_run=True,
                export_review=False,
                collect_candidate_trace=True,
                concept_names=["diabetes"],
                max_candidates=3,
            )

        self.assertEqual(stats["concepts_seen"], 1)
        self.assertEqual(stats["concepts_normalized"], 1)
        self.assertEqual(stats["candidate_trace_records"], 3)
        self.assertTrue(concepts[0].candidate_trace)
        self.assertEqual(concepts[1].candidate_trace, [])
        self.assertEqual(driver.fake_session.write_calls, [])


if __name__ == "__main__":
    unittest.main()
