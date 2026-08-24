"""Offline regressions for UMLS ranking-policy v2.

These tests must not call the UMLS API and must not connect to Neo4j.
"""

from __future__ import annotations

import unittest

from knowledge_graph.umls_normalization import (
    SEMANTIC_COMPATIBLE,
    ConceptRecord,
    UMLSAPIClient,
    UMLSMatch,
    build_alias_provenance_for_concept,
    build_aliases_for_concept,
    classify_semantic_compatibility,
    compute_umls_selection_score,
    select_best_umls_api_match,
    select_best_umls_api_match_with_trace,
    should_include_secondary_alias,
)


class PayloadClient:
    """UMLSAPIClient instance without constructor or network side effects."""

    @staticmethod
    def create(payloads):
        client = object.__new__(UMLSAPIClient)
        client.request_search = lambda alias, search_type: payloads[search_type]
        return client


class SemanticPolicyExpansionTests(unittest.TestCase):
    def test_clinical_attribute_is_valid_for_clinical_finding(self):
        self.assertEqual(
            classify_semantic_compatibility(
                "clinical_finding",
                ["Clinical Attribute"],
            ),
            SEMANTIC_COMPATIBLE,
        )

    def test_organism_function_is_valid_for_clinical_finding(self):
        self.assertEqual(
            classify_semantic_compatibility(
                "clinical_finding",
                ["Organism Function"],
            ),
            SEMANTIC_COMPATIBLE,
        )

    def test_finding_is_valid_for_score_or_risk_model(self):
        self.assertEqual(
            classify_semantic_compatibility(
                "score_or_risk_model",
                ["Finding"],
            ),
            SEMANTIC_COMPATIBLE,
        )


class SecondaryAliasSafetyTests(unittest.TestCase):
    def test_strict_subset_secondary_alias_is_rejected(self):
        self.assertFalse(
            should_include_secondary_alias(
                "left ventricular hypertrabeculation",
                "left ventricular",
            )
        )

    def test_non_subset_secondary_alias_can_be_retained(self):
        self.assertTrue(
            should_include_secondary_alias(
                "implantable defibrillator procedure",
                "implantable cardioverter defibrillator",
            )
        )

    def test_secondary_lv_expansion_is_not_queried_independently(self):
        concept = ConceptRecord(
            concept_id="c1",
            name="left ventricular hypertrabeculation",
            canonical_type="clinical_finding",
            relationship_acronyms=[
                {"short": "LV", "definition": "left ventricular"}
            ],
        )

        aliases = build_aliases_for_concept(concept, {})
        provenance = build_alias_provenance_for_concept(
            concept,
            {},
            aliases,
        )

        self.assertEqual(aliases, ["left ventricular hypertrabeculation"])
        self.assertEqual(len(provenance), 1)
        self.assertEqual(provenance[0]["alias_sources"], ["concept_name"])

    def test_primary_acronym_expansion_is_unchanged(self):
        concept = ConceptRecord(
            concept_id="c1",
            name="cmr",
            canonical_type="diagnostic_test",
            relationship_acronyms=[
                {"short": "CMR", "definition": "cardiac magnetic resonance"}
            ],
        )
        aliases = build_aliases_for_concept(concept, {})
        self.assertEqual(aliases[0], "cardiac magnetic resonance")
        self.assertIn("cmr", aliases)


class SelectionScoreTests(unittest.TestCase):
    @staticmethod
    def match(
        *,
        alias="term",
        cui="C1",
        score=0.5,
        search_type="exact",
        api_rank=1,
    ):
        return UMLSMatch(
            alias=alias,
            cui=cui,
            canonical_name=alias,
            definition=None,
            aliases=[],
            score=score,
            semantic_types=["Finding"],
            search_type=search_type,
            type_compatible=True,
            semantic_compatibility=SEMANTIC_COMPATIBLE,
            api_rank=api_rank,
        )

    def test_words_search_receives_small_selection_penalty(self):
        match = self.match(score=0.76, search_type="words", api_rank=1)
        self.assertEqual(compute_umls_selection_score(match), 0.73)

    def test_exact_is_not_an_unconditional_cross_strategy_winner(self):
        exact = self.match(
            cui="C_EXACT",
            score=0.40,
            search_type="exact",
            api_rank=2,
        )
        normalized = self.match(
            cui="C_NORMALIZED",
            score=0.90,
            search_type="normalizedString",
            api_rank=1,
        )

        class Client:
            def search_alias(self, alias, search_type, canonical_type=None):
                if search_type == "exact":
                    return exact
                if search_type == "normalizedString":
                    return normalized
                return None

        selected = select_best_umls_api_match(
            aliases=["term"],
            client=Client(),
            canonical_type="clinical_finding",
        )
        self.assertEqual(selected.cui, "C_NORMALIZED")


class ExactIdentityRecoveryTests(unittest.TestCase):
    def _select(self, alias, canonical_type, semantic_type):
        payloads = {
            search_type: {
                "result": {
                    "results": [
                        {
                            "ui": "C_EXACT",
                            "name": alias,
                            "semanticTypes": [semantic_type],
                        }
                    ]
                }
            }
            for search_type in ("exact", "normalizedString", "words")
        }
        client = PayloadClient.create(payloads)
        return select_best_umls_api_match_with_trace(
            aliases=[alias],
            client=client,
            canonical_type=canonical_type,
            trace_limit=3,
        )

    def test_left_ventricular_ejection_fraction_is_selectable(self):
        match, trace = self._select(
            "left ventricular ejection fraction",
            "clinical_finding",
            "Clinical Attribute",
        )
        self.assertEqual(match.cui, "C_EXACT")
        self.assertTrue(match.type_compatible)
        self.assertEqual(match.score, 1.0)
        self.assertTrue(any(row["selected_final"] for row in trace))

    def test_pregnancy_is_selectable(self):
        match, _ = self._select(
            "pregnancy",
            "clinical_finding",
            "Organism Function",
        )
        self.assertEqual(match.cui, "C_EXACT")
        self.assertTrue(match.type_compatible)
        self.assertEqual(match.score, 1.0)

    def test_polygenic_risk_score_is_selectable(self):
        match, trace = self._select(
            "polygenic risk score",
            "score_or_risk_model",
            "Finding",
        )
        self.assertEqual(match.cui, "C_EXACT")
        self.assertTrue(match.type_compatible)
        self.assertEqual(match.score, 1.0)
        self.assertTrue(all("selection_score" in row for row in trace))


if __name__ == "__main__":
    unittest.main()
