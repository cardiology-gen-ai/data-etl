import unittest

from knowledge_graph.umls_normalization import (
    ConceptRecord,
    UMLSMatch,
    UMLSAPIClient,
    build_aliases_for_concept,
    compute_umls_candidate_score,
    select_best_umls_api_match,
    semantic_types_are_compatible,
)


class FakeTx:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))


class UMLSRankingTests(unittest.TestCase):
    def test_acronym_expansion_is_prioritized(self):
        concept = ConceptRecord(
            concept_id="1",
            name="icd",
            canonical_type="device",
            relationship_acronyms=[
                {
                    "short": "ICD",
                    "definition": "implantable cardioverter defibrillator",
                }
            ],
        )

        aliases = build_aliases_for_concept(
            concept,
            acronyms_by_doc_id={},
        )

        self.assertEqual(
            aliases[0],
            "implantable cardioverter defibrillator",
        )
        self.assertIn("icd", aliases)

    def test_semantic_type_rejects_wrong_icd_meaning(self):
        self.assertFalse(
            semantic_types_are_compatible(
                "device",
                ["Clinical Attribute"],
            )
        )
        self.assertTrue(
            semantic_types_are_compatible(
                "device",
                ["Medical Device"],
            )
        )

    def test_numeric_subtype_is_penalized(self):
        score = compute_umls_candidate_score(
            "arrhythmogenic right ventricular cardiomyopathy",
            "ARRHYTHMOGENIC RIGHT VENTRICULAR CARDIOMYOPATHY 15",
            "exact",
        )
        self.assertLess(score, 0.9)

    def test_reordered_equivalent_term_remains_above_threshold(self):
        score = compute_umls_candidate_score(
            "left ventricular outflow tract obstruction",
            "Ventricular Outflow Obstruction, Left",
            "normalizedString",
        )
        self.assertGreaterEqual(score, 0.9)

    def test_api_search_skips_incompatible_first_result(self):
        client = object.__new__(UMLSAPIClient)
        client.request_search = lambda alias, search_type: {
            "result": {
                "results": [
                    {
                        "ui": "C_WRONG",
                        "name": "Pathology diagnosis ICD code",
                        "semanticTypes": ["Clinical Attribute"],
                    },
                    {
                        "ui": "C_RIGHT",
                        "name": "Implantable Cardioverter-Defibrillator",
                        "semanticTypes": ["Medical Device"],
                    },
                ]
            }
        }

        match = client.search_alias(
            alias="implantable cardioverter defibrillator",
            search_type="normalizedString",
            canonical_type="device",
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.cui, "C_RIGHT")
        self.assertTrue(match.type_compatible)

    def test_api_search_prefers_general_term_over_numbered_subtype(self):
        client = object.__new__(UMLSAPIClient)
        client.request_search = lambda alias, search_type: {
            "result": {
                "results": [
                    {
                        "ui": "C_SUBTYPE",
                        "name": (
                            "ARRHYTHMOGENIC RIGHT VENTRICULAR "
                            "CARDIOMYOPATHY 15"
                        ),
                        "semanticTypes": ["Disease or Syndrome"],
                    },
                    {
                        "ui": "C_GENERAL",
                        "name": (
                            "Arrhythmogenic right ventricular "
                            "cardiomyopathy"
                        ),
                        "semanticTypes": ["Disease or Syndrome"],
                    },
                ]
            }
        }

        match = client.search_alias(
            alias="arrhythmogenic right ventricular cardiomyopathy",
            search_type="exact",
            canonical_type="disease",
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.cui, "C_GENERAL")
        self.assertGreaterEqual(match.score, 0.9)

    def test_api_selection_allows_materially_better_words_candidate(self):
        class FakeClient:
            def search_alias(self, alias, search_type, canonical_type=None):
                if search_type == "exact":
                    return UMLSMatch(
                        alias=alias,
                        cui="C_EXACT",
                        canonical_name=(
                            "Arrhythmogenic Right Ventricular Dysplasia"
                        ),
                        definition=None,
                        aliases=[],
                        score=0.1,
                        search_type="exact",
                        type_compatible=None,
                    )
                if search_type == "words":
                    return UMLSMatch(
                        alias=alias,
                        cui="C_WORDS",
                        canonical_name=(
                            "High scoring related words candidate"
                        ),
                        definition=None,
                        aliases=[],
                        score=0.99,
                        search_type="words",
                        type_compatible=True,
                    )
                return None

        match = select_best_umls_api_match(
            aliases=["ARVC"],
            client=FakeClient(),
            canonical_type="disease",
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.cui, "C_WORDS")
        self.assertEqual(match.search_type, "words")

    def test_api_selection_does_not_use_rank_one_exact_score_floor(self):
        class FakeClient:
            def search_alias(self, alias, search_type, canonical_type=None):
                if search_type == "exact":
                    return UMLSMatch(
                        alias=alias,
                        cui="C_EXACT",
                        canonical_name=(
                            "Arrhythmogenic Right Ventricular Dysplasia"
                        ),
                        definition=None,
                        aliases=[],
                        score=0.70,
                        search_type="exact",
                        type_compatible=True,
                        api_rank=1,
                    )
                if search_type == "words":
                    return UMLSMatch(
                        alias=alias,
                        cui="C_WORDS",
                        canonical_name=(
                            "Marginally higher words candidate"
                        ),
                        definition=None,
                        aliases=[],
                        score=0.76,
                        search_type="words",
                        type_compatible=True,
                        api_rank=1,
                    )
                return None

        match = select_best_umls_api_match(
            aliases=["ARVC"],
            client=FakeClient(),
            canonical_type="disease",
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.cui, "C_WORDS")
        self.assertEqual(match.search_type, "words")


if __name__ == "__main__":
    unittest.main()
