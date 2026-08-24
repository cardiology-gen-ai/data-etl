"""Offline regression tests for the current UMLS candidate-quality policy.

Consolidates the historical v3-v7 regression cases.
No network or Neo4j access.
"""

from __future__ import annotations

import unittest

from knowledge_graph.umls_normalization import (
    CANDIDATE_TRACE_VERSION,
    DEFAULT_EXACT_UMLS_THRESHOLD,
    NO_PLAUSIBLE_MATCH_STATUS,
    RANKING_POLICY_VERSION,
    SEMANTIC_COMPATIBLE,
    ConceptRecord,
    UMLSAPIClient,
    UMLSMatch,
    build_review_record,
    has_disease_specificity_conflict,
    is_confident_umls_match,
    is_plausible_umls_match,
    is_strong_compatible_exact_match,
    select_best_umls_api_match_with_trace,
)



class PayloadClient:
    @staticmethod
    def create(search_payloads, atoms_by_cui=None):
        client = object.__new__(UMLSAPIClient)
        client.enable_atom_enrichment = True
        client.request_search = (
            lambda alias, search_type: search_payloads[search_type]
        )
        client.request_atoms = lambda cui: {
            "result": list((atoms_by_cui or {}).get(cui, []))
        }
        return client


def empty_payload():
    return {"result": {"results": []}}


def payload(results):
    return {"result": {"results": list(results)}}


class AtomEvidenceTests(unittest.TestCase):
    def test_multi_source_exact_atom_supports_synonym(self):
        payloads = {
            "exact": {
                "result": {
                    "results": [
                        {
                            "ui": "C0038454",
                            "name": "Cerebrovascular accident",
                            "semanticTypes": ["Disease or Syndrome"],
                        }
                    ]
                }
            },
            "normalizedString": {"result": {"results": []}},
            "words": {"result": {"results": []}},
        }
        atoms = {
            "C0038454": [
                {
                    "name": "Stroke",
                    "rootSource": "SNOMEDCT_US",
                    "termType": "PT",
                },
                {
                    "name": "stroke",
                    "rootSource": "MSH",
                    "termType": "MH",
                },
            ]
        }
        client = PayloadClient.create(payloads, atoms)

        selected, trace = select_best_umls_api_match_with_trace(
            aliases=["stroke"],
            client=client,
            canonical_type="disease",
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.cui, "C0038454")
        self.assertTrue(selected.synonym_supported)
        self.assertEqual(selected.matched_atom_source_count, 2)
        self.assertTrue(
            is_confident_umls_match(
                selected,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )
        final = next(row for row in trace if row["selected_final"])
        self.assertEqual(final["matched_atom_name"].lower(), "stroke")
        self.assertTrue(final["synonym_supported"])


    def test_multi_source_atom_does_not_auto_accept_non_disease_type(self):
        payloads = {
            "exact": {
                "result": {
                    "results": [
                        {
                            "ui": "C1623258",
                            "name": "Electrocardiography",
                            "semanticTypes": ["Diagnostic Procedure"],
                        }
                    ]
                }
            },
            "normalizedString": {"result": {"results": []}},
            "words": {"result": {"results": []}},
        }
        atoms = {
            "C1623258": [
                {
                    "name": "Electrocardiogram",
                    "rootSource": "SNOMEDCT_US",
                    "termType": "PT",
                },
                {
                    "name": "electrocardiogram",
                    "rootSource": "LNC",
                    "termType": "LN",
                },
            ]
        }
        client = PayloadClient.create(payloads, atoms)

        selected, _ = select_best_umls_api_match_with_trace(
            aliases=["electrocardiogram"],
            client=client,
            canonical_type="diagnostic_test",
        )

        self.assertIsNotNone(selected)
        self.assertFalse(selected.synonym_supported)
        self.assertFalse(
            is_confident_umls_match(
                selected,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )

    def test_single_source_atom_remains_review_only(self):
        match = UMLSMatch(
            alias="right ventricular enlargement",
            cui="C0162770",
            canonical_name="Right Ventricular Hypertrophy",
            definition=None,
            aliases=[],
            score=0.6962,
            semantic_types=["Disease or Syndrome"],
            search_type="exact",
            type_compatible=True,
            semantic_compatibility=SEMANTIC_COMPATIBLE,
            api_rank=1,
            matched_atom_name="right ventricular enlargement",
            matched_atom_score=1.0,
            matched_atom_count=1,
            matched_atom_source_count=1,
            synonym_supported=False,
        )

        self.assertFalse(
            is_confident_umls_match(
                match,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )
        self.assertTrue(is_plausible_umls_match(match))


class PlausibilityTests(unittest.TestCase):
    def test_very_weak_candidate_is_not_plausible(self):
        match = UMLSMatch(
            alias="myocardial inflammation",
            cui="C0027059",
            canonical_name="Myocarditis",
            definition=None,
            aliases=[],
            score=0.1241,
            semantic_types=["Disease or Syndrome"],
            search_type="exact",
            type_compatible=True,
            semantic_compatibility=SEMANTIC_COMPATIBLE,
            api_rank=1,
        )
        self.assertFalse(is_plausible_umls_match(match))

    def test_no_plausible_review_hides_official_mapping(self):
        match = UMLSMatch(
            alias="myocardial inflammation",
            cui="C0027059",
            canonical_name="Myocarditis",
            definition=None,
            aliases=[],
            score=0.1241,
            semantic_types=["Disease or Syndrome"],
            search_type="exact",
            type_compatible=True,
            semantic_compatibility=SEMANTIC_COMPATIBLE,
            api_rank=1,
        )
        concept = ConceptRecord(
            concept_id="c1",
            name="myocardial inflammation",
            canonical_type="clinical_finding",
            normalization_status=NO_PLAUSIBLE_MATCH_STATUS,
            best_match=match,
        )
        record = build_review_record(
            concept=concept,
            run_id="run",
            doc_id="doc",
            model_name="model",
            linker_name="linker",
            backend="umls_api",
            threshold=0.85,
            fuzzy_threshold=90,
            exact_threshold=0.75,
        )

        self.assertIsNone(record["umls_cui"])
        self.assertEqual(record["rejected_umls_cui"], "C0027059")
        self.assertEqual(
            record["candidate_trace_version"],
            CANDIDATE_TRACE_VERSION,
        )
        self.assertEqual(
            record["ranking_policy_version"],
            RANKING_POLICY_VERSION,
        )


class SelectionFlagTests(unittest.TestCase):
    def test_one_alias_winner_and_one_strategy_winner_per_query(self):
        payloads = {
            search_type: {
                "result": {
                    "results": [
                        {
                            "ui": "C1",
                            "name": "Pregnancy",
                            "semanticTypes": ["Organism Function"],
                        }
                    ]
                }
            }
            for search_type in ("exact", "normalizedString", "words")
        }
        client = PayloadClient.create(payloads)
        client.enable_atom_enrichment = False

        selected, trace = select_best_umls_api_match_with_trace(
            aliases=["pregnancy"],
            client=client,
            canonical_type="clinical_finding",
        )

        self.assertEqual(selected.cui, "C1")
        self.assertEqual(
            sum(bool(row["selected_for_search_strategy"]) for row in trace),
            3,
        )
        self.assertEqual(
            sum(bool(row["selected_for_alias"]) for row in trace),
            1,
        )
        self.assertEqual(
            sum(bool(row["selected_final"]) for row in trace),
            1,
        )


class ExactRankingRegressionTests(unittest.TestCase):
    def test_identity_exact_beats_broader_non_primary_exact(self):
        payloads = {
            "exact": {
                "result": {
                    "results": [
                        {
                            "ui": "C0002658",
                            "name": "amphetamine",
                            "semanticTypes": [
                                "Organic Chemical",
                                "Pharmacologic Substance",
                            ],
                        },
                        {
                            "ui": "C0002667",
                            "name": "Amphetamines",
                            "semanticTypes": [
                                "Organic Chemical",
                                "Pharmacologic Substance",
                                "Biologically Active Substance",
                            ],
                        },
                    ]
                }
            },
            "normalizedString": empty_payload(),
            "words": empty_payload(),
        }
        client = PayloadClient.create(payloads)

        selected, trace = select_best_umls_api_match_with_trace(
            aliases=["amphetamine"],
            client=client,
            canonical_type="exposure_or_lifestyle_factor",
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.cui, "C0002658")
        self.assertEqual(selected.canonical_name, "amphetamine")

        exact_rows = [
            row for row in trace
            if row["search_type"] == "exact"
        ]
        by_cui = {row["cui"]: row for row in exact_rows}
        self.assertEqual(by_cui["C0002658"]["selection_score"], 0.98)
        self.assertEqual(by_cui["C0002667"]["selection_score"], 0.96)


class HGNCEvidenceRegressionTests(unittest.TestCase):
    def test_rank_two_exact_gene_is_enriched_and_auto_accepted(self):
        payloads = {
            "exact": {
                "result": {
                    "results": [
                        {
                            "ui": "C_PROTEIN",
                            "name": "GAA protein, human",
                            "semanticTypes": [
                                "Amino Acid, Peptide, or Protein",
                                "Enzyme",
                            ],
                        },
                        {
                            "ui": "C1414899",
                            "name": "GAA gene",
                            "semanticTypes": ["Gene or Genome"],
                        },
                    ]
                }
            },
            "normalizedString": empty_payload(),
            "words": empty_payload(),
        }
        atoms = {
            "C1414899": [
                {
                    "name": "GAA",
                    "rootSource": "HGNC",
                    "termType": "ACR",
                }
            ]
        }
        client = PayloadClient.create(payloads, atoms)

        selected, trace = select_best_umls_api_match_with_trace(
            aliases=["GAA"],
            client=client,
            canonical_type="genetic_factor",
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.cui, "C1414899")
        self.assertEqual(selected.api_rank, 2)
        self.assertEqual(selected.matched_atom_name, "GAA")
        self.assertEqual(selected.matched_atom_source, "HGNC")
        self.assertTrue(selected.synonym_supported)
        self.assertTrue(
            is_confident_umls_match(
                selected,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )

        final = next(row for row in trace if row["selected_final"])
        self.assertEqual(final["cui"], "C1414899")
        self.assertTrue(final["synonym_supported"])

    def test_hgnc_is_required_for_gene_auto_acceptance(self):
        payloads = {
            "exact": {
                "result": {
                    "results": [
                        {
                            "ui": "C_GENE",
                            "name": "TEST gene",
                            "semanticTypes": ["Gene or Genome"],
                        }
                    ]
                }
            },
            "normalizedString": empty_payload(),
            "words": empty_payload(),
        }
        atoms = {
            "C_GENE": [
                {
                    "name": "TEST",
                    "rootSource": "SNOMEDCT_US",
                    "termType": "SY",
                },
                {
                    "name": "TEST",
                    "rootSource": "MSH",
                    "termType": "SY",
                },
            ]
        }
        client = PayloadClient.create(payloads, atoms)

        selected, _ = select_best_umls_api_match_with_trace(
            aliases=["TEST"],
            client=client,
            canonical_type="genetic_factor",
        )

        self.assertIsNotNone(selected)
        self.assertFalse(selected.synonym_supported)
        self.assertFalse(
            is_confident_umls_match(
                selected,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )

    def test_hgnc_evidence_does_not_override_non_gene_semantic_type(self):
        payloads = {
            "exact": {
                "result": {
                    "results": [
                        {
                            "ui": "C_PROTEIN",
                            "name": "TEST protein",
                            "semanticTypes": [
                                "Amino Acid, Peptide, or Protein"
                            ],
                        }
                    ]
                }
            },
            "normalizedString": empty_payload(),
            "words": empty_payload(),
        }
        atoms = {
            "C_PROTEIN": [
                {
                    "name": "TEST",
                    "rootSource": "HGNC",
                    "termType": "ACR",
                }
            ]
        }
        client = PayloadClient.create(payloads, atoms)

        selected, _ = select_best_umls_api_match_with_trace(
            aliases=["TEST"],
            client=client,
            canonical_type="genetic_factor",
        )

        self.assertIsNone(selected)


class ExactRankRegressionTests(unittest.TestCase):
    def test_bradyarrhythmia_beats_weak_exact_rank_one(self):
        search_payloads = {
            "exact": payload(
                [
                    {
                        "ui": "C0428977",
                        "name": "Bradycardia",
                        "semanticTypes": ["Finding"],
                    },
                    {
                        "ui": "C0079035",
                        "name": "Bradyarrhythmia (disorder)",
                        "semanticTypes": ["Disease or Syndrome"],
                    },
                ]
            ),
            "normalizedString": payload([]),
            "words": payload([]),
        }
        client = PayloadClient.create(search_payloads)

        selected, trace = select_best_umls_api_match_with_trace(
            aliases=["bradyarrhythmias"],
            client=client,
            canonical_type="clinical_finding",
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.cui, "C0079035")
        self.assertEqual(
            selected.canonical_name,
            "Bradyarrhythmia (disorder)",
        )

        exact = {
            row["cui"]: row
            for row in trace
            if row["search_type"] == "exact"
        }
        self.assertLess(
            exact["C0428977"]["selection_score"],
            exact["C0079035"]["selection_score"],
        )


class StrongSynonymPriorityTests(unittest.TestCase):
    def test_supported_exact_disease_synonym_beats_broader_words_candidate(self):
        search_payloads = {
            "exact": payload(
                [
                    {
                        "ui": "C0008312",
                        "name": "Primary Biliary Cholangitis",
                        "semanticTypes": ["Disease or Syndrome"],
                    }
                ]
            ),
            "normalizedString": payload([]),
            "words": payload(
                [
                    {
                        "ui": "C2931878",
                        "name": "Familial primary biliary cirrhosis",
                        "semanticTypes": ["Disease or Syndrome"],
                    }
                ]
            ),
        }
        atoms = {
            "C0008312": [
                {
                    "name": "Primary Biliary Cirrhosis",
                    "rootSource": "NCI",
                    "termType": "PT",
                },
                {
                    "name": "Primary Biliary Cirrhosis",
                    "rootSource": "MSH",
                    "termType": "MH",
                },
            ]
        }
        client = PayloadClient.create(search_payloads, atoms)

        selected, trace = select_best_umls_api_match_with_trace(
            aliases=["primary biliary cirrhosis"],
            client=client,
            canonical_type="disease",
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.cui, "C0008312")
        self.assertTrue(selected.synonym_supported)
        self.assertTrue(
            is_confident_umls_match(
                selected,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )

        final = [row for row in trace if row["selected_final"]]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["cui"], "C0008312")
        self.assertTrue(final[0]["synonym_supported"])

    def test_unsupported_words_candidate_still_uses_normal_scoring(self):
        search_payloads = {
            "exact": payload(
                [
                    {
                        "ui": "C_EXACT",
                        "name": "Weak exact candidate",
                        "semanticTypes": ["Disease or Syndrome"],
                    }
                ]
            ),
            "normalizedString": payload([]),
            "words": payload(
                [
                    {
                        "ui": "C_WORDS",
                        "name": "target disease",
                        "semanticTypes": ["Disease or Syndrome"],
                    }
                ]
            ),
        }
        client = PayloadClient.create(search_payloads)

        selected, _ = select_best_umls_api_match_with_trace(
            aliases=["target disease"],
            client=client,
            canonical_type="disease",
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.cui, "C_WORDS")


class StrongCompatibleExactTests(unittest.TestCase):
    def test_tobacco_smoking_behavior_beats_substance_candidate(self):
        search_payloads = {
            "exact": payload(
                [
                    {
                        "ui": "C0453996",
                        "name": "Tobacco smoking behavior",
                        "semanticTypes": ["Individual Behavior"],
                    }
                ]
            ),
            "normalizedString": payload(
                [
                    {
                        "ui": "C0439994",
                        "name": "Tobacco smoke",
                        "semanticTypes": ["Substance"],
                    },
                    {
                        "ui": "C0302836",
                        "name": "Smoking tobacco",
                        "semanticTypes": [
                            "Hazardous or Poisonous Substance"
                        ],
                    },
                ]
            ),
            "words": payload(
                [
                    {
                        "ui": "C0453996",
                        "name": "Tobacco smoking behavior",
                        "semanticTypes": ["Individual Behavior"],
                    },
                    {
                        "ui": "C0302836",
                        "name": "Smoking tobacco",
                        "semanticTypes": [
                            "Hazardous or Poisonous Substance"
                        ],
                    },
                ]
            ),
        }
        client = PayloadClient.create(search_payloads)

        selected, trace = select_best_umls_api_match_with_trace(
            aliases=["tobacco smoking"],
            client=client,
            canonical_type="exposure_or_lifestyle_factor",
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.cui, "C0453996")
        self.assertEqual(
            selected.canonical_name,
            "Tobacco smoking behavior",
        )
        self.assertEqual(selected.search_type, "exact")
        self.assertEqual(
            selected.semantic_compatibility,
            "compatible",
        )
        self.assertGreaterEqual(
            selected.score,
            DEFAULT_EXACT_UMLS_THRESHOLD,
        )

        final = [row for row in trace if row["selected_final"]]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["cui"], "C0453996")

    def test_below_threshold_exact_gets_no_special_priority(self):
        weak_exact = UMLSMatch(
            alias="target",
            cui="C_EXACT",
            canonical_name="Target related finding",
            definition=None,
            aliases=[],
            score=DEFAULT_EXACT_UMLS_THRESHOLD - 0.01,
            search_type="exact",
            semantic_compatibility="compatible",
            type_compatible=True,
            api_rank=1,
        )
        self.assertFalse(
            is_strong_compatible_exact_match(weak_exact)
        )

    def test_contextual_mismatch_exact_gets_no_special_priority(self):
        contextual = UMLSMatch(
            alias="amphetamine",
            cui="C0002658",
            canonical_name="amphetamine",
            definition=None,
            aliases=[],
            score=0.98,
            search_type="exact",
            semantic_compatibility="contextual_mismatch",
            type_compatible=None,
            api_rank=1,
        )
        self.assertFalse(
            is_strong_compatible_exact_match(contextual)
        )


class DiseaseSpecificityGuardTests(unittest.TestCase):
    def test_hereditary_added_to_generic_disease_is_conflict(self):
        match = UMLSMatch(
            alias="transthyretin amyloidosis",
            cui="C2751492",
            canonical_name="AMYLOIDOSIS, HEREDITARY, TRANSTHYRETIN-RELATED",
            definition=None,
            aliases=[],
            score=0.7007,
            search_type="exact",
            semantic_compatibility="compatible",
            type_compatible=True,
            canonical_type="disease",
            synonym_supported=True,
        )
        self.assertTrue(has_disease_specificity_conflict(match))
        self.assertFalse(
            is_confident_umls_match(
                match,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )

    def test_senile_added_to_generic_disease_is_conflict(self):
        match = UMLSMatch(
            alias="cardiac amyloidosis",
            cui="C0268407",
            canonical_name="Senile cardiac amyloidosis",
            definition=None,
            aliases=[],
            score=0.8564,
            search_type="exact",
            semantic_compatibility="compatible",
            type_compatible=True,
            canonical_type="disease",
        )
        self.assertTrue(has_disease_specificity_conflict(match))
        self.assertFalse(
            is_confident_umls_match(
                match,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )

    def test_modifier_already_present_is_not_conflict(self):
        match = UMLSMatch(
            alias="hereditary transthyretin amyloidosis",
            cui="C2751492",
            canonical_name="AMYLOIDOSIS, HEREDITARY, TRANSTHYRETIN-RELATED",
            definition=None,
            aliases=[],
            score=0.90,
            search_type="exact",
            semantic_compatibility="compatible",
            type_compatible=True,
            canonical_type="disease",
        )
        self.assertFalse(has_disease_specificity_conflict(match))

    def test_primary_is_intentionally_not_a_narrowing_modifier(self):
        match = UMLSMatch(
            alias="primary biliary cirrhosis",
            cui="C0008312",
            canonical_name="Primary Biliary Cholangitis",
            definition=None,
            aliases=[],
            score=0.738,
            search_type="exact",
            semantic_compatibility="compatible",
            type_compatible=True,
            canonical_type="disease",
            synonym_supported=True,
        )
        self.assertFalse(has_disease_specificity_conflict(match))
        self.assertTrue(
            is_confident_umls_match(
                match,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )

    def test_non_disease_is_unchanged(self):
        match = UMLSMatch(
            alias="familial support",
            cui="C_TEST",
            canonical_name="Familial support program",
            definition=None,
            aliases=[],
            score=0.95,
            search_type="exact",
            semantic_compatibility="compatible",
            type_compatible=True,
            canonical_type="care_strategy",
        )
        self.assertFalse(has_disease_specificity_conflict(match))


class CrossStrategySpecificityTests(unittest.TestCase):
    def test_generic_attr_candidate_beats_narrower_supported_hereditary_cui(self):
        search_payloads = {
            "exact": payload(
                [
                    {
                        "ui": "C5959210",
                        "name": "ATTR Amyloidosis",
                        "semanticTypes": ["Disease or Syndrome"],
                    },
                    {
                        "ui": "C2751492",
                        "name": "AMYLOIDOSIS, HEREDITARY, TRANSTHYRETIN-RELATED",
                        "semanticTypes": ["Disease or Syndrome"],
                    },
                ]
            ),
            "normalizedString": payload(
                [
                    {
                        "ui": "C5959210",
                        "name": "ATTR Amyloidosis",
                        "semanticTypes": ["Disease or Syndrome"],
                    }
                ]
            ),
            "words": payload([]),
        }
        atoms = {
            "C2751492": [
                {
                    "name": "Transthyretin amyloidosis",
                    "rootSource": "MEDLINEPLUS",
                    "termType": "PT",
                },
                {
                    "name": "Transthyretin amyloidosis",
                    "rootSource": "NCI",
                    "termType": "SY",
                },
            ]
        }
        client = PayloadClient.create(search_payloads, atoms)

        selected, trace = select_best_umls_api_match_with_trace(
            aliases=["transthyretin amyloidosis"],
            client=client,
            canonical_type="disease",
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.cui, "C5959210")
        self.assertFalse(has_disease_specificity_conflict(selected))
        self.assertFalse(
            is_confident_umls_match(
                selected,
                threshold=0.85,
                exact_threshold=0.75,
            )
        )
        final = [row for row in trace if row["selected_final"]]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["cui"], "C5959210")


class MetadataRegressionTests(unittest.TestCase):
    def test_ranking_policy_version_is_current(self):
        self.assertEqual(
            RANKING_POLICY_VERSION,
            "umls_candidate_quality_v8_acronym_provenance",
        )


if __name__ == "__main__":
    unittest.main()
