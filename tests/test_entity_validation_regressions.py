import unittest

from knowledge_graph.entity_schema import ALLOWED_TYPES, BLOCKLIST_NAMES
from knowledge_graph.validate_entities import (
    deduplicate_validated_concepts,
    validate_concepts_against_source,
    validate_single_concept,
)


class EntityValidationRegressionTests(unittest.TestCase):
    def validate(self, concept, source_text, acronyms=None):
        return validate_single_concept(
            concept=concept,
            source_text=source_text,
            allowed_types=ALLOWED_TYPES,
            blocklist_names=BLOCKLIST_NAMES,
            acronyms=acronyms,
        )

    def test_named_trial_is_rejected_for_any_proposed_type(self):
        for concept_type in (
            "clinical_outcome",
            "score_or_risk_model",
            "care_strategy",
        ):
            with self.subTest(concept_type=concept_type):
                result = self.validate(
                    {
                        "name": "COMPASS trial",
                        "type": concept_type,
                        "raw_name": "COMPASS trial",
                        "raw_type": concept_type,
                    },
                    "The COMPASS trial evaluated rivaroxaban.",
                )
                self.assertFalse(result["accepted"])
                self.assertEqual(
                    result["reason"],
                    "nonclinical_research_or_variable",
                )

    def test_acronym_expansion_cannot_bypass_trial_exclusion(self):
        result = self.validate(
            {
                "name": "COMPASS",
                "type": "score_or_risk_model",
                "raw_name": "COMPASS",
                "raw_type": "score_or_risk_model",
            },
            "The COMPASS trial evaluated rivaroxaban.",
            acronyms={"COMPASS": "COMPASS trial"},
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(
            result["reason"],
            "nonclinical_research_or_variable",
        )

    def test_electrophysiological_study_remains_a_diagnostic_test(self):
        result = self.validate(
            {
                "name": "electrophysiological study",
                "type": "diagnostic_test",
                "raw_name": "electrophysiological study",
                "raw_type": "diagnostic_test",
            },
            "An electrophysiological study was performed.",
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "accepted_by_direct_source_match")

    def test_named_score_with_organization_words_is_not_rejected(self):
        result = self.validate(
            {
                "name": (
                    "Society of Thoracic Surgeons Predicted Risk of Mortality"
                ),
                "type": "score_or_risk_model",
                "raw_name": (
                    "Society of Thoracic Surgeons Predicted Risk of Mortality"
                ),
                "raw_type": "score_or_risk_model",
            },
            (
                "The Society of Thoracic Surgeons Predicted Risk of Mortality "
                "was calculated."
            ),
        )
        self.assertTrue(result["accepted"])

    def test_task_force_criteria_is_not_rejected_as_organization(self):
        result = self.validate(
            {
                "name": "revised Task Force criteria",
                "type": "score_or_risk_model",
                "raw_name": "revised Task Force criteria",
                "raw_type": "score_or_risk_model",
            },
            "Diagnosis used the revised Task Force criteria.",
        )
        self.assertTrue(result["accepted"])

    def test_guideline_title_is_rejected_but_gdmt_is_not(self):
        title_result = self.validate(
            {
                "name": "2021 ESC Guidelines for heart failure",
                "type": "care_strategy",
                "raw_name": "2021 ESC Guidelines for heart failure",
                "raw_type": "care_strategy",
            },
            "The 2021 ESC Guidelines for heart failure were consulted.",
        )
        self.assertFalse(title_result["accepted"])
        self.assertEqual(
            title_result["reason"],
            "document_or_publication_not_entity",
        )

        gdmt_result = self.validate(
            {
                "name": "guideline-directed medical therapy",
                "type": "care_strategy",
                "raw_name": "guideline-directed medical therapy",
                "raw_type": "care_strategy",
            },
            "Guideline-directed medical therapy was recommended.",
        )
        self.assertTrue(gdmt_result["accepted"])

    def test_direct_source_evidence_wins_without_mixed_acronym_metadata(self):
        direct = {
            "name": "hypertrophic cardiomyopathy",
            "type": "disease",
            "raw_name": "hypertrophic cardiomyopathy",
            "raw_type": "disease",
            "support_method": "direct_source",
            "validation_reason": "accepted_by_direct_source_match",
            "matched_text": "hypertrophic cardiomyopathy",
        }
        expanded = {
            "name": "hypertrophic cardiomyopathy",
            "type": "disease",
            "raw_name": "HCM",
            "raw_type": "disease",
            "support_method": "acronym",
            "validation_reason": "accepted_by_acronym_expansion",
            "acronym_short": "HCM",
            "acronym_definition": "Hypertrophic cardiomyopathy",
            "acronym_match_method": "raw_short_matches_cached_acronym",
            "expanded_from_acronym": True,
        }

        for concepts in ([direct, expanded], [expanded, direct]):
            with self.subTest(order=concepts[0]["support_method"]):
                deduped = deduplicate_validated_concepts(concepts)
                self.assertEqual(len(deduped), 1)
                mention = deduped[0]
                self.assertEqual(mention["support_method"], "direct_source")
                self.assertEqual(
                    mention["raw_name"],
                    "hypertrophic cardiomyopathy",
                )
                self.assertFalse(mention["expanded_from_acronym"])
                self.assertNotIn("acronym_short", mention)
                self.assertNotIn("acronym_definition", mention)
                self.assertNotIn("acronym_match_method", mention)

    def test_incoherent_acronym_expansion_is_downgraded_not_rewritten(self):
        record = {
            "name": "index of microcirculatory resistance",
            "type": "clinical_finding",
            "raw_name": "index of microcirculatory resistance",
            "raw_type": "clinical_finding",
            "support_method": "acronym",
            "validation_reason": "accepted_by_acronym_support",
            "acronym_short": "IMR",
            "acronym_definition": "Index of microcirculatory resistance",
            "acronym_match_method": (
                "short_in_source_and_definition_matches_concept"
            ),
            "expanded_from_acronym": True,
        }
        deduped = deduplicate_validated_concepts([record])
        self.assertEqual(len(deduped), 1)
        self.assertFalse(deduped[0]["expanded_from_acronym"])
        self.assertEqual(deduped[0]["raw_name"], record["raw_name"])

    def test_gene_symbol_path_is_unchanged(self):
        result = self.validate(
            {
                "name": "LMNA",
                "type": "genetic_factor",
                "raw_name": "LMNA",
                "raw_type": "genetic_factor",
            },
            "Pathogenic variants in LMNA are clinically relevant.",
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(
            result["reason"],
            "accepted_gene_symbol_direct_source",
        )
        self.assertEqual(result["concept"]["name"], "LMNA")
    def validate_batch(self, concept, source_text, acronyms=None):
        return validate_concepts_against_source(
            concepts=[concept],
            source_text=source_text,
            allowed_types=ALLOWED_TYPES,
            blocklist_names=BLOCKLIST_NAMES,
            acronyms=acronyms or {},
        )

    def test_legacy_gene_symbol_regressions(self):
        cases = (
            ("SCN5A", None),
            ("TAZ", None),
            ("TTN", "Titin"),
            ("TTR", "Transthyretin"),
            ("TMEM43", "transmembrane protein 43"),
        )

        for symbol, expansion in cases:
            with self.subTest(symbol=symbol):
                result = self.validate_batch(
                    {
                        "name": symbol.lower(),
                        "type": "genetic_factor",
                        "raw_name": symbol,
                        "raw_type": "genetic_factor",
                    },
                    f"Pathogenic variants in {symbol} may be clinically relevant.",
                    {symbol: expansion} if expansion else {},
                )

                self.assertEqual(len(result["accepted"]), 1)
                accepted = result["accepted"][0]

                self.assertEqual(accepted["name"], symbol)
                self.assertEqual(
                    accepted["support_method"],
                    "direct_source",
                )
                self.assertEqual(
                    accepted["validation_reason"],
                    "accepted_gene_symbol_direct_source",
                )
                self.assertIn(
                    "expanded_from_acronym",
                    accepted,
                )
                self.assertFalse(
                    accepted["expanded_from_acronym"]
                )

    def test_mace_acronym_expands_to_clinical_outcome(self):
        result = self.validate_batch(
            {
                "name": "mace",
                "type": "clinical_outcome",
                "raw_name": "MACE",
                "raw_type": "clinical_outcome",
            },
            "MACE was reduced during follow-up.",
            {
                "MACE": "Major adverse cardiovascular events",
            },
        )

        self.assertEqual(len(result["accepted"]), 1)

        accepted = result["accepted"][0]

        self.assertEqual(
            accepted["name"],
            "major adverse cardiovascular event",
        )
        self.assertEqual(
            accepted["support_method"],
            "acronym",
        )
        self.assertTrue(
            accepted["expanded_from_acronym"]
        )

    def test_short_as_without_valid_expansion_is_rejected(self):
        result = self.validate_batch(
            {
                "name": "as",
                "type": "disease",
                "raw_name": "AS",
                "raw_type": "disease",
            },
            "This was described as clinically relevant.",
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(len(result["rejected"]), 1)
        self.assertEqual(
            result["rejected"][0]["reason"],
            "acronym_short_without_valid_expansion",
        )

    def test_primary_care_is_not_a_population(self):
        result = self.validate_batch(
            {
                "name": "primary care",
                "type": "population_or_patient_group",
                "raw_name": "primary care",
                "raw_type": "population_or_patient_group",
            },
            "Screening may be performed in primary care.",
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(len(result["rejected"]), 1)

        rejected = result["rejected"][0]

        self.assertEqual(
            rejected["reason"],
            "care_setting_not_population",
        )
        self.assertIn(
            "non_population_care_setting",
            rejected.get("quality_flags", []),
        )

    def test_unsupported_particulate_phrase_is_rejected(self):
        result = self.validate_batch(
            {
                "name": "fine particulate matter",
                "type": "exposure_or_lifestyle_factor",
                "raw_name": "fine particulate matter",
                "raw_type": "exposure_or_lifestyle_factor",
            },
            "Electronic cigarettes may emit fine and ultrafine particulates.",
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(len(result["rejected"]), 1)
        self.assertEqual(
            result["rejected"][0]["reason"],
            "not_supported_by_source_or_acronym",
        )


if __name__ == "__main__":
    unittest.main()
