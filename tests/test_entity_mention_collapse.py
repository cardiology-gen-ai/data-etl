import unittest

from knowledge_graph.validate_entities import (
    collapse_validated_concepts_by_name,
)


class CollapseValidatedConceptsTests(unittest.TestCase):
    def test_same_concept_multiple_types_becomes_one_mention(self):
        concepts = [
            {
                "name": "dual antiplatelet therapy",
                "type": "drug_or_drug_class",
                "raw_name": "dual antiplatelet therapy",
                "raw_type": "drug_or_drug_class",
                "support_method": "direct_source",
                "validation_reason": "accepted_by_direct_source_match",
            },
            {
                "name": "dual antiplatelet therapy",
                "type": "care_strategy",
                "raw_name": "dual antiplatelet therapy",
                "raw_type": "care_strategy",
                "support_method": "direct_source",
                "validation_reason": "accepted_by_direct_source_match",
            },
        ]

        collapsed = collapse_validated_concepts_by_name(concepts)

        self.assertEqual(len(collapsed), 1)
        self.assertEqual(
            collapsed[0]["observed_types"],
            ["drug_or_drug_class", "care_strategy"],
        )
        self.assertEqual(
            collapsed[0]["support_method"],
            "direct_source",
        )

    def test_direct_evidence_wins_over_acronym_evidence(self):
        concepts = [
            {
                "name": "heart failure",
                "type": "clinical_finding",
                "raw_name": "HF",
                "raw_type": "clinical_finding",
                "support_method": "acronym",
                "validation_reason": "accepted_by_acronym_expansion",
                "acronym_short": "HF",
                "acronym_definition": "Heart failure",
                "expanded_from_acronym": True,
            },
            {
                "name": "heart failure",
                "type": "disease",
                "raw_name": "heart failure",
                "raw_type": "disease",
                "support_method": "direct_source",
                "validation_reason": "accepted_by_direct_source_match",
            },
        ]

        collapsed = collapse_validated_concepts_by_name(concepts)

        self.assertEqual(len(collapsed), 1)

        row = collapsed[0]

        self.assertEqual(
            row["observed_types"],
            ["clinical_finding", "disease"],
        )
        self.assertEqual(row["support_method"], "direct_source")
        self.assertEqual(row["raw_name"], "heart failure")
        self.assertNotIn("acronym_short", row)
        self.assertFalse(row["expanded_from_acronym"])


if __name__ == "__main__":
    unittest.main()
