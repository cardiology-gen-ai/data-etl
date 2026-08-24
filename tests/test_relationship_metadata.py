import unittest
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from knowledge_graph.relationship_metadata import (
    VALID_RELATIONSHIP_FAMILIES,
    build_mention_relationship_metadata,
    build_normalization_relationship_metadata,
    build_ontology_relationship_metadata,
    build_structural_relationship_metadata,
)


class RelationshipMetadataTests(unittest.TestCase):
    def test_structural_metadata_omits_empty_doc_id(self):
        metadata = build_structural_relationship_metadata("HAS_CHILD", doc_id=" ")

        self.assertEqual(metadata["relationship_family"], "structural")
        self.assertEqual(metadata["provenance"], "graph_loader")
        self.assertEqual(metadata["provenance_source"], "source_document")
        self.assertEqual(metadata["provenance_method"], "hierarchical_chunking")
        self.assertNotIn("doc_id", metadata)

    def test_structural_metadata_accepts_only_structural_types(self):
        with self.assertRaises(ValueError):
            build_structural_relationship_metadata("MENTIONS")

    def test_mention_metadata_includes_non_empty_doc_id(self):
        metadata = build_mention_relationship_metadata(doc_id=" doc-a ")

        self.assertEqual(metadata["relationship_family"], "mention")
        self.assertEqual(metadata["provenance"], "entity_extraction")
        self.assertEqual(metadata["provenance_method"], "llm_assisted_entity_extraction")
        self.assertEqual(metadata["doc_id"], "doc-a")

    def test_normalization_metadata_distinguishes_same_as_and_candidates(self):
        same_as = build_normalization_relationship_metadata("SAME_AS")
        possibly_same_as = build_normalization_relationship_metadata("POSSIBLY_SAME_AS")

        self.assertEqual(same_as["relationship_family"], "normalization")
        self.assertEqual(same_as["provenance_source"], "umls_metathesaurus")
        self.assertEqual(same_as["provenance_method"], "umls_cui")
        self.assertEqual(possibly_same_as["provenance_source"], "local_matching")
        self.assertEqual(possibly_same_as["provenance_method"], "fuzzy_name")

    def test_normalization_metadata_accepts_only_normalization_types(self):
        with self.assertRaises(ValueError):
            build_normalization_relationship_metadata("UMLS_ISA")

    def test_ontology_metadata_requires_source_vocabulary(self):
        with self.assertRaises(ValueError):
            build_ontology_relationship_metadata("")

    def test_ontology_metadata_uses_source_vocabulary_only_for_ontology(self):
        metadata = build_ontology_relationship_metadata(" SNOMEDCT_US ")

        self.assertEqual(metadata["relationship_family"], "ontology")
        self.assertEqual(metadata["provenance"], "umls_connections")
        self.assertEqual(metadata["provenance_source"], "umls_metathesaurus")
        self.assertEqual(metadata["provenance_method"], "umls_relations_api")
        self.assertEqual(metadata["source_vocabulary"], "SNOMEDCT_US")

    def test_declared_families_are_the_allowed_values(self):
        self.assertEqual(
            VALID_RELATIONSHIP_FAMILIES,
            {"structural", "mention", "normalization", "ontology"},
        )


if __name__ == "__main__":
    unittest.main()
