from __future__ import annotations

import unittest

from main_graph import resolve_phase_kwargs


def resolve(
    phase: str,
    *,
    sanity_mode: str | None = "standard",
    normalization: bool = False,
    umls_connections: bool = False,
    clear_neo4j: bool = False,
    acronyms: bool = False,
):
    return resolve_phase_kwargs(
        phase,
        sanity_mode=sanity_mode,
        run_entity_normalization=normalization,
        run_umls_connections=umls_connections,
        clear_neo4j_before_run=clear_neo4j,
        run_acronym_extraction=acronyms,
    )


class MainGraphPhaseResolutionTests(unittest.TestCase):
    def test_preprocess_enables_only_preprocessing_and_optional_acronyms(self):
        result = resolve(
            "preprocess",
            acronyms=True,
            clear_neo4j=True,
            normalization=True,
            umls_connections=True,
        )

        self.assertTrue(result["run_preprocessing"])
        self.assertTrue(result["run_acronym_extraction"])

        self.assertFalse(result["clear_neo4j_before_run"])
        self.assertFalse(result["run_graph_loader"])
        self.assertFalse(result["run_entity_extraction"])
        self.assertFalse(result["run_embeddings"])
        self.assertFalse(result["run_entity_disambiguation"])
        self.assertFalse(result["run_entity_normalization"])
        self.assertFalse(result["run_umls_connections"])

        # Preprocessing has no graph to sanity-check yet.
        self.assertFalse(result["run_sanity_checks"])

    def test_graph_enables_loader_and_honours_clear_flag(self):
        result = resolve(
            "graph",
            clear_neo4j=True,
            acronyms=True,
            normalization=True,
            umls_connections=True,
        )

        self.assertTrue(result["clear_neo4j_before_run"])
        self.assertTrue(result["run_graph_loader"])

        self.assertFalse(result["run_preprocessing"])
        self.assertFalse(result["run_acronym_extraction"])
        self.assertFalse(result["run_entity_extraction"])
        self.assertFalse(result["run_embeddings"])
        self.assertFalse(result["run_entity_disambiguation"])
        self.assertFalse(result["run_entity_normalization"])
        self.assertFalse(result["run_umls_connections"])

        self.assertTrue(result["run_sanity_checks"])

    def test_entities_enables_extraction_disambiguation_and_optional_normalization(self):
        without_normalization = resolve(
            "entities",
            normalization=False,
        )
        with_normalization = resolve(
            "entities",
            normalization=True,
        )

        for result in (
            without_normalization,
            with_normalization,
        ):
            self.assertTrue(result["run_entity_extraction"])
            self.assertTrue(result["run_entity_disambiguation"])

            self.assertFalse(result["run_preprocessing"])
            self.assertFalse(result["run_graph_loader"])
            self.assertFalse(result["run_embeddings"])
            self.assertFalse(result["run_umls_connections"])

        self.assertFalse(
            without_normalization["run_entity_normalization"]
        )
        self.assertTrue(
            with_normalization["run_entity_normalization"]
        )

    def test_embeddings_phase_is_isolated(self):
        result = resolve(
            "embeddings",
            clear_neo4j=True,
            acronyms=True,
            normalization=True,
            umls_connections=True,
        )

        self.assertTrue(result["run_embeddings"])

        self.assertFalse(result["clear_neo4j_before_run"])
        self.assertFalse(result["run_preprocessing"])
        self.assertFalse(result["run_acronym_extraction"])
        self.assertFalse(result["run_graph_loader"])
        self.assertFalse(result["run_entity_extraction"])
        self.assertFalse(result["run_entity_disambiguation"])
        self.assertFalse(result["run_entity_normalization"])
        self.assertFalse(result["run_umls_connections"])

    def test_normalization_always_normalizes_and_optionally_runs_connections(self):
        without_connections = resolve(
            "normalization",
            umls_connections=False,
        )
        with_connections = resolve(
            "normalization",
            umls_connections=True,
        )

        self.assertTrue(
            without_connections["run_entity_normalization"]
        )
        self.assertFalse(
            without_connections["run_umls_connections"]
        )

        self.assertTrue(
            with_connections["run_entity_normalization"]
        )
        self.assertTrue(
            with_connections["run_umls_connections"]
        )

        for result in (
            without_connections,
            with_connections,
        ):
            self.assertFalse(result["run_preprocessing"])
            self.assertFalse(result["run_graph_loader"])
            self.assertFalse(result["run_entity_extraction"])
            self.assertFalse(result["run_embeddings"])
            self.assertFalse(result["run_entity_disambiguation"])

    def test_umls_connections_phase_runs_only_connections(self):
        result = resolve(
            "umls_connections",
            normalization=True,
            clear_neo4j=True,
            acronyms=True,
        )

        self.assertTrue(result["run_umls_connections"])

        self.assertFalse(result["clear_neo4j_before_run"])
        self.assertFalse(result["run_preprocessing"])
        self.assertFalse(result["run_acronym_extraction"])
        self.assertFalse(result["run_graph_loader"])
        self.assertFalse(result["run_entity_extraction"])
        self.assertFalse(result["run_embeddings"])
        self.assertFalse(result["run_entity_disambiguation"])
        self.assertFalse(result["run_entity_normalization"])

    def test_full_enables_entire_requested_pipeline(self):
        result = resolve(
            "full",
            clear_neo4j=True,
            acronyms=True,
            normalization=True,
            umls_connections=True,
        )

        self.assertTrue(result["clear_neo4j_before_run"])
        self.assertTrue(result["run_preprocessing"])
        self.assertTrue(result["run_acronym_extraction"])
        self.assertTrue(result["run_graph_loader"])
        self.assertTrue(result["run_entity_extraction"])
        self.assertTrue(result["run_embeddings"])
        self.assertTrue(result["run_entity_disambiguation"])
        self.assertTrue(result["run_entity_normalization"])
        self.assertTrue(result["run_umls_connections"])
        self.assertTrue(result["run_sanity_checks"])

    def test_sanity_mode_is_propagated_without_changing_phase_selection(self):
        result = resolve(
            "graph",
            sanity_mode="post_graph",
        )

        self.assertEqual(
            result["sanity_mode"],
            "post_graph",
        )
        self.assertTrue(result["run_sanity_checks"])

        preprocess = resolve(
            "preprocess",
            sanity_mode="post_graph",
        )

        self.assertEqual(
            preprocess["sanity_mode"],
            "post_graph",
        )
        self.assertFalse(
            preprocess["run_sanity_checks"]
        )

    def test_unknown_phase_fails_closed(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported PIPELINE_PHASE",
        ):
            resolve("definitely_not_a_phase")


if __name__ == "__main__":
    unittest.main()
