import unittest
import json
import os
import tempfile
from pathlib import Path

from knowledge_graph import umls_normalization as norm


class FakeTx:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))


class FakeFuzz:
    @staticmethod
    def token_sort_ratio(left, right):
        if {left, right} == {
            "left ventricular hypertrophy",
            "left ventricle hypertrophy",
        }:
            return 94
        return 10

    @staticmethod
    def token_set_ratio(left, right):
        return FakeFuzz.token_sort_ratio(left, right)


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        if not self.responses:
            raise AssertionError("Unexpected API call")
        return self.responses.pop(0)


def make_concept(name, concept_id=None, canonical_type="clinical_finding"):
    return norm.ConceptRecord(
        concept_id=concept_id or name,
        name=name,
        canonical_type=canonical_type,
        doc_ids=["doc"],
        properties={},
    )


def make_match(
    cui="C123",
    score=0.99,
    search_type=None,
    type_compatible=None,
):
    return norm.UMLSMatch(
        alias="left ventricular hypertrophy",
        cui=cui,
        canonical_name="Left Ventricular Hypertrophy",
        definition="Increased thickness of the left ventricular wall.",
        aliases=["LVH"],
        score=score,
        search_type=search_type,
        type_compatible=type_compatible,
    )


def sample_search_payload(cui="C0004238", name="Atrial Fibrillation"):
    return {
        "result": {
            "results": [
                {
                    "ui": cui,
                    "name": name,
                    "semanticTypes": ["Disease or Syndrome"],
                }
            ]
        }
    }


class UMLSNormalizationTests(unittest.TestCase):
    def test_is_confident_umls_match_policy(self):
        cases = [
            (None, False),
            (
                make_match(
                    score=0.0,
                    search_type="exact",
                    type_compatible=True,
                ),
                False,
            ),
            (
                make_match(
                    score=0.756,
                    search_type="exact",
                    type_compatible=None,
                ),
                True,
            ),
            (
                make_match(
                    score=1.0,
                    search_type="exact",
                    type_compatible=False,
                ),
                False,
            ),
            (
                make_match(
                    score=0.95,
                    search_type="words",
                    type_compatible=True,
                ),
                False,
            ),
            (
                make_match(
                    score=0.85,
                    search_type="words",
                    type_compatible=True,
                ),
                False,
            ),
            (
                make_match(
                    score=0.86,
                    search_type="normalizedString",
                    type_compatible=True,
                ),
                True,
            ),
            (
                make_match(
                    score=0.84,
                    search_type="normalizedString",
                    type_compatible=True,
                ),
                False,
            ),
        ]

        for match, expected in cases:
            with self.subTest(match=match):
                self.assertIs(
                    norm.is_confident_umls_match(match, threshold=0.85),
                    expected,
                )

    def test_confident_umls_match_writes_umls_fields(self):
        concept = make_concept("left ventricular hypertrophy")
        concept.best_match = make_match(score=0.91)
        tx = FakeTx()

        norm.update_concept_from_result(
            tx=tx,
            concept=concept,
            model_name="en_core_sci_sm",
            linker_name="umls",
            threshold=0.85,
            force=False,
            normalized_at="2026-05-29T00:00:00+00:00",
        )

        self.assertEqual(concept.normalization_status, "umls_matched")
        self.assertEqual(tx.calls[0][1]["umls_cui"], "C123")
        self.assertEqual(tx.calls[0][1]["umls_score"], 0.91)

    def test_exact_umls_match_below_exact_threshold_but_plausible_is_low_confidence(self):
        concept = make_concept("NT-proBNP")
        concept.best_match = make_match(
            cui="C0754710",
            score=0.60,
            search_type="exact",
            type_compatible=None,
        )
        tx = FakeTx()

        norm.update_concept_from_result(
            tx=tx,
            concept=concept,
            model_name="UMLS REST API",
            linker_name="umls_api",
            threshold=0.85,
            force=False,
            normalized_at="2026-05-29T00:00:00+00:00",
            match_method=norm.API_NORMALIZATION_METHOD,
            low_confidence_method=norm.API_LOW_CONFIDENCE_METHOD,
            no_match_method=norm.API_NO_MATCH_METHOD,
        )

        self.assertEqual(concept.normalization_status, "umls_low_confidence")
        self.assertEqual(concept.normalization_method, norm.API_LOW_CONFIDENCE_METHOD)
        self.assertNotIn("umls_cui", tx.calls[0][1])

    def test_exact_umls_match_below_plausible_threshold_is_no_plausible_match(self):
        concept = make_concept("NT-proBNP")
        concept.best_match = make_match(
            cui="C0754710",
            score=0.0,
            search_type="exact",
            type_compatible=None,
        )
        tx = FakeTx()

        norm.update_concept_from_result(
            tx=tx,
            concept=concept,
            model_name="UMLS REST API",
            linker_name="umls_api",
            threshold=0.85,
            force=False,
            normalized_at="2026-05-29T00:00:00+00:00",
            match_method=norm.API_NORMALIZATION_METHOD,
            low_confidence_method=norm.API_LOW_CONFIDENCE_METHOD,
            no_match_method=norm.API_NO_MATCH_METHOD,
        )

        self.assertEqual(
            concept.normalization_status,
            norm.NO_PLAUSIBLE_MATCH_STATUS,
        )
        self.assertEqual(
            concept.normalization_method,
            norm.NO_PLAUSIBLE_MATCH_METHOD,
        )
        self.assertNotIn("umls_cui", tx.calls[0][1])

    def test_exact_umls_match_at_exact_threshold_writes_umls_fields(self):
        concept = make_concept("ARVC")
        concept.best_match = make_match(
            cui="C0349788",
            score=0.75,
            search_type="exact",
            type_compatible=None,
        )
        tx = FakeTx()

        norm.update_concept_from_result(
            tx=tx,
            concept=concept,
            model_name="UMLS REST API",
            linker_name="umls_api",
            threshold=0.85,
            force=False,
            normalized_at="2026-05-29T00:00:00+00:00",
            match_method=norm.API_NORMALIZATION_METHOD,
            low_confidence_method=norm.API_LOW_CONFIDENCE_METHOD,
            no_match_method=norm.API_NO_MATCH_METHOD,
        )

        self.assertEqual(concept.normalization_status, "umls_matched")
        self.assertEqual(tx.calls[0][1]["umls_cui"], "C0349788")

    def test_words_umls_match_remains_low_confidence_at_high_score(self):
        concept = make_concept("left ventricular ejection fraction")
        concept.best_match = make_match(
            score=0.95,
            search_type="words",
            type_compatible=True,
        )
        tx = FakeTx()

        norm.update_concept_from_result(
            tx=tx,
            concept=concept,
            model_name="UMLS REST API",
            linker_name="umls_api",
            threshold=0.85,
            force=False,
            normalized_at="2026-05-29T00:00:00+00:00",
            match_method=norm.API_NORMALIZATION_METHOD,
            low_confidence_method=norm.API_LOW_CONFIDENCE_METHOD,
            no_match_method=norm.API_NO_MATCH_METHOD,
        )

        self.assertEqual(concept.normalization_status, "umls_low_confidence")
        self.assertEqual(concept.normalization_method, norm.API_LOW_CONFIDENCE_METHOD)
        self.assertNotIn("umls_cui", tx.calls[0][1])

    def test_low_confidence_candidate_does_not_write_umls_fields(self):
        concept = make_concept("left ventricular hypertrophy")
        concept.best_match = make_match(score=0.5)
        tx = FakeTx()

        norm.update_concept_from_result(
            tx=tx,
            concept=concept,
            model_name="en_core_sci_sm",
            linker_name="umls",
            threshold=0.85,
            force=False,
            normalized_at="2026-05-29T00:00:00+00:00",
        )

        self.assertEqual(concept.normalization_status, "umls_low_confidence")
        self.assertNotIn("umls_cui", tx.calls[0][1])

    def test_same_cui_creates_same_as_candidates(self):
        left = make_concept("left ventricular hypertrophy", concept_id="1")
        right = make_concept("lv hypertrophy", concept_id="2")
        left.best_match = make_match(cui="C123")
        right.best_match = make_match(cui="C123")
        left.normalization_status = "umls_matched"
        right.normalization_status = "umls_matched"

        pairs = norm.compute_same_cui_pairs([left, right])

        self.assertEqual(pairs, [(left, right)])

    def test_same_as_edge_writes_common_normalization_metadata(self):
        tx = FakeTx()

        norm.create_same_as_edge(
            tx=tx,
            source_id="source",
            target_id="target",
            normalized_at="2026-05-29T00:00:00+00:00",
        )

        metadata = tx.calls[0][1]["relationship_metadata"]
        self.assertEqual(metadata["relationship_family"], "normalization")
        self.assertEqual(metadata["provenance"], "umls_normalization")
        self.assertEqual(metadata["provenance_source"], "umls_metathesaurus")
        self.assertEqual(metadata["provenance_method"], "umls_cui")
        self.assertNotIn("source_vocabulary", metadata)

    def test_possibly_same_as_edge_writes_common_normalization_metadata(self):
        tx = FakeTx()

        norm.create_possibly_same_as_edge(
            tx=tx,
            source_id="source",
            target_id="target",
            score=92.5,
            normalized_at="2026-05-29T00:00:00+00:00",
        )

        metadata = tx.calls[0][1]["relationship_metadata"]
        self.assertEqual(metadata["relationship_family"], "normalization")
        self.assertEqual(metadata["provenance"], "umls_normalization")
        self.assertEqual(metadata["provenance_source"], "local_matching")
        self.assertEqual(metadata["provenance_method"], "fuzzy_name")
        self.assertNotIn("source_vocabulary", metadata)

    def test_fuzzy_similarity_creates_candidate_only(self):
        old_fuzz = norm.fuzz
        norm.fuzz = FakeFuzz
        try:
            left = make_concept("left ventricular hypertrophy", concept_id="1")
            right = make_concept("left ventricle hypertrophy", concept_id="2")

            pairs = norm.compute_fuzzy_pairs(
                concepts=[left, right],
                fuzzy_threshold=90,
                same_as_keys=set(),
            )
        finally:
            norm.fuzz = old_fuzz

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][2], 94)

    def test_short_acronyms_are_not_fuzzily_merged_without_evidence(self):
        old_fuzz = norm.fuzz
        norm.fuzz = FakeFuzz
        try:
            left = make_concept("as", concept_id="1")
            right = make_concept("mr", concept_id="2")

            pairs = norm.compute_fuzzy_pairs(
                concepts=[left, right],
                fuzzy_threshold=1,
                same_as_keys=set(),
            )
        finally:
            norm.fuzz = old_fuzz

        self.assertEqual(pairs, [])

    def test_local_cache_resolver_uses_metadata_without_network_head(self):
        url = "https://example.test/scispacy/resource.bin"

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            data_path = cache_dir / "cached.resource.bin"
            meta_path = cache_dir / "cached.resource.bin.json"
            data_path.write_bytes(b"cached")
            meta_path.write_text(
                json.dumps({"url": url, "etag": "fake-etag"}),
                encoding="utf-8",
            )

            resolved = norm.find_cached_scispacy_file_without_head(
                url,
                str(cache_dir),
            )

        self.assertEqual(resolved, str(data_path))

    def test_memory_guard_fails_before_loading_heavy_umls_linker(self):
        old_get_available_memory_gb = norm.get_available_memory_gb
        norm.get_available_memory_gb = lambda: 1.0
        try:
            with self.assertRaisesRegex(RuntimeError, "Not enough available memory"):
                norm.ensure_available_memory(8.0)
        finally:
            norm.get_available_memory_gb = old_get_available_memory_gb

    def test_umls_api_client_successful_match(self):
        old_key = os.environ.get("UMLS_API_KEY")
        os.environ["UMLS_API_KEY"] = "secret-test-key"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                client = norm.UMLSAPIClient(
                    cache_dir=Path(tmp),
                    rate_limit_per_second=0,
                    session=FakeSession([FakeResponse(200, sample_search_payload())]),
                )

                match = client.search_alias(
                    "atrial fibrillation",
                    "normalizedString",
                )

            self.assertIsNotNone(match)
            self.assertEqual(match.cui, "C0004238")
            self.assertEqual(match.canonical_name, "Atrial Fibrillation")
            self.assertEqual(match.score, 1.0)
            self.assertEqual(match.semantic_types, ["Disease or Syndrome"])
        finally:
            if old_key is None:
                os.environ.pop("UMLS_API_KEY", None)
            else:
                os.environ["UMLS_API_KEY"] = old_key

    def test_umls_api_cache_hit_avoids_api_call(self):
        old_key = os.environ.get("UMLS_API_KEY")
        os.environ["UMLS_API_KEY"] = "secret-test-key"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                client = norm.UMLSAPIClient(
                    cache_dir=Path(tmp),
                    rate_limit_per_second=0,
                    session=FakeSession([FakeResponse(200, sample_search_payload())]),
                )

                first = client.search_alias("atrial fibrillation", "normalizedString")
                client.session = FakeSession([])
                second = client.search_alias("atrial fibrillation", "normalizedString")

            self.assertEqual(first.cui, second.cui)
            self.assertEqual(client.stats["api_requests"], 1)
            self.assertEqual(client.stats["api_cache_hits"], 1)
        finally:
            if old_key is None:
                os.environ.pop("UMLS_API_KEY", None)
            else:
                os.environ["UMLS_API_KEY"] = old_key

    def test_umls_api_auth_error_does_not_include_key(self):
        old_key = os.environ.get("UMLS_API_KEY")
        os.environ["UMLS_API_KEY"] = "do-not-print-this-key"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                client = norm.UMLSAPIClient(
                    cache_dir=Path(tmp),
                    rate_limit_per_second=0,
                    session=FakeSession([FakeResponse(401, {})]),
                )
                with self.assertRaises(norm.UMLSAPIAuthError) as raised:
                    client.search_alias("atrial fibrillation", "normalizedString")

            self.assertNotIn("do-not-print-this-key", str(raised.exception))
            self.assertIn("UMLS_API_KEY is missing/invalid", str(raised.exception))
        finally:
            if old_key is None:
                os.environ.pop("UMLS_API_KEY", None)
            else:
                os.environ["UMLS_API_KEY"] = old_key

    def test_fuzzy_only_backend_does_not_require_umls_api_key(self):
        old_key = os.environ.pop("UMLS_API_KEY", None)
        try:
            self.assertEqual(norm.normalize_backend_name("fuzzy_only"), "fuzzy_only")
        finally:
            if old_key is not None:
                os.environ["UMLS_API_KEY"] = old_key

    def test_api_low_confidence_uses_status_only(self):
        concept = make_concept("some weak concept")
        concept.best_match = make_match(score=0.5)
        tx = FakeTx()

        norm.update_concept_from_result(
            tx=tx,
            concept=concept,
            model_name="UMLS REST API",
            linker_name="umls_api",
            threshold=0.85,
            force=False,
            normalized_at="2026-05-29T00:00:00+00:00",
            match_method=norm.API_NORMALIZATION_METHOD,
            low_confidence_method=norm.API_LOW_CONFIDENCE_METHOD,
            no_match_method=norm.API_NO_MATCH_METHOD,
        )

        self.assertEqual(concept.normalization_status, "umls_low_confidence")
        self.assertEqual(concept.normalization_method, norm.API_LOW_CONFIDENCE_METHOD)
        self.assertNotIn("umls_cui", tx.calls[0][1])

    def test_api_no_match_uses_status_only(self):
        concept = make_concept("unmatched concept")
        tx = FakeTx()

        norm.update_concept_from_result(
            tx=tx,
            concept=concept,
            model_name="UMLS REST API",
            linker_name="umls_api",
            threshold=0.85,
            force=False,
            normalized_at="2026-05-29T00:00:00+00:00",
            match_method=norm.API_NORMALIZATION_METHOD,
            low_confidence_method=norm.API_LOW_CONFIDENCE_METHOD,
            no_match_method=norm.API_NO_MATCH_METHOD,
        )

        self.assertEqual(concept.normalization_status, "umls_no_match")
        self.assertEqual(concept.normalization_method, norm.API_NO_MATCH_METHOD)
        self.assertNotIn("umls_cui", tx.calls[0][1])


if __name__ == "__main__":
    unittest.main()
