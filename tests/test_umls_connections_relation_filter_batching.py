import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge_graph import umls_connections as conn


class _FakeBatchClient:
    def __init__(self):
        self.calls = []

    def get_relations(self, *, cui, source_vocab, max_records, include_additional_relation_labels):
        labels = tuple(include_additional_relation_labels)
        self.calls.append((cui, source_vocab, max_records, labels))
        records = [
            {
                "ui": f"R-{label}",
                "additionalRelationLabel": label,
            }
            for label in labels
        ]
        return conn.RelationFetchResult(
            records=records,
            fetched_records=len(records),
            page_count=1,
        )


class RelationFilterBatchingTests(unittest.TestCase):
    def test_relation_labels_are_stably_batched(self):
        labels = [f"rel_{i:03d}" for i in range(45)] + ["rel_000"]
        batches = conn.relation_label_batches(labels, batch_size=20)
        self.assertEqual([len(batch) for batch in batches], [20, 20, 5])
        flattened = [label for batch in batches for label in batch]
        self.assertEqual(flattened, sorted(set(labels)))

    def test_broad_filter_is_fetched_in_exhaustive_batches_then_merged(self):
        client = _FakeBatchClient()
        labels = [f"rel_{i:03d}" for i in range(45)]
        result = conn.get_relations_with_batched_filter(
            client,
            cui="C0000001",
            source_vocab="SNOMEDCT_US",
            max_records=100,
            relation_labels=labels,
            batch_size=20,
        )
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all(call[2] is None for call in client.calls))
        self.assertEqual(result.fetched_records, 45)
        self.assertEqual(len(result.records), 45)
        self.assertFalse(result.truncated_by_limit)

    def test_global_guardrail_is_applied_after_batch_merge(self):
        client = _FakeBatchClient()
        labels = [f"rel_{i:03d}" for i in range(45)]
        result = conn.get_relations_with_batched_filter(
            client,
            cui="C0000001",
            source_vocab="SNOMEDCT_US",
            max_records=30,
            relation_labels=labels,
            batch_size=20,
        )
        self.assertEqual(result.fetched_records, 45)
        self.assertEqual(len(result.records), 30)
        self.assertTrue(result.truncated_by_limit)
        self.assertEqual(result.skipped_by_limit, 15)

    def test_http_403_is_not_mislabeled_as_invalid_api_key(self):
        class Response:
            status_code = 403

        class Session:
            def get(self, *args, **kwargs):
                return Response()

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"UMLS_API_KEY": "test-key"},
            clear=False,
        ):
            client = conn.UMLSRelationsClient(
                cache_dir=Path(tmp),
                rate_limit_per_second=0,
                session=Session(),
            )
            with self.assertRaises(conn.UMLSAPIHTTPStatusError) as ctx:
                client.request_uncached(
                    "https://example.invalid",
                    params={},
                    max_retries=0,
                )
            self.assertEqual(ctx.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
