import os
import types
import unittest
from unittest.mock import MagicMock, patch





from knowledge_graph import add_embeddings, build_graph, llm_utils


class FakeOpenAIEmbeddingItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class FakeOpenAIEmbeddingResponse:
    def __init__(self, data):
        self.data = data
        self.usage = None


class FakeArray(list):
    def tolist(self):
        return list(self)


class FakeTx:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))


class FakeSession:
    def __init__(self, tx):
        self.tx = tx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute_write(self, fn, *args):
        return fn(self.tx, *args)


class FakeDriver:
    def __init__(self):
        self.tx = FakeTx()

    def session(self):
        return FakeSession(self.tx)


class EmbeddingConfigOpenAITests(unittest.TestCase):
    def setUp(self):
        llm_utils.get_openai_client.cache_clear()
        llm_utils.get_embedding_model.cache_clear()

    def tearDown(self):
        llm_utils.get_openai_client.cache_clear()
        llm_utils.get_embedding_model.cache_clear()

    def _mock_openai_client(self, response):
        client = MagicMock()
        client.embeddings.create.return_value = response
        return client


    def test_openai_provider_dispatches_without_dimensions_when_none(self):
        response = FakeOpenAIEmbeddingResponse([
            FakeOpenAIEmbeddingItem(index=0, embedding=[3.0, 0.0]),
        ])
        client = self._mock_openai_client(response)

        with patch("knowledge_graph.llm_utils.get_openai_client", return_value=client):
            vectors = llm_utils.embed_texts(
                ["alpha"],
                batch_size=1,
                provider="openai",
                model_name="text-embedding-3-small",
                dimensions=None,
            )

        self.assertEqual(vectors, [[1.0, 0.0]])
        request_kwargs = client.embeddings.create.call_args.kwargs
        self.assertEqual(request_kwargs["model"], "text-embedding-3-small")
        self.assertEqual(request_kwargs["input"], ["alpha"])
        self.assertNotIn("dimensions", request_kwargs)

    def test_openai_provider_includes_numeric_dimensions(self):
        response = FakeOpenAIEmbeddingResponse([
            FakeOpenAIEmbeddingItem(index=0, embedding=[0.0, 2.0]),
        ])
        client = self._mock_openai_client(response)

        with patch("knowledge_graph.llm_utils.get_openai_client", return_value=client):
            llm_utils.embed_texts(
                ["alpha"],
                batch_size=1,
                provider="openai",
                model_name="text-embedding-3-small",
                dimensions=256,
            )

        self.assertEqual(client.embeddings.create.call_args.kwargs["dimensions"], 256)

    def test_local_provider_dispatches_to_sentence_transformer_backend(self):
        model = MagicMock()
        model.encode.return_value = FakeArray([[1.0, 0.0]])

        with patch("knowledge_graph.llm_utils.get_embedding_model", return_value=model) as get_model:
            with patch("knowledge_graph.llm_utils.get_openai_client") as get_openai_client:
                vectors = llm_utils.embed_texts(
                    ["local text"],
                    batch_size=4,
                    provider="local_hf",
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                )

        self.assertEqual(vectors, [[1.0, 0.0]])
        get_model.assert_called_once()
        get_openai_client.assert_not_called()
        model.encode.assert_called_once_with(
            ["local text"],
            batch_size=4,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def test_missing_openai_api_key_raises_only_for_openai_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                llm_utils.embed_texts(
                    ["alpha"],
                    provider="openai",
                    model_name="text-embedding-3-small",
                )

        model = MagicMock()
        model.encode.return_value = FakeArray([[1.0]])
        with patch.dict(os.environ, {}, clear=True):
            with patch("knowledge_graph.llm_utils.get_embedding_model", return_value=model):
                vectors = llm_utils.embed_texts(
                    ["alpha"],
                    provider="local_hf",
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                )

        self.assertEqual(vectors, [[1.0]])

    def test_openai_response_ordering_uses_index(self):
        response = FakeOpenAIEmbeddingResponse([
            FakeOpenAIEmbeddingItem(index=1, embedding=[0.0, 2.0]),
            FakeOpenAIEmbeddingItem(index=0, embedding=[3.0, 0.0]),
        ])
        client = self._mock_openai_client(response)

        with patch("knowledge_graph.llm_utils.get_openai_client", return_value=client):
            vectors = llm_utils.embed_texts(
                ["first", "second"],
                batch_size=2,
                provider="openai",
                model_name="text-embedding-3-small",
            )

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])

    def test_openai_response_rejects_invalid_indexes(self):
        cases = [
            [
                FakeOpenAIEmbeddingItem(index=0, embedding=[1.0]),
                FakeOpenAIEmbeddingItem(index=0, embedding=[1.0]),
            ],
            [
                FakeOpenAIEmbeddingItem(index=0, embedding=[1.0]),
                FakeOpenAIEmbeddingItem(index=2, embedding=[1.0]),
            ],
            [
                FakeOpenAIEmbeddingItem(index="0", embedding=[1.0]),
                FakeOpenAIEmbeddingItem(index=1, embedding=[1.0]),
            ],
        ]

        for data in cases:
            with self.subTest(indexes=[item.index for item in data]):
                client = self._mock_openai_client(FakeOpenAIEmbeddingResponse(data))
                with patch("knowledge_graph.llm_utils.get_openai_client", return_value=client):
                    with self.assertRaisesRegex(RuntimeError, "index"):
                        llm_utils.embed_texts(
                            ["first", "second"],
                            batch_size=2,
                            provider="openai",
                            model_name="text-embedding-3-small",
                        )

    def test_build_graph_passes_embedding_config_to_add_embeddings(self):
        config = types.SimpleNamespace(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=256,
        )

        with patch.object(
            build_graph,
            "add_embeddings_to_sections",
            return_value={"written_embeddings": 0},
        ) as add_embeddings_to_sections:
            build_graph.process_document_embeddings(
                driver=MagicMock(),
                config=config,
                doc_id="doc-1",
            )

        kwargs = add_embeddings_to_sections.call_args.kwargs
        self.assertEqual(kwargs["embedding_provider"], "openai")
        self.assertEqual(kwargs["embedding_model"], "text-embedding-3-small")
        self.assertEqual(kwargs["embedding_dimensions"], 256)

    def test_add_embeddings_passes_config_to_request_and_writes_model(self):
        driver = FakeDriver()
        segment_text = "section text"

        prepared_row = {
            "uid": "section-1",
            "doc_id": "doc-1",
            "section_id": "1",
            "embedding_segments": [
                {
                    "text": segment_text,
                    "weight_chars": len(segment_text),
                }
            ],
            "embedding_input_chars": len(segment_text),
            "embedding_content_hash": "test-content-hash",
        }

        with patch(
            "knowledge_graph.add_embeddings.fetch_sections_to_embed",
            return_value=([prepared_row], 0),
        ):
            with patch(
                "knowledge_graph.add_embeddings.request_embeddings",
                return_value=[[1.0, 2.0, 3.0]],
            ) as request_embeddings:
                stats = add_embeddings.add_embeddings_to_sections(
                    driver=driver,
                    doc_id="doc-1",
                    embedding_provider="openai",
                    embedding_model="text-embedding-3-small",
                    embedding_dimensions=256,
                    batch_size=1,
                )

        request_embeddings.assert_called_once_with(
            texts=[segment_text],
            batch_size=1,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=256,
        )
        self.assertEqual(stats["written_embeddings"], 1)
        self.assertEqual(stats["embedding_segments"], 1)
        self.assertEqual(
            driver.tx.calls[-1][1]["embedding_model"],
            "text-embedding-3-small",
        )
        written_row = driver.tx.calls[-1][1]["rows"][0]
        self.assertEqual(written_row["embedding_content_hash"], "test-content-hash")
        self.assertEqual(written_row["embedding_segment_count"], 1)


if __name__ == "__main__":
    unittest.main()
