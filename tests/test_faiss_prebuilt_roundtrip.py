import pathlib
import tempfile
import unittest
import warnings

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from cardiology_gen_ai import (
    DistanceTypeNames,
    EmbeddingConfig,
    IndexingConfig,
    IndexTypeNames,
    RetrievalTypeNames,
)
from src.managers.index_manager import (
    EditableFaissVectorstore,
    IndexManager,
    L2NormalizedEmbeddings,
)


class _DeterministicEmbeddings(Embeddings):
    """Tiny deterministic embedding backend for a real FAISS round-trip."""

    @staticmethod
    def _vector(text: str):
        lowered = text.lower()
        if "alpha" in lowered:
            return [2.0, 0.0, 0.0, 0.0]
        if "beta" in lowered:
            return [0.0, 3.0, 0.0, 0.0]
        return [0.0, 0.0, 4.0, 0.0]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class FaissPrebuiltRoundTripTests(unittest.TestCase):
    def test_cosine_faiss_round_trip_overwrite_move_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = pathlib.Path(tmpdir)
            indexing = IndexingConfig(
                name="test_prebuilt",
                description="test",
                folder=folder,
                type=IndexTypeNames.faiss,
                distance=DistanceTypeNames.cosine,
                retrieval_mode=RetrievalTypeNames.dense,
            )
            embeddings = EmbeddingConfig(
                model_name="deterministic-test",
                ollama=False,
                model=_DeterministicEmbeddings(),
                kwargs={},
                dim=4,
            )
            manager = IndexManager(config=indexing, embeddings=embeddings)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                manager.create_index()
            self.assertFalse(
                any(
                    "Normalizing L2 is not applicable" in str(item.message)
                    for item in caught
                )
            )

            documents = [
                Document(
                    page_content="Title: Alpha\n\nBody:\nalpha evidence",
                    metadata={
                        "filename": "machine-a/artifact.json",
                        "doc_id": "Doc",
                        "prebuilt_source_type": "fixed_chunks",
                        "record_id": "alpha-record",
                    },
                ),
                Document(
                    page_content="Title: Beta\n\nBody:\nbeta evidence",
                    metadata={
                        "filename": "machine-a/artifact.json",
                        "doc_id": "Doc",
                        "prebuilt_source_type": "fixed_chunks",
                        "record_id": "beta-record",
                    },
                ),
            ]

            manager.add_document(documents)
            self.assertEqual(manager.get_n_documents_in_vectorstore(), 2)
            self.assertTrue((folder / "test_prebuilt.faiss").is_file())
            self.assertTrue((folder / "test_prebuilt.pkl").is_file())
            self.assertIsInstance(
                manager.vectorstore.vectorstore.embedding_function,
                L2NormalizedEmbeddings,
            )

            result = manager.vectorstore.vectorstore.similarity_search("alpha", k=1)
            self.assertEqual(result[0].metadata["record_id"], "alpha-record")

            moved_documents = [
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "filename": "machine-b/artifact.json",
                        "source_key": "Doc::fixed_chunks",
                    },
                )
                for document in documents
            ]
            manager.add_document(moved_documents)
            self.assertEqual(manager.get_n_documents_in_vectorstore(), 2)
            stored_filenames = {
                document.metadata["filename"]
                for document in manager.vectorstore.vectorstore.docstore._dict.values()
            }
            self.assertEqual(stored_filenames, {"machine-b/artifact.json"})

            reloaded = EditableFaissVectorstore(config=indexing)
            reloaded.load_vectorstore(embeddings_model=embeddings)
            self.assertEqual(reloaded.get_n_documents_in_vectorstore(), 2)
            self.assertIsInstance(
                reloaded.vectorstore.embedding_function,
                L2NormalizedEmbeddings,
            )
            result = reloaded.vectorstore.similarity_search("beta", k=1)
            self.assertEqual(result[0].metadata["record_id"], "beta-record")


if __name__ == "__main__":
    unittest.main(verbosity=2)
