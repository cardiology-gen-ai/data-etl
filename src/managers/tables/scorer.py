import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import List, Optional


# --- Stop-word list tuned to ESC recommendation captions --------------------
# We strip the words that appear in EVERY caption ("Recommendations for ...")
# so the score reflects topic-bearing tokens only.
_STOPWORDS = {
    "recommendation", "recommendations", "for", "in", "with", "of", "the",
    "a", "and", "or", "to", "patients", "individuals", "from", "without",
    "their", "those", "be", "is", "are", "an", "on",
}


def _content_tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


# ============================================================================
# Abstract scorer
# ============================================================================


class CaptionScorer(ABC):
    """Score how well a section title matches a recommendation table caption.

    Returns a value in [0, 1]. Higher = better match.
    """

    name: str = "abstract"

    @abstractmethod
    def score(self, caption: str, title: str) -> float:
        ...

    def score_many(self, caption: str, titles: List[str]) -> List[float]:
        """Override for batch-friendly scorers (embeddings benefit a lot)."""
        return [self.score(caption, t) for t in titles]

    def prime(self, titles: List[str]) -> None:
        """Optional hook for pre-computing anything that depends on titles."""
        return None


class LexicalScorer(CaptionScorer):
    """Jaccard + sequence ratio on content tokens."""

    name = "lexical"

    def __init__(self, jaccard_weight: float = 0.7):
        self.w_jac = jaccard_weight
        self.w_seq = 1.0 - jaccard_weight

    def score(self, caption: str, title: str) -> float:
        c, t = _content_tokens(caption), _content_tokens(title)
        if not c or not t:
            return 0.0

        jac = len(c & t) / len(c | t)
        seq = SequenceMatcher(
            None,
            " ".join(sorted(c)),
            " ".join(sorted(t)),
        ).ratio()

        return self.w_jac * jac + self.w_seq * seq


class EmbeddingScorer(CaptionScorer):
    """Cosine similarity between caption and title sentence embeddings.

    Uses a biomedical sentence encoder via langchain-huggingface. This improves
    cases where lexical overlap is zero but clinical meaning is close
    (e.g. dyslipidaemia ↔ lipids).
    """

    DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        cache_folder: Optional[str] = None,
    ):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            raise ImportError(
                "EmbeddingScorer requires langchain-huggingface. "
                "Install with: pip install langchain-huggingface sentence-transformers"
            ) from e

        self.model_name = model_name
        self.name = f"embedding:{model_name}"
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=cache_folder,
        )
        self._title_cache: dict[str, list[float]] = {}

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def prime(self, titles: List[str]) -> None:
        """Embed all unique section titles up front."""
        new = [t for t in set(titles) if t not in self._title_cache]
        if not new:
            return

        vectors = self._embeddings.embed_documents(new)

        for t, v in zip(new, vectors):
            self._title_cache[t] = v

    def score(self, caption: str, title: str) -> float:
        cap_vec = self._embeddings.embed_query(caption)

        if title not in self._title_cache:
            self._title_cache[title] = self._embeddings.embed_documents([title])[0]

        sim = self._dot(cap_vec, self._title_cache[title])

        # Cosine on normalized vectors is in [-1, 1]; map to [0, 1].
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def score_many(self, caption: str, titles: List[str]) -> List[float]:
        cap_vec = self._embeddings.embed_query(caption)

        missing = [t for t in titles if t not in self._title_cache]
        if missing:
            vectors = self._embeddings.embed_documents(missing)
            for t, v in zip(missing, vectors):
                self._title_cache[t] = v

        return [
            max(0.0, min(1.0, (self._dot(cap_vec, self._title_cache[t]) + 1.0) / 2.0))
            for t in titles
        ]
