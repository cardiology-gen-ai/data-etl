"""
llm_utils.py

Local LLM/embedding utilities for the knowledge-graph pipeline.
"""

import gc
import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_optional_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _get_required_env(name: str) -> str:
    value = _get_optional_env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_env_int(name: str, default: int) -> int:
    value = _get_optional_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as e:
        raise RuntimeError(
            f"Environment variable {name} must be an integer, got: {value!r}"
        ) from e


def _ensure_json_only_instruction(text: str) -> str:
    """
    Append a JSON-only instruction if none is already present.
    """
    text = text.rstrip()

    normalized = text.lower()
    already_requests_json_only = (
        "return json only" in normalized
        or "return valid json only" in normalized
    )

    if not already_requests_json_only:
        text += "\n\nReturn valid JSON only."

    return text


def _clear_cuda_cache() -> None:
    """
    Best-effort CUDA cache cleanup.

    This does not magically free tensors that are still referenced somewhere,
    but after lru_cache.clear_cache() and gc.collect(), it helps release cached
    GPU memory back to PyTorch/CUDA.
    """
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning("torch.cuda.empty_cache() failed: %s", e)

    try:
        torch.cuda.ipc_collect()
    except Exception as e:
        logger.debug("torch.cuda.ipc_collect() failed or unavailable: %s", e)


def get_chat_model_ref() -> str:
    """
    Return the reference used to load the chat model.

    If KG_CHAT_MODEL_PATH is set, loading uses that local/custom path.
    Otherwise KG_CHAT_MODEL is used.
    """
    return _get_optional_env("KG_CHAT_MODEL_PATH") or _get_required_env("KG_CHAT_MODEL")


def get_embedding_model_ref() -> str:
    """
    Return the reference used to load the embedding model.

    If KG_EMBEDDING_MODEL_PATH is set, loading uses that local/custom path.
    Otherwise KG_EMBEDDING_MODEL is used.
    """
    return _get_optional_env("KG_EMBEDDING_MODEL_PATH") or _get_required_env(
        "KG_EMBEDDING_MODEL"
    )


def get_chat_model_name() -> str:
    """
    Return the canonical chat model name to be logged/stored in metadata.
    """
    return _get_required_env("KG_CHAT_MODEL")


def get_embedding_model_name() -> str:
    """
    Return the canonical embedding model name to be logged/stored in metadata.
    """
    return _get_required_env("KG_EMBEDDING_MODEL")


@lru_cache(maxsize=1)
def get_chat_tokenizer_and_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    model_ref = get_chat_model_ref()
    hf_token = _get_optional_env("HF_TOKEN")
    local_files_only = _get_env_bool("KG_LOCAL_FILES_ONLY", True)

    logger.info(
        "Loading chat model | model_ref=%s | model_name=%s | local_files_only=%s",
        model_ref,
        get_chat_model_name(),
        local_files_only,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        token=hf_token,
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # torch_dtype is the most widely compatible argument name across
    # Transformers versions. Your previous dtype="auto" may work in newer
    # versions, but torch_dtype="auto" is safer on many clusters.
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        token=hf_token,
        local_files_only=local_files_only,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    model_ref = get_embedding_model_ref()
    hf_token = _get_optional_env("HF_TOKEN")
    local_files_only = _get_env_bool("KG_LOCAL_FILES_ONLY", True)

    # Optional override is useful on clusters, but defaults to CUDA when present.
    device = _get_optional_env("KG_EMBEDDING_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Loading embedding model | model_ref=%s | model_name=%s | local_files_only=%s | device=%s",
        model_ref,
        get_embedding_model_name(),
        local_files_only,
        device,
    )

    model = SentenceTransformer(
        model_ref,
        token=hf_token,
        local_files_only=local_files_only,
        device=device,
    )

    return model


def clear_chat_model_cache() -> None:
    """
    Release the cached chat tokenizer/model and clear CUDA cache.

    Use this after entity extraction and before embedding generation when
    running both stages in the same Python process.
    """
    logger.info("Clearing cached chat model/tokenizer")

    get_chat_tokenizer_and_model.cache_clear()
    gc.collect()
    _clear_cuda_cache()


def clear_embedding_model_cache() -> None:
    """
    Release the cached embedding model and clear CUDA cache.

    Useful if a later stage in the same process needs to load the chat model again.
    """
    logger.info("Clearing cached embedding model")

    get_embedding_model.cache_clear()
    gc.collect()
    _clear_cuda_cache()


def clear_all_model_caches() -> None:
    """
    Release all cached model objects managed by this module.
    """
    logger.info("Clearing all cached LLM/embedding models")

    get_chat_tokenizer_and_model.cache_clear()
    get_embedding_model.cache_clear()
    gc.collect()
    _clear_cuda_cache()


def generate_chat_text(
    messages: List[Dict[str, str]],
    json_mode: bool = False,
    max_new_tokens: Optional[int] = None,
) -> str:
    if not messages:
        raise ValueError("messages must not be empty")

    tokenizer, model = get_chat_tokenizer_and_model()

    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            f"Tokenizer for model '{get_chat_model_name()}' has no chat template. "
            "Use a chat/instruct model or implement a manual prompt formatter."
        )

    prepared_messages = [dict(m) for m in messages]

    if json_mode and prepared_messages:
        last = prepared_messages[-1]
        if last.get("role") == "user":
            last["content"] = _ensure_json_only_instruction(last.get("content") or "")

    input_ids = tokenizer.apply_chat_template(
        prepared_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    attention_mask = torch.ones_like(input_ids)

    try:
        model_device = model.device
    except Exception:
        model_device = next(model.parameters()).device

    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)

    effective_max_new_tokens = (
        max_new_tokens
        if max_new_tokens is not None
        else _get_env_int("KG_CHAT_MAX_NEW_TOKENS", 512)
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=effective_max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    generated_ids = outputs[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text


def embed_texts(
    texts: List[str],
    batch_size: int = 8,
) -> List[List[float]]:
    if not texts:
        return []

    model = get_embedding_model()

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return vectors.tolist()