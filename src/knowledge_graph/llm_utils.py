import gc
import logging
import math
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)

LOCAL_HF_PROVIDER = "local_hf"
OPENAI_PROVIDER = "openai"

_PROVIDER_ALIASES = {
    "hf": LOCAL_HF_PROVIDER,
    "huggingface": LOCAL_HF_PROVIDER,
    "local": LOCAL_HF_PROVIDER,
    "local_hf": LOCAL_HF_PROVIDER,
    "openai": OPENAI_PROVIDER,
}


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


def _normalize_provider(value: Optional[str], env_name: str) -> str:
    raw = (value or LOCAL_HF_PROVIDER).strip().lower().replace("-", "_")
    provider = _PROVIDER_ALIASES.get(raw)

    if provider is None:
        supported = ", ".join(sorted(set(_PROVIDER_ALIASES.values())))
        raise RuntimeError(
            f"Unsupported {env_name}={value!r}. Supported providers: {supported}"
        )

    return provider


def get_chat_provider() -> str:
    """
    Return the configured chat provider.

    KG_CHAT_PROVIDER is preferred. KG_MODEL_PROVIDER is accepted as a shared
    fallback for runs where chat and embeddings use the same backend.
    """
    return _normalize_provider(
        _get_optional_env("KG_CHAT_PROVIDER") or _get_optional_env("KG_MODEL_PROVIDER"),
        "KG_CHAT_PROVIDER",
    )


def get_embedding_provider() -> str:
    """
    Return the configured embedding provider.

    KG_EMBEDDING_PROVIDER is preferred. KG_MODEL_PROVIDER is accepted as a
    shared fallback for runs where chat and embeddings use the same backend.
    """
    return _normalize_provider(
        _get_optional_env("KG_EMBEDDING_PROVIDER")
        or _get_optional_env("KG_MODEL_PROVIDER"),
        "KG_EMBEDDING_PROVIDER",
    )


def resolve_embedding_provider(provider: Optional[str] = None) -> str:
    """
    Resolve an explicit embedding provider, falling back to legacy env settings.
    """
    return _normalize_provider(
        provider
        or _get_optional_env("KG_EMBEDDING_PROVIDER")
        or _get_optional_env("KG_MODEL_PROVIDER"),
        "embedding_provider",
    )


def get_chat_model_ref() -> str:
    """
    Return the reference used to load the chat model.

    If KG_CHAT_MODEL_PATH is set, loading uses that local/custom path.
    Otherwise KG_CHAT_MODEL is used.
    """
    if get_chat_provider() != LOCAL_HF_PROVIDER:
        return _get_required_env("KG_CHAT_MODEL")

    return _get_optional_env("KG_CHAT_MODEL_PATH") or _get_required_env("KG_CHAT_MODEL")


def get_embedding_model_ref(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
) -> str:
    """
    Return the reference used to load the embedding model.

    If KG_EMBEDDING_MODEL_PATH is set, loading uses that local/custom path.
    Otherwise KG_EMBEDDING_MODEL is used.
    """
    if resolve_embedding_provider(provider) != LOCAL_HF_PROVIDER:
        return get_embedding_model_name(model_name)

    return (
        model_path
        or _get_optional_env("KG_EMBEDDING_MODEL_PATH")
        or get_embedding_model_name(model_name)
    )


def get_chat_model_name() -> str:
    """
    Return the canonical chat model name to be logged/stored in metadata.
    """
    return _get_required_env("KG_CHAT_MODEL")


def get_embedding_model_name(model_name: Optional[str] = None) -> str:
    """
    Return the canonical embedding model name to be logged/stored in metadata.
    """
    if model_name:
        return model_name
    return _get_required_env("KG_EMBEDDING_MODEL")


@lru_cache(maxsize=1)
def get_chat_tokenizer_and_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    if get_chat_provider() != LOCAL_HF_PROVIDER:
        raise RuntimeError(
            "get_chat_tokenizer_and_model() is only available for KG_CHAT_PROVIDER=local_hf"
        )

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


@lru_cache(maxsize=4)
def get_embedding_model(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    local_files_only: Optional[bool] = None,
) -> SentenceTransformer:
    if resolve_embedding_provider(provider) != LOCAL_HF_PROVIDER:
        raise RuntimeError(
            "get_embedding_model() is only available for embedding provider local_hf"
        )

    model_ref = get_embedding_model_ref(
        provider=provider,
        model_name=model_name,
        model_path=model_path,
    )
    hf_token = _get_optional_env("HF_TOKEN")
    if local_files_only is None:
        local_files_only = _get_env_bool("KG_LOCAL_FILES_ONLY", True)

    # Optional override is useful on clusters, but defaults to CUDA when present.
    device = _get_optional_env("KG_EMBEDDING_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Loading embedding model | model_ref=%s | model_name=%s | local_files_only=%s | device=%s",
        model_ref,
        get_embedding_model_name(model_name),
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


@lru_cache(maxsize=1)
def get_openai_client():
    """
    Lazily create the plain OpenAI client.

    OPENAI_API_KEY is required only when an OpenAI provider is selected.
    """
    _get_required_env("OPENAI_API_KEY")

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "The OpenAI Python package is required for KG_*_PROVIDER=openai. "
            "Install the data-etl project dependencies first."
        ) from e

    return OpenAI()


def _log_openai_usage(response: Any, operation: str, model_name: str) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    input_tokens = (
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
    )
    output_tokens = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
    )
    total_tokens = getattr(usage, "total_tokens", None)

    logger.info(
        "OpenAI usage | operation=%s | model=%s | input_tokens=%s | output_tokens=%s | total_tokens=%s",
        operation,
        model_name,
        input_tokens,
        output_tokens,
        total_tokens,
    )


def clear_chat_model_cache() -> None:
    """
    Release the cached chat tokenizer/model and clear CUDA cache.

    Use this after entity extraction and before embedding generation when
    running both stages in the same Python process.
    """
    logger.info("Clearing cached chat model/tokenizer")

    get_chat_tokenizer_and_model.cache_clear()
    get_openai_client.cache_clear()
    gc.collect()
    _clear_cuda_cache()


def clear_embedding_model_cache() -> None:
    """
    Release the cached embedding model and clear CUDA cache.

    Useful if a later stage in the same process needs to load the chat model again.
    """
    logger.info("Clearing cached embedding model")

    get_embedding_model.cache_clear()
    get_openai_client.cache_clear()
    gc.collect()
    _clear_cuda_cache()


def clear_all_model_caches() -> None:
    """
    Release all cached model objects managed by this module.
    """
    logger.info("Clearing all cached LLM/embedding models")

    get_chat_tokenizer_and_model.cache_clear()
    get_embedding_model.cache_clear()
    get_openai_client.cache_clear()
    gc.collect()
    _clear_cuda_cache()


def _build_openai_response_format(
    json_mode: bool,
    json_schema: Optional[Dict[str, Any]],
    json_schema_name: str,
) -> Optional[Dict[str, Any]]:
    if json_schema is not None:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": json_schema_name,
                "strict": True,
                "schema": json_schema,
            },
        }

    if json_mode:
        return {"type": "json_object"}

    return None


def _generate_openai_chat_text(
    messages: List[Dict[str, str]],
    json_mode: bool,
    max_new_tokens: Optional[int],
    json_schema: Optional[Dict[str, Any]],
    json_schema_name: str,
) -> str:
    client = get_openai_client()
    model_name = get_chat_model_name()

    request: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": 0,
    }

    effective_max_tokens = (
        max_new_tokens
        if max_new_tokens is not None
        else _get_env_int("KG_CHAT_MAX_NEW_TOKENS", 512)
    )
    request["max_tokens"] = effective_max_tokens

    response_format = _build_openai_response_format(
        json_mode=json_mode,
        json_schema=json_schema,
        json_schema_name=json_schema_name,
    )
    if response_format is not None:
        request["response_format"] = response_format

    logger.info(
        "Calling OpenAI chat model | model=%s | json_mode=%s | schema=%s | max_tokens=%d",
        model_name,
        json_mode,
        json_schema_name if json_schema is not None else None,
        effective_max_tokens,
    )

    response = client.chat.completions.create(**request)
    _log_openai_usage(response, operation="chat", model_name=model_name)

    if not response.choices:
        raise RuntimeError("OpenAI chat response did not contain choices")

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI chat response content was empty")

    return content.strip()


def generate_chat_text(
    messages: List[Dict[str, str]],
    json_mode: bool = False,
    max_new_tokens: Optional[int] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    json_schema_name: str = "kg_response",
) -> str:
    if not messages:
        raise ValueError("messages must not be empty")

    if get_chat_provider() == OPENAI_PROVIDER:
        return _generate_openai_chat_text(
            messages=messages,
            json_mode=json_mode,
            max_new_tokens=max_new_tokens,
            json_schema=json_schema,
            json_schema_name=json_schema_name,
        )

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


def _iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _l2_normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    if norm == 0:
        return vector

    return [float(x) / norm for x in vector]


def _embed_texts_openai(
    texts: List[str],
    batch_size: int,
    model_name: str,
    dimensions: Optional[int] = None,
) -> List[List[float]]:
    client = get_openai_client()
    vectors: List[List[float]] = []

    for batch in _iter_batches(texts, batch_size):
        logger.info(
            "Calling OpenAI embedding model | model=%s | dimensions=%s | batch_size=%d",
            model_name,
            dimensions,
            len(batch),
        )

        request_kwargs: Dict[str, Any] = {
            "model": model_name,
            "input": batch,
        }
        if dimensions is not None:
            request_kwargs["dimensions"] = dimensions

        response = client.embeddings.create(**request_kwargs)
        _log_openai_usage(response, operation="embedding", model_name=model_name)

        response_data = list(response.data)
        if len(response_data) != len(batch):
            raise RuntimeError(
                "OpenAI embedding response size mismatch | "
                f"expected={len(batch)} | received={len(response_data)}"
            )

        indexes = [getattr(item, "index", None) for item in response_data]
        if not all(type(index) is int for index in indexes):
            raise RuntimeError("OpenAI embedding response contained a non-integer index")

        expected_indexes = list(range(len(batch)))
        if sorted(indexes) != expected_indexes:
            raise RuntimeError(
                "OpenAI embedding response indexes were not unique and contiguous | "
                f"expected={expected_indexes} | received={indexes}"
            )

        batch_vectors = [
            _l2_normalize(item.embedding)
            for item in sorted(response_data, key=lambda item: item.index)
        ]

        vectors.extend(batch_vectors)

    return vectors


def embed_texts(
    texts: List[str],
    batch_size: int = 8,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> List[List[float]]:
    if not texts:
        return []

    resolved_provider = resolve_embedding_provider(provider)
    resolved_model_name = get_embedding_model_name(model_name)

    if resolved_provider == OPENAI_PROVIDER:
        return _embed_texts_openai(
            texts=texts,
            batch_size=batch_size,
            model_name=resolved_model_name,
            dimensions=dimensions,
        )

    model = get_embedding_model(
        provider=resolved_provider,
        model_name=resolved_model_name,
    )

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return vectors.tolist()
