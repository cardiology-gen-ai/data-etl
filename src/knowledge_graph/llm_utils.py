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


def get_chat_model_ref() -> str:
    """
    Return the model reference used to load the chat model.

    Loading can still come from a local path if KG_CHAT_MODEL_PATH is set,
    but model metadata elsewhere in the repo should use KG_CHAT_MODEL,
    not the filesystem path.
    """
    return _get_optional_env("KG_CHAT_MODEL_PATH") or _get_required_env("KG_CHAT_MODEL")


def get_embedding_model_ref() -> str:
    """
    Return the model reference used to load the embedding model.

    Loading can still come from a local path if KG_EMBEDDING_MODEL_PATH is set,
    but model metadata elsewhere in the repo should use KG_EMBEDDING_MODEL,
    not the filesystem path.
    """
    return _get_optional_env("KG_EMBEDDING_MODEL_PATH") or _get_required_env("KG_EMBEDDING_MODEL")


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

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        token=hf_token,
        local_files_only=local_files_only,
        dtype="auto",
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    model_ref = get_embedding_model_ref()
    hf_token = _get_optional_env("HF_TOKEN")
    local_files_only = _get_env_bool("KG_LOCAL_FILES_ONLY", True)
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


def generate_chat_text(
    messages: List[Dict[str, str]],
    json_mode: bool = False,
    max_new_tokens: Optional[int] = None,
) -> Optional[str]:
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
            content = (last.get("content") or "").rstrip()
            if "Return JSON only." not in content:
                content += "\n\nReturn JSON only."
            last["content"] = content

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

    effective_max_new_tokens = max_new_tokens or int(
        _get_optional_env("KG_CHAT_MAX_NEW_TOKENS") or "512"
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
) -> Optional[List[List[float]]]:
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