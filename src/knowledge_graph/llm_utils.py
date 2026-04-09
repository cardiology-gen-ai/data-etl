import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)

load_dotenv()


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


def get_chat_model_name() -> str:
    return _get_optional_env("KG_CHAT_MODEL_PATH") or _get_required_env("KG_CHAT_MODEL")


def get_embedding_model_name() -> str:
    return _get_optional_env("KG_EMBEDDING_MODEL_PATH") or _get_required_env("KG_EMBEDDING_MODEL")


@lru_cache(maxsize=1)
def get_chat_tokenizer_and_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    model_ref = get_chat_model_name()
    hf_token = _get_optional_env("HF_TOKEN")
    local_files_only = _get_env_bool("KG_LOCAL_FILES_ONLY", True)

    logger.info(
        "Loading chat model | model=%s | local_files_only=%s",
        model_ref,
        local_files_only,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        token=hf_token,
        local_files_only=local_files_only,
    )

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
    model_ref = get_embedding_model_name()
    hf_token = _get_optional_env("HF_TOKEN")
    local_files_only = _get_env_bool("KG_LOCAL_FILES_ONLY", True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Loading embedding model | model=%s | local_files_only=%s | device=%s",
        model_ref,
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

    try:
        model_device = model.device
    except Exception:
        model_device = next(model.parameters()).device

    input_ids = input_ids.to(model_device)

    effective_max_new_tokens = max_new_tokens or int(
        _get_optional_env("KG_CHAT_MAX_NEW_TOKENS") or "512"
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=effective_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
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