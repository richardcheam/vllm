#!/usr/bin/env python3
"""Helpers for building exact-length pre-tokenized benchmark prompts."""

from __future__ import annotations

import os
from typing import Any


def load_tokenizer(model: str) -> Any:
    from vllm.tokenizers import get_tokenizer

    tokenizer_path = (
        os.environ.get("TOKENIZER_PATH")
        or os.environ.get("MODEL_PATH")
        or model
    )
    return get_tokenizer(tokenizer_path, trust_remote_code=True)


def tokenize_to_length(
    text: str,
    prompt_len: int,
    tokenizer: Any,
    filler: str,
) -> list[int]:
    """Tokenize text and pad or truncate it to exactly prompt_len tokens."""
    if prompt_len < 1:
        raise ValueError("prompt_len must be positive")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    filler_ids = tokenizer.encode(" " + filler, add_special_tokens=False)
    if not filler_ids:
        raise ValueError("filler must produce at least one token")
    while len(token_ids) < prompt_len:
        token_ids.extend(filler_ids)
    return token_ids[:prompt_len]
