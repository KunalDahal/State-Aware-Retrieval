from __future__ import annotations

import re

def _clean_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def chunk_text(text: str, *, max_tokens: int = 350, overlap_tokens: int = 50, tokenizer=None, allow_whitespace_fallback: bool = False) -> list[str]:
    """Chunk text with the specified tokenizer; whitespace fallback is allowed only when explicitly requested."""
    if not text.strip():
        return []
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("require max_tokens > 0 and 0 <= overlap_tokens < max_tokens")
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            step = max_tokens - overlap_tokens
            out = []
            for s in range(0, len(ids), step):
                piece = tokenizer.decode(ids[s:s + max_tokens], skip_special_tokens=True).strip()
                if piece:
                    out.append(piece)
                if s + max_tokens >= len(ids):
                    break
            return out
        except Exception as exc:
            if not allow_whitespace_fallback:
                raise RuntimeError("configured tokenizer failed during scientific corpus construction") from exc
    if tokenizer is None and not allow_whitespace_fallback:
        raise RuntimeError("scientific corpus construction requires the specified tokenizer; select explicit offline mode for whitespace tokenization")
    words = text.split()
    step = max_tokens - overlap_tokens
    return [" ".join(words[s:s + max_tokens]) for s in range(0, len(words), step) if words[s:s + max_tokens]]


def _load_tokenizer(name: str | None, *, allow_whitespace_fallback: bool = False):
    if not name:
        if allow_whitespace_fallback:
            return None, "whitespace_fallback"
        raise RuntimeError("scientific corpus construction requires a tokenizer name")
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name, use_fast=True), name
    except Exception as exc:
        if allow_whitespace_fallback:
            return None, "whitespace_fallback"
        raise RuntimeError(f"cannot load configured tokenizer {name!r}") from exc
