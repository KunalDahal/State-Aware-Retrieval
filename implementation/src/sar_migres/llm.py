"""LLM provider abstraction (no credentials in code; keys come from environment variables).

Real providers : ``OpenAIClient`` (OPENAI_API_KEY [+ OPENAI_BASE_URL]) and ``GeminiClient`` (GEMINI_API_KEY).
Offline        : ``DeterministicOfflineLLM`` - a rule-based *fake* that emits correctly formatted
                 MIGRES-style replies so the whole loop can run without any API. Its answers are
                 meaningless and runs using it are labelled ``llm_provider=offline_deterministic``.

The model name is never defaulted: it must come from ``llm.model`` or the env var ``LLM_MODEL``.
The provider SDK calls below follow the public SDK documentation but have NOT been exercised
against the live services in this repository's CI (no credentials there).
"""
from __future__ import annotations

import ast
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Callable

from .errors import (ConfigError, DependencyMissingError, LLMError, LLMResponseError, MissingCredentialError,
                     retry_call)
from .logging_utils import get_logger

log = get_logger("llm")

_DEFAULT_KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


class LLMClient(ABC):
    provider: str = "abstract"
    model: str = ""

    @abstractmethod
    def _generate(self, prompt: str, n: int) -> list[str]: ...

    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Return ``n`` completions. Empty/None completions raise ``LLMResponseError``."""
        out = self._generate(prompt, n)
        if not out or any((not isinstance(o, str)) or not o.strip() for o in out):
            raise LLMResponseError(f"{self.provider}/{self.model}: empty or non-text response")
        return out


def _need_key(env_name: str) -> str:
    key = os.environ.get(env_name, "")
    if not key:
        raise MissingCredentialError(f"environment variable {env_name} is not set; copy .env.example to .env "
                                     "and fill it in (or use llm.provider=offline_deterministic for the smoke test)")
    return key


class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY", base_url_env: str = "OPENAI_BASE_URL",
                 temperature: float = 0.0, timeout_s: float = 120.0, max_retries: int = 5, client=None):
        self.model, self.temperature, self.max_retries = model, temperature, max_retries
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise DependencyMissingError("openai SDK not installed: pip install 'sar-migres[openai]'") from e
            kw = {"api_key": _need_key(api_key_env), "timeout": timeout_s, "max_retries": 0}
            if os.environ.get(base_url_env):
                kw["base_url"] = os.environ[base_url_env]
            client = OpenAI(**kw)
        self._client = client

    def _generate(self, prompt: str, n: int) -> list[str]:
        def call():
            r = self._client.chat.completions.create(
                model=self.model, n=n, temperature=self.temperature,
                messages=[{"role": "system", "content": "You are ChatGPT, a model trained by OpenAI."},  # as in upstream MIGRES
                          {"role": "user", "content": prompt}])
            return [c.message.content for c in r.choices]
        return retry_call(call, attempts=self.max_retries, what=f"openai:{self.model}")


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, model: str, api_key_env: str = "GEMINI_API_KEY", temperature: float = 0.0,
                 timeout_s: float = 120.0, max_retries: int = 5, client=None):
        self.model, self.temperature, self.max_retries, self.timeout_s = model, temperature, max_retries, timeout_s
        if client is None:
            try:
                from google import genai
            except ImportError as e:  # pragma: no cover
                raise DependencyMissingError("google-genai SDK not installed: pip install 'sar-migres[gemini]'") from e
            client = genai.Client(api_key=_need_key(api_key_env))
        self._client = client

    def _one(self, prompt: str) -> str:
        from google.genai import types  # local import: only needed for real calls

        cfg = types.GenerateContentConfig(temperature=self.temperature,
                                          http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)))
        r = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        return r.text or ""

    def _generate(self, prompt: str, n: int) -> list[str]:
        return [retry_call(lambda: self._one(prompt), attempts=self.max_retries, what=f"gemini:{self.model}")
                for _ in range(n)]


class ScriptedLLM(LLMClient):
    """Test double: ``handler(prompt) -> str``; records every prompt."""

    provider = "scripted"
    model = "scripted"

    def __init__(self, handler: Callable[[str], str]):
        self.handler, self.prompts = handler, []

    def _generate(self, prompt, n):
        self.prompts.append(prompt)
        return [self.handler(prompt) for _ in range(n)]


# ------------------------------------------------------------------------------------------
_STOP = set("""a an the of in on at to for from by with and or is are was were be been being who what which when where why how
does do did has have had that this these those it its as into than then their there his her he she they them not no yes""".split())
_W = re.compile(r"[A-Za-z0-9']+")


def _content_tokens(text: str) -> list[str]:
    return [t for t in (w.lower() for w in _W.findall(text)) if len(t) > 2 and t not in _STOP]


def _first_sentence(text: str) -> str:
    text = re.sub(r"^\(Title:[^)]*\)\s*", "", text.strip())
    m = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return m[0].strip()


class DeterministicOfflineLLM(LLMClient):
    """Rule-based stand-in that recognises MIGRES prompt kinds and replies in MIGRES' expected formats.

    NOT a language model: it copies the first sentence of passages that share words with the
    question and 'answers' once most question words are covered.  Only for offline smoke tests.
    """

    provider = "offline_deterministic"
    model = "offline-rule-based-v1"

    def __init__(self, answer_coverage: float = 0.8, max_info_items: int = 3):
        self.answer_coverage, self.max_info_items = answer_coverage, max_info_items

    def _generate(self, prompt: str, n: int) -> list[str]:
        return [self._reply(prompt)] * n

    @staticmethod
    def _last(pattern: str, prompt: str) -> str:
        m = re.findall(pattern, prompt, flags=re.S)
        return m[-1].strip() if m else ""

    def _reply(self, p: str) -> str:
        head = p.lstrip()[:80]
        if head.startswith("Generate relevant information to the given Question"):
            return "unknown"  # <5 words -> MIGRES drops it; no parametric knowledge offline
        if "distillate useful information from the Passages" in p:
            q = self._last(r"Question:\s*(.*?)\nYour response:", p)
            qt = set(_content_tokens(q))
            infos = []
            for m in re.finditer(r"^Passage (\d+): (.*)$", p, flags=re.M):
                idx, text = int(m.group(1)), m.group(2)
                if qt & set(_content_tokens(text)):
                    infos.append({"info": _first_sentence(text), "support_passages": [idx]})
            return json.dumps({"useful_information": infos[: self.max_info_items]})
        if head.startswith("Based on the Original Question, Historical Questions"):
            orig = self._last(r"Original Question:\s*(.*?)\nHistorical Questions:", p)
            hist = self._last(r"Historical Questions:\s*(.*?)\nKnown Information:", p)
            known = self._last(r"Known Information:\s*(.*?)\nMissing Information:", p)
            missing = self._last(r"Missing Information:\s*(.*?)\nNew queries:", p)
            uncovered = [t for t in dict.fromkeys(_content_tokens(orig)) if t not in known.lower()]
            terms = uncovered or _content_tokens(missing) or _content_tokens(orig)
            qs = []
            for i in range(0, min(len(terms), 4), 2):
                cand = f"What is the information about {' and '.join(terms[i:i + 2])}?"
                if cand not in hist and cand != orig and cand not in qs:
                    qs.append(cand)
            return "\n".join(qs) if qs else f"What else is known about {' '.join(terms[:2])} in this context?"
        if head.startswith("Answer the Question based solely on the provided Information") or \
                head.startswith("Answer the Question based on the provided"):
            q = self._last(r"Question:\s*(.*?)\nYour response:", p) or self._last(r"Question:\s*(.*)$", p)
            info = self._last(r"Information:\s*(.*?)\nQuestion:", p)
            if not info:  # first main call is fed passages instead of information
                info = " ".join(m.group(2) for m in re.finditer(r"^Passage (\d+): (.*)$", p, flags=re.M))
            qt = list(dict.fromkeys(_content_tokens(q)))
            cov = [t for t in qt if t in info.lower()]
            last_call = "own knowledge" in head or "If the Information is not sufficient to answer the question" in p[:600]
            ok = bool(info) and info != "None" and (len(cov) / max(len(qt), 1) >= self.answer_coverage)
            if ok or last_call:
                text = _first_sentence(info.split(";")[0]) if info and info != "None" else "unknown"
                return json.dumps({"answer": {"text": text[:80], "confidence": 1},
                                   "explanation": "offline deterministic stand-in; not a real model answer"})
            missing = "Information about: " + ", ".join(t for t in qt if t not in cov)
            return json.dumps({"answer": {"text": "unanswerable", "confidence": 1},
                               "explanation": missing, "missing_information": missing})
        raise LLMResponseError(f"offline LLM does not recognise this prompt: {p[:80]!r}")


def parse_dict_reply(text: str) -> dict:
    """Best-effort structured parse of a dict-like LLM reply using ``literal_eval`` (never ``eval``)."""
    s = text.strip()
    s = re.sub(r"^```(?:json|python)?\s*|\s*```$", "", s)
    for cand in (s, s[s.find("{"): s.rfind("}") + 1] if "{" in s else ""):
        if not cand:
            continue
        try:
            v = ast.literal_eval(cand)
            if isinstance(v, dict):
                return v
        except (ValueError, SyntaxError):
            pass
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    raise LLMResponseError(f"malformed LLM response (not a dict): {text[:120]!r}")


def build_llm(cfg) -> LLMClient:
    """cfg: Config (top-level)."""
    l = cfg.llm
    if l.provider == "offline_deterministic":
        return DeterministicOfflineLLM()
    model = cfg.resolve_llm_model()
    key_env = l.api_key_env or _DEFAULT_KEY_ENV.get(l.provider, "")
    if l.provider == "openai":
        return OpenAIClient(model, key_env, l.base_url_env, l.temperature, l.timeout_s, l.max_retries)
    if l.provider == "gemini":
        return GeminiClient(model, key_env, l.temperature, l.timeout_s, l.max_retries)
    raise ConfigError(f"unknown llm.provider {l.provider!r}")
