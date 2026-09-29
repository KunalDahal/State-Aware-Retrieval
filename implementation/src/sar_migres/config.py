"""Configuration: YAML files -> typed dataclasses.  Secrets are NEVER stored here; the
config only names the *environment variables* that hold them."""
from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

ENV_FILE_DEFAULT = ".env"


def load_dotenv_file(path: str | os.PathLike = ENV_FILE_DEFAULT, override: bool = False) -> list[str]:
    """Minimal .env reader (KEY=VALUE, '#' comments). Returns the *names* loaded (never values)."""
    p = Path(path)
    loaded: list[str] = []
    if not p.exists():
        return loaded
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and (override or k not in os.environ):
            os.environ[k] = v
            loaded.append(k)
    return loaded


@dataclass
class EmbeddingConfig:
    type: str = "sentence_transformer"  # sentence_transformer | hashing
    model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    device: str = ""  # "" = auto
    batch_size: int = 64
    expected_dim: int | None = 768
    hashing_dim: int = 256
    hashing_seed: int = 0
    embed_template: str = "{title}. {text}"


@dataclass
class VectorStoreConfig:
    type: str = "pinecone"  # pinecone | local
    on_cache_corruption: str = "rebuild"  # rebuild | raise
    pinecone_index: str = "sar-migres"
    pinecone_namespace: str = ""
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_api_key_env: str = "PINECONE_API_KEY"
    pinecone_create_if_missing: bool = True
    remove_missing: bool = True

    def local_dir_name(self, embedder_name: str) -> str:
        return "emb_" + "".join(c if c.isalnum() or c in "-_." else "_" for c in embedder_name)


@dataclass
class SARConfig:
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    k: int = 5
    weighting: str = "static"  # static | dynamic (dynamic = FUTURE)
    chunk_size: int = 50_000
    # NOTE: default weights/K are DEVELOPMENT PLACEHOLDERS, not tuned or optimal values.
    # The thesis selects alpha/beta/gamma by Bayesian optimisation on validation data (October stage).


@dataclass
class LLMConfig:
    provider: str = "gemini"  # gemini | openai | offline_deterministic
    model: str = ""  # or set env var named by model_env
    model_env: str = "LLM_MODEL"
    api_key_env: str = ""  # "" -> provider default (GEMINI_API_KEY / OPENAI_API_KEY)
    base_url_env: str = "OPENAI_BASE_URL"
    temperature: float = 0.0
    max_retries: int = 5
    timeout_s: float = 120.0
    max_workers: int = 4


@dataclass
class MigresConfig:
    backend: str = "upstream"  # upstream (official AdelWang/MIGRES code) | standin (offline smoke test only)
    repo_dir: str = ""  # or env MIGRES_REPO_DIR
    max_iter: int = 5
    top_k: int = 5  # passages placed in each MIGRES prompt (upstream --top_k)
    relevance: float = 3.0  # upstream reranker threshold delta (--relevance)
    reranker: str = "upstream_cross_encoder"  # upstream_cross_encoder | offline_standin
    ranker_model_path: str = "BAAI/bge-reranker-base"  # upstream default string is 'BAAI/bge-rerank-base' (unverified id)
    sent_rerank: bool = False
    entail_judge: bool = False  # upstream migres.sh enables it (needs google/t5_xxl_true_nli_mixture, ~11B params)
    gpt_knowledge: bool = True
    add_demon: bool = False
    demon_path: str = ""
    num_return: int = 1
    gap_source: str = "queries"  # queries | missing_info | both  (how MIGRES state maps to G_t)
    strict_upstream_hash: bool = False


@dataclass
class DataConfig:
    corpus_path: str = "data/sample/corpus.jsonl"
    corpus_lazy: bool = False
    max_malformed_fraction: float = 0.01
    dataset: str = "sample"  # sample | wikihop | hotpotqa | musique | natural_questions | triviaqa | strategyqa | jsonl
    dataset_path: str = "data/sample/questions.jsonl"
    limit: int | None = None


@dataclass
class RunConfig:
    seed: int = 13
    output_dir: str = "outputs"
    name: str = "sar_migres"
    log_level: str = "INFO"
    resume: bool = False
    snippet_chars: int = 240
    cache_root: str = "data/cache"


@dataclass
class Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    sar: SARConfig = field(default_factory=SARConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    migres: MigresConfig = field(default_factory=MigresConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunConfig = field(default_factory=RunConfig)
    future: dict[str, Any] = field(default_factory=dict)  # FUTURE/October settings, disabled by default

    def snapshot(self) -> dict[str, Any]:
        """Serialisable copy for storing next to run outputs (contains no secret values)."""
        return dataclasses.asdict(self)

    def resolve_llm_model(self) -> str:
        m = self.llm.model or os.environ.get(self.llm.model_env, "")
        if not m:
            raise ConfigError(f"LLM model name not configured: set llm.model in the YAML or export {self.llm.model_env}. "
                              "The model name is intentionally not defaulted (it must be one your provider supports).")
        return m

    def resolve_migres_repo(self) -> Path | None:
        p = self.migres.repo_dir or os.environ.get("MIGRES_REPO_DIR", "")
        return Path(p).expanduser() if p else None


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _build(cls, d: dict, path: str):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ConfigError(f"unknown config key(s) under '{path}': {sorted(unknown)} (valid: {sorted(names)})")
    return cls(**d)


def config_from_dict(d: dict[str, Any]) -> Config:
    sections = {"embedding": EmbeddingConfig, "vector_store": VectorStoreConfig, "sar": SARConfig,
                "llm": LLMConfig, "migres": MigresConfig, "data": DataConfig, "run": RunConfig}
    unknown = set(d) - set(sections) - {"future"}
    if unknown:
        raise ConfigError(f"unknown top-level config section(s): {sorted(unknown)}")
    kw = {name: _build(cls, d.get(name, {}) or {}, name) for name, cls in sections.items()}
    cfg = Config(**kw, future=d.get("future", {}) or {})
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    from .scoring import SARWeights

    SARWeights(cfg.sar.alpha, cfg.sar.beta, cfg.sar.gamma)  # raises ConfigError on violation
    if cfg.sar.k < 1:
        raise ConfigError("sar.k must be >= 1")
    if cfg.embedding.type not in ("sentence_transformer", "hashing"):
        raise ConfigError(f"embedding.type must be sentence_transformer|hashing, got {cfg.embedding.type!r}")
    if cfg.vector_store.type not in ("pinecone", "local"):
        raise ConfigError("vector_store.type must be pinecone|local")
    if cfg.migres.backend not in ("upstream", "standin"):
        raise ConfigError("migres.backend must be upstream|standin")
    if cfg.migres.gap_source not in ("queries", "missing_info", "both"):
        raise ConfigError("migres.gap_source must be queries|missing_info|both")
    if cfg.sar.weighting not in ("static", "dynamic"):
        raise ConfigError("sar.weighting must be static|dynamic")


def load_config(*paths: str | os.PathLike, overrides: dict[str, Any] | None = None) -> Config:
    """Load and deep-merge YAML files in order (later wins), then apply ``overrides``."""
    merged: dict[str, Any] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
        merged = _deep_merge(merged, data)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return config_from_dict(merged)


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """``["sar.k=10", "llm.provider=openai"]`` -> nested dict (values parsed as YAML scalars)."""
    out: dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p or "." not in p.split("=", 1)[0]:
            raise ConfigError(f"bad override {p!r}; expected section.key=value")
        key, val = p.split("=", 1)
        section, name = key.split(".", 1)
        out.setdefault(section, {})[name] = yaml.safe_load(val)
    return out
