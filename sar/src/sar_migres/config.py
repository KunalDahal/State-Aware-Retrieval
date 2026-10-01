"""Typed configuration for the SAR-MIGRES experiments.

The configuration deliberately separates SAR components from baseline retrievers.  SAR owns
its Sentence Transformer + state-aware scoring, while baseline methods are drop-in retrievers
that are fed into the same MIGRES iteration loop.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

ENV_FILE_DEFAULT = ".env"


def load_dotenv_file(path: str | os.PathLike = ENV_FILE_DEFAULT, override: bool = False) -> list[str]:
    p = Path(path)
    loaded: list[str] = []
    if not p.exists():
        return loaded
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if v[:1] in ('"', "'"):  # quoted value: take what is inside the quotes, ignore anything after
            end = v.find(v[0], 1)
            v = v[1:end] if end != -1 else v[1:]
        else:  # unquoted: drop a trailing "  # comment"
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
        if v and (override or k not in os.environ):
            os.environ[k] = v
            loaded.append(k)
    return loaded


@dataclass
class EmbeddingConfig:
    type: str = "sentence_transformer"  # sentence_transformer | hashing
    model_name: str = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    device: str = ""
    batch_size: int = 64
    expected_dim: int | None = 768
    hashing_dim: int = 256
    hashing_seed: int = 0
    embed_template: str = "{title}. {text}"


@dataclass
class VectorStoreConfig:
    type: str = "local_memmap"  # local_memmap is the primary disk-backed exact-scoring store
    on_cache_corruption: str = "rebuild"
    remove_missing: bool = True

    def local_dir_name(self, embedder_name: str) -> str:
        return "emb_" + "".join(c if c.isalnum() or c in "-_." else "_" for c in embedder_name)


@dataclass
class SARConfig:
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    k: int = 5
    weighting: str = "static"  # static | dynamic
    dynamic_rule: str = ""  # methodology does not define a rule; must be explicit to enable dynamic mode
    chunk_size: int = 50_000


@dataclass
class RetrievalConfig:
    """Which retriever is plugged into MIGRES.

    ``sar`` is the proposed method.  The other values are baseline retrievers from Table 5.6.
    Only ``sentence_transformer_search`` needs the project Sentence Transformer embedder/store.
    """

    method: str = "sar"  # sar | bm25 | mmr | sentence_transformer_search | infogain_rag | dpr | contriever | splade | colbertv2


@dataclass
class BaselineConfig:
    k: int = 5
    cache_dir: str = "data/cache/baselines"
    dpr_question_model: str = "facebook/dpr-question_encoder-single-nq-base"
    dpr_context_model: str = "facebook/dpr-ctx_encoder-single-nq-base"
    contriever_model: str = "facebook/contriever"
    splade_model: str = "naver/splade-cocondenser-ensembledistil"
    colbert_model: str = "colbert-ir/colbertv2.0"
    batch_size: int = 16
    max_length: int = 384
    colbert_query_max_tokens: int = 32
    colbert_doc_max_tokens: int = 180
    score_chunk_size: int = 50_000
    mmr_lambda: float = 0.6
    mmr_candidate_multiplier: int = 10
    infogain_checkpoint: str = ""
    infogain_candidate_k: int = 50


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str = ""
    model_env: str = "LLM_MODEL"
    api_key_env: str = ""
    base_url_env: str = "OPENAI_BASE_URL"
    temperature: float = 0.0
    max_retries: int = 5
    timeout_s: float = 120.0
    max_workers: int = 4


@dataclass
class MigresConfig:
    backend: str = "upstream"
    repo_dir: str = ""
    max_iter: int = 5
    top_k: int = 5
    relevance: float = 3.0
    reranker: str = "upstream_cross_encoder"
    ranker_model_path: str = "BAAI/bge-reranker-base"
    sent_rerank: bool = False
    entail_judge: bool = False
    gpt_knowledge: bool = True
    add_demon: bool = False
    demon_path: str = ""
    num_return: int = 1
    gap_source: str = "queries"
    strict_upstream_hash: bool = False


@dataclass
class DataConfig:
    corpus_path: str = "data/processed/pooled_train_corpus.jsonl"
    corpus_lazy: bool = True
    max_malformed_fraction: float = 0.01
    dataset: str = "hotpotqa"
    dataset_path: str = "data/raw/hotpot.json"
    dataset_paths: dict[str, str] = field(default_factory=lambda: {
        "wikihop": "data/raw/wikihop.json",
        "hotpotqa": "data/raw/hotpot.json",
        "musique": "data/raw/musique.json",
        "natural_questions": "data/raw/nq.json",
        "triviaqa": "data/raw/triviaqa.json",
        "strategyqa": "data/raw/strategyqa.json",
    })
    datasets: list[str] = field(default_factory=lambda: ["wikihop", "hotpotqa", "musique", "natural_questions", "triviaqa", "strategyqa"])
    limit: int | None = None
    split: str = "dev"
    final_split: str = "dev"
    validation_fraction: float = 0.10
    validation_seed: int = 13
    corpus_source_split: str = "train"
    corpus_chunk_max_tokens: int = 350
    corpus_chunk_overlap_tokens: int = 50
    max_documents: int | None = 100_000


@dataclass
class RunConfig:
    seed: int = 13
    statistics_seed: int = 13
    bootstrap_resamples: int = 2000
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
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    migres: MigresConfig = field(default_factory=MigresConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run: RunConfig = field(default_factory=RunConfig)
    future: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def resolve_llm_model(self) -> str:
        m = self.llm.model or os.environ.get(self.llm.model_env, "")
        if not m:
            raise ConfigError(
                f"LLM model name not configured: set llm.model in the YAML or export {self.llm.model_env}. "
                "The model id is intentionally not guessed."
            )
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
    sections = {
        "embedding": EmbeddingConfig,
        "vector_store": VectorStoreConfig,
        "sar": SARConfig,
        "retrieval": RetrievalConfig,
        "baseline": BaselineConfig,
        "llm": LLMConfig,
        "migres": MigresConfig,
        "data": DataConfig,
        "run": RunConfig,
    }
    unknown = set(d) - set(sections) - {"future"}
    if unknown:
        raise ConfigError(f"unknown top-level config section(s): {sorted(unknown)}")
    kw = {name: _build(cls, d.get(name, {}) or {}, name) for name, cls in sections.items()}
    cfg = Config(**kw, future=d.get("future", {}) or {})
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    from .scoring import SARWeights

    SARWeights(cfg.sar.alpha, cfg.sar.beta, cfg.sar.gamma)
    if cfg.sar.k < 1 or cfg.baseline.k < 1:
        raise ConfigError("retrieval K values must be >= 1")
    if cfg.sar.chunk_size < 1 or cfg.baseline.score_chunk_size < 1:
        raise ConfigError("chunk sizes must be >= 1")
    if cfg.embedding.type not in ("sentence_transformer", "hashing"):
        raise ConfigError("embedding.type must be sentence_transformer|hashing")
    if cfg.vector_store.type not in ("local", "local_memmap"):
        raise ConfigError("vector_store.type must be local|local_memmap")
    if cfg.retrieval.method not in {"sar", "bm25", "mmr", "sentence_transformer_search", "infogain_rag", "dpr", "contriever", "splade", "colbertv2"}:
        raise ConfigError(f"unsupported retrieval.method={cfg.retrieval.method!r}")
    if cfg.migres.backend not in ("upstream", "standin"):
        raise ConfigError("migres.backend must be upstream|standin")
    if cfg.migres.gap_source not in ("queries", "missing_info", "both"):
        raise ConfigError("migres.gap_source must be queries|missing_info|both")
    if cfg.sar.weighting not in ("static", "dynamic"):
        raise ConfigError("sar.weighting must be static|dynamic")
    if cfg.sar.weighting == "dynamic" and not cfg.sar.dynamic_rule.strip():
        raise ConfigError(
            "sar.weighting=dynamic requires sar.dynamic_rule. The methodology states that weights may change "
            "with retrieval state but does not define a rule; no implicit rule is used."
        )
    if cfg.retrieval.method != "sar" and cfg.sar.weighting == "dynamic":
        raise ConfigError("sar.weighting applies only when retrieval.method=sar")
    if cfg.data.max_documents is not None and cfg.data.max_documents < 1:
        raise ConfigError("data.max_documents must be >= 1 when provided")
    if not (0.0 < cfg.data.validation_fraction < 1.0):
        raise ConfigError("data.validation_fraction must be in (0,1)")
    if cfg.data.corpus_chunk_max_tokens < 1 or cfg.data.corpus_chunk_overlap_tokens < 0:
        raise ConfigError("corpus chunking values must be positive/non-negative")
    if cfg.data.corpus_chunk_overlap_tokens >= cfg.data.corpus_chunk_max_tokens:
        raise ConfigError("corpus_chunk_overlap_tokens must be smaller than corpus_chunk_max_tokens")
    if cfg.baseline.mmr_candidate_multiplier < 1 or cfg.baseline.infogain_candidate_k < 1:
        raise ConfigError("baseline candidate settings must be >= 1")
    if not (0.0 <= cfg.baseline.mmr_lambda <= 1.0):
        raise ConfigError("baseline.mmr_lambda must be in [0,1]")


def load_config(*paths: str | os.PathLike, overrides: dict[str, Any] | None = None) -> Config:
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
    out: dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p or "." not in p.split("=", 1)[0]:
            raise ConfigError(f"bad override {p!r}; expected section.key=value")
        key, val = p.split("=", 1)
        section, name = key.split(".", 1)
        out.setdefault(section, {})[name] = yaml.safe_load(val)
    return out
