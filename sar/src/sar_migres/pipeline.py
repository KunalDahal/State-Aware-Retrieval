"""End-to-end orchestration: corpus -> embeddings -> vector store -> SAR retriever -> MIGRES loop -> outputs."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .config import Config
from .corpus import Corpus
from .datasets import load_questions
from .embeddings import Embedder, build_embedder
from .errors import ConfigError
from .llm import LLMClient, build_llm
from .logging_utils import get_logger, setup_logging
from .migres_adapter import MigresBackend, SARSearchProvider, build_backend, question_id_of
from .outputs import RunWriter
from .retrieval import DynamicWeights, StateAwareRetriever, StaticWeights
from .schemas import QuestionResult
from .scoring import SARWeights
from .utils import make_run_id, peak_rss_mb, set_seed, utc_now_iso
from .vector_store import SyncReport, VectorStore, build_vector_store, sync_corpus_embeddings

log = get_logger("pipeline")


def _pkg_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "sar_migres": __version__}
    for name in ("numpy", "torch", "sentence_transformers", "pinecone", "openai", "transformers", "bayes_opt"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - optional dependency absent
            pass
    try:
        from google import genai  # noqa: F401
        import importlib.metadata as md
        out["google-genai"] = md.version("google-genai")
    except Exception:  # noqa: BLE001
        pass
    return out


class SARMigresPipeline:
    def __init__(self, cfg: Config, *, run_id: str | None = None, embedder: Embedder | None = None,
                 store: VectorStore | None = None, corpus: Corpus | None = None, llm: LLMClient | None = None,
                 backend: MigresBackend | None = None, run_dir: str | Path | None = None):
        if cfg.sar.weighting == "dynamic":
            raise ConfigError("sar.weighting=dynamic is a FUTURE (October) experiment and is not implemented")
        self.cfg = cfg
        self.run_id = run_id or make_run_id(cfg.run.name)
        self.run_dir = Path(run_dir) if run_dir else Path(cfg.run.output_dir) / "runs" / self.run_id
        self.writer = RunWriter(self.run_dir)
        setup_logging(cfg.run.log_level, self.run_dir / "logs" / "run.log")
        set_seed(cfg.run.seed)
        self.corpus, self.embedder, self.store, self.llm, self.backend = corpus, embedder, store, llm, backend
        self.retriever: StateAwareRetriever | None = None
        self.sync_report: SyncReport | None = None
        self.warnings: list[str] = []
        self.started = utc_now_iso()

    # ---- setup ----------------------------------------------------------------------------------------
    def prepare(self, rebuild: bool = False) -> "SARMigresPipeline":
        cfg = self.cfg
        if self.corpus is None:
            self.corpus = Corpus.from_file(cfg.data.corpus_path, embed_template=cfg.embedding.embed_template,
                                           lazy=cfg.data.corpus_lazy, max_malformed_fraction=cfg.data.max_malformed_fraction)
        if self.embedder is None:
            self.embedder = build_embedder(cfg.embedding)
        if not self.embedder.is_semantic:
            self.warnings.append(f"embedder {self.embedder.name} is a LEXICAL HASHING stand-in for smoke tests, not a semantic model")
        if self.store is None:
            self.store = build_vector_store(cfg.vector_store, self.embedder, cfg.run.cache_root, rebuild=rebuild)
        self.sync_report = sync_corpus_embeddings(self.store, self.corpus, self.embedder, cfg.embedding.batch_size,
                                                  remove_missing=cfg.vector_store.remove_missing)
        w = SARWeights(cfg.sar.alpha, cfg.sar.beta, cfg.sar.gamma)
        self.retriever = StateAwareRetriever(self.store, self.corpus, self.embedder, StaticWeights(w), k=cfg.sar.k,
                                             chunk_size=cfg.sar.chunk_size, snippet_chars=cfg.run.snippet_chars)
        if self.llm is None:
            self.llm = build_llm(cfg)
        if cfg.llm.provider == "offline_deterministic":
            self.warnings.append("LLM is the OFFLINE DETERMINISTIC rule-based fake: answers are not meaningful")
        if self.backend is None:
            self.backend = build_backend(cfg)
        if self.backend.name == "standin":
            self.warnings.append("MIGRES backend is the offline STAND-IN loop, not the official MIGRES code")
        if getattr(self.backend, "reranker_label", "") == "lexical_offline_standin":
            self.warnings.append("MIGRES' cross-encoder filter is replaced by a lexical offline stand-in")
        self.warnings.append(f"SAR weights alpha/beta/gamma={w.as_tuple()} and K={cfg.sar.k} are configured values, not optimised ones")
        return self

    def metadata(self, status: str, finished: str | None = None) -> dict[str, Any]:
        cfg = self.cfg
        return {
            "run_id": self.run_id, "status": status, "started": self.started, "finished": finished,
            "seed": cfg.run.seed, "sar": {"alpha": cfg.sar.alpha, "beta": cfg.sar.beta, "gamma": cfg.sar.gamma, "k": cfg.sar.k,
                                          "weighting": cfg.sar.weighting},
            "embedder": getattr(self.embedder, "name", None), "embedder_is_semantic": getattr(self.embedder, "is_semantic", None),
            "vector_store": cfg.vector_store.type,
            "llm_provider": getattr(self.llm, "provider", cfg.llm.provider), "llm_model": getattr(self.llm, "model", None),
            "migres": self.backend.describe() if self.backend else {},
            "corpus": {"path": cfg.data.corpus_path, "n_docs": len(self.corpus) if self.corpus else None,
                       "version": self.corpus.version if self.corpus else None,
                       "load_report": self.corpus.report.to_dict() if self.corpus else None},
            "dataset": {"name": cfg.data.dataset, "path": cfg.data.dataset_path},
            "embedding_sync": self.sync_report.to_dict() if self.sync_report else None,
            "versions": _pkg_versions(), "argv": sys.argv[:1], "peak_rss_mb": peak_rss_mb(),
            "warnings": self.warnings}

    # ---- execution ----------------------------------------------------------------------------------------
    def run_question(self, item: dict[str, Any]) -> QuestionResult:
        if self.retriever is None:
            raise RuntimeError("call prepare() first")
        qid = question_id_of(item)
        provider = SARSearchProvider(self.retriever, self.run_id, lambda _q, _id=qid: _id,
                                     gap_source=self.cfg.migres.gap_source, snippet_chars=self.cfg.run.snippet_chars)
        res = self.backend.run_question(item, provider, self.llm, self.run_id, self.run_dir / "upstream_state")
        self.writer.append_result(res)
        return res

    def run_single(self, item: dict[str, Any]) -> QuestionResult:
        """One question with the same saved artefacts as a batch (config snapshot, JSONL, CSV, report)."""
        self.writer.write_json("config_snapshot.json", self.cfg.snapshot())
        self.writer.write_json("metadata.json", self.metadata("running"))
        status = "completed"
        try:
            return self.run_question(item)
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except BaseException:
            status = "failed"
            raise
        finally:
            self.finalize(status)

    def run_batch(self, items: Iterable[dict[str, Any]], resume: bool | None = None) -> list[QuestionResult]:
        items = list(items)
        resume = self.cfg.run.resume if resume is None else resume
        self.writer.write_json("config_snapshot.json", self.cfg.snapshot())
        self.writer.write_json("metadata.json", self.metadata("running"))
        done = self.writer.finished_ids() if resume else set()
        if done:
            log.info("resuming: %d questions already finished", len(done))
        results: list[QuestionResult] = []
        status = "completed"
        try:
            for i, it in enumerate(items, 1):
                if question_id_of(it) in done:
                    continue
                r = self.run_question(it)
                results.append(r)
                log.info("[%d/%d] %s -> %s (%s, %.2fs)", i, len(items), r.question_id, r.final_answer, r.status, r.elapsed_seconds)
        except KeyboardInterrupt:
            status = "interrupted"
            log.error("run interrupted; finished results are saved and the run can be resumed with resume=True")
            raise
        except BaseException:
            status = "failed"
            raise
        finally:
            self.finalize(status)
        return results

    def finalize(self, status: str) -> dict[str, Path]:
        all_results = self.writer.load_results()
        meta = self.metadata(status, utc_now_iso())
        meta["n_questions"] = len(all_results)
        meta["n_failed"] = sum(r["status"] == "failed" for r in all_results)
        self.writer.write_json("metadata.json", meta)
        paths = self.writer.write_tables(all_results, Path(self.cfg.run.output_dir) / "tables")
        paths["report.md"] = self.writer.write_report(meta, all_results)
        return paths

    def run_dataset(self, limit: int | None = None) -> list[QuestionResult]:
        items = load_questions(self.cfg.data.dataset, self.cfg.data.dataset_path, limit or self.cfg.data.limit)
        return self.run_batch(items)
