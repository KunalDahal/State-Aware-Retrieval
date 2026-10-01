"""End-to-end orchestration with a single MIGRES loop and swappable retriever.

Retrieval method selection happens before MIGRES starts.  SAR builds its own
Sentence-Transformer -> state-aware algorithm -> score stack; baselines replace that whole
retrieval component and do not construct the SAR embedding stack.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

from . import __version__
from .baselines import build_baseline
from .config import Config
from .corpus import Corpus
from .datasets import load_questions
from .embeddings import Embedder, build_embedder
from .errors import ConfigError
from .llm import LLMClient, build_llm
from .logging_utils import get_logger, setup_logging
from .migres_adapter import MigresBackend, RetrieverSearchProvider, build_backend, question_id_of
from .outputs import RunWriter
from .retrieval import DynamicWeights, StateAwareRetriever, StaticWeights
from .schemas import QuestionResult
from .scoring import SARWeights
from .utils import make_run_id, peak_rss_mb, set_seed, utc_now_iso
from .vector_store import SyncReport, VectorStore, build_vector_store, sync_corpus_embeddings

log = get_logger("pipeline")


def _pkg_versions() -> dict[str, str]:
    names = ("numpy", "torch", "sentence-transformers", "openai", "transformers",
             "bayesian-optimization", "google-genai", "jupyter", "nbformat")
    out = {"python": platform.python_version(), "sar_migres": __version__}
    for name in names:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out


class SARMigresPipeline:
    def __init__(self, cfg: Config, *, run_id: str | None = None, embedder: Embedder | None = None,
                 store: VectorStore | None = None, corpus: Corpus | None = None, llm: LLMClient | None = None,
                 backend: MigresBackend | None = None, run_dir: str | Path | None = None):
        self.cfg = cfg
        self.run_id = run_id or make_run_id(cfg.run.name)
        self.run_dir = Path(run_dir) if run_dir else Path(cfg.run.output_dir) / "runs" / self.run_id
        self.writer = RunWriter(self.run_dir)
        setup_logging(cfg.run.log_level, self.run_dir / "logs" / "run.log")
        set_seed(cfg.run.seed)
        self.corpus, self.embedder, self.store, self.llm, self.backend = corpus, embedder, store, llm, backend
        self.retriever = None
        self.sync_report: SyncReport | None = None
        self.sync_history: list[SyncReport] = []
        self.warnings: list[str] = []
        self.started = utc_now_iso()

    @property
    def retrieval_method(self) -> str:
        return self.cfg.retrieval.method

    def _load_corpus(self) -> Corpus:
        c = self.corpus or Corpus.from_file(
            self.cfg.data.corpus_path,
            embed_template=self.cfg.embedding.embed_template,
            lazy=self.cfg.data.corpus_lazy,
            max_malformed_fraction=self.cfg.data.max_malformed_fraction,
            max_documents=self.cfg.data.max_documents,
        )
        return c

    def _sync_sar_corpus(self) -> SyncReport:
        if self.retrieval_method != "sar":
            return SyncReport()
        if self.store is None or self.embedder is None or self.corpus is None:
            raise RuntimeError("SAR corpus sync requested before SAR store/embedder setup")

        # Methodology Sec. 5.2.2 requires a corpus check after every retrieval iteration.
        # ``refresh_if_changed`` performs a cheap size/mtime check and reparses only when the
        # source file actually changed.  The retriever is then rebound to the refreshed corpus.
        refreshed = self.corpus.refresh_if_changed()
        if refreshed is not self.corpus:
            self.corpus = refreshed
            if self.retriever is not None and hasattr(self.retriever, "corpus"):
                self.retriever.corpus = refreshed

        rep = sync_corpus_embeddings(
            self.store, self.corpus, self.embedder, self.cfg.embedding.batch_size,
            remove_missing=self.cfg.vector_store.remove_missing,
        )
        self.sync_history.append(rep)
        self.sync_report = rep
        return rep

    def prepare(self, rebuild: bool = False) -> "SARMigresPipeline":
        cfg = self.cfg
        self.corpus = self._load_corpus()
        if cfg.data.max_documents is not None and len(self.corpus) >= cfg.data.max_documents:
            self.warnings.append(f"corpus reached configured max_documents={cfg.data.max_documents}; this bound is part of the executable pooled-corpus protocol")

        method = self.retrieval_method
        if method == "sar":
            if self.embedder is None:
                self.embedder = build_embedder(cfg.embedding)
            if not self.embedder.is_semantic:
                self.warnings.append("SAR is using a non-semantic hashing embedder; this is smoke-test mode, not the thesis model")
            if self.store is None:
                self.store = build_vector_store(cfg.vector_store, self.embedder, cfg.run.cache_root, rebuild=rebuild)
            self.sync_report = self._sync_sar_corpus()
            w = SARWeights(cfg.sar.alpha, cfg.sar.beta, cfg.sar.gamma)
            schedule = StaticWeights(w) if cfg.sar.weighting == "static" else DynamicWeights(cfg.sar.dynamic_rule, base=w)
            self.retriever = StateAwareRetriever(
                self.store, self.corpus, self.embedder, schedule, k=cfg.sar.k,
                chunk_size=cfg.sar.chunk_size, snippet_chars=cfg.run.snippet_chars,
            )
            self.warnings.append(f"SAR weights={w.as_tuple()} K={cfg.sar.k}; tune/freeze these on validation before test evaluation")
            if cfg.sar.weighting == "dynamic":
                self.warnings.append(
                    f"dynamic SAR rule={cfg.sar.dynamic_rule!r} is an explicit project extension because the supplied methodology names no dynamic rule"
                )
        elif method == "sentence_transformer_search":
            # This one baseline intentionally shares the SAR embedding model, but it does not use SAR state/scoring.
            if self.embedder is None:
                self.embedder = build_embedder(cfg.embedding)
            if self.store is None:
                self.store = build_vector_store(cfg.vector_store, self.embedder, cfg.run.cache_root, rebuild=rebuild)
            self.sync_report = sync_corpus_embeddings(self.store, self.corpus, self.embedder, cfg.embedding.batch_size,
                                                      remove_missing=cfg.vector_store.remove_missing)
            self.retriever = build_baseline(cfg, method, self.corpus, store=self.store, embedder=self.embedder)
            setattr(self.retriever, "corpus", self.corpus)
            self.warnings.append("Sentence Transformer Search is the explicit ST baseline; it bypasses SAR state signals")
        else:
            # Baselines replace the *entire* SAR retrieval component. They do not construct a Sentence Transformer.
            self.retriever = build_baseline(cfg, method, self.corpus)
            self.warnings.append(f"baseline={method}: SAR Sentence Transformer, state algorithm and SAR score are not constructed")

        if self.llm is None:
            self.llm = build_llm(cfg)
        if cfg.llm.provider == "offline_deterministic":
            self.warnings.append("LLM is the offline deterministic test double; outputs have no scientific meaning")
        if self.backend is None:
            self.backend = build_backend(cfg)
        if self.backend.name == "standin":
            self.warnings.append("MIGRES backend is the offline stand-in, not the official MIGRES implementation")
        if getattr(self.backend, "reranker_label", "") == "lexical_offline_standin":
            self.warnings.append("MIGRES cross-encoder filter is replaced by the lexical offline stand-in")
        return self

    @staticmethod
    def _file_sha256(path: str | Path) -> str | None:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        import hashlib
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def metadata(self, status: str, finished: str | None = None) -> dict[str, Any]:
        cfg = self.cfg
        ret_meta = {"method": self.retrieval_method,
                    "name": getattr(self.retriever, "name", None),
                    "score_name": getattr(self.retriever, "score_name", None),
                    "is_sar": bool(getattr(self.retriever, "is_sar", False))}
        if getattr(self.retriever, "stats", None) is not None:
            ret_meta["stats"] = self.retriever.stats.to_dict()
        corpus_manifest = None
        manifest_path = Path(str(cfg.data.corpus_path) + ".manifest.json")
        if manifest_path.exists():
            try:
                corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                corpus_manifest = None
        return {
            "run_id": self.run_id, "status": status, "started": self.started, "finished": finished,
            "seed": cfg.run.seed,
            "sar": {"alpha": cfg.sar.alpha, "beta": cfg.sar.beta, "gamma": cfg.sar.gamma,
                    "k": cfg.sar.k, "t_max": cfg.migres.max_iter, "weighting": cfg.sar.weighting, "dynamic_rule": cfg.sar.dynamic_rule},
            "retrieval": ret_meta,
            "embedder": getattr(self.embedder, "name", None),
            "embedder_is_semantic": getattr(self.embedder, "is_semantic", None),
            "vector_store": cfg.vector_store.type if self.store else None,
            "llm_provider": getattr(self.llm, "provider", cfg.llm.provider),
            "llm_model": getattr(self.llm, "model", None),
            "llm_stats": self.llm.stats.to_dict() if hasattr(self.llm, "stats") else None,
            "migres": self.backend.describe() if self.backend else {},
            "corpus": {"path": cfg.data.corpus_path, "n_docs": len(self.corpus) if self.corpus else None,
                       "version": self.corpus.version if self.corpus else None,
                       "load_report": self.corpus.report.to_dict() if self.corpus else None},
            "dataset": {"name": cfg.data.dataset, "path": cfg.data.dataset_path, "split": cfg.data.split,
                        "datasets_in_study": cfg.data.datasets, "dataset_paths": cfg.data.dataset_paths,
                        "final_split_protocol": cfg.data.final_split, "validation_from_train_fraction": cfg.data.validation_fraction,
                        "validation_seed": cfg.data.validation_seed, "file_sha256": self._file_sha256(cfg.data.dataset_path)},
            "corpus_protocol": {"type": "pooled_train_bounded", "source_split": cfg.data.corpus_source_split,
                                "max_documents": cfg.data.max_documents, "chunk_max_tokens": cfg.data.corpus_chunk_max_tokens,
                                "chunk_overlap_tokens": cfg.data.corpus_chunk_overlap_tokens, "manifest": corpus_manifest},
            "embedding_sync": self.sync_report.to_dict() if self.sync_report else None,
            "embedding_sync_history": [x.to_dict() for x in self.sync_history],
            "versions": _pkg_versions(), "argv": sys.argv[:1], "peak_rss_mb": peak_rss_mb(),
            "warnings": self.warnings,
        }

    def _provider(self, item: dict[str, Any]) -> RetrieverSearchProvider:
        qid = question_id_of(item)
        sync_cb = self._sync_sar_corpus if self.retrieval_method == "sar" else None
        return RetrieverSearchProvider(
            self.retriever, self.run_id, lambda _q, _id=qid: _id,
            gap_source=self.cfg.migres.gap_source, snippet_chars=self.cfg.run.snippet_chars,
            k=(self.cfg.sar.k if self.retrieval_method == "sar" else self.cfg.baseline.k),
            sync_callback=sync_cb,
        )

    def run_question(self, item: dict[str, Any]) -> QuestionResult:
        if self.retriever is None or self.backend is None or self.llm is None:
            raise RuntimeError("call prepare() first")
        provider = self._provider(item)
        before_llm = self.llm.stats.to_dict() if hasattr(self.llm, "stats") else {"calls": 0, "estimated_total_tokens": 0}
        result = self.backend.run_question(item, provider, self.llm, self.run_id, self.run_dir / "upstream_state")
        after_llm = self.llm.stats.to_dict() if hasattr(self.llm, "stats") else before_llm
        result.llm_calls = int(after_llm.get("calls", 0) - before_llm.get("calls", 0))
        result.llm_estimated_tokens = int(after_llm.get("estimated_total_tokens", 0) - before_llm.get("estimated_total_tokens", 0))
        # The MIGRES provider performs SAR corpus checks at iteration boundaries and once after
        # the final retrieval, so the synchronization happens exactly at the methodology-defined points.
        self.writer.append_result(result)
        return result

    def run_single(self, item: dict[str, Any]) -> QuestionResult:
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
        results: list[QuestionResult] = []
        status = "completed"
        try:
            for i, item in enumerate(items, 1):
                if question_id_of(item) in done:
                    continue
                r = self.run_question(item)
                results.append(r)
                log.info("[%d/%d] %s -> %s (%s, %.2fs)", i, len(items), r.question_id, r.final_answer, r.status, r.elapsed_seconds)
        except KeyboardInterrupt:
            status = "interrupted"
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
        # Persist terminal metadata before generating derived report/table artifacts.
        self.writer.write_json("metadata.json", meta)
        paths = self.writer.write_tables(all_results, Path(self.cfg.run.output_dir) / "tables")
        paths["report.md"] = self.writer.write_report(meta, all_results)
        return paths

    def run_dataset(self, limit: int | None = None) -> list[QuestionResult]:
        items = load_questions(self.cfg.data.dataset, self.cfg.data.dataset_path, limit or self.cfg.data.limit, split=self.cfg.data.split)
        return self.run_batch(items)
