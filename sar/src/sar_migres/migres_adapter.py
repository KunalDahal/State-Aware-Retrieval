"""Integration layer between SAR and MIGRES.

Where SAR replaces the original MIGRES retriever
------------------------------------------------
In the official code (AdelWang/MIGRES, ``code/pipeline.py``) every retrieval goes through
``RetGBatched.search(questions, rerank)`` which (for ``--search_type bm25``) calls a BM25
searcher (``search_bm25``) and then applies MIGRES' own cross-encoder filter
(``Reranker.sentenceRerank``) and the ``QueryForward`` fallback.  :class:`UpstreamMigres`
loads the *unmodified* upstream ``RetGBatched`` and overrides exactly one method,
``search``: the BM25 candidate lookup is replaced by full-corpus SAR Top-K, and the
remaining steps (``sentenceRerank`` filtering, ``QueryForward``, evidence extraction, gap
generation, prompts, stopping rule) are the upstream code, untouched.

State mapping (methodology (q, E_t, G_t, D)  <->  MIGRES variables)
    q    = the original question key of the batch item
    E_t  = ``self.known_info[q]``   (list[str] of extracted useful information)
    G_t  = the sub-queries MIGRES generates for the missing information (``questions[q]``
           in later ``search`` calls) - config ``migres.gap_source`` can switch to the
           ``self.missing_info[q]`` summary string, or both.  G_0 = {} (first call).

Compatibility adapters (isolated here, not applied to the upstream files)
  1. ``pipeline.py`` cannot be imported: it parses CLI args at import time, imports
     deepspeed/pyserini/faiss at module level and (in the inspected commit) contains a
     syntax error at line 1104.  We therefore extract the needed top-level definitions by AST
     and execute them in a private namespace.
  2. Upstream ``send_post_request`` is hard-wired to an OpenAI client with empty credentials and
     returns a *string* although downstream code iterates over a *list* of outputs;
     ``askGPT`` is overridden to route through :mod:`sar_migres.llm` and return ``list[str]``.
  3. Upstream ``extractMain``/``extractLeaf`` call ``eval`` on raw LLM text (arbitrary code
     execution risk).  In the private namespace ``eval`` is bound to ``ast.literal_eval``, which
     accepts the same valid dict/list literals and fails (into the same ``except`` path) otherwise.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import copy
import json
import re
import subprocess
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .errors import (ConfigError, DependencyMissingError, LLMError, LLMResponseError, MissingCredentialError,
                     UpstreamMigresError)
from .llm import LLMClient, _content_tokens, parse_dict_reply
from .logging_utils import get_logger
from .retrieval import StateAwareRetriever
from .schemas import QuestionResult, RetrievalState, TraceRecord
from .utils import sha256_file, sha256_text, snippet

log = get_logger("migres_adapter")

UPSTREAM_URL = "https://github.com/AdelWang/MIGRES"
# Upstream commit / file hash this adapter was developed and tested against.
TESTED_UPSTREAM_COMMIT = "1a67b8769ca3fdf2b497e109e529143822248ec5"
TESTED_UPSTREAM_SHA256 = {
    "code/pipeline.py": "c0371457158abc586067f986747991b4b01832293bd9b38dea3cb254cc124a26",
    "code/utils/search_engine.py": "75461e8e249e37d0cb87e4170cb1ae93938f4d66b1a3c5774ecca395dc04b23b",
}


# ----------------------------------------------------------------------------------------------
# State mapping + generic retrieval provider
# ----------------------------------------------------------------------------------------------
def derive_gaps(question: str, queries: list[str], missing_info: str, gap_source: str, iteration: int) -> list[str]:
    """Map the MIGRES state to G_t.  G_0 is empty, as specified in methodology Sec. 5.2.6."""
    if iteration == 0:
        return []
    gaps: list[str] = []
    if gap_source in ("queries", "both"):
        gaps.extend(x.strip() for x in queries if isinstance(x, str) and x.strip() and x.strip() != question.strip())
    if gap_source in ("missing_info", "both") and missing_info and missing_info.strip():
        gaps.append(missing_info.strip())
    return list(dict.fromkeys(gaps))


class RetrieverSearchProvider:
    """Adapter shared by SAR and every baseline retriever.

    SAR consumes (q, E_t, G_t) and applies its own scoring.  Baselines receive the current
    MIGRES-generated query list and are fused with ``retrieve_many``.  This means changing the
    retrieval method no longer changes the surrounding MIGRES loop, prompts, filtering,
    evidence extraction or stopping behaviour.
    """

    def __init__(self, retriever, run_id: str, qid_of: Callable[[str], str], *,
                 gap_source: str = "queries", snippet_chars: int = 240, k: int = 5,
                 sync_callback: Callable[[], Any] | None = None):
        self.retriever = retriever
        self.run_id = run_id
        self.qid_of = qid_of
        self.gap_source = gap_source
        self.snippet_chars = snippet_chars
        self.k = int(k)
        self.sync_callback = sync_callback
        self.traces: dict[str, list[TraceRecord]] = {}
        self._last_docs: dict[str, list[dict[str, Any]]] = {}

    @property
    def is_sar(self) -> bool:
        return bool(getattr(self.retriever, "is_sar", False))

    @property
    def name(self) -> str:
        return str(getattr(self.retriever, "name", "retriever"))

    def _make_trace_documents(self, rows: list[tuple[str, float]], iteration: int) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for rank, (did, score) in enumerate(rows, start=1):
            d = self.retriever.corpus.get(did) if hasattr(self.retriever, "corpus") else None
            docs.append({
                "doc_id": str(did), "rank": rank, "title": d.title if d else "",
                "text": snippet(d.text, self.snippet_chars) if d else "",
                "relevance": 0.0, "gap_coverage": 0.0, "novelty": 0.0,
                "score": float(score), "retrieval_score": float(score),
                "retriever": self.name, "score_name": getattr(self.retriever, "score_name", "baseline_score"),
                "is_sar": False,
            })
        return docs

    def retrieve_for(self, question: str, queries: list[str], known_info: list[str], missing_info: str = "") -> list[dict]:
        traces = self.traces.setdefault(question, [])
        iteration = len(traces)
        # At iteration t>0, MIGRES has already completed the previous retrieval's filtering,
        # evidence extraction and missing-information update before asking for the next retrieval.
        # Syncing here therefore implements the methodology's end-of-iteration corpus check without
        # forcing a refresh before the first retrieval.
        if self.is_sar and iteration > 0 and self.sync_callback is not None:
            self.sync_callback()
        t0 = time.perf_counter()
        stats = getattr(self.retriever, "stats", None)
        before_scored = int(getattr(stats, "n_docs_scored", 0)) if stats else 0
        before_sim = int(getattr(stats, "n_similarity_calcs", 0)) if stats else 0
        before_retrieval = float(getattr(stats, "retrieval_seconds", 0.0)) if stats else 0.0
        if self.is_sar:
            gaps = derive_gaps(question, list(queries), missing_info, self.gap_source, iteration)
            res = self.retriever.retrieve(
                RetrievalState(question, [str(e) for e in known_info], gaps), iteration=iteration, k=self.k)
            docs = []
            for d in res.documents:
                row = d.to_dict()
                row["text"] = snippet(row["text"], self.snippet_chars)
                row["retriever"] = "sar"; row["score_name"] = "sar_score"; row["is_sar"] = True
                row["retrieval_score"] = row["score"]
                docs.append(row)
            total = time.perf_counter() - t0
            trace = TraceRecord(
                run_id=self.run_id, question_id=self.qid_of(question), question=question, iteration=iteration,
                gaps=gaps, evidence=[str(e) for e in known_info], alpha=res.alpha, beta=res.beta, gamma=res.gamma,
                k=res.k, documents=docs, n_docs_scored=res.n_docs_scored,
                n_similarity_calcs=res.n_similarity_calcs, embed_seconds=res.embed_seconds,
                score_seconds=res.score_seconds, total_seconds=total, retriever="sar", score_name="sar_score", is_sar=True)
            self._last_docs[question] = docs
            traces.append(trace)
            return [{"title": d.title, "text": d.text, "score": d.score, "doc_id": d.doc_id,
                     "sar_rank": d.rank} for d in res.documents]

        clean_queries = list(dict.fromkeys(q.strip() for q in queries if isinstance(q, str) and q.strip())) or [question]
        rows = self.retriever.retrieve_many(clean_queries, self.k)
        docs = self._make_trace_documents(rows, iteration)
        stats = getattr(self.retriever, "stats", None)
        n_scored = int(getattr(stats, "n_docs_scored", 0)) - before_scored if stats else 0
        n_sim = int(getattr(stats, "n_similarity_calcs", 0)) - before_sim if stats else 0
        score_s = float(getattr(stats, "retrieval_seconds", 0.0)) - before_retrieval if stats else None
        total = time.perf_counter() - t0
        if score_s is None:
            score_s = total
        trace = TraceRecord(
            run_id=self.run_id, question_id=self.qid_of(question), question=question, iteration=iteration,
            gaps=derive_gaps(question, clean_queries, missing_info, self.gap_source, iteration),
            evidence=[str(e) for e in known_info], alpha=0.0, beta=0.0, gamma=0.0, k=self.k, documents=docs,
            n_docs_scored=n_scored, n_similarity_calcs=n_sim, embed_seconds=0.0,
            score_seconds=min(score_s, total), total_seconds=total, retriever=self.name,
            score_name=getattr(self.retriever, "score_name", "baseline_score"), is_sar=False)
        self._last_docs[question] = docs
        traces.append(trace)
        return [{"title": d["title"], "text": d["text"], "score": d["score"], "doc_id": d["doc_id"],
                 "sar_rank": d["rank"]} for d in docs]

    def record_filter(self, question: str, kept_titles: list[str]) -> None:
        if question not in self.traces or not self.traces[question]:
            return
        tr = self.traces[question][-1]
        kept = set(kept_titles)
        tr.migres_state_after["filter_kept"] = len(kept_titles)
        tr.migres_state_after["filter_kept_doc_ids"] = [d["doc_id"] for d in self._last_docs[question] if d.get("title") in kept]

    def finalize(self, question: str, final_evidence: list[str], final_missing: str = "") -> None:
        trs = self.traces.get(question, [])
        for i, tr in enumerate(trs):
            before = tr.evidence
            after = trs[i + 1].evidence if i + 1 < len(trs) else [str(e) for e in final_evidence]
            tr.migres_state_after.update({
                "evidence_added": [e for e in after if e not in before],
                "evidence_size_after": len(after),
                "next_gaps": trs[i + 1].gaps if i + 1 < len(trs) else None,
                "is_last_retrieval": i + 1 == len(trs),
            })
        if trs and final_missing:
            trs[-1].migres_state_after["missing_info_at_end"] = final_missing
        if trs and self.is_sar and self.sync_callback is not None:
            # No subsequent retrieval call exists after the final iteration, so perform the final
            # corpus check after MIGRES has finished the question.
            self.sync_callback()


class SARSearchProvider(RetrieverSearchProvider):
    """Backward-compatible SAR-specific provider name."""

    def __init__(self, retriever: StateAwareRetriever, run_id: str, qid_of: Callable[[str], str],
                 gap_source: str = "queries", snippet_chars: int = 240, k: int | None = None,
                 sync_callback: Callable[[], Any] | None = None):
        super().__init__(retriever, run_id, qid_of, gap_source=gap_source, snippet_chars=snippet_chars,
                         k=k or retriever.k, sync_callback=sync_callback)


class LexicalStandInReranker:
    """OFFLINE stand-in for MIGRES' cross-encoder filter."""

    is_stand_in = True

    def __init__(self, args=None, min_overlap: float = 0.15):
        self.min_overlap = min_overlap

    def sentenceRerank(self, ret_res, questions, historical_choosed):
        out = defaultdict(list)
        for ori_q, raw in ret_res.items():
            ctx = [{"title": c["title"], "text": c["text"].replace("\n", "").replace("<br>", "").replace("<b>", "").replace("</b>", "")}
                   for c in raw]
            title_of = {c["text"]: c["title"] for c in ctx}
            chosen: dict[str, list] = {}
            for q in questions[ori_q]:
                qt = set(_content_tokens(q))
                for c in ctx:
                    score = len(qt & set(_content_tokens(c["title"] + " " + c["text"]))) / max(len(qt), 1)
                    if score < self.min_overlap or historical_choosed[ori_q][c["text"]] >= 2:
                        continue
                    if c["text"] not in chosen or chosen[c["text"]][0] < score:
                        chosen[c["text"]] = [score, q]
            rows = sorted(chosen.items(), key=lambda kv: kv[1][0], reverse=True)
            out[ori_q] = [[t, sc, q, title_of[t], t] for t, (sc, q) in rows]
        return out


# ----------------------------------------------------------------------------------------------
# Backend interface
# ----------------------------------------------------------------------------------------------
class MigresBackend(ABC):
    name: str = "abstract"
    reranker_label: str = ""

    @abstractmethod
    def run_question(self, item: dict[str, Any], provider: SARSearchProvider, llm: LLMClient, run_id: str,
                     workdir: Path) -> QuestionResult: ...

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name}


def question_id_of(item: dict[str, Any]) -> str:
    for k in ("id", "_id", "qid", "question_id"):
        if item.get(k) not in (None, ""):
            return str(item[k])
    return "q_" + sha256_text(item["question"])[:12]


def _final_answer(d: dict[str, Any]) -> str | None:
    fr = d.get("final_res")
    if isinstance(fr, dict) and isinstance(fr.get("answer"), dict):
        t = fr["answer"].get("text")
        return None if t is None else str(t)
    return None


# ----------------------------------------------------------------------------------------------
# Upstream (official MIGRES code) backend
# ----------------------------------------------------------------------------------------------
def _safe_eval(source, *_a, **_k):
    """Replacement for ``eval`` inside the upstream namespace (see module docstring, item 3)."""
    if not isinstance(source, str):
        raise TypeError("eval() arg must be a string")
    return ast.literal_eval(source)


def _slice_before_argparse(src: str) -> str:
    m = re.search(r"^parser\s*=\s*argparse\.ArgumentParser", src, flags=re.M)
    if not m:
        raise UpstreamMigresError("could not locate the argparse section in upstream pipeline.py; upstream layout changed")
    return src[: m.start()]


def _extract_defs(source: str, names: set[str]) -> dict[str, str]:
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            found[node.name] = ast.get_source_segment(source, node)
    missing = names - set(found)
    if missing:
        raise UpstreamMigresError(f"upstream definitions not found: {sorted(missing)} - upstream layout changed?")
    return found


class UpstreamCode:
    """Loads official MIGRES definitions from a local clone (never vendored in this repo)."""

    def __init__(self, repo_dir: str | Path, reranker: str, ranker_model_path: str, strict_hash: bool = False):
        self.repo = Path(repo_dir)
        self.reranker_mode = reranker
        self.pipeline_py = self.repo / "code" / "pipeline.py"
        self.engine_py = self.repo / "code" / "utils" / "search_engine.py"
        for p in (self.pipeline_py, self.engine_py):
            if not p.exists():
                raise UpstreamMigresError(
                    f"{p} not found. Clone the official implementation:  git clone {UPSTREAM_URL} <dir>  "
                    "and set migres.repo_dir or MIGRES_REPO_DIR.")
        self.info = self._provenance(strict_hash)
        self.ns = self._load()
        self.RetGBatched = self.ns["RetGBatched"]

    def _provenance(self, strict: bool) -> dict[str, Any]:
        info: dict[str, Any] = {"repo": UPSTREAM_URL, "tested_commit": TESTED_UPSTREAM_COMMIT}
        try:
            info["commit"] = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True,
                                            text=True, timeout=10).stdout.strip() or None
        except Exception:  # noqa: BLE001 - git may be absent; hashes below still identify the code
            info["commit"] = None
        info["sha256"] = {"code/pipeline.py": sha256_file(self.pipeline_py),
                          "code/utils/search_engine.py": sha256_file(self.engine_py)}
        ok = all(info["sha256"][k] == v for k, v in TESTED_UPSTREAM_SHA256.items())
        info["matches_tested_version"] = ok
        if not ok:
            msg = ("upstream MIGRES files differ from the version this adapter was tested against "
                   f"(commit {TESTED_UPSTREAM_COMMIT}); behaviour is unverified for the loaded version")
            if strict:
                raise UpstreamMigresError(msg)
            log.warning(msg)
        return info

    def _load(self) -> dict[str, Any]:
        b = dict(builtins.__dict__)
        b["eval"] = _safe_eval
        import concurrent.futures

        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover
            tqdm = lambda x, *a, **k: x  # noqa: E731
        ns: dict[str, Any] = {"__name__": "migres_upstream_pipeline", "__builtins__": b, "json": json,
                              "copy": copy, "defaultdict": defaultdict, "Counter": Counter, "time": time,
                              "concurrent": concurrent, "tqdm": tqdm, "os": __import__("os")}
        for mod, names in (("sklearn.cluster", ["KMeans"]), ("sklearn.feature_extraction.text", ["TfidfVectorizer"]),
                           ("scipy.spatial.distance", ["cdist"])):
            try:
                m = __import__(mod, fromlist=names)
                for n in names:
                    ns[n] = getattr(m, n)
            except ImportError:
                pass  # only needed when num_return > 1 (upstream clusterFilter)
        pipe_src = _slice_before_argparse(self.pipeline_py.read_text(encoding="utf-8"))
        defs = _extract_defs(pipe_src, {"multi_return_process", "clusterFilter", "RetGBatched"})
        eng_src = self.engine_py.read_text(encoding="utf-8")
        eng_names = {"verify_info"} | ({"Reranker"} if self.reranker_mode == "upstream_cross_encoder" else set())
        eng = _extract_defs(eng_src, eng_names)
        if self.reranker_mode == "upstream_cross_encoder":
            try:
                import nltk
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as e:
                raise DependencyMissingError(
                    "upstream cross-encoder filtering needs torch, transformers and nltk "
                    "(pip install 'sar-migres[migres]'); use migres.reranker=offline_standin for the offline smoke test") from e
            ns.update(nltk=nltk, torch=torch, AutoModelForSequenceClassification=AutoModelForSequenceClassification,
                      AutoTokenizer=AutoTokenizer)
        else:
            ns["Reranker"] = LexicalStandInReranker
        try:
            import torch as _t
            ns.setdefault("torch", _t)
        except ImportError:
            pass
        for name, code in {**eng, **defs}.items():
            exec(compile(code, f"<upstream:{name}>", "exec"), ns)  # noqa: S102 - trusted, hash-recorded source
        return ns


def make_upstream_namespace_args(mcfg, data_path_hint: str, save_path: str, model_name: str, search_type: str = "custom") -> argparse.Namespace:
    """Mirrors the flags of upstream ``code/migres.sh`` (see README for the deliberate deviations)."""
    return argparse.Namespace(
        data_path=data_path_hint, save_path=save_path, top_k=mcfg.top_k, max_iter=mcfg.max_iter,
        relevance=mcfg.relevance, gpt_knowledge=mcfg.gpt_knowledge, entail_judge=mcfg.entail_judge,
        add_demon=mcfg.add_demon, demon_path=mcfg.demon_path or None, num_return=mcfg.num_return,
        sentRerank=mcfg.sent_rerank, ranker_model_path=mcfg.ranker_model_path, model_name_or_path=None,
        summ_snippet=None, search_type=search_type, normalize=True, num_process=1, aug=False, retrieval=False,
        model_name=model_name, query_prompt="", encoder_path=None, encoded_data_path=None, index_dir=None,
        sar_retrieval_method=search_type)


class UpstreamMigres(MigresBackend):
    name = "upstream"

    def __init__(self, cfg):
        repo = cfg.resolve_migres_repo()
        if repo is None:
            raise UpstreamMigresError(f"MIGRES repo not configured: git clone {UPSTREAM_URL} and set MIGRES_REPO_DIR")
        self.cfg = cfg
        self.code = UpstreamCode(repo, cfg.migres.reranker, cfg.migres.ranker_model_path, cfg.migres.strict_upstream_hash)
        self.reranker_label = "lexical_offline_standin" if cfg.migres.reranker == "offline_standin" else "upstream_cross_encoder"
        self._nli = None
        self.cls = self._make_class()

    def describe(self):
        return {"backend": self.name, "upstream": self.code.info, "reranker": self.reranker_label,
                "gap_source": self.cfg.migres.gap_source}

    def _nli_component(self):
        if not self.cfg.migres.entail_judge:
            return None
        if self._nli is None:
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as e:
                raise DependencyMissingError("entail_judge needs torch+transformers") from e
            name = "google/t5_xxl_true_nli_mixture"  # as hard-coded in upstream pipeline.py
            log.info("loading NLI model %s (large)", name)
            self._nli = [AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto"),
                         AutoTokenizer.from_pretrained(name, use_fast=False)]
        return self._nli

    def _make_class(self):
        Base = self.code.RetGBatched
        backend = self

        class SARMigres(Base):  # noqa: D401 - subclass of the UNMODIFIED upstream class
            """Upstream RetGBatched with retrieval (`search`) and LLM transport (`askGPT`) replaced."""

            def __init__(self, data, args, provider, llm, failures):
                self._retrieval, self._llm, self._failures = provider, llm, failures
                super().__init__(data, args, searcher=None, model_name=args.model_name, world_size=1,
                                 wikiData=None, nli_component=backend._nli_component())

            # ---- the only retrieval replacement: route to SAR OR one Table-5.6 baseline --------
            def search(self, questions, rerank=True):
                raw = {}
                for ori_q, queries in questions.items():
                    raw[ori_q] = self._retrieval.retrieve_for(ori_q, list(queries), self.known_info[ori_q],
                                                             self.missing_info.get(ori_q, ""))
                if not rerank:
                    return None, raw
                ranked = self.reranker.sentenceRerank(raw, questions, self.history_passages)
                for ori_q, rows in ranked.items():
                    self._retrieval.record_filter(ori_q, [r[3] for r in rows])
                ranked = self.QueryForward(ranked, questions, raw)
                return ranked, raw

            # ---- LLM transport (compat adapter #2) -----------------------------------------------
            def askGPT(self, inputs, num_return=1):
                out = {}
                for q, prompt in inputs.items():
                    try:
                        out[q] = self._llm.generate(prompt, n=num_return)
                    except (MissingCredentialError, ConfigError, DependencyMissingError):
                        raise
                    except Exception as e:  # noqa: BLE001 - recorded, and the item is marked failed (upstream convention)
                        log.error("LLM call failed for question %r: %s: %s", q[:60], type(e).__name__, e)
                        self._failures.append({"question": q, "error": f"{type(e).__name__}: {e}"})
                for d in self.data:
                    if d["question"] in inputs and d["question"] not in out:
                        d["fail"] = True
                return out

        return SARMigres

    def run_question(self, item, provider, llm, run_id, workdir):
        q = item["question"]
        qid = question_id_of(item)
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
        data_hint = self.cfg.data.dataset_path if "strategy" in str(self.cfg.data.dataset).lower() else "custom_dataset.json"
        if "strategy" in str(self.cfg.data.dataset).lower():
            data_hint = "./data/strategy.json"  # upstream selects its StrategyQA prompts via this substring
        args = make_upstream_namespace_args(self.cfg.migres, data_hint, str(work / f"{qid}.json"), llm.model, search_type=getattr(provider, "name", "custom"))
        d0 = {k: v for k, v in copy.deepcopy(item).items() if k not in ("prompt_to_ask",)}
        failures: list[dict] = []
        t0 = time.perf_counter()
        status, err = "ok", None
        runner = self.cls([d0], args, provider, llm, failures)
        try:
            runner._iteration()
        except (MissingCredentialError, ConfigError, DependencyMissingError, KeyboardInterrupt):
            raise
        except Exception as e:  # noqa: BLE001 - reported, never hidden
            log.exception("upstream MIGRES raised for question %s", qid)
            status, err = "failed", f"{type(e).__name__}: {e}"
        d = runner.data[0]
        provider.finalize(q, runner.known_info.get(q, []), runner.missing_info.get(q, ""))
        ans = _final_answer(d)
        if d.get("fail") or failures:
            status = "failed"
            err = err or (failures[0]["error"] if failures else "MIGRES marked the item as failed (see migres_record)")
        elif ans is None and status == "ok":
            status = "no_answer"
        rec = json.loads(json.dumps(d, default=repr))
        rec["llm_failures"] = failures
        return QuestionResult(run_id=run_id, question_id=qid, question=q, gold_answer=item.get("answer"),
                              final_answer=ans, traces=provider.traces.get(q, []), status=status, error=err,
                              elapsed_seconds=time.perf_counter() - t0, backend=self.name, migres_record=rec)


# ----------------------------------------------------------------------------------------------
# Stand-in backend (offline smoke test / unit tests ONLY - not MIGRES)
# ----------------------------------------------------------------------------------------------
class StandInMigres(MigresBackend):
    """Minimal iterative loop with the same *interface* as MIGRES (evidence E_t, gaps G_t, extraction,
    gap update) and the same prompt heads, so the offline LLM can drive it.  It exists so the
    project's smoke test and unit tests run without the upstream repo or any API.
    It is NOT an implementation of MIGRES and its outputs must never be reported as MIGRES results."""

    name = "standin"
    reranker_label = "lexical_offline_standin"

    def __init__(self, cfg):
        self.cfg = cfg
        self.filter = LexicalStandInReranker()

    def describe(self):
        return {"backend": self.name, "note": "offline stand-in loop, NOT the official MIGRES"}

    def run_question(self, item, provider, llm, run_id, workdir):
        q, qid = item["question"], question_id_of(item)
        m = self.cfg.migres
        known: list[str] = []
        hist_q: list[str] = []
        missing = ""
        queries = [q]
        used: defaultdict = defaultdict(int)
        transitions: list[dict] = []
        answer, status, err = None, "ok", None
        t0 = time.perf_counter()
        try:
            for t in range(m.max_iter + 1):
                raw = provider.retrieve_for(q, queries, known, missing)
                ranked = self.filter.sentenceRerank({q: raw}, {q: queries}, {q: used})[q]
                provider.record_filter(q, [r[3] for r in ranked])
                passages = []
                for r in ranked:
                    if len(passages) == m.top_k:
                        break
                    if used[r[4]] >= 2 or len(r[0].split()) < 5:
                        continue
                    used[r[4]] += 1
                    passages.append(f"(Title: {r[3]}) {r[0]}")
                hist_q.extend(queries)
                if passages:
                    leaf_prompt = ("Given the following Question and Passages, please distillate useful information from the Passages "
                                   "to address the Question effectively.\n\n" + "\n".join(f"Passage {i}: {p}" for i, p in enumerate(passages))
                                   + f"\nQuestion: {' '.join(queries)}\nYour response:")
                    infos = parse_dict_reply(llm.generate(leaf_prompt)[0]).get("useful_information", [])
                    for it in infos:
                        s = str(it.get("info", "")).strip()
                        if s and it.get("support_passages") and s not in known:
                            known.append(s)
                last = t == m.max_iter
                info = "; ".join(known) if known else "None"
                head = ("Answer the Question based on the provided Information. If the Information is not sufficient to answer the question, "
                        "answer it based on your own knowledge." if last else
                        "Answer the Question based solely on the provided Information. If the Information is insufficient, answer unanswerable.")
                main = parse_dict_reply(llm.generate(f"{head}\n\nInformation: {info}\nQuestion: {q}\nYour response:")[0])
                text = str(main.get("answer", {}).get("text", ""))
                transitions.append({"iteration": t, "answer_candidate": text})
                if "unanswerable" not in text.lower() or last:
                    answer = text
                    break
                missing = str(main.get("missing_information") or main.get("explanation") or "")
                qprompt = ("Based on the Original Question, Historical Questions, Known Information and Missing Information, write no more than 3 queries "
                           f"that ask for Missing Information.\n\nOriginal Question: {q}\nHistorical Questions: {' '.join(hist_q)}\n"
                           f"Known Information: {info}\nMissing Information: {missing}\nNew queries:")
                new = [x.strip() for x in llm.generate(qprompt)[0].split("\n") if len(x.strip()) > 10 and x.strip() not in hist_q]
                queries = new or [q]
        except (MissingCredentialError, ConfigError, DependencyMissingError, KeyboardInterrupt):
            raise
        except (LLMError, ValueError) as e:
            status, err = "failed", f"{type(e).__name__}: {e}"
            log.error("stand-in loop failed for %s: %s", qid, err)
        provider.finalize(q, known, missing)
        if status == "ok" and answer is None:
            status = "no_answer"
        return QuestionResult(run_id=run_id, question_id=qid, question=q, gold_answer=item.get("answer"), final_answer=answer,
                              traces=provider.traces.get(q, []), status=status, error=err,
                              elapsed_seconds=time.perf_counter() - t0, backend=self.name,
                              migres_record={"known_information": known, "history_questions": hist_q, "transitions": transitions})


def build_backend(cfg) -> MigresBackend:
    if cfg.migres.backend == "standin":
        return StandInMigres(cfg)
    return UpstreamMigres(cfg)
