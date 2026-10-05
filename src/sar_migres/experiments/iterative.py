from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
import json

from ..utils.config import Config, load_config
from ..corpus.models import Corpus
from ..datasets.loader import dataset_statistics, load_phase_questions, load_questions
from ..datasets.relevance import relevant_doc_ids
from ..evaluation.retrieval import cumulative_recall, hit_at_k, mean_metric, marginal_recall_gain, ndcg_at_k, new_document_rate, precision_at_k, recall_at_k, reciprocal_rank, retrieval_overlap
from ..evaluation.answer import evaluate_answers
from ..evaluation.cost import retrieval_overhead_pct
from ..evaluation.aggregation import paired_bootstrap_ci, paired_permutation_pvalue
from ..utils.errors import ConfigError
from .runner import SARMigresPipeline

def load_config_and_items(config_paths: Sequence[str | Path], *, split: str | None = None,
                          limit: int | None = None, overrides: dict[str, Any] | None = None) -> tuple[Config, list[dict[str, Any]]]:
    cfg = load_config(*config_paths, overrides=overrides)
    items = load_questions(cfg.data.dataset, cfg.data.dataset_path,
                           limit if limit is not None else cfg.data.limit,
                           split=split or cfg.data.split)
    return cfg, items


def run_pipeline_experiment(config_paths: Sequence[str | Path], items: Sequence[dict[str, Any]], *,
                            overrides: dict[str, Any] | None = None,
                            experiment_name: str = "experiment") -> tuple[SARMigresPipeline, list[Any]]:
    pipe = SARMigresPipeline(
        load_config(*config_paths, overrides=overrides),
        run_id=None,
    ).prepare()
    results = pipe.run_batch(items, resume=False)
    return pipe, results


def load_run_results(run_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "results.jsonl"
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_run_traces(run_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "traces.jsonl"
    rows = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _trace_by_q(run_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in load_run_traces(run_dir):
        out.setdefault(str(row["question_id"]), []).append(row)
    for rows in out.values():
        rows.sort(key=lambda x: int(x.get("iteration", 0)))
    return out


def _retrieved_ids(trace: dict[str, Any]) -> list[str]:
    return [str(d["doc_id"]) for d in trace.get("documents", []) if d.get("doc_id") is not None]


def retrieval_metrics_from_run(run_dir: str | Path, items: Sequence[dict[str, Any]], corpus: Corpus) -> dict[str, Any]:
    traces = _trace_by_q(run_dir)
    per_q_final: list[dict[str, float]] = []
    by_iteration: dict[int, list[dict[str, float]]] = {}
    item_map = {str(x["id"]): x for x in items}
    n_seen = 0
    n_annotated = 0
    for qid, rows in traces.items():
        item = item_map.get(qid)
        if item is None:
            continue
        n_seen += 1
        rel = relevant_doc_ids(item, corpus)
        if not rel:
            continue
        n_annotated += 1
        for t in rows:
            ret = _retrieved_ids(t)
            k = int(t.get("k") or len(ret) or 1)
            metrics = {
                "recall_at_k": recall_at_k(ret, rel, k),
                "precision_at_k": precision_at_k(ret, rel, k),
                "hit_at_k": hit_at_k(ret, rel, k),
                "mrr": reciprocal_rank(ret, rel),
                "ndcg_at_k": ndcg_at_k(ret, {d: 1.0 for d in rel}, k),
            }
            by_iteration.setdefault(int(t.get("iteration", 0)), []).append(metrics)
            if t is rows[-1]:
                per_q_final.append(metrics)
    return {
        "n_questions_seen": n_seen,
        "n_questions_with_relevance_annotations": n_annotated,
        "final_mean": {m: mean_metric([x[m] for x in per_q_final]) for m in ("recall_at_k", "precision_at_k", "hit_at_k", "mrr", "ndcg_at_k")},
        "by_iteration": {
            str(i): {m: mean_metric([x[m] for x in vals]) for m in ("recall_at_k", "precision_at_k", "hit_at_k", "mrr", "ndcg_at_k")}
            for i, vals in sorted(by_iteration.items())
        },
        "note": "Retrieval metrics are computed only for questions with explicit/derivable relevant document IDs, matching methodology §5.6.4.",
    }


def iterative_metrics_from_run(run_dir: str | Path, items: Sequence[dict[str, Any]], corpus: Corpus) -> dict[str, Any]:
    traces = _trace_by_q(run_dir)
    item_map = {str(x["id"]): x for x in items}
    series = []
    for qid, rows in traces.items():
        item = item_map.get(qid)
        if item is None:
            continue
        rel = relevant_doc_ids(item, corpus)
        if not rel:
            continue
        per_iter = [_retrieved_ids(t) for t in rows]
        series.append({
            "question_id": qid,
            "cumulative_recall": cumulative_recall(per_iter, rel),
            "marginal_recall_gain": marginal_recall_gain(cumulative_recall(per_iter, rel)),
            "new_document_rate": new_document_rate(per_iter),
            "retrieval_overlap": [retrieval_overlap(per_iter[i - 1], per_iter[i]) for i in range(1, len(per_iter))],
        })
    return {"n_questions": len(series), "per_question": series}


def answer_metrics_from_run(run_dir: str | Path) -> dict[str, float]:
    return evaluate_answers(load_run_results(run_dir))


def cost_metrics_from_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    traces = load_run_traces(run_dir)
    results = load_run_results(run_dir)
    trace_counts: dict[str, int] = {}
    for t in traces:
        qid = str(t.get("question_id", ""))
        trace_counts[qid] = trace_counts.get(qid, 0) + 1
    iterations = [trace_counts.get(str(r.get("question_id", "")), 0) for r in results]
    llm_calls = sum(int(r.get("llm_calls", 0)) for r in results)
    llm_tokens = sum(int(r.get("llm_estimated_tokens", 0)) for r in results)
    retr_stats = meta.get("retrieval", {}).get("stats", {}) or {}
    sync_hist = meta.get("embedding_sync_history", []) or []
    initial_sync_embed = float(sync_hist[0].get("embed_seconds", 0.0)) if sync_hist else 0.0
    index_build = max(float(retr_stats.get("index_build_seconds", 0.0)), initial_sync_embed)
    retrieval_latency = sum(float(t.get("total_seconds", 0.0)) for t in traces)
    baseline_latency = meta.get("comparison", {}).get("baseline_retrieval_latency_total_s")
    overhead = retrieval_overhead_pct(retrieval_latency, float(baseline_latency)) if baseline_latency is not None else float("nan")
    return {
        "retrieval_latency_total_s": retrieval_latency,
        "embedding_time_total_s": sum(float(t.get("embed_seconds", 0.0)) for t in traces),
        "scoring_time_total_s": sum(float(t.get("score_seconds", 0.0)) for t in traces),
        "index_build_seconds": index_build,
        "documents_scored": sum(int(t.get("n_docs_scored", 0)) for t in traces),
        "similarity_calculations": sum(int(t.get("n_similarity_calcs", 0)) for t in traces),
        "iterations_to_termination_mean": float(sum(iterations)/len(iterations)) if iterations else float("nan"),
        "iterations_to_termination_median": float(__import__("statistics").median(iterations)) if iterations else float("nan"),
        "llm_calls": llm_calls,
        "llm_estimated_tokens": llm_tokens,
        "retrieval_overhead_pct": overhead,
        "n_questions": len(results),
        "peak_rss_mb": meta.get("peak_rss_mb"),
        "retrieval_method": meta.get("retrieval", {}).get("method"),
    }


def paired_run_comparison(run_a: str | Path, run_b: str | Path, items: Sequence[dict[str, Any]], corpus: Corpus,
                          *, metric: str = "recall_at_k", seed: int = 13, n_resamples: int = 2000) -> dict[str, Any]:
    """Compare two completed runs on the same question IDs with paired bootstrap and permutation test."""
    def final_values(run_dir):
        rows = _trace_by_q(run_dir)
        item_map = {str(x["id"]): x for x in items}
        values = {}
        for qid, ts in rows.items():
            item = item_map.get(qid)
            if item is None or not ts:
                continue
            rel = relevant_doc_ids(item, corpus)
            if not rel:
                continue
            t = ts[-1]; ret = _retrieved_ids(t); k = int(t.get("k") or len(ret) or 1)
            vals = {"recall_at_k": recall_at_k(ret, rel, k), "precision_at_k": precision_at_k(ret, rel, k),
                    "hit_at_k": hit_at_k(ret, rel, k), "mrr": reciprocal_rank(ret, rel),
                    "ndcg_at_k": ndcg_at_k(ret, {d: 1.0 for d in rel}, k)}
            values[qid] = vals[metric]
        return values
    a, b = final_values(run_a), final_values(run_b)
    common = sorted(set(a) & set(b))
    av = [a[q] for q in common]; bv = [b[q] for q in common]
    return {"metric": metric, "question_ids": common,
            "bootstrap": paired_bootstrap_ci(av, bv, n_resamples=n_resamples, seed=seed),
            "permutation": paired_permutation_pvalue(av, bv, n_resamples=n_resamples, seed=seed)}
