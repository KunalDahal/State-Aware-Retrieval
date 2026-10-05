"""Machine-readable and human-readable run outputs for SAR and baseline runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .retrieval.common import QuestionResult
from .utils.paths import atomic_write_text, utc_now_iso
from .utils.common import dumps

TRACE_COLUMNS = [
    "run_id", "question_id", "question", "retriever", "score_name", "is_sar", "iteration", "gaps", "evidence",
    "rank", "doc_id", "title", "snippet", "R_relevance", "C_gap_coverage", "N_novelty",
    "alpha", "beta", "gamma", "retrieval_score", "K", "n_docs_scored", "n_similarity_calcs",
    "embed_seconds", "score_seconds", "iteration_seconds", "passed_migres_filter", "evidence_added_after",
]
SUMMARY_COLUMNS = [
    "run_id", "question_id", "question", "status", "backend", "retriever", "is_sar",
    "n_retrieval_iterations", "termination_iteration", "final_answer", "gold_answer", "total_retrieval_seconds",
    "elapsed_seconds", "llm_calls", "llm_estimated_tokens", "error",
]


class RunWriter:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        atomic_write_text(p, json.dumps(obj, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else repr(o),
                                        indent=2, ensure_ascii=False, allow_nan=False))
        return p

    def read_json(self, name: str) -> Any:
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    def append_result(self, r: QuestionResult) -> None:
        # The implementation keeps question-level results and iteration traces in separate JSONL files.
        summary = {
            "run_id": r.run_id, "question_id": r.question_id, "question": r.question,
            "status": r.status, "final_answer": r.final_answer, "gold_answer": r.gold_answer,
            "elapsed_seconds": r.elapsed_seconds, "backend": r.backend,
            "llm_calls": r.llm_calls, "llm_estimated_tokens": r.llm_estimated_tokens,
            "error": r.error,
        }
        with open(self.dir / "results.jsonl", "a", encoding="utf-8") as fh:
            fh.write(dumps(summary) + "\n"); fh.flush()
        with open(self.dir / "traces.jsonl", "a", encoding="utf-8") as fh:
            for t in r.traces:
                fh.write(dumps(t.to_dict()) + "\n")
            fh.flush()

    def finished_ids(self) -> set[str]:
        p = self.dir / "results.jsonl"
        ids = set()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") in ("ok", "no_answer"):
                    ids.add(rec["question_id"])
        return ids

    def load_results(self) -> list[dict[str, Any]]:
        p = self.dir / "results.jsonl"
        out = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def _trace_rows_by_question(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        p = self.dir / "traces.jsonl"
        if not p.exists():
            return grouped
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            grouped.setdefault(str(row.get("question_id", "")), []).append(row)
        for rows in grouped.values():
            rows.sort(key=lambda x: int(x.get("iteration", 0)))
        return grouped

    def write_tables(self, results: Iterable[dict[str, Any]]) -> dict[str, Path]:
        results = list(results)
        traces_by_q = self._trace_rows_by_question()
        trace_rows, summary_rows = [], []
        for r in results:
            traces = traces_by_q.get(str(r.get("question_id", "")), [])
            for t in traces:
                after = t.get("migres_state_after", {})
                kept = set(after.get("filter_kept_doc_ids", []))
                for d in t.get("documents", []):
                    trace_rows.append({
                        "run_id": r["run_id"], "question_id": r["question_id"], "question": r["question"],
                        "retriever": t.get("retriever", d.get("retriever", "")), "score_name": t.get("score_name", d.get("score_name", "")),
                        "is_sar": t.get("is_sar", d.get("is_sar", False)), "iteration": t["iteration"],
                        "gaps": json.dumps(t.get("gaps", []), ensure_ascii=False), "evidence": json.dumps(t.get("evidence", []), ensure_ascii=False),
                        "rank": d.get("rank"), "doc_id": d.get("doc_id"), "title": d.get("title", ""), "snippet": d.get("text", ""),
                        "R_relevance": d.get("relevance", ""), "C_gap_coverage": d.get("gap_coverage", ""), "N_novelty": d.get("novelty", ""),
                        "alpha": t.get("alpha", ""), "beta": t.get("beta", ""), "gamma": t.get("gamma", ""),
                        "retrieval_score": d.get("retrieval_score", d.get("score", "")), "K": t.get("k", ""),
                        "n_docs_scored": t.get("n_docs_scored", ""), "n_similarity_calcs": t.get("n_similarity_calcs", ""),
                        "embed_seconds": t.get("embed_seconds", ""), "score_seconds": t.get("score_seconds", ""),
                        "iteration_seconds": t.get("total_seconds", ""),
                        "passed_migres_filter": (d.get("doc_id") in kept) if "filter_kept_doc_ids" in after else "",
                        "evidence_added_after": json.dumps(after.get("evidence_added", []), ensure_ascii=False),
                    })
            summary_rows.append({
                "run_id": r["run_id"], "question_id": r["question_id"], "question": r["question"], "status": r["status"],
                "backend": r.get("backend", ""), "retriever": (traces[0].get("retriever") if traces else r.get("retriever", "")),
                "is_sar": (traces[0].get("is_sar") if traces else r.get("is_sar", False)),
                "n_retrieval_iterations": len(traces), "termination_iteration": (traces[-1].get("iteration") if traces else ""),
                "final_answer": r.get("final_answer"),
                "gold_answer": json.dumps(r.get("gold_answer"), ensure_ascii=False),
                "total_retrieval_seconds": round(sum(float(t.get("total_seconds", 0)) for t in traces), 6),
                "elapsed_seconds": round(float(r.get("elapsed_seconds", 0.0)), 6),
                "llm_calls": r.get("llm_calls", 0), "llm_estimated_tokens": r.get("llm_estimated_tokens", 0),
                "error": r.get("error") or "",
            })
        out = {}
        for name, cols, rows in (("traces.csv", TRACE_COLUMNS, trace_rows), ("summary.csv", SUMMARY_COLUMNS, summary_rows)):
            p = self.dir / name
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
            out[name] = p
        return out

    def write_report(self, meta: dict[str, Any], results: Iterable[dict[str, Any]]) -> Path:
        results = list(results)
        ret = meta.get("retrieval", {})
        L = [
            f"# SAR-MIGRES run report - `{meta.get('run_id')}`", "",
            f"- status: **{meta.get('status')}**", f"- retriever: `{ret.get('method')}` (SAR={ret.get('is_sar')})",
            f"- MIGRES backend: `{meta.get('migres', {}).get('backend')}` (filter: `{meta.get('migres', {}).get('reranker', 'n/a')}`)",
            f"- LLM: `{meta.get('llm_provider')}` / `{meta.get('llm_model')}`", f"- dataset: `{meta.get('dataset', {}).get('name')}` split=`{meta.get('dataset', {}).get('split')}`",
            f"- corpus: `{meta.get('corpus', {}).get('path')}` n_docs={meta.get('corpus', {}).get('n_docs')}",
            "",
        ]
        if ret.get("is_sar"):
            s = meta.get("sar", {})
            L += [f"SAR weights: alpha={s.get('alpha')} beta={s.get('beta')} gamma={s.get('gamma')} K={s.get('k')} weighting={s.get('weighting')}", ""]
        if meta.get("warnings"):
            L += ["## Warnings", ""] + [f"- {w}" for w in meta["warnings"]] + [""]
        traces_by_q = self._trace_rows_by_question()
        for r in results:
            traces = traces_by_q.get(str(r.get("question_id", "")), [])
            L += [f"## Question `{r['question_id']}`", "", f"**Q:** {r['question']}", "",
                  f"status={r['status']} answer=`{r.get('final_answer')}` gold=`{r.get('gold_answer')}` time={r.get('elapsed_seconds',0):.3f}s"]
            for t in traces:
                L += ["", f"### Iteration {t['iteration']} — `{t.get('retriever')}` / `{t.get('score_name')}`",
                      f"gaps={t.get('gaps', []) or '∅'}; evidence={t.get('evidence', []) or '∅'}; time={t.get('total_seconds',0)*1000:.1f}ms", "",
                      "| rank | doc_id | title | score | R | C | N | filter |", "|---:|---|---|---:|---:|---:|---:|:---:|"]
                kept = set(t.get("migres_state_after", {}).get("filter_kept_doc_ids", []))
                for d in t.get("documents", []):
                    L.append(f"| {d.get('rank')} | {d.get('doc_id')} | {d.get('title','')} | {float(d.get('retrieval_score', d.get('score',0))):.4f} | "
                             f"{float(d.get('relevance',0)):.4f} | {float(d.get('gap_coverage',0)):.4f} | {float(d.get('novelty',0)):.4f} | "
                             f"{d.get('doc_id') in kept if 'filter_kept_doc_ids' in t.get('migres_state_after', {}) else 'n/a'} |")
        p = self.dir / "report.md"; atomic_write_text(p, "\n".join(L)); return p
