"""Machine-readable (JSON/JSONL/CSV) and human-readable (Markdown) run outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import QuestionResult
from .utils import atomic_write_text, dumps, utc_now_iso

TRACE_COLUMNS = ["run_id", "question_id", "question", "iteration", "gaps", "evidence", "rank", "doc_id", "title",
                 "snippet", "R_relevance", "C_gap_coverage", "N_novelty", "alpha", "beta", "gamma", "sar_score", "K",
                 "n_docs_scored", "n_similarity_calcs", "embed_seconds", "score_seconds", "iteration_seconds",
                 "passed_migres_filter", "evidence_added_after"]
SUMMARY_COLUMNS = ["run_id", "question_id", "question", "status", "backend", "n_sar_iterations", "final_answer",
                   "gold_answer", "total_retrieval_seconds", "elapsed_seconds", "error"]


class RunWriter:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)

    # -- small json files ------------------------------------------------------------------
    def write_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        atomic_write_text(p, json.dumps(obj, default=lambda o: o.to_dict() if hasattr(o, "to_dict") else repr(o),
                                        indent=2, ensure_ascii=False))
        return p

    def read_json(self, name: str) -> Any:
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    # -- incremental JSONL (crash-safe: one flushed line per finished question) -------------------
    def append_result(self, r: QuestionResult) -> None:
        with open(self.dir / "results.jsonl", "a", encoding="utf-8") as fh:
            fh.write(dumps(r.to_dict()) + "\n")
            fh.flush()
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
                    continue  # partial last line of an interrupted run -> question will be redone
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

    # -- tables ---------------------------------------------------------------------------------
    def write_tables(self, results: Iterable[dict[str, Any]], tables_dir: str | Path | None = None) -> dict[str, Path]:
        results = list(results)
        trace_rows, summary_rows = [], []
        for r in results:
            traces = r.get("traces", [])
            for t in traces:
                after = t.get("migres_state_after", {})
                kept = set(after.get("filter_kept_doc_ids", []))
                for d in t["documents"]:
                    trace_rows.append({
                        "run_id": r["run_id"], "question_id": r["question_id"], "question": r["question"],
                        "iteration": t["iteration"], "gaps": json.dumps(t["gaps"], ensure_ascii=False),
                        "evidence": json.dumps(t["evidence"], ensure_ascii=False), "rank": d["rank"],
                        "doc_id": d["doc_id"], "title": d["title"], "snippet": d["text"],
                        "R_relevance": d["relevance"], "C_gap_coverage": d["gap_coverage"], "N_novelty": d["novelty"],
                        "alpha": t["alpha"], "beta": t["beta"], "gamma": t["gamma"], "sar_score": d["score"], "K": t["k"],
                        "n_docs_scored": t["n_docs_scored"], "n_similarity_calcs": t["n_similarity_calcs"],
                        "embed_seconds": t["embed_seconds"], "score_seconds": t["score_seconds"],
                        "iteration_seconds": t["total_seconds"],
                        "passed_migres_filter": (d["doc_id"] in kept) if "filter_kept_doc_ids" in after else "",
                        "evidence_added_after": json.dumps(after.get("evidence_added", []), ensure_ascii=False)})
            summary_rows.append({
                "run_id": r["run_id"], "question_id": r["question_id"], "question": r["question"], "status": r["status"],
                "backend": r["backend"], "n_sar_iterations": len(traces), "final_answer": r.get("final_answer"),
                "gold_answer": json.dumps(r.get("gold_answer"), ensure_ascii=False),
                "total_retrieval_seconds": round(sum(t["total_seconds"] for t in traces), 6),
                "elapsed_seconds": round(r.get("elapsed_seconds", 0.0), 6), "error": r.get("error") or ""})
        out = {}
        dirs = [self.dir] + ([Path(tables_dir)] if tables_dir else [])
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            prefix = "" if d == self.dir else f"{results[0]['run_id']}_" if results else ""
            for name, cols, rows in (("traces.csv", TRACE_COLUMNS, trace_rows), ("summary.csv", SUMMARY_COLUMNS, summary_rows)):
                p = d / f"{prefix}{name}"
                with open(p, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=cols)
                    w.writeheader()
                    w.writerows(rows)
                out[f"{prefix}{name}"] = p
        return out

    # -- human readable report -----------------------------------------------------------------------
    def write_report(self, meta: dict[str, Any], results: Iterable[dict[str, Any]]) -> Path:
        results = list(results)
        L = [f"# SAR-MIGRES run report - `{meta.get('run_id')}`", "",
             "> Development/design-stage output. It documents what was executed; it contains **no benchmark "
             "evaluation** and no claim about SAR's effectiveness.", "",
             "## Run", "",
             f"- started: {meta.get('started')}   finished: {meta.get('finished')}   status: **{meta.get('status')}**",
             f"- MIGRES backend: `{meta.get('migres', {}).get('backend')}` (reranker/filter: `{meta.get('migres', {}).get('reranker', 'n/a')}`)",
             f"- LLM: provider=`{meta.get('llm_provider')}` model=`{meta.get('llm_model')}`",
             f"- embedder: `{meta.get('embedder')}` (semantic: {meta.get('embedder_is_semantic')}); vector store: `{meta.get('vector_store')}`",
             f"- corpus: `{meta.get('corpus', {}).get('path')}`  n_docs={meta.get('corpus', {}).get('n_docs')}  version=`{str(meta.get('corpus', {}).get('version'))[:16]}…`",
             f"- SAR: alpha={meta.get('sar', {}).get('alpha')} beta={meta.get('sar', {}).get('beta')} gamma={meta.get('sar', {}).get('gamma')} K={meta.get('sar', {}).get('k')} (weights are development placeholders unless you set them)",
             f"- seed: {meta.get('seed')}", ""]
        if meta.get("warnings"):
            L += ["## Warnings", ""] + [f"- {w}" for w in meta["warnings"]] + [""]
        for r in results:
            L += [f"## Question `{r['question_id']}`", "", f"**Q:** {r['question']}", "",
                  f"- status: {r['status']}   final answer: `{r.get('final_answer')}`   gold: `{r.get('gold_answer')}`   time: {r.get('elapsed_seconds', 0):.3f}s"]
            if r.get("error"):
                L.append(f"- error: {r['error']}")
            for t in r.get("traces", []):
                after = t.get("migres_state_after", {})
                L += ["", f"### Iteration {t['iteration']}  (alpha={t['alpha']:.2f}, beta={t['beta']:.2f}, gamma={t['gamma']:.2f}, K={t['k']}, "
                      f"{t['total_seconds'] * 1000:.1f} ms, {t['n_docs_scored']} docs scored)", "",
                      f"- gaps G_t: {t['gaps'] or '∅'}", f"- evidence E_t: {t['evidence'] or '∅'}",
                      f"- evidence added by MIGRES afterwards: {after.get('evidence_added', 'n/a')}", "",
                      "| rank | doc_id | title | R | C | N | S_t | passed filter |", "|---|---|---|---|---|---|---|---|"]
                kept = set(after.get("filter_kept_doc_ids", []))
                for d in t["documents"]:
                    L.append(f"| {d['rank']} | {d['doc_id']} | {d['title']} | {d['relevance']:.4f} | {d['gap_coverage']:.4f} | "
                             f"{d['novelty']:.4f} | {d['score']:.4f} | {(d['doc_id'] in kept) if 'filter_kept_doc_ids' in after else 'n/a'} |")
            L.append("")
        p = self.dir / "report.md"
        atomic_write_text(p, "\n".join(L))
        return p
