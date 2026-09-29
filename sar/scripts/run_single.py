#!/usr/bin/env python
"""Run ONE question through MIGRES+SAR and print the per-iteration retrieval trace."""
from _common import base_parser, load

from sar_migres.pipeline import SARMigresPipeline


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--question", help="free-text question (default: first question of the configured dataset)")
    ap.add_argument("--id", default="adhoc")
    a = ap.parse_args()
    cfg = load(a)
    pipe = SARMigresPipeline(cfg).prepare()
    if a.question:
        item = {"id": a.id, "question": a.question}
    else:
        from sar_migres.datasets import load_questions
        item = load_questions(cfg.data.dataset, cfg.data.dataset_path, 1)[0]
    r = pipe.run_single(item)
    print(f"\nQ: {r.question}\nstatus={r.status} answer={r.final_answer!r} gold={r.gold_answer!r}")
    for t in r.traces:
        print(f"\n-- iteration {t.iteration}  gaps={t.gaps}  evidence={t.evidence}")
        for d in t.documents:
            print(f"   #{d['rank']} {d['doc_id']:<10} R={d['relevance']:.3f} C={d['gap_coverage']:.3f} N={d['novelty']:.3f} S={d['score']:.3f}  {d['title']}")
    print(f"\nrun dir: {pipe.run_dir}")


if __name__ == "__main__":
    main()
