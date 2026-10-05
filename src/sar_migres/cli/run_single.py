from __future__ import annotations

from sar_migres.cli.common import base_parser, load
from sar_migres.experiments.runner import SARMigresPipeline
from sar_migres.datasets.loader import load_questions

def main():
    ap=base_parser(__doc__); ap.add_argument("--question"); ap.add_argument("--id",default="adhoc"); a=ap.parse_args(); cfg=load(a); pipe=SARMigresPipeline(cfg).prepare(); item={"id":a.id,"question":a.question} if a.question else load_questions(cfg.data.dataset,cfg.data.dataset_path,1,split=cfg.data.split)[0]
    r=pipe.run_single(item); print(f"\nQ: {r.question}\nstatus={r.status} answer={r.final_answer!r} gold={r.gold_answer!r}")
    for t in r.traces:
        print(f"\n-- iteration {t.iteration} gaps={t.gaps} evidence={t.evidence}")
        for d in t.documents: print(f"   #{d['rank']} {d['doc_id']:<10} R={d['relevance']:.3f} C={d['gap_coverage']:.3f} N={d['novelty']:.3f} S={d['score']:.3f}  {d['title']}")
    print(f"\nrun dir: {pipe.run_dir}")

if __name__ == "__main__": main()
