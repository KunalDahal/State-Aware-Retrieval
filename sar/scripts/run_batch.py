#!/usr/bin/env python
"""Run the configured dataset (or --limit N questions) through MIGRES+SAR. Resumable."""
from _common import base_parser, load

from sar_migres.pipeline import SARMigresPipeline


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None, help="reuse a run id together with --resume to continue an interrupted run")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = load(a)
    pipe = SARMigresPipeline(cfg, run_id=a.run_id).prepare()
    from sar_migres.datasets import load_questions
    items = load_questions(cfg.data.dataset, cfg.data.dataset_path, a.limit or cfg.data.limit, split=cfg.data.split)
    res = pipe.run_batch(items, resume=a.resume or cfg.run.resume)
    print(f"finished {len(res)} question(s) this session; outputs in {pipe.run_dir}")


if __name__ == "__main__":
    main()
