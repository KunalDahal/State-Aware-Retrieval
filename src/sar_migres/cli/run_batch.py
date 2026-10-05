from __future__ import annotations

from sar_migres.cli.common import base_parser, load
from sar_migres.experiments.runner import SARMigresPipeline
from sar_migres.datasets.loader import load_questions

def main():
    ap=base_parser(__doc__); ap.add_argument("--limit",type=int,default=None); ap.add_argument("--run-id",default=None); ap.add_argument("--resume",action="store_true"); a=ap.parse_args(); cfg=load(a)
    pipe=SARMigresPipeline(cfg,run_id=a.run_id).prepare(); items=load_questions(cfg.data.dataset,cfg.data.dataset_path,a.limit or cfg.data.limit,split=cfg.data.split); res=pipe.run_batch(items,resume=a.resume or cfg.run.resume)
    print(f"finished {len(res)} question(s) this session; outputs in {pipe.run_dir}")

if __name__ == "__main__": main()
