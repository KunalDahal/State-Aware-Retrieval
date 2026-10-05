from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from sar_migres.cli.common import ROOT, base_parser, load

def check(label, ok, detail=""):
    print(f"[{'OK ' if ok else 'MISSING'}] {label} {detail}"); return ok

def main():
    ap=base_parser(__doc__); a=ap.parse_args(); load(a)
    hard=[check("python >= 3.10",sys.version_info>=(3,10),sys.version.split()[0]),check("required package numpy",importlib.util.find_spec("numpy") is not None),check("required package yaml",importlib.util.find_spec("yaml") is not None)]
    from sar_migres.utils.config import load_config
    from sar_migres.datasets.loader import load_questions
    from sar_migres.experiments.runner import SARMigresPipeline
    sample=ROOT/"data"/"sample"
    with tempfile.TemporaryDirectory() as td:
        cfg=load_config(ROOT/"configs"/"local.yaml",overrides={"run":{"output_dir":td,"cache_root":td,"log_level":"WARNING"}})
        pipe=SARMigresPipeline(cfg).prepare(); res=pipe.run_batch(load_questions("sample",cfg.data.dataset_path,1,split=cfg.data.split)); ok=len(res)==1 and len(res[0].traces)>=1 and (pipe.run_dir/"report.md").exists()
    hard.append(check("offline smoke test",ok)); print("\nRESULT:","offline path OK" if all(hard) else "PROBLEMS FOUND"); raise SystemExit(0 if all(hard) else 1)

if __name__ == "__main__": main()
