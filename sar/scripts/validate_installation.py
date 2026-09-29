#!/usr/bin/env python
"""Check the environment. Exit code 0 = the OFFLINE smoke path works. Optional components are reported, not required."""
import importlib
import importlib.util
import os
import sys
import tempfile
from _common import ROOT, base_parser, load

from sar_migres.config import load_config


def check(label, ok, detail=""):
    print(f"[{'OK ' if ok else 'MISSING'}] {label} {detail}")
    return ok


def main():
    ap = base_parser(__doc__)
    a = ap.parse_args()
    load(a)
    hard = []
    hard.append(check("python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    for m in ("numpy", "yaml"):
        hard.append(check(f"required package {m}", importlib.util.find_spec(m) is not None))
    print("\noptional packages (needed only for real runs):")
    for m, why in (("sentence_transformers", "real embeddings"), ("torch", "embeddings / MIGRES filter"), ("pinecone", "Pinecone store"),
                   ("google.genai", "Gemini"), ("openai", "OpenAI"), ("transformers", "MIGRES filter"), ("nltk", "MIGRES filter"),
                   ("bayes_opt", "October: Bayesian optimisation"), ("pytest", "tests")):
        try:
            ok = importlib.util.find_spec(m) is not None
        except ModuleNotFoundError:
            ok = False
        check(f"{m:<22}", ok, f"({why})")
    print("\nenvironment variables (names only; values are never printed):")
    for v in ("PINECONE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "MIGRES_REPO_DIR"):
        check(f"{v:<18}", bool(os.environ.get(v)))
    repo = os.environ.get("MIGRES_REPO_DIR", "")
    check("official MIGRES clone", bool(repo) and os.path.exists(os.path.join(repo, "code", "pipeline.py")), f"({repo or 'MIGRES_REPO_DIR unset'})")
    print("\noffline smoke run (hashing embedder + fake LLM + stand-in loop):")
    from sar_migres.datasets import load_questions
    from sar_migres.pipeline import SARMigresPipeline
    with tempfile.TemporaryDirectory() as td:
        cfg = load_config(ROOT / "configs" / "local.yaml", overrides={"run": {"output_dir": td, "cache_root": td, "log_level": "WARNING"}})
        os.chdir(ROOT)
        pipe = SARMigresPipeline(cfg).prepare()
        res = pipe.run_batch(load_questions("sample", cfg.data.dataset_path, 1))
        ok = len(res) == 1 and len(res[0].traces) >= 1 and (pipe.run_dir / "report.md").exists()
    hard.append(check("offline smoke test", ok))
    print("\nRESULT:", "offline path OK" if all(hard) else "PROBLEMS FOUND")
    sys.exit(0 if all(hard) else 1)


if __name__ == "__main__":
    main()
