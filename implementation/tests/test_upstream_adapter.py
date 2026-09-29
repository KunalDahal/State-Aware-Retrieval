import json
import os
import textwrap
from pathlib import Path

import pytest

from sar_migres import migres_adapter as ma
from sar_migres.config import load_config
from sar_migres.errors import UpstreamMigresError
from sar_migres.pipeline import SARMigresPipeline

ROOT = Path(__file__).resolve().parents[1]


def test_safe_eval_never_executes_code():
    assert ma._safe_eval("{'a': [1, 2]}") == {"a": [1, 2]}
    with pytest.raises((ValueError, SyntaxError)):
        ma._safe_eval("__import__('os').getcwd()")


def test_ast_extraction_and_argparse_slicing():
    src = textwrap.dedent('''
        import argparse
        def helper(x):
            return eval(x)
        class RetGBatched:
            pass
        parser = argparse.ArgumentParser(description="")
        BROKEN = 'unterminated
    ''')
    head = ma._slice_before_argparse(src)
    defs = ma._extract_defs(head, {"helper", "RetGBatched"})
    assert set(defs) == {"helper", "RetGBatched"}
    with pytest.raises(UpstreamMigresError):
        ma._extract_defs(head, {"nope"})
    with pytest.raises(UpstreamMigresError):
        ma._slice_before_argparse("x = 1")


def test_missing_upstream_repo_gives_actionable_error(tmp_path):
    with pytest.raises(UpstreamMigresError, match="git clone"):
        ma.UpstreamCode(tmp_path, "offline_standin", "x")


REPO = os.environ.get("MIGRES_REPO_DIR", "")


@pytest.mark.skipif(not (REPO and (Path(REPO) / "code" / "pipeline.py").exists()),
                    reason="set MIGRES_REPO_DIR to a clone of AdelWang/MIGRES to run the upstream-code integration test")
def test_upstream_retgbatched_runs_with_sar_replacing_search(tmp_path):
    cfg = load_config(ROOT / "configs" / "local.yaml", overrides={
        "migres": {"backend": "upstream", "repo_dir": REPO, "max_iter": 2},
        "run": {"output_dir": str(tmp_path / "o"), "cache_root": str(tmp_path / "c")}})
    pipe = SARMigresPipeline(cfg).prepare()
    assert pipe.backend.name == "upstream"
    # the class used is a subclass of the unmodified upstream RetGBatched, overriding only search/askGPT
    cls = pipe.backend.cls
    assert cls.__mro__[1].__name__ == "RetGBatched"
    assert {"search", "askGPT"} <= set(vars(cls))
    item = json.loads((ROOT / "data/sample/questions.jsonl").read_text().splitlines()[0])
    r = pipe.run_question(item)
    assert r.status in ("ok", "no_answer") and len(r.traces) >= 1
    assert r.traces[0].gaps == [] and r.traces[0].evidence == []          # first upstream retrieval: G_0 = {}, E_0 = {}
    assert all(d["score"] >= 0 for t in r.traces for d in t.documents)     # documents come from SAR
    assert "history" in r.migres_record and "retrieval_trace" in r.migres_record or r.status == "no_answer"
