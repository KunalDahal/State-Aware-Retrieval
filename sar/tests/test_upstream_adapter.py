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


def test_baseline_provider_uses_current_migres_queries_not_one_shot_question():
    from sar_migres.migres_adapter import RetrieverSearchProvider

    class R:
        name = "bm25"
        score_name = "bm25_score"
        is_sar = False
        def __init__(self):
            self.corpus = type("C", (), {"get": lambda self, did: type("D", (), {"title": did, "text": "text"})()})()
            self.calls = []
            self.stats = type("S", (), {"n_docs_scored":0,"n_similarity_calcs":0,"retrieval_seconds":0.0})()
        def retrieve_many(self, queries, k):
            self.calls.append(list(queries))
            return [("doc-gap", 2.0)]

    r = R()
    p = RetrieverSearchProvider(r, "run", lambda q: "q1", gap_source="queries", k=1)
    p.retrieve_for("original?", ["original?"], [], "")
    p.retrieve_for("original?", ["who founded it?"], ["evidence"], "missing")
    assert r.calls == [["original?"], ["who founded it?"]]
    assert p.traces["original?"][1].documents[0]["doc_id"] == "doc-gap"


def test_sar_sync_callback_runs_after_each_completed_iteration():
    from sar_migres.embeddings import HashingEmbedder
    from sar_migres.corpus import Corpus
    from sar_migres.schemas import Document
    from sar_migres.scoring import SARWeights
    from sar_migres.retrieval import StateAwareRetriever, StaticWeights
    from sar_migres.vector_store import LocalVectorStore, sync_corpus_embeddings
    from sar_migres.migres_adapter import RetrieverSearchProvider

    emb = HashingEmbedder(32)
    corpus = Corpus.from_documents([Document("a", "alpha", "A"), Document("b", "beta", "B")])
    store = LocalVectorStore("/tmp/sar-sync-test", emb.dim, emb.name)
    store.clear()
    sync_corpus_embeddings(store, corpus, emb)
    retriever = StateAwareRetriever(store, corpus, emb, StaticWeights(SARWeights(1, 0, 0)), k=1)
    calls = []
    provider = RetrieverSearchProvider(retriever, "r", lambda q: "q1", k=1, sync_callback=lambda: calls.append("sync"))
    provider.retrieve_for("q", ["q"], [], "")
    provider.retrieve_for("q", ["gap"], ["evidence"], "missing")
    provider.finalize("q", ["evidence"], "missing")
    assert calls == ["sync", "sync"]
