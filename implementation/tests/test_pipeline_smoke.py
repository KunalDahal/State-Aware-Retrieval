"""End-to-end smoke tests with mocked MIGRES / LLM / vector store. No credentials, no network."""
import json
from pathlib import Path

import numpy as np
import pytest

from sar_migres.config import load_config
from sar_migres.corpus import Corpus
from sar_migres.embeddings import HashingEmbedder
from sar_migres.llm import DeterministicOfflineLLM, ScriptedLLM
from sar_migres.migres_adapter import MigresBackend, SARSearchProvider, derive_gaps
from sar_migres.pipeline import SARMigresPipeline
from sar_migres.schemas import Document, QuestionResult
from sar_migres.vector_store import LocalVectorStore

ROOT = Path(__file__).resolve().parents[1]


def cfg_for(tmp_path, **over):
    base = {"run": {"output_dir": str(tmp_path / "out"), "cache_root": str(tmp_path / "cache")}}
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return load_config(ROOT / "configs" / "local.yaml", overrides=base)


class MockBackend(MigresBackend):
    """Mocked MIGRES: two SAR calls with an evidence/gap update between them."""
    name = "mock"

    def run_question(self, item, provider: SARSearchProvider, llm, run_id, workdir):
        q = item["question"]
        provider.retrieve_for(q, [q], [], "")
        provider.retrieve_for(q, ["who founded it?"], ["Evidence one."], "founder unknown")
        provider.finalize(q, ["Evidence one.", "Evidence two."], "")
        return QuestionResult(run_id, item["id"], q, item.get("answer"), "mock-answer", provider.traces[q], "ok", backend="mock")


FAKE_OPENAI = "sk-" + "fixture-not-a-real-key-000111"      # assembled at runtime: fake, never a real credential
FAKE_PINECONE = "pc" + "sk_" + "fixture_not_a_real_key_000111"


def test_full_local_pipeline_with_mocked_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI)
    monkeypatch.setenv("PINECONE_API_KEY", FAKE_PINECONE)
    cfg = cfg_for(tmp_path)
    pipe = SARMigresPipeline(cfg, backend=MockBackend(), llm=ScriptedLLM(lambda p: "x")).prepare()
    items = [json.loads(l) for l in (ROOT / "data/sample/questions.jsonl").read_text().splitlines()]
    res = pipe.run_batch(items)
    assert [r.status for r in res] == ["ok"] * 3
    run = pipe.run_dir
    for name in ("results.jsonl", "traces.jsonl", "traces.csv", "summary.csv", "report.md", "metadata.json", "config_snapshot.json"):
        assert (run / name).exists(), name
    meta = json.loads((run / "metadata.json").read_text())
    assert meta["status"] == "completed" and meta["corpus"]["version"] and meta["embedder"].startswith("hashing")
    assert any("LEXICAL HASHING" in w for w in meta["warnings"])
    t0 = json.loads((run / "traces.jsonl").read_text().splitlines()[0])
    for key in ("question", "iteration", "gaps", "evidence", "alpha", "beta", "gamma", "k", "documents", "total_seconds"):
        assert key in t0
    d0 = t0["documents"][0]
    for key in ("doc_id", "text", "relevance", "gap_coverage", "novelty", "score"):
        assert key in d0
    assert t0["gaps"] == [] and t0["evidence"] == []  # iteration 0: G_0 = {} and E_0 = {}
    t1 = json.loads((run / "traces.jsonl").read_text().splitlines()[1])
    assert t1["gaps"] == ["who founded it?"] and t1["evidence"] == ["Evidence one."]
    assert t0["migres_state_after"]["evidence_added"] == ["Evidence one."]
    for f in run.rglob("*"):  # no secret VALUE may appear in any output file (env var *names* are fine)
        if f.is_file():
            txt = f.read_text(errors="ignore")
            assert FAKE_OPENAI not in txt and FAKE_PINECONE not in txt, f


def test_offline_standin_backend_end_to_end(tmp_path):
    cfg = cfg_for(tmp_path)
    pipe = SARMigresPipeline(cfg).prepare()
    items = [json.loads(l) for l in (ROOT / "data/sample/questions.jsonl").read_text().splitlines()]
    res = pipe.run_batch(items)
    assert len(res) == 3 and all(r.status in ("ok", "no_answer") for r in res)
    assert all(len(r.traces) >= 1 for r in res)
    assert (pipe.run_dir / "report.md").read_text().startswith("# SAR-MIGRES run report")


def test_resume_skips_finished_questions(tmp_path):
    cfg = cfg_for(tmp_path)
    pipe = SARMigresPipeline(cfg, backend=MockBackend(), llm=ScriptedLLM(lambda p: "x"), run_id="fixed_run").prepare()
    items = [{"id": f"q{i}", "question": f"Question number {i} about capitals?"} for i in range(3)]
    pipe.run_batch(items[:2])
    pipe2 = SARMigresPipeline(cfg, backend=MockBackend(), llm=ScriptedLLM(lambda p: "x"), run_id="fixed_run").prepare()
    res = pipe2.run_batch(items, resume=True)
    assert [r.question_id for r in res] == ["q2"]
    assert len((pipe2.run_dir / "results.jsonl").read_text().splitlines()) == 3


def test_interrupted_run_keeps_partial_results(tmp_path):
    class Boom(MockBackend):
        def run_question(self, item, provider, llm, run_id, workdir):
            if item["id"] == "q1":
                raise KeyboardInterrupt
            return super().run_question(item, provider, llm, run_id, workdir)

    cfg = cfg_for(tmp_path)
    pipe = SARMigresPipeline(cfg, backend=Boom(), llm=ScriptedLLM(lambda p: "x")).prepare()
    with pytest.raises(KeyboardInterrupt):
        pipe.run_batch([{"id": f"q{i}", "question": f"Question number {i} about capitals?"} for i in range(3)])
    assert json.loads((pipe.run_dir / "metadata.json").read_text())["status"] == "interrupted"
    assert len(pipe.writer.load_results()) == 1


def test_mocked_vector_store_and_embedder_injection(tmp_path):
    emb = HashingEmbedder(dim=32)
    corpus = Corpus.from_documents([Document(f"d{i}", f"document about topic {i} and things", f"T{i}") for i in range(6)])
    store = LocalVectorStore(tmp_path / "st", emb.dim, emb.name)
    pipe = SARMigresPipeline(cfg_for(tmp_path), corpus=corpus, embedder=emb, store=store, backend=MockBackend(),
                             llm=ScriptedLLM(lambda p: "x")).prepare()
    r = pipe.run_question({"id": "a", "question": "what is topic 3"})
    assert r.traces[0].n_docs_scored == 6


def test_dynamic_weighting_refused_as_future(tmp_path):
    from sar_migres.errors import ConfigError
    with pytest.raises(ConfigError):
        SARMigresPipeline(cfg_for(tmp_path, sar={"weighting": "dynamic"}))


def test_derive_gaps_mapping():
    assert derive_gaps("q", ["q", "a"], "miss", "queries", 0) == []  # G_0 empty
    assert derive_gaps("q", ["q", "a", "a"], "miss", "queries", 1) == ["a"]
    assert derive_gaps("q", ["a"], "miss", "both", 2) == ["a", "miss"]
    assert derive_gaps("q", ["a"], "miss", "missing_info", 2) == ["miss"]


def test_run_single_writes_full_artefacts(tmp_path):
    pipe = SARMigresPipeline(cfg_for(tmp_path), backend=MockBackend(), llm=ScriptedLLM(lambda p: "x")).prepare()
    r = pipe.run_single({"id": "one", "question": "What is the capital of France?"})
    assert r.status == "ok"
    for name in ("config_snapshot.json", "metadata.json", "results.jsonl", "traces.jsonl", "traces.csv", "summary.csv", "report.md"):
        assert (pipe.run_dir / name).exists(), name
    assert json.loads((pipe.run_dir / "metadata.json").read_text())["status"] == "completed"
