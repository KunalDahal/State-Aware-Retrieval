import json
import os

import numpy as np
import pytest

from sar_migres import baselines, evaluation
from sar_migres.config import load_config, load_dotenv_file
from sar_migres.corpus import Corpus
from sar_migres.datasets import contexts_to_documents, load_questions, load_rgb
from sar_migres.embeddings import HashingEmbedder
from sar_migres.errors import (ConfigError, DatasetNotFoundError, LLMResponseError, MissingCredentialError,
                               VectorStoreConnectionError, retry_call)
from sar_migres.llm import DeterministicOfflineLLM, GeminiClient, OpenAIClient, ScriptedLLM, build_llm, parse_dict_reply
from sar_migres.logging_utils import RedactingFilter
from sar_migres.optimization import optimize_weights
from sar_migres.schemas import Document
from sar_migres.vector_store import PineconeVectorStore, sync_corpus_embeddings


# ---------------- Pinecone store with an injected in-memory fake index ----------------
class FakeIndex:
    def __init__(self):
        self.data = {}
        self.fail_upserts = 0

    def upsert(self, vectors, namespace=""):
        if self.fail_upserts:
            self.fail_upserts -= 1
            raise RuntimeError("429 rate limit")
        for v in vectors:
            self.data[v["id"]] = v

    def delete(self, ids=None, namespace="", delete_all=False):
        if delete_all:
            self.data.clear()
        for i in ids or []:
            self.data.pop(i, None)

    def describe_index_stats(self):
        return {"namespaces": {"": {"vector_count": len(self.data)}} if self.data else {}}

    def list(self, namespace=""):
        yield list(self.data)

    def fetch(self, ids, namespace=""):
        return {"vectors": {i: {"values": self.data[i]["values"], "metadata": self.data[i]["metadata"]} for i in ids if i in self.data}}


DOCS = [Document(f"d{i}", f"text of document {i} about subject {i}", f"Title {i}") for i in range(5)]


def test_pinecone_store_sync_and_reload(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    emb, idx = HashingEmbedder(dim=32), FakeIndex()
    store = PineconeVectorStore("idx", emb.dim, emb.name, index=idx, mirror_dir=tmp_path / "m")
    corpus = Corpus.from_documents(DOCS)
    idx.fail_upserts = 1  # first upsert hits a rate limit and is retried
    rep = sync_corpus_embeddings(store, corpus, emb)
    assert rep.n_embedded == 5 and len(idx.data) == 5 and len(store) == 5
    assert idx.data["d0"]["metadata"]["embedder"] == emb.name
    # a fresh process: mirror restored from disk without re-embedding
    store2 = PineconeVectorStore("idx", emb.dim, emb.name, index=idx, mirror_dir=tmp_path / "m")
    store2.load_mirror()
    assert sync_corpus_embeddings(store2, corpus, emb).n_embedded == 0
    # mirror deleted -> rebuilt from Pinecone fetch, still identical
    import shutil; shutil.rmtree(tmp_path / "m")
    store3 = PineconeVectorStore("idx", emb.dim, emb.name, index=idx, mirror_dir=tmp_path / "m")
    store3.load_mirror()
    np.testing.assert_allclose(store3.get_vectors(["d3"]), store.get_vectors(["d3"]), atol=1e-6)
    # update propagates to Pinecone
    corpus.add_or_update([Document("d3", "completely new content", "Title 3")])
    assert sync_corpus_embeddings(store3, corpus, emb).n_changed == 1
    assert idx.data["d3"]["metadata"]["content_hash"] == corpus.content_hashes["d3"]


def test_pinecone_missing_key_and_connection_failure(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    s = PineconeVectorStore("idx", 8, "m")
    with pytest.raises(MissingCredentialError, match="PINECONE_API_KEY"):
        s.connect()

    class Down(FakeIndex):
        def describe_index_stats(self):
            raise ConnectionError("connection refused")

    monkeypatch.setattr("time.sleep", lambda s: None)
    s2 = PineconeVectorStore("idx", 8, "m", index=Down(), retry_attempts=2)
    with pytest.raises(VectorStoreConnectionError):
        s2.load_mirror()


# ---------------- LLM layer ----------------
def test_missing_api_keys_raise_clear_errors(monkeypatch):
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    cfg = load_config("configs/local.yaml", overrides={"llm": {"provider": "gemini"}})
    with pytest.raises(ConfigError, match="LLM model name not configured"):
        build_llm(cfg)
    cfg = load_config("configs/local.yaml", overrides={"llm": {"provider": "openai", "model": "some-model"}})
    with pytest.raises((MissingCredentialError, Exception)) as e:
        build_llm(cfg)
    assert "OPENAI_API_KEY" in str(e.value) or "openai" in str(e.value).lower()


def test_openai_client_with_injected_fake_and_rate_limit_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    class Msg:  # minimal shape of the SDK response
        def __init__(self, c): self.message = type("M", (), {"content": c})()

    class Chat:
        def create(self, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Error 429: rate limit exceeded")
            return type("R", (), {"choices": [Msg("hello")] * kw["n"]})()

    fake = type("C", (), {"chat": type("X", (), {"completions": Chat()})()})()
    llm = OpenAIClient("any-model", client=fake, max_retries=5)
    assert llm.generate("hi", n=2) == ["hello", "hello"] and calls["n"] == 3


def test_empty_llm_response_is_an_error_not_silence():
    with pytest.raises(LLMResponseError):
        ScriptedLLM(lambda p: "  ").generate("x")


def test_gemini_client_with_injected_fake():
    pytest.importorskip("google.genai")
    fake = type("C", (), {"models": type("M", (), {"generate_content": lambda self, **kw: type("R", (), {"text": "ok"})()})()})()
    assert GeminiClient("m", client=fake).generate("hi") == ["ok"]


def test_retry_gives_up_on_non_transient_immediately():
    n = {"c": 0}
    def f():
        n["c"] += 1
        raise ValueError("bad request")
    with pytest.raises(ValueError):
        retry_call(f, sleep=lambda s: None)
    assert n["c"] == 1


def test_parse_dict_reply_variants_and_no_code_execution():
    assert parse_dict_reply('{"answer": {"text": "x"}}')["answer"]["text"] == "x"
    assert parse_dict_reply('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_dict_reply("Sure! {'a': 2} done") == {"a": 2}
    with pytest.raises(LLMResponseError):
        parse_dict_reply("__import__('os').system('echo pwned')")
    with pytest.raises(LLMResponseError):
        parse_dict_reply("no dict here")


def test_offline_llm_rejects_unknown_prompt():
    with pytest.raises(LLMResponseError):
        DeterministicOfflineLLM().generate("some unrelated prompt")


# ---------------- logging / secrets ----------------
def test_redaction(monkeypatch, caplog):
    fake_openai = "sk-" + "fixturevalue1234"          # fake values assembled at runtime
    fake_google = "AI" + "za" + "SyFixture0123456789012345"
    monkeypatch.setenv("OPENAI_API_KEY", fake_openai)
    import logging
    rec = logging.LogRecord("x", logging.INFO, "f", 1, f"key={fake_openai} and {fake_google}", None, None)
    RedactingFilter().filter(rec)
    assert "fixturevalue" not in rec.msg and "Fixture0123" not in rec.msg and "REDACTED" in rec.msg


def test_dotenv_loader_returns_names_only(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# c\nFOO_TEST_KEY=abc\nEMPTY=\n")
    monkeypatch.delenv("FOO_TEST_KEY", raising=False)
    assert load_dotenv_file(p) == ["FOO_TEST_KEY"]
    assert os.environ["FOO_TEST_KEY"] == "abc"


# ---------------- config ----------------
def test_config_validation(tmp_path):
    with pytest.raises(ConfigError):
        load_config("configs/local.yaml", overrides={"sar": {"alpha": 0.9, "beta": 0.9, "gamma": 0.9}})
    with pytest.raises(ConfigError):
        load_config("configs/local.yaml", overrides={"sar": {"typo_key": 1}})
    with pytest.raises(ConfigError):
        load_config("configs/does_not_exist.yaml")


def test_default_config_is_valid_and_uses_pinecone_and_thesis_model():
    cfg = load_config("configs/default.yaml")
    assert cfg.vector_store.type == "pinecone" and cfg.embedding.model_name.endswith("multi-qa-mpnet-base-cos-v1")
    assert cfg.embedding.expected_dim == 768 and cfg.llm.model == ""  # model never silently assumed


# ---------------- datasets ----------------
@pytest.fixture
def upstream_like(tmp_path):
    (tmp_path / "hotpot.json").write_text("\n".join(json.dumps(x) for x in [
        {"_id": "a", "question": "Q1?", "answer": "A1", "context": [["Title X", ["Sent one.", "Sent two."]], ["Title Y", ["Other."]]]},
        {"_id": "b", "question": "Q2?", "answer": "A2", "context": [["Title X", ["Sent one.", "Sent two."]]]}]))
    (tmp_path / "odqa.json").write_text("\n".join(json.dumps(x) for x in [
        {"id": "1", "question": "nq q", "answer": "['x']", "dataset": "nq"}, {"id": "2", "question": "tv q", "answer": "['y']", "dataset": "trivia"}]))
    return tmp_path


def test_dataset_loaders(upstream_like):
    hp = load_questions("hotpotqa", upstream_like / "hotpot.json")
    assert [q["id"] for q in hp] == ["a", "b"] and hp[0]["dataset"] == "hotpotqa"
    assert [q["id"] for q in load_questions("natural_questions", upstream_like / "odqa.json")] == ["1"]
    tv = load_questions("triviaqa", upstream_like / "odqa.json")
    assert tv[0]["answer"] == ["y"]
    docs = list(contexts_to_documents("hotpotqa", hp))
    assert len(docs) == 2 and {d.title for d in docs} == {"Title X", "Title Y"}  # deduplicated across questions


def test_missing_dataset_message(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="data/README.md"):
        load_questions("musique", tmp_path / "nope.json")
    with pytest.raises(ValueError):
        load_questions("squad", tmp_path / "x")


def test_rgb_loader(tmp_path):
    p = tmp_path / "en.json"
    p.write_text(json.dumps({"id": 0, "query": "when?", "answer": [["2022"]], "positive": [], "negative": []}) + "\n")
    assert load_rgb(p)[0]["question"] == "when?"
    with pytest.raises(NotImplementedError):
        evaluation.RGBRobustnessEvaluation()


# ---------------- future modules ----------------
def test_metric_formulas_on_hand_computed_examples():
    ret, rel = ["a", "b", "c", "d"], ["b", "d", "z"]
    assert evaluation.recall_at_k(ret, rel, 4) == pytest.approx(2 / 3)
    assert evaluation.precision_at_k(ret, rel, 4) == 0.5
    assert evaluation.hit_at_k(ret, rel, 1) == 0.0 and evaluation.reciprocal_rank(ret, rel) == 0.5
    assert evaluation.ndcg_at_k(["a", "b"], {"b": 1.0}, 2) == pytest.approx((1 / np.log2(3)) / 1.0)
    cum = evaluation.cumulative_recall([["a", "b"], ["b", "d"]], rel)
    assert cum == pytest.approx([1 / 3, 2 / 3]) and evaluation.marginal_recall_gain(cum) == pytest.approx([1 / 3, 1 / 3])
    assert evaluation.new_document_rate([["a", "b"], ["b", "d"]]) == [1.0, 0.5]
    assert evaluation.retrieval_overlap(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert evaluation.exact_match("The Rush!", ["rush"]) == 1.0 and evaluation.token_f1("big red dog", "red dog") == pytest.approx(0.8)
    assert evaluation.n_similarity_calculations(100, 2, 3) == 600 and evaluation.retrieval_overhead_pct(1.5, 1.0) == 50.0


def test_bm25_and_st_search_baselines_and_stubs(tmp_path):
    corpus = Corpus.from_documents([Document("a", "the cat sat on the mat", "Cat"), Document("b", "stock markets fell sharply", "Finance"),
                                    Document("c", "a dog chased the cat", "Dog")])
    assert baselines.BM25Baseline(corpus).retrieve("stock market", 1)[0][0] == "b"
    from sar_migres.vector_store import LocalVectorStore
    emb = HashingEmbedder(64)
    st = LocalVectorStore(tmp_path / "s", emb.dim, emb.name)
    sync_corpus_embeddings(st, corpus, emb)
    assert baselines.SentenceTransformerSearch(st, emb).retrieve("cat on a mat", 1)[0][0] == "a"
    for cls in (baselines.DPRBaseline, baselines.ContrieverBaseline, baselines.SPLADEBaseline, baselines.ColBERTv2Baseline):
        with pytest.raises(NotImplementedError, match="FUTURE"):
            cls()


def test_bayesian_optimisation_respects_simplex_on_toy_objective():
    pytest.importorskip("bayes_opt")
    seen = []
    def toy(w):  # analytic toy function, NOT a research objective
        seen.append(w.as_tuple())
        return -((w.alpha - 0.2) ** 2 + (w.beta - 0.5) ** 2)
    best, val, hist = optimize_weights(toy, init_points=4, n_iter=6, seed=1)
    assert all(min(t) >= 0 and abs(sum(t) - 1) < 1e-9 for t in seen) and len(seen) == 10
    assert abs(sum(best.as_tuple()) - 1) < 1e-9


def test_ablation_weights_are_valid_simplex_points():
    from sar_migres.scoring import SARWeights
    for name, t in evaluation.AblationStudy.ablation_weights(SARWeights(0.4, 0.3, 0.3)).items():
        SARWeights(*t)


def test_contexts_skip_records_without_context_but_fail_if_none(upstream_like):
    recs = [{"id": "a", "question": "q", "context": None},
            {"id": "b", "question": "q", "context": [["T", ["Sentence one.", "Sentence two."]]]}]
    docs = list(contexts_to_documents("wikihop", recs))
    assert len(docs) == 1 and docs[0].title == "T"
    with pytest.raises(DatasetNotFoundError, match="supply a corpus file"):
        list(contexts_to_documents("wikihop", [recs[0]]))
    with pytest.raises(ValueError):
        list(contexts_to_documents("strategyqa", recs))
