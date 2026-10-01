import json

import numpy as np
import pytest

from sar_migres.corpus import Corpus
from sar_migres.embeddings import HashingEmbedder
from sar_migres.errors import CorpusFormatError, EmbeddingCacheCorruptionError, StoreIncompatibleError
from sar_migres.retrieval import StateAwareRetriever, StaticWeights
from sar_migres.schemas import Document, RetrievalState
from sar_migres.scoring import SARWeights
from sar_migres.vector_store import LocalVectorStore, _open_local, sync_corpus_embeddings

DOCS = [
    Document("d1", "Paris is the capital of France.", "Paris"),
    Document("d2", "The Eiffel Tower was built in 1889.", "Eiffel Tower"),
    Document("d3", "Berlin is the capital of Germany.", "Berlin"),
    Document("d4", "Photosynthesis converts light into chemical energy.", "Photosynthesis"),
]


def make(tmp_path, docs=DOCS, dim=64):
    emb = HashingEmbedder(dim=dim)
    corpus = Corpus.from_documents(docs)
    store = LocalVectorStore(tmp_path / "s", emb.dim, emb.name)
    sync_corpus_embeddings(store, corpus, emb, batch_size=2)
    return emb, corpus, store


def test_incremental_new_changed_unchanged(tmp_path):
    emb, corpus, store = make(tmp_path)
    calls = []
    orig = emb.encode
    emb.encode = lambda texts, batch_size=None: (calls.append(list(texts)), orig(texts))[1]
    corpus.add_or_update([Document("d5", "Madrid is the capital of Spain.", "Madrid"),  # new
                          Document("d2", "The Eiffel Tower is 330 metres tall.", "Eiffel Tower")])  # changed
    rep = sync_corpus_embeddings(store, corpus, emb)
    assert (rep.n_new, rep.n_changed, rep.n_unchanged, rep.n_embedded) == (1, 1, 3, 2)
    assert sum(len(c) for c in calls) == 2  # unchanged docs were NOT re-embedded
    assert len(store) == 5
    # changed vector really replaced
    np.testing.assert_allclose(store.get_vectors(["d2"])[0], orig([corpus.get("d2").embed_text()])[0], atol=1e-6)
    rep2 = sync_corpus_embeddings(store, corpus, emb)
    assert rep2.n_embedded == 0 and rep2.n_unchanged == 5


def test_removed_and_renamed_documents(tmp_path):
    emb, corpus, store = make(tmp_path)
    corpus.remove(["d3"])
    corpus.add_or_update([Document("d3_new_id", "Berlin is the capital of Germany.", "Berlin")])  # same content, new id
    rep = sync_corpus_embeddings(store, corpus, emb)
    assert rep.n_removed == 1 and rep.n_reused_same_content == 1 and rep.n_embedded == 0
    assert "d3" not in store.ids() and "d3_new_id" in store.ids()


def test_persistence_roundtrip_and_reuse(tmp_path):
    emb, corpus, store = make(tmp_path)
    store2 = LocalVectorStore(tmp_path / "s", emb.dim, emb.name)
    assert store2.ids() == store.ids()
    assert sync_corpus_embeddings(store2, corpus, emb).n_embedded == 0


def test_cache_corruption_detected_and_quarantined(tmp_path):
    emb, corpus, store = make(tmp_path)
    vp = tmp_path / "s" / "vectors.npy"
    raw = bytearray(vp.read_bytes()); raw[-5] ^= 0xFF; vp.write_bytes(bytes(raw))
    with pytest.raises(EmbeddingCacheCorruptionError):
        LocalVectorStore(tmp_path / "s", emb.dim, emb.name)
    with pytest.raises(EmbeddingCacheCorruptionError):
        _open_local(tmp_path / "s", emb.dim, emb.name, on_corruption="raise")
    fresh = _open_local(tmp_path / "s", emb.dim, emb.name, on_corruption="rebuild")
    assert len(fresh) == 0 and list((tmp_path / "s").glob("*.corrupt-*"))
    assert sync_corpus_embeddings(fresh, corpus, emb).n_embedded == len(DOCS)


def test_model_mismatch_refused_unless_rebuild(tmp_path):
    emb, corpus, store = make(tmp_path)
    with pytest.raises(StoreIncompatibleError):
        LocalVectorStore(tmp_path / "s", emb.dim, "some-other-model")
    ok = _open_local(tmp_path / "s", emb.dim, "some-other-model", rebuild=True)
    assert len(ok) == 0


def test_duplicate_documents(tmp_path):
    c = Corpus.from_documents([DOCS[0], DOCS[0]])
    assert len(c) == 1 and c.report.n_duplicate_identical == 1
    with pytest.raises(CorpusFormatError):
        Corpus.from_documents([DOCS[0], Document("d1", "different text entirely", "Paris")])


def test_malformed_records_reported_and_thresholded(tmp_path):
    p = tmp_path / "c.jsonl"
    lines = [json.dumps({"id": i, "title": "t", "text": f"doc number {i}"}) for i in range(200)]
    lines += ['{"id": "bad1"}', "not json at all"]
    p.write_text("\n".join(lines))
    c = Corpus.from_file(p, max_malformed_fraction=0.05)
    assert len(c) == 200 and c.report.n_malformed == 2 and len(c.report.malformed_examples) == 2
    with pytest.raises(CorpusFormatError):
        Corpus.from_file(p, max_malformed_fraction=0.0)


def test_lazy_corpus_matches_eager(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps({"id": d.doc_id, "title": d.title, "text": d.text}) for d in DOCS))
    lazy, eager = Corpus.from_file(p, lazy=True), Corpus.from_file(p, lazy=False)
    assert lazy._docs == {} and lazy.version == eager.version
    assert lazy.get("d3").text == eager.get("d3").text


def test_corpus_version_changes_with_content():
    a = Corpus.from_documents(DOCS)
    b = Corpus.from_documents(DOCS[:3] + [Document("d4", "changed", "Photosynthesis")])
    assert a.version != b.version and a.version == Corpus.from_documents(list(reversed(DOCS))).version


def test_retriever_edge_cases_and_ordering(tmp_path):
    emb, corpus, store = make(tmp_path)
    r = StateAwareRetriever(store, corpus, emb, StaticWeights(SARWeights(0.4, 0.3, 0.3)), k=3, chunk_size=2)
    res = r.retrieve(RetrievalState("What is the capital of France?"), return_all=True)  # empty G and E
    assert len(res.documents) == 3
    assert [d.score for d in res.documents] == sorted([d.score for d in res.documents], reverse=True)
    assert all(d.gap_coverage == 0.0 and d.novelty == 1.0 for d in res.documents)
    assert res.n_docs_scored == 4 and res.n_similarity_calcs == 4
    # top-K equals brute-force argsort of the full score vector (no hidden candidate stage)
    order = np.lexsort((np.arange(4), -res.all_scores))[:3]
    assert [store.ids()[i] for i in order] == [d.doc_id for d in res.documents]
    # chunking must not change the result
    r2 = StateAwareRetriever(store, corpus, emb, StaticWeights(SARWeights(0.4, 0.3, 0.3)), k=3, chunk_size=100)
    assert [d.doc_id for d in r2.retrieve(RetrievalState("What is the capital of France?")).documents] == [d.doc_id for d in res.documents]


def test_state_changes_ranking_and_counts(tmp_path):
    emb, corpus, store = make(tmp_path)
    r = StateAwareRetriever(store, corpus, emb, StaticWeights(SARWeights(0.0, 0.0, 1.0)), k=4)
    st = RetrievalState("capital of France", evidence=["Paris is the capital of France."], gaps=["height of the Eiffel Tower", "Berlin"])
    res = r.retrieve(st, iteration=1)
    assert res.documents[-1].doc_id == "d1"  # pure novelty: the doc identical to the evidence ranks last
    assert res.n_similarity_calcs == 4 * (1 + 2 + 1)


def test_k_larger_than_corpus_and_bad_input(tmp_path):
    emb, corpus, store = make(tmp_path)
    r = StateAwareRetriever(store, corpus, emb, StaticWeights(SARWeights(1, 0, 0)), k=50)
    assert len(r.retrieve(RetrievalState("capital")).documents) == 4
    with pytest.raises(ValueError):
        r.retrieve(RetrievalState("   "))


def test_out_of_sync_index_detected(tmp_path):
    emb, corpus, store = make(tmp_path)
    corpus.add_or_update([Document("d9", "New document not embedded yet", "X")])
    r = StateAwareRetriever(store, corpus, emb, StaticWeights(SARWeights(1, 0, 0)), k=2)
    with pytest.raises(Exception):
        r._check_index_matches_corpus()


def test_corpus_source_refresh_is_stat_guarded(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"id": "d1", "title": "T", "text": "old"}) + "\n")
    c = Corpus.from_file(p, lazy=True)
    assert c.refresh_if_changed() is c
    # Make the source visibly newer and changed.
    p.write_text(json.dumps({"id": "d1", "title": "T", "text": "new"}) + "\n" +
                 json.dumps({"id": "d2", "title": "T2", "text": "added"}) + "\n")
    st = p.stat(); import os; os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    c2 = c.refresh_if_changed()
    assert c2 is not c and len(c2) == 2 and c2.get("d1").text == "new"
