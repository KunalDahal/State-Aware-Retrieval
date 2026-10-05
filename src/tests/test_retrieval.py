import json
import numpy as np
from sar_migres.corpus.models import Corpus, Document
from sar_migres.embeddings.encoder import HashingEmbedder
from sar_migres.retrieval.sar import StateAwareRetriever, StaticWeights
from sar_migres.retrieval.common import RetrievalState
from sar_migres.scoring.weights import SARWeights
from sar_migres.vector_store import LocalVectorStore, sync_corpus_embeddings
from sar_migres.utils.config import VectorStoreConfig

def _retriever(chunk_size):
    docs=[Document("z", "alpha"), Document("a", "alpha"), Document("b", "beta"), Document("c", "gamma")]
    corpus=Corpus.from_documents(docs)
    emb=HashingEmbedder(dim=64, seed=0)
    store=LocalVectorStore("/tmp/sar_test_store", emb.dim, emb.name)
    store.clear(); sync_corpus_embeddings(store, corpus, emb, 16, remove_missing=True)
    return StateAwareRetriever(store, corpus, emb, StaticWeights(SARWeights(0.4,0.3,0.3)), k=3, chunk_size=chunk_size)

def test_blockwise_and_non_blockwise_retrieval_equivalence():
    a=_retriever(2).retrieve(RetrievalState("alpha"), k=3)
    b=_retriever(100).retrieve(RetrievalState("alpha"), k=3)
    assert [d.doc_id for d in a.documents] == [d.doc_id for d in b.documents]
    assert np.allclose([d.score for d in a.documents], [d.score for d in b.documents])

def test_sar_tie_breaking_uses_document_identity():
    docs=[Document("z", "same"), Document("a", "same"), Document("m", "same")]
    corpus=Corpus.from_documents(docs); emb=HashingEmbedder(dim=32, seed=0)
    store=LocalVectorStore("/tmp/sar_tie_store", emb.dim, emb.name); store.clear(); sync_corpus_embeddings(store, corpus, emb, 16, remove_missing=True)
    r=StateAwareRetriever(store, corpus, emb, StaticWeights(SARWeights(0,0,1)), k=2, chunk_size=10)
    got=[d.doc_id for d in r.retrieve(RetrievalState("same"), k=2).documents]
    assert got == ["a", "m"]


def test_iterative_metrics_skip_unlabeled(tmp_path):
    from sar_migres.experiments.iterative import iterative_metrics_from_run
    from sar_migres.corpus.models import Corpus, Document
    rd=tmp_path/'r'; rd.mkdir()
    (rd/'traces.jsonl').write_text(json.dumps({"question_id":"q1","iteration":0,"k":2,"documents":[]})+'\n')
    corpus=Corpus.from_documents([Document(doc_id="d1",text="relevant",title="T")])
    items=[{"id":"q1","question":"q"}]
    got=iterative_metrics_from_run(rd,items,corpus)
    assert got["n_questions"] == 0


def test_bm25_tie_order_uses_stable_document_id(tmp_path):
    from sar_migres.corpus.models import Corpus, Document
    from sar_migres.retrieval.baselines import BM25Baseline
    corpus = Corpus.from_documents([Document("z", "alpha", "Z"), Document("a", "alpha", "A")])
    bm = BM25Baseline(corpus, cache_dir=tmp_path)
    rows = bm.retrieve("alpha", 2)
    assert [x[0] for x in rows] == ["a", "z"]
