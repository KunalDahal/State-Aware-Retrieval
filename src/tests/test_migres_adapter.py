from sar_migres.migres.adapter import reset_initial_evidence_state, derive_gaps, UpstreamCode
from sar_migres.utils.errors import UpstreamMigresError

def test_migres_initial_state_is_empty():
    class R:
        def __init__(self):
            self.known_info={"q":["seed"]}; self.missing_info={"q":"missing"}; self.backup_info={"q":["seed"]}; self.history_passages={"q":{"old":1}}
    r=R(); reset_initial_evidence_state(r,"q")
    assert r.known_info["q"] == []
    assert r.missing_info["q"] == ""
    assert r.backup_info["q"] == []
    assert r.history_passages["q"] == {}

def test_initial_gap_set_is_empty():
    assert derive_gaps("q", ["q", "g1"], "missing", "both", 0) == []

def test_upstream_version_validation_is_strict(tmp_path):
    repo=tmp_path/'MIGRES'; (repo/'code'/'utils').mkdir(parents=True)
    (repo/'code'/'pipeline.py').write_text('# wrong')
    (repo/'code'/'utils'/'search_engine.py').write_text('# wrong')
    try:
        UpstreamCode(repo, 'upstream_cross_encoder', 'BAAI/bge-reranker-base', strict_hash=True)
    except UpstreamMigresError:
        return
    assert False, 'strict upstream validation accepted mismatched provenance'

def test_baseline_trace_snippet_does_not_replace_full_migres_text():
    from sar_migres.corpus.models import Corpus, Document
    from sar_migres.migres.adapter import RetrieverSearchProvider
    class R:
        name='bm25'; score_name='bm25_score'; is_sar=False
        def __init__(self, corpus): self.corpus=corpus; self.stats=None
        def retrieve_many(self, queries, k): return [('d1', 1.0)]
    text='x' * 400
    corpus=Corpus.from_documents([Document('d1', text, 'Title')])
    r=R(corpus); provider=RetrieverSearchProvider(r,'run',lambda q:'q1',snippet_chars=240,k=1)
    rows=provider.retrieve_for('question',['question'],[])
    assert rows[0]['text'] == text
    assert len(provider.traces['question'][0].documents[0]['text']) <= 240
