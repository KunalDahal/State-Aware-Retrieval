from sar_migres.datasets.loader import load_questions, gold_answer_value, split_holdout_from_train
import json
from sar_migres.corpus.builder import build_pooled_corpus
from sar_migres.corpus.models import Corpus
from sar_migres.datasets.relevance import relevant_doc_ids

def test_musique_relevance_maps_all_chunks_of_supporting_parent(tmp_path):
    raw=tmp_path/'m.json'; raw.write_text(json.dumps([{"id":"q1","paragraphs":[{"title":"T","paragraph_text":"one two three four five six","is_supporting":True}],"split":"train"}]))
    out=tmp_path/'c.jsonl'; build_pooled_corpus({'musique':raw},output=out,max_documents=10,max_tokens=2,overlap_tokens=0,allow_whitespace_fallback=True)
    corpus=Corpus.from_file(out,lazy=False)
    item={"dataset":"musique","id":"q1","paragraphs":[{"title":"T","paragraph_text":"one two three four five six","is_supporting":True}]}
    ids=relevant_doc_ids(item,corpus)
    assert len(ids) == len(corpus)

def test_natural_questions_positive_contexts_normalize_to_corpus_ids(tmp_path):
    ctx={"title":"T","text":"the positive answer context"}
    raw=tmp_path/'nq.json'; raw.write_text(json.dumps([{"id":"nq1","positive_ctxs":[ctx],"question":"q","split":"train"}]))
    out=tmp_path/'c.jsonl'; build_pooled_corpus({'natural_questions':raw},output=out,max_documents=10,max_tokens=50,overlap_tokens=0,allow_whitespace_fallback=True)
    corpus=Corpus.from_file(out,lazy=False)
    item={"dataset":"natural_questions","id":"nq1","positive_ctxs":[ctx]}
    ids=relevant_doc_ids(item,corpus)
    assert ids and all(i in corpus.ids for i in ids)


def test_natural_questions_native_schema(tmp_path):
    p=tmp_path/'nq.json'
    p.write_text(json.dumps([{"id":"nq1","question_text":"who sings hello","short_answers":[{"text":"Adele"}],"yes_no_answer":"NONE"}]))
    rows=load_questions("natural_questions", p)
    assert rows[0]["question"] == "who sings hello"
    assert rows[0]["answer"] == "Adele"


def test_musique_answer_aliases_are_preserved(tmp_path):
    p=tmp_path/'m.json'
    p.write_text(json.dumps([{"id":"m1","question":"q","answer":"Main Answer","answer_aliases":["main answer","alt"]}]))
    rows=load_questions("musique", p)
    assert rows[0]["answer_aliases"] == ["main answer","alt"]


def test_natural_questions_yes_no_gold_is_normalized(tmp_path):
    p = tmp_path / 'nq_yesno.json'
    p.write_text(json.dumps([{'id':'nq2','question_text':'is the sky blue','short_answers':[],'yes_no_answer':'YES'}]))
    rows = load_questions('natural_questions', p)
    assert rows[0]['answer'] == 'YES'


def test_natural_questions_gold_answer_value_handles_native_yes_no():
    assert gold_answer_value({"dataset": "natural_questions", "yes_no_answer": "YES"}) == "YES"


def test_training_holdout_is_exact_deterministic_fraction():
    rows = [{"id": f"q{i}"} for i in range(100)]
    a = split_holdout_from_train(rows, 0.10, 13)
    b = split_holdout_from_train(rows, 0.10, 13)
    assert len(a) == 10
    assert [x["id"] for x in a] == [x["id"] for x in b]
