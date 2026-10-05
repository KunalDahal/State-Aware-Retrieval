import json
from pathlib import Path
from sar_migres.corpus.models import Corpus, Document
from sar_migres.corpus.builder import build_pooled_corpus

def test_stable_corpus_ids_and_document_representation(tmp_path):
    docs = [Document("d1", "text", "T")]
    a = Corpus.from_documents(docs)
    b = Corpus.from_documents(docs)
    assert list(a.ids) == list(b.ids)
    assert a.version == b.version
    assert a.get("d1").embed_text() == "T. text"

def test_chunk_parent_identity_is_stable(tmp_path):
    raw = tmp_path / "musique.json"
    raw.write_text(json.dumps([{"id":"q1","paragraphs":[{"title":"T","paragraph_text":"one two three four","is_supporting":True}],"split":"train"}]))
    out1 = tmp_path / "c1.jsonl"; out2 = tmp_path / "c2.jsonl"
    m1=build_pooled_corpus({"musique":raw}, output=out1, max_documents=5, max_tokens=2, overlap_tokens=0, tokenizer_name=None, allow_whitespace_fallback=True)
    m2=build_pooled_corpus({"musique":raw}, output=out2, max_documents=5, max_tokens=2, overlap_tokens=0, tokenizer_name=None, allow_whitespace_fallback=True)
    ids1=[json.loads(x)["id"] for x in out1.read_text().splitlines()]
    ids2=[json.loads(x)["id"] for x in out2.read_text().splitlines()]
    assert ids1 == ids2
    assert all("-p" in x and "-c" in x for x in ids1)
    assert m1["n_documents"] == m2["n_documents"]


def test_scientific_chunking_requires_explicit_tokenizer():
    from sar_migres.corpus.chunking import chunk_text
    import pytest
    with pytest.raises(RuntimeError):
        chunk_text("one two three", max_tokens=2, overlap_tokens=0, tokenizer=None)
    assert chunk_text("one two three", max_tokens=2, overlap_tokens=0, tokenizer=None, allow_whitespace_fallback=True)
