#!/usr/bin/env python
"""Validate/normalise a corpus into data/processed/corpus.jsonl (id, title, text).

Sources
  --from-file PATH          JSONL/JSON/TSV in MIGRES/DPR layout (id, title, text|contents). Malformed rows are reported.
  --from-dataset NAME PATH  build a DEVELOPMENT corpus from the per-question contexts of hotpotqa / wikihop / musique.
                            (This is NOT the Wikipedia dump that MIGRES retrieves from - see data/README.md.)
"""
import json
from _common import ROOT, base_parser, load

from sar_migres.corpus import Corpus
from sar_migres.datasets import contexts_to_documents, load_questions
from sar_migres.utils import atomic_write_text


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--from-file")
    ap.add_argument("--from-dataset", nargs=2, metavar=("NAME", "PATH"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "corpus.jsonl"))
    ap.add_argument("--limit-questions", type=int, default=None)
    a = ap.parse_args()
    cfg = load(a)
    if bool(a.from_file) == bool(a.from_dataset):
        ap.error("give exactly one of --from-file / --from-dataset")
    if a.from_file:
        corpus = Corpus.from_file(a.from_file, embed_template=cfg.embedding.embed_template,
                                  max_malformed_fraction=cfg.data.max_malformed_fraction)
        docs = corpus.get_many(corpus.ids)
    else:
        name, path = a.from_dataset
        recs = load_questions(name, path, a.limit_questions)
        docs = list(contexts_to_documents(name, recs))
        corpus = Corpus.from_documents(docs, cfg.embedding.embed_template)
    atomic_write_text(a.out, "\n".join(json.dumps({"id": d.doc_id, "title": d.title, "text": d.text}, ensure_ascii=False)
                                       for d in docs) + "\n")
    print(f"wrote {len(docs)} documents -> {a.out}\ncorpus version: {corpus.version}\nreport: {corpus.report.to_dict()}")


if __name__ == "__main__":
    main()
