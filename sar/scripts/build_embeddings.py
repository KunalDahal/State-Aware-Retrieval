#!/usr/bin/env python
"""Build / incrementally update the embedding store for the configured corpus."""
import json
from _common import base_parser, load

from sar_migres.corpus import Corpus
from sar_migres.embeddings import build_embedder
from sar_migres.logging_utils import setup_logging
from sar_migres.vector_store import build_vector_store, sync_corpus_embeddings


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--rebuild", action="store_true", help="discard an incompatible cache (different embedder) and re-embed")
    a = ap.parse_args()
    cfg = load(a)
    setup_logging(cfg.run.log_level)
    corpus = Corpus.from_file(cfg.data.corpus_path, embed_template=cfg.embedding.embed_template, lazy=cfg.data.corpus_lazy,
                              max_malformed_fraction=cfg.data.max_malformed_fraction)
    emb = build_embedder(cfg.embedding)
    store = build_vector_store(cfg.vector_store, emb, cfg.run.cache_root, rebuild=a.rebuild)
    rep = sync_corpus_embeddings(store, corpus, emb, cfg.embedding.batch_size, remove_missing=cfg.vector_store.remove_missing)
    print(json.dumps({"embedder": emb.name, "dim": emb.dim, "corpus_version": corpus.version, **rep.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
