"""Baseline retriever interfaces (FUTURE / OCTOBER EXPERIMENTS - methodology Table 5.6).

Nothing here is run or compared automatically.  ``BaselineRetriever`` is the plug-in contract:
given the ORIGINAL question (baselines are state-agnostic) return Top-K (doc_id, score).

* ``BM25Baseline``            - implemented (pure NumPy, Robertson/Zaragoza BM25, k1/b configurable).
* ``SentenceTransformerSearch`` - implemented (cosine Top-K on the same embeddings; the methodology's
                                 "Sentence Transformer Search" row; no state signals).
* DPR / Contriever / SPLADE / ColBERTv2 - interface stubs.  They raise ``NotImplementedError`` with the
  checkpoint/library each would need; no checkpoint names are assumed here.
"""
from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from .corpus import Corpus
from .embeddings import Embedder
from .scoring import top_k_indices
from .similarity import cosine_matrix
from .vector_store import VectorStore

_TOK = re.compile(r"\w+", re.UNICODE)


class BaselineRetriever(ABC):
    name: str = "baseline"

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return [(doc_id, score)] sorted by descending score."""


class BM25Baseline(BaselineRetriever):
    name = "bm25"

    def __init__(self, corpus: Corpus, k1: float = 0.9, b: float = 0.4):
        # defaults follow upstream migres.sh (--k1 0.9 --bm25_b 0.4); Robertson's classic defaults are 1.2/0.75
        self.k1, self.b = k1, b
        self.ids = corpus.ids
        toks = [_TOK.findall(t.lower()) for t in corpus.embed_texts(self.ids)]
        self.tf = [Counter(t) for t in toks]
        self.dl = np.array([len(t) for t in toks], dtype=np.float64)
        self.avgdl = float(self.dl.mean()) if len(toks) else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(toks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = np.zeros(len(self.ids))
        for t in set(_TOK.findall(query.lower())):
            if t not in self.idf:
                continue
            f = np.array([c.get(t, 0) for c in self.tf], dtype=np.float64)
            scores += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl / max(self.avgdl, 1e-9)))
        return [(self.ids[i], float(scores[i])) for i in top_k_indices(scores, k)]


class SentenceTransformerSearch(BaselineRetriever):
    name = "sentence_transformer_search"

    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store, self.embedder = store, embedder

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self.embedder.encode([query])
        best: list[tuple[float, int]] = []
        ids = self.store.ids()
        for start, mat, _ in self.store.iter_chunks(50_000):
            s = cosine_matrix(q, mat)[0]
            for i in top_k_indices(s, k):
                best.append((float(s[i]), start + int(i)))
        best.sort(key=lambda x: (-x[0], x[1]))
        return [(ids[i], sc) for sc, i in best[:k]]


class _NotImplementedBaseline(BaselineRetriever):
    requires = ""

    def __init__(self, *_, **__):
        raise NotImplementedError(f"{self.name}: FUTURE (October) baseline - not implemented. Needs: {self.requires}")

    def retrieve(self, query, k):  # pragma: no cover
        raise NotImplementedError


class DPRBaseline(_NotImplementedBaseline):
    name, requires = "dpr", "DPR question+context encoders (Karpukhin et al. 2020) and a passage index"


class ContrieverBaseline(_NotImplementedBaseline):
    name, requires = "contriever", "a Contriever checkpoint (Izacard et al. 2022) and a passage index"


class SPLADEBaseline(_NotImplementedBaseline):
    name, requires = "splade", "a SPLADE checkpoint (Formal et al. 2021) and an inverted index"


class ColBERTv2Baseline(_NotImplementedBaseline):
    name, requires = "colbertv2", "the ColBERTv2 library/checkpoint (Santhanam et al. 2022) and a compressed index"


BASELINES = {"bm25": BM25Baseline, "sentence_transformer_search": SentenceTransformerSearch, "dpr": DPRBaseline,
             "contriever": ContrieverBaseline, "splade": SPLADEBaseline, "colbertv2": ColBERTv2Baseline}
