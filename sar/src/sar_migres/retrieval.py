"""State Aware Retrieval over the COMPLETE corpus (no candidate stage, no threshold).

At iteration t:  embed q, G_t, E_t  ->  score every document with S_t  ->  Top-K.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Sequence

import numpy as np

from .corpus import Corpus
from .embeddings import Embedder
from .errors import ConfigError
from .logging_utils import get_logger
from .schemas import RetrievalResult, RetrievalState, ScoredDocument
from .scoring import SARWeights, score_components, sar_score, top_k_indices
from .utils import snippet
from .vector_store import VectorStore

log = get_logger("retrieval")


class WeightSchedule(ABC):
    @abstractmethod
    def weights_for(self, iteration: int, state: RetrievalState) -> SARWeights: ...


class StaticWeights(WeightSchedule):
    """alpha_t = alpha, beta_t = beta, gamma_t = gamma for every iteration (methodology 5.6.1)."""

    def __init__(self, weights: SARWeights):
        self.weights = weights

    def weights_for(self, iteration, state):
        return self.weights


class DynamicWeights(WeightSchedule):
    """FUTURE / OCTOBER EXPERIMENT: state-dependent (alpha_t, beta_t, gamma_t).

    The methodology only states that the weights "may change according to the current
    retrieval state"; no rule is specified, so none is invented here.  Subclass and
    implement ``weights_for`` when the dynamic-weighting study is defined.
    """

    def weights_for(self, iteration, state):
        raise NotImplementedError("Dynamic weighting is a FUTURE (October) experiment; no rule is specified in methodology.tex")


class _EmbeddingMemo:
    """Per-retriever memo of text -> vector so evidence/gaps already embedded are not recomputed.
    (Same model + same text => identical vector, so results are unchanged.)"""

    def __init__(self, embedder: Embedder, capacity: int = 8192):
        self.embedder, self.capacity = embedder, capacity
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self._d))
        if missing:
            vecs = self.embedder.encode(missing)
            if vecs.shape != (len(missing), self.embedder.dim):
                raise ValueError(f"embedder returned {vecs.shape}, expected {(len(missing), self.embedder.dim)}")
            for t, v in zip(missing, vecs):
                if not np.any(v):
                    log.warning("zero embedding for text %r; its cosine similarities are defined as 0", t[:60])
                self._d[t] = v
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)
        out = np.stack([self._d[t] for t in texts]) if texts else np.zeros((0, self.embedder.dim), np.float32)
        for t in texts:
            self._d.move_to_end(t)
        return out


class StateAwareRetriever:
    def __init__(self, store: VectorStore, corpus: Corpus, embedder: Embedder,
                 schedule: WeightSchedule, k: int = 5, chunk_size: int = 50_000, snippet_chars: int = 240):
        if k < 1:
            raise ConfigError("K must be >= 1")
        if store.dim != embedder.dim:
            raise ConfigError(f"vector store dim {store.dim} != embedder dim {embedder.dim}")
        self.store, self.corpus, self.embedder = store, corpus, embedder
        self.schedule, self.k, self.chunk_size, self.snippet_chars = schedule, k, chunk_size, snippet_chars
        self._memo = _EmbeddingMemo(embedder)

    def _check_index_matches_corpus(self) -> None:
        stored = self.store.hashes()
        d = self.corpus.diff(stored)
        if d.new or d.changed or d.removed:
            raise ConfigError(
                f"embedding store is out of sync with the corpus (new={len(d.new)}, changed={len(d.changed)}, "
                f"removed={len(d.removed)}). Run sync_corpus_embeddings()/scripts/build_embeddings.py first.")

    def retrieve(self, state: RetrievalState, iteration: int = 0, k: int | None = None,
                 return_all: bool = False, include_text: bool = True) -> RetrievalResult:
        q = (state.question or "").strip()
        if not q:
            raise ValueError("question must be non-empty")
        gaps = [g for g in (x.strip() for x in state.gaps) if g]
        evid = [e for e in (x.strip() for x in state.evidence) if e]
        k = k or self.k
        w = self.schedule.weights_for(iteration, RetrievalState(q, evid, gaps))
        n_docs = len(self.store)
        if n_docs == 0:
            raise ConfigError("vector store is empty; build the corpus embeddings first")

        t0 = time.perf_counter()
        E = self._memo.encode([q] + gaps + evid)  # one batched embedding call for the whole state
        embed_s = time.perf_counter() - t0
        q_emb, g_emb, e_emb = E[:1], E[1:1 + len(gaps)], E[1 + len(gaps):]

        t1 = time.perf_counter()
        cand_idx = np.zeros(0, dtype=np.int64)
        cand_sc = np.zeros(0)
        cand_R, cand_C, cand_N = (np.zeros(0),) * 3
        all_scores = [] if return_all else None
        for start, mat, norms in self.store.iter_chunks(self.chunk_size):
            comp = score_components(mat, q_emb, g_emb, e_emb, doc_norms=norms)
            sc = sar_score(comp, w)
            if return_all:
                all_scores.append(sc)
            local = top_k_indices(sc, k)
            cand_idx = np.concatenate([cand_idx, local + start])
            cand_sc = np.concatenate([cand_sc, sc[local]])
            cand_R = np.concatenate([cand_R, comp.relevance[local]])
            cand_C = np.concatenate([cand_C, comp.gap_coverage[local]])
            cand_N = np.concatenate([cand_N, comp.novelty[local]])
            if cand_idx.size > k:  # keep running top-k (exact: any global top-k doc is in its chunk's top-k)
                keep = np.lexsort((cand_idx, -cand_sc))[:k]
                cand_idx, cand_sc, cand_R, cand_C, cand_N = (a[keep] for a in (cand_idx, cand_sc, cand_R, cand_C, cand_N))
        order = np.lexsort((cand_idx, -cand_sc))[:k]
        score_s = time.perf_counter() - t1

        ids = self.store.ids()
        docs: list[ScoredDocument] = []
        for r, o in enumerate(order, start=1):
            did = ids[int(cand_idx[o])]
            title = text = ""
            if include_text:
                d = self.corpus.get(did)
                title, text = d.title, d.text
            docs.append(ScoredDocument(doc_id=did, rank=r, relevance=float(cand_R[o]), gap_coverage=float(cand_C[o]),
                                       novelty=float(cand_N[o]), score=float(cand_sc[o]), title=title,
                                       text=text if include_text else ""))
        if len(docs) < k:
            log.warning("corpus has only %d documents; returning %d < K=%d", n_docs, len(docs), k)
        return RetrievalResult(
            documents=docs, alpha=w.alpha, beta=w.beta, gamma=w.gamma, k=k, n_docs_scored=n_docs,
            n_similarity_calcs=n_docs * (1 + len(gaps) + len(evid)),  # |D|(1+m+n), methodology 5.6.7
            embed_seconds=embed_s, score_seconds=score_s,
            all_scores=np.concatenate(all_scores) if return_all else None)

    def snippet(self, text: str) -> str:
        return snippet(text, self.snippet_chars)
