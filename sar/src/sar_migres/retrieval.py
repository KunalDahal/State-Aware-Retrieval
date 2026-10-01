"""Retrieval implementations used by the MIGRES adapter.

SAR is deliberately decomposed into exactly three pieces:
1. Sentence Transformer embeddings,
2. the state-aware algorithm, and
3. the SAR score.

All non-SAR baselines bypass this stack and implement only the retriever contract.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Sequence

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
    def __init__(self, weights: SARWeights):
        self.weights = weights

    def weights_for(self, iteration: int, state: RetrievalState) -> SARWeights:
        return self.weights


class DynamicWeights(WeightSchedule):
    """Explicitly configured state-dependent rule.

    The report's methodology only says that dynamic weights *may* depend on the current state;
    it does not define a rule. To make the code executable without silently inventing a rule,
    this class exposes a named project extension. ``state_balanced_v1`` uses the exact formula
    recorded in configs/experiments.yaml:

        alpha : beta : gamma = 1 : (1 + m/(1+n)) : (1 + 1/(1+n))

    followed by normalisation to sum to one, where m=|G_t| and n=|E_t|. The rule is deterministic,
    non-negative, and is NOT claimed to be specified by the supplied methodology.
    """

    def __init__(self, rule: str, base: SARWeights | None = None):
        rule = rule.strip().lower()
        if rule not in {"state_balanced_v1", "fixed"}:
            raise ConfigError(
                f"unsupported dynamic rule {rule!r}; use 'state_balanced_v1' or 'fixed'. "
                "The methodology itself does not define a dynamic rule."
            )
        self.rule = rule
        self.base = base

    def weights_for(self, iteration: int, state: RetrievalState) -> SARWeights:
        if self.rule == "fixed":
            if self.base is None:
                raise ConfigError("dynamic rule 'fixed' needs a base SARWeights")
            return self.base
        m = len(state.gaps)
        n = len(state.evidence)
        raw = np.array([
            1.0,
            1.0 + m / (1.0 + n),
            1.0 + 1.0 / (1.0 + n),
        ], dtype=float)
        raw /= raw.sum()
        return SARWeights(float(raw[0]), float(raw[1]), float(raw[2]))


class CallableWeights(WeightSchedule):
    """Adapter for notebooks or applications supplying their own documented rule."""

    def __init__(self, fn: Callable[[int, RetrievalState], SARWeights]):
        self.fn = fn

    def weights_for(self, iteration: int, state: RetrievalState) -> SARWeights:
        w = self.fn(iteration, state)
        if not isinstance(w, SARWeights):
            raise ConfigError("custom weight schedule must return SARWeights")
        return w


class _EmbeddingMemo:
    """Per-retriever memo of state text -> vector."""

    def __init__(self, embedder: Embedder, capacity: int = 8192):
        self.embedder, self.capacity = embedder, capacity
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        missing = list(dict.fromkeys(t for t in texts if t not in self._d))
        if missing:
            vecs = self.embedder.encode(missing)
            if vecs.shape != (len(missing), self.embedder.dim):
                raise ValueError(f"embedder returned {vecs.shape}, expected {(len(missing), self.embedder.dim)}")
            for t, v in zip(missing, vecs):
                if not np.any(v):
                    log.warning("zero embedding for text %r; cosine similarities are defined as 0", t[:60])
                self._d[t] = v
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)
        out = np.stack([self._d[t] for t in texts]) if texts else np.zeros((0, self.embedder.dim), np.float32)
        for t in texts:
            self._d.move_to_end(t)
        return out


class StateAwareRetriever:
    """Exact full-corpus SAR scorer with chunked document vectors."""

    name = "sar"
    score_name = "sar_score"
    is_sar = True

    def __init__(self, store: VectorStore, corpus: Corpus, embedder: Embedder,
                 schedule: WeightSchedule, k: int = 5, chunk_size: int = 50_000, snippet_chars: int = 240):
        if k < 1:
            raise ConfigError("K must be >= 1")
        if store.dim != embedder.dim:
            raise ConfigError(f"vector store dim {store.dim} != embedder dim {embedder.dim}")
        self.store, self.corpus, self.embedder = store, corpus, embedder
        self.schedule, self.k, self.chunk_size, self.snippet_chars = schedule, k, chunk_size, snippet_chars
        self._memo = _EmbeddingMemo(embedder)

    def retrieve(self, state: RetrievalState, iteration: int = 0, k: int | None = None,
                 return_all: bool = False, include_text: bool = True) -> RetrievalResult:
        q = (state.question or "").strip()
        if not q:
            raise ValueError("question must be non-empty")
        gaps = [g for g in (x.strip() for x in state.gaps) if g]
        evid = [e for e in (x.strip() for x in state.evidence) if e]
        k = min(k or self.k, len(self.store))
        if k < 1:
            raise ConfigError("vector store is empty; build the corpus embeddings first")

        w = self.schedule.weights_for(iteration, RetrievalState(q, evid, gaps))
        t0 = time.perf_counter()
        E = self._memo.encode([q] + gaps + evid)
        embed_s = time.perf_counter() - t0
        q_emb, g_emb, e_emb = E[:1], E[1:1 + len(gaps)], E[1 + len(gaps):]

        t1 = time.perf_counter()
        candidate_idx: list[np.ndarray] = []
        candidate_scores: list[np.ndarray] = []
        candidate_R: list[np.ndarray] = []
        candidate_C: list[np.ndarray] = []
        candidate_N: list[np.ndarray] = []
        all_scores = [] if return_all else None
        for start, mat, norms, chunk_ids in self.store.iter_chunks(self.chunk_size):
            comp = score_components(mat, q_emb, g_emb, e_emb, doc_norms=norms)
            sc = sar_score(comp, w)
            if return_all:
                all_scores.append(sc)
            local = top_k_indices(sc, k)
            candidate_idx.append(np.asarray(chunk_ids, dtype=object)[local])
            candidate_scores.append(sc[local])
            candidate_R.append(comp.relevance[local])
            candidate_C.append(comp.gap_coverage[local])
            candidate_N.append(comp.novelty[local])

        doc_ids = np.concatenate(candidate_idx).tolist() if candidate_idx else []
        scores = np.concatenate(candidate_scores) if candidate_scores else np.zeros(0)
        Rs = np.concatenate(candidate_R) if candidate_R else np.zeros(0)
        Cs = np.concatenate(candidate_C) if candidate_C else np.zeros(0)
        Ns = np.concatenate(candidate_N) if candidate_N else np.zeros(0)
        rank_order = sorted(range(len(doc_ids)), key=lambda i: (-float(scores[i]), str(doc_ids[i])))[:k]
        score_s = time.perf_counter() - t1

        docs: list[ScoredDocument] = []
        for r, idx in enumerate(rank_order, start=1):
            did = str(doc_ids[idx])
            title = text = ""
            if include_text:
                d = self.corpus.get(did)
                title, text = d.title, d.text
            docs.append(ScoredDocument(
                doc_id=did, rank=r, relevance=float(Rs[idx]), gap_coverage=float(Cs[idx]), novelty=float(Ns[idx]),
                score=float(scores[idx]), title=title, text=text if include_text else ""))
        if len(docs) < (k or self.k):
            log.warning("corpus has only %d documents; returning %d", len(self.store), len(docs))
        return RetrievalResult(
            documents=docs, alpha=w.alpha, beta=w.beta, gamma=w.gamma, k=k,
            n_docs_scored=len(self.store),
            n_similarity_calcs=len(self.store) * (1 + len(gaps) + len(evid)),
            embed_seconds=embed_s, score_seconds=score_s,
            all_scores=np.concatenate(all_scores) if return_all and all_scores else None)

    def snippet(self, text: str) -> str:
        return snippet(text, self.snippet_chars)
