"""Persistent document-embedding stores.

``LocalVectorStore``   NumPy matrix + JSON manifest on disk (development / smoke tests).
``PineconeVectorStore`` Pinecone as the persistent store (methodology default).

IMPORTANT (design decision, see README "Compatibility decisions"):
SAR scores the COMPLETE corpus with a non-linear term (max over evidence).  A Pinecone
query returns only an ANN top-k, so exact full-corpus SAR scoring cannot be done
server-side.  ``PineconeVectorStore`` therefore keeps Pinecone as the durable source of
truth and maintains an in-process (optionally disk-cached) mirror of the vectors; the
scoring itself is the same exact NumPy code path for both stores, so the scoring
semantics are identical.
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .corpus import Corpus
from .embeddings import Embedder
from .errors import (DependencyMissingError, EmbeddingCacheCorruptionError, MissingCredentialError,
                     StoreIncompatibleError, VectorStoreConnectionError, retry_call)
from .logging_utils import get_logger
from .similarity import l2_norms
from .utils import atomic_write_text, sha256_file, utc_now_iso

log = get_logger("vector_store")


@dataclass
class SyncReport:
    n_total: int = 0
    n_new: int = 0
    n_changed: int = 0
    n_unchanged: int = 0
    n_removed: int = 0
    n_reused_same_content: int = 0
    n_embedded: int = 0
    embed_seconds: float = 0.0
    rebuilt_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class VectorStore(ABC):
    dim: int
    embedder_name: str

    @abstractmethod
    def hashes(self) -> dict[str, str]: ...
    @abstractmethod
    def upsert(self, ids: Sequence[str], vectors: np.ndarray, hashes: Sequence[str]) -> None: ...
    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None: ...
    @abstractmethod
    def get_vectors(self, ids: Sequence[str]) -> np.ndarray: ...
    @abstractmethod
    def ids(self) -> list[str]: ...
    @abstractmethod
    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield (start_row, vectors[chunk,d], l2_norms[chunk]) in a stable id order."""
    @abstractmethod
    def flush(self) -> None: ...
    @abstractmethod
    def clear(self) -> None: ...

    def __len__(self) -> int:
        return len(self.ids())


class _Matrix:
    """Shared in-memory id/vector bookkeeping."""

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[str] = []
        self._hash: list[str] = []
        self._mat = np.zeros((0, dim), dtype=np.float32)
        self._row: dict[str, int] = {}
        self._norms: np.ndarray | None = None

    def hashes(self) -> dict[str, str]:
        return dict(zip(self._ids, self._hash))

    def ids(self) -> list[str]:
        return list(self._ids)

    def upsert(self, ids, vectors, hashes):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim or len(ids) != vectors.shape[0]:
            raise ValueError(f"bad vectors shape {vectors.shape} for dim={self.dim}, n_ids={len(ids)}")
        if not np.isfinite(vectors).all():
            raise ValueError("non-finite values in embeddings")
        if not self._mat.flags.writeable:
            self._mat = np.array(self._mat)
        new_ids, new_vecs, new_h = [], [], []
        for i, (d, v, h) in enumerate(zip(ids, vectors, hashes)):
            if d in self._row:
                r = self._row[d]
                self._mat[r] = v
                self._hash[r] = h
            else:
                if d in new_ids:  # duplicate inside the batch -> last wins
                    j = new_ids.index(d)
                    new_vecs[j], new_h[j] = v, h
                else:
                    new_ids.append(d); new_vecs.append(v); new_h.append(h)
        if new_ids:
            start = len(self._ids)
            self._ids.extend(new_ids)
            self._hash.extend(new_h)
            self._mat = np.vstack([self._mat, np.stack(new_vecs)])
            for j, d in enumerate(new_ids):
                self._row[d] = start + j
        self._norms = None

    def delete(self, ids):
        drop = {i for i in ids if i in self._row}
        if not drop:
            return
        keep = [r for r, d in enumerate(self._ids) if d not in drop]
        self._mat = np.array(self._mat[keep]) if keep else np.zeros((0, self.dim), np.float32)
        self._ids = [self._ids[r] for r in keep]
        self._hash = [self._hash[r] for r in keep]
        self._row = {d: r for r, d in enumerate(self._ids)}
        self._norms = None

    def get_vectors(self, ids):
        return np.stack([self._mat[self._row[i]] for i in ids]) if ids else np.zeros((0, self.dim), np.float32)

    def norms(self) -> np.ndarray:
        if self._norms is None:
            self._norms = l2_norms(np.asarray(self._mat))
        return self._norms

    def iter_chunks(self, chunk_size):
        n = len(self._ids)
        nr = self.norms()
        for s in range(0, n, chunk_size):
            yield s, np.asarray(self._mat[s:s + chunk_size]), nr[s:s + chunk_size]


class LocalVectorStore(_Matrix, VectorStore):
    """Directory layout:  <dir>/vectors.npy  +  <dir>/manifest.json (ids, hashes, sha256, embedder)."""

    def __init__(self, directory: str | Path, dim: int, embedder_name: str):
        _Matrix.__init__(self, dim)
        self.directory = Path(directory)
        self.embedder_name = embedder_name
        self.events: list[str] = []
        self._load_if_present()

    def _paths(self):
        return self.directory / "vectors.npy", self.directory / "manifest.json"

    def _load_if_present(self) -> None:
        vp, mp = self._paths()
        if not vp.exists() and not mp.exists():
            return
        try:
            if not (vp.exists() and mp.exists()):
                raise EmbeddingCacheCorruptionError("vectors.npy or manifest.json missing")
            man = json.loads(mp.read_text(encoding="utf-8"))
            if sha256_file(vp) != man["vectors_sha256"]:
                raise EmbeddingCacheCorruptionError("vectors.npy checksum mismatch")
            mat = np.load(vp, mmap_mode="r")
            if mat.shape != (len(man["ids"]), man["dim"]) or len(man["hashes"]) != len(man["ids"]):
                raise EmbeddingCacheCorruptionError(f"shape {mat.shape} inconsistent with manifest")
        except EmbeddingCacheCorruptionError:
            raise
        except Exception as e:  # noqa: BLE001
            raise EmbeddingCacheCorruptionError(f"cannot read cache in {self.directory}: {e}") from e
        if man["embedder"] != self.embedder_name or man["dim"] != self.dim:
            raise StoreIncompatibleError(
                f"cache in {self.directory} was built with embedder={man['embedder']!r} dim={man['dim']}, "
                f"but current embedder={self.embedder_name!r} dim={self.dim}. Re-run with rebuild=True.")
        self._ids, self._hash, self._mat = list(man["ids"]), list(man["hashes"]), mat
        self._row = {d: r for r, d in enumerate(self._ids)}
        self.events = man.get("events", [])
        log.info("loaded %d cached embeddings from %s", len(self._ids), self.directory)

    def flush(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        vp, mp = self._paths()
        tmp = vp.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(self._mat, dtype=np.float32))
        os.replace(tmp, vp)
        man = {"embedder": self.embedder_name, "dim": self.dim, "ids": self._ids, "hashes": self._hash,
               "vectors_sha256": sha256_file(vp), "updated": utc_now_iso(), "events": self.events[-50:]}
        atomic_write_text(mp, json.dumps(man))
        # the mmap (if any) points at the old inode; reload lazily is unnecessary since data is in RAM

    def clear(self) -> None:
        _Matrix.__init__(self, self.dim)
        for p in self._paths():
            if p.exists():
                p.unlink()


class PineconeVectorStore(_Matrix, VectorStore):
    """Pinecone-backed durable store with an exact in-process mirror used for scoring.

    ``client`` / ``index`` may be injected (unit tests use an in-memory fake); otherwise the
    official ``pinecone`` SDK is imported lazily and the API key is read from the environment
    variable named in the config (never from the config file itself).
    """

    FETCH_BATCH = 100
    UPSERT_BATCH = 100

    def __init__(self, index_name: str, dim: int, embedder_name: str, namespace: str = "",
                 metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1",
                 api_key_env: str = "PINECONE_API_KEY", create_if_missing: bool = True,
                 mirror_dir: str | Path | None = None, index: Any = None, retry_attempts: int = 5):
        _Matrix.__init__(self, dim)
        self.embedder_name = embedder_name
        self.index_name, self.namespace, self.metric = index_name, namespace, metric
        self.cloud, self.region, self.api_key_env = cloud, region, api_key_env
        self.create_if_missing, self.retry_attempts = create_if_missing, retry_attempts
        self.mirror = _open_local(mirror_dir, dim, embedder_name, on_corruption="rebuild", rebuild=True) if mirror_dir else None
        self._index = index
        self._connected = index is not None

    # -- connection ------------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        key = os.environ.get(self.api_key_env)
        if not key:
            raise MissingCredentialError(f"environment variable {self.api_key_env} is not set "
                                         "(required for the Pinecone vector store; use vector_store.type=local for offline runs)")
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as e:  # pragma: no cover
            raise DependencyMissingError("pinecone SDK not installed: pip install 'sar-migres[pinecone]'") from e
        try:
            pc = Pinecone(api_key=key)
            if not pc.has_index(self.index_name):
                if not self.create_if_missing:
                    raise VectorStoreConnectionError(f"Pinecone index {self.index_name!r} does not exist")
                log.info("creating Pinecone index %s (dim=%d, metric=%s)", self.index_name, self.dim, self.metric)
                pc.create_index(name=self.index_name, dimension=self.dim, metric=self.metric,
                                spec=ServerlessSpec(cloud=self.cloud, region=self.region))
                for _ in range(60):
                    if pc.describe_index(self.index_name).status["ready"]:
                        break
                    time.sleep(2)
            desc = pc.describe_index(self.index_name)
            if int(desc.dimension) != self.dim:
                raise StoreIncompatibleError(f"Pinecone index dimension {desc.dimension} != embedder dim {self.dim}")
            self._index = pc.Index(self.index_name)
            self._connected = True
        except (StoreIncompatibleError, VectorStoreConnectionError):
            raise
        except Exception as e:  # noqa: BLE001
            raise VectorStoreConnectionError(f"could not connect to Pinecone index {self.index_name!r}: "
                                             f"{type(e).__name__}: {e}") from e

    def _call(self, fn, what):
        try:
            return retry_call(fn, attempts=self.retry_attempts, what=f"pinecone {what}")
        except Exception as e:  # noqa: BLE001
            raise VectorStoreConnectionError(f"Pinecone {what} failed: {type(e).__name__}: {e}") from e

    def _remote_count(self) -> int:
        stats = self._call(lambda: self._index.describe_index_stats(), "describe_index_stats")
        ns = (stats.get("namespaces") if isinstance(stats, dict) else getattr(stats, "namespaces", {})) or {}
        entry = ns.get(self.namespace)
        if entry is None:
            return 0
        return int(entry["vector_count"] if isinstance(entry, dict) else entry.vector_count)

    def load_mirror(self, refresh: bool = False) -> None:
        """Populate the scoring mirror from the disk cache (if consistent) or from Pinecone."""
        self.connect()
        remote_n = self._remote_count()
        if self.mirror is not None and not refresh and len(self.mirror) == remote_n and remote_n > 0:
            self._mat, self._ids, self._hash = np.asarray(self.mirror._mat), list(self.mirror._ids), list(self.mirror._hash)
            self._row = {d: r for r, d in enumerate(self._ids)}
            self._norms = None
            log.info("Pinecone mirror restored from disk cache (%d vectors)", remote_n)
            return
        ids: list[str] = []
        for page in self._call(lambda: list(self._index.list(namespace=self.namespace)), "list"):
            ids.extend(page if isinstance(page, (list, tuple)) else [page])
        _Matrix.__init__(self, self.dim)
        for s in range(0, len(ids), self.FETCH_BATCH):
            batch = ids[s:s + self.FETCH_BATCH]
            resp = self._call(lambda b=batch: self._index.fetch(ids=b, namespace=self.namespace), "fetch")
            vecs = resp["vectors"] if isinstance(resp, dict) else resp.vectors
            got_ids, mat, hs = [], [], []
            for vid, v in vecs.items():
                values = v["values"] if isinstance(v, dict) else v.values
                meta = (v["metadata"] if isinstance(v, dict) else v.metadata) or {}
                if meta.get("embedder", self.embedder_name) != self.embedder_name:
                    raise StoreIncompatibleError(f"vector {vid!r} was written by embedder {meta.get('embedder')!r}, "
                                                 f"current is {self.embedder_name!r}")
                got_ids.append(vid); mat.append(values); hs.append(meta.get("content_hash", ""))
            if got_ids:
                _Matrix.upsert(self, got_ids, np.asarray(mat, dtype=np.float32), hs)
        log.info("fetched %d vectors from Pinecone index %s", len(self._ids), self.index_name)
        self._sync_disk_mirror()

    def _sync_disk_mirror(self):
        if self.mirror is not None:
            self.mirror.clear()
            self.mirror.upsert(self._ids, np.asarray(self._mat), self._hash)
            self.mirror.flush()

    # -- VectorStore API -----------------------------------------------------------
    def upsert(self, ids, vectors, hashes):
        self.connect()
        vectors = np.asarray(vectors, dtype=np.float32)
        for s in range(0, len(ids), self.UPSERT_BATCH):
            payload = [{"id": str(ids[j]), "values": vectors[j].tolist(),
                        "metadata": {"content_hash": hashes[j], "embedder": self.embedder_name}}
                       for j in range(s, min(s + self.UPSERT_BATCH, len(ids)))]
            self._call(lambda p=payload: self._index.upsert(vectors=p, namespace=self.namespace), "upsert")
        _Matrix.upsert(self, ids, vectors, hashes)

    def delete(self, ids):
        self.connect()
        ids = [i for i in ids]
        for s in range(0, len(ids), 1000):
            b = ids[s:s + 1000]
            self._call(lambda b=b: self._index.delete(ids=b, namespace=self.namespace), "delete")
        _Matrix.delete(self, ids)

    def flush(self) -> None:
        self._sync_disk_mirror()

    def clear(self) -> None:
        self.connect()
        self._call(lambda: self._index.delete(delete_all=True, namespace=self.namespace), "delete_all")
        _Matrix.__init__(self, self.dim)
        if self.mirror is not None:
            self.mirror.clear()


def _open_local(directory, dim, embedder_name, on_corruption: str = "rebuild", rebuild: bool = False) -> LocalVectorStore:
    """Open a local cache; corrupt caches are quarantined (never silently reused)."""
    directory = Path(directory)
    try:
        return LocalVectorStore(directory, dim, embedder_name)
    except EmbeddingCacheCorruptionError as e:
        if on_corruption != "rebuild":
            raise
        stamp = time.strftime("%Y%m%dT%H%M%S")
        log.error("EMBEDDING CACHE CORRUPT (%s). Quarantining files in %s and rebuilding from the corpus.", e, directory)
        for name in ("vectors.npy", "manifest.json"):
            p = directory / name
            if p.exists():
                p.rename(directory / f"{name}.corrupt-{stamp}")
        return LocalVectorStore(directory, dim, embedder_name)
    except StoreIncompatibleError as e:
        if not rebuild:
            raise
        log.warning("cache incompatible with current embedder (%s); rebuilding because rebuild=True", e)
        for name in ("vectors.npy", "manifest.json"):
            p = directory / name
            if p.exists():
                p.unlink()
        return LocalVectorStore(directory, dim, embedder_name)


def build_vector_store(cfg, embedder: Embedder, cache_root: str | Path = "data/cache", rebuild: bool = False) -> VectorStore:
    """cfg: VectorStoreConfig."""
    if cfg.type == "local":
        return _open_local(Path(cache_root) / cfg.local_dir_name(embedder.name), embedder.dim, embedder.name,
                           on_corruption=cfg.on_cache_corruption, rebuild=rebuild)
    if cfg.type == "pinecone":
        store = PineconeVectorStore(cfg.pinecone_index, embedder.dim, embedder.name, namespace=cfg.pinecone_namespace,
                                    metric="cosine", cloud=cfg.pinecone_cloud, region=cfg.pinecone_region,
                                    api_key_env=cfg.pinecone_api_key_env, create_if_missing=cfg.pinecone_create_if_missing,
                                    mirror_dir=Path(cache_root) / "pinecone_mirror" / cfg.pinecone_index)
        store.load_mirror()
        return store
    raise ValueError(f"unknown vector_store.type {cfg.type!r}")


def sync_corpus_embeddings(store: VectorStore, corpus: Corpus, embedder: Embedder, batch_size: int = 64,
                           remove_missing: bool = True, checkpoint_every: int = 20) -> SyncReport:
    """Incremental maintenance (methodology 5.2.2):
    new -> embed+add;  changed -> re-embed+replace;  unchanged -> reuse stored vector.
    A new id whose content hash already exists under another id reuses that vector (renamed doc).
    """
    t0 = time.perf_counter()
    stored = store.hashes()
    diff = corpus.diff(stored)
    rep = SyncReport(n_total=len(corpus), n_new=len(diff.new), n_changed=len(diff.changed),
                     n_unchanged=len(diff.unchanged), n_removed=len(diff.removed))
    by_hash: dict[str, str] = {}
    for did, h in stored.items():
        by_hash.setdefault(h, did)
    todo = diff.changed + diff.new
    reuse, embed = [], []
    for did in todo:
        src = by_hash.get(corpus.content_hashes[did])
        # identical embed-text under the same model => identical vector; reuse it (also covers renamed ids)
        (reuse if src is not None else embed).append((did, src) if src is not None else did)
    if reuse:
        vecs = store.get_vectors([s_ for _, s_ in reuse])
        store.upsert([d for d, _ in reuse], vecs, [corpus.content_hashes[d] for d, _ in reuse])
        rep.n_reused_same_content = len(reuse)
    groups: dict[str, list[str]] = {}
    for did in embed:
        groups.setdefault(corpus.content_hashes[did], []).append(did)
    reps = [g[0] for g in groups.values()]  # embed each distinct text once
    emb_t = 0.0
    for bi, s0 in enumerate(range(0, len(reps), batch_size)):
        rids = reps[s0:s0 + batch_size]
        t1 = time.perf_counter()
        vecs = embedder.encode(corpus.embed_texts(rids), batch_size=batch_size)
        emb_t += time.perf_counter() - t1
        if vecs.shape != (len(rids), embedder.dim):
            raise ValueError(f"embedder returned shape {vecs.shape}, expected {(len(rids), embedder.dim)}")
        all_ids, all_vecs, all_h = [], [], []
        for rid, v in zip(rids, vecs):
            h = corpus.content_hashes[rid]
            for did in groups[h]:
                all_ids.append(did); all_vecs.append(v); all_h.append(h)
        store.upsert(all_ids, np.stack(all_vecs), all_h)
        rep.n_embedded += len(rids)
        if checkpoint_every and (bi + 1) % checkpoint_every == 0:
            store.flush()  # an interrupted build keeps its progress
    if remove_missing and diff.removed:
        store.delete(diff.removed)
    store.flush()
    rep.embed_seconds = emb_t
    log.info("embedding sync: %s (%.2fs total)", rep.to_dict(), time.perf_counter() - t0)
    return rep
