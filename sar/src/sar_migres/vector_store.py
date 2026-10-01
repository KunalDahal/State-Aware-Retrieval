"""Persistent vector stores with chunked scoring support.

Two local modes exist:

* ``LocalVectorStore`` keeps the matrix in memory for tiny/offline tests.
* ``DiskMemmapVectorStore`` keeps vectors on disk and metadata in SQLite, so the dense
  matrix is never materialised in RAM.  This is the memory-safe store for large corpora.

All stores are local; ``DiskMemmapVectorStore`` is the primary store for exact full-corpus scoring.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .corpus import Corpus
from .embeddings import Embedder
from .errors import EmbeddingCacheCorruptionError, StoreIncompatibleError
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
    def hashes(self) -> Mapping[str, str]: ...

    @abstractmethod
    def upsert(self, ids: Sequence[str], vectors: np.ndarray, hashes: Sequence[str]) -> None: ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None: ...

    @abstractmethod
    def get_vectors(self, ids: Sequence[str]) -> np.ndarray: ...

    @abstractmethod
    def ids(self) -> list[str]: ...

    @abstractmethod
    def iter_chunks(self, chunk_size: int) -> Iterator[tuple[int, np.ndarray, np.ndarray, Sequence[str]]]: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    def __len__(self) -> int:
        return len(self.ids())

    def find_id_by_hash(self, content_hash: str) -> str | None:
        for did, h in self.hashes().items():
            if h == content_hash:
                return did
        return None


class _Matrix:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self._ids: list[str] = []
        self._hash: list[str] = []
        self._mat = np.zeros((0, dim), dtype=np.float32)
        self._row: dict[str, int] = {}
        self._norms: np.ndarray | None = None

    def hashes(self) -> Mapping[str, str]:
        return dict(zip(self._ids, self._hash))

    def ids(self) -> list[str]:
        return list(self._ids)

    def upsert(self, ids, vectors, hashes):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim or len(ids) != vectors.shape[0] or len(hashes) != len(ids):
            raise ValueError(f"bad vectors shape {vectors.shape} for dim={self.dim}, n_ids={len(ids)}")
        if not np.isfinite(vectors).all():
            raise ValueError("non-finite values in embeddings")
        pending: dict[str, tuple[np.ndarray, str]] = {}
        for did, vec, h in zip(ids, vectors, hashes):
            pending[str(did)] = (vec, str(h))
        for did, (vec, h) in pending.items():
            if did in self._row:
                r = self._row[did]
                self._mat[r] = vec
                self._hash[r] = h
            else:
                self._row[did] = len(self._ids)
                self._ids.append(did)
                self._hash.append(h)
                self._mat = np.vstack([self._mat, vec[None, :]])
        self._norms = None

    def delete(self, ids):
        drop = {str(i) for i in ids if str(i) in self._row}
        if not drop:
            return
        keep = [r for r, did in enumerate(self._ids) if did not in drop]
        self._mat = np.array(self._mat[keep], dtype=np.float32) if keep else np.zeros((0, self.dim), np.float32)
        self._ids = [self._ids[r] for r in keep]
        self._hash = [self._hash[r] for r in keep]
        self._row = {did: r for r, did in enumerate(self._ids)}
        self._norms = None

    def get_vectors(self, ids):
        return np.stack([self._mat[self._row[str(i)]] for i in ids]) if ids else np.zeros((0, self.dim), np.float32)

    def iter_chunks(self, chunk_size):
        nr = l2_norms(self._mat)
        for s in range(0, len(self._ids), chunk_size):
            e = min(s + chunk_size, len(self._ids))
            yield s, np.asarray(self._mat[s:e]), nr[s:e], self._ids[s:e]


class LocalVectorStore(_Matrix, VectorStore):
    """Small-corpus store: NumPy matrix + JSON manifest. Use DiskMemmap for large corpora."""

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
            mat = np.load(vp, mmap_mode="r+")
            if mat.shape != (len(man["ids"]), man["dim"]) or len(man["hashes"]) != len(man["ids"]):
                raise EmbeddingCacheCorruptionError(f"shape {mat.shape} inconsistent with manifest")
        except EmbeddingCacheCorruptionError:
            raise
        except Exception as e:
            raise EmbeddingCacheCorruptionError(f"cannot read cache in {self.directory}: {e}") from e
        if man["embedder"] != self.embedder_name or man["dim"] != self.dim:
            raise StoreIncompatibleError(
                f"cache in {self.directory} was built with embedder={man['embedder']!r} dim={man['dim']}, "
                f"but current embedder={self.embedder_name!r} dim={self.dim}. Re-run with rebuild=True."
            )
        self._ids, self._hash, self._mat = list(man["ids"]), list(man["hashes"]), mat
        self._row = {d: r for r, d in enumerate(self._ids)}
        self.events = man.get("events", [])

    def flush(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        vp, mp = self._paths()
        tmp = vp.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(self._mat, dtype=np.float32))
        os.replace(tmp, vp)
        man = {"embedder": self.embedder_name, "dim": self.dim, "ids": self._ids, "hashes": self._hash,
               "vectors_sha256": sha256_file(vp), "updated": utc_now_iso(), "events": self.events[-50:]}
        atomic_write_text(mp, json.dumps(man))

    def clear(self) -> None:
        _Matrix.__init__(self, self.dim)
        for p in self._paths():
            if p.exists():
                p.unlink()


_SQL_IN_BATCH = 500  # well below SQLite's bound-parameter limit (999 on old builds)


def _batches(items: list, n: int = _SQL_IN_BATCH):
    for s in range(0, len(items), n):
        yield items[s:s + n]


class _SQLiteHashMapping(Mapping[str, str]):
    """Read-only Mapping view over DiskMemmapVectorStore metadata without building a dict."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def __len__(self) -> int:
        with sqlite3.connect(self.db_path) as con:
            return int(con.execute("SELECT COUNT(*) FROM vectors WHERE active=1").fetchone()[0])

    def __iter__(self):
        with sqlite3.connect(self.db_path) as con:
            for (did,) in con.execute("SELECT doc_id FROM vectors WHERE active=1 ORDER BY row_idx"):
                yield str(did)

    def __getitem__(self, key: str) -> str:
        with sqlite3.connect(self.db_path) as con:
            row = con.execute("SELECT content_hash FROM vectors WHERE doc_id=? AND active=1", (str(key),)).fetchone()
        if row is None:
            raise KeyError(key)
        return str(row[0])

    def __contains__(self, key: object) -> bool:
        with sqlite3.connect(self.db_path) as con:
            return con.execute("SELECT 1 FROM vectors WHERE doc_id=? AND active=1", (str(key),)).fetchone() is not None

    # The Mapping defaults call __getitem__ (one SQLite connection each) per key; these stream
    # the whole table with a single query instead.
    def items(self):
        with sqlite3.connect(self.db_path) as con:
            for did, h in con.execute("SELECT doc_id, content_hash FROM vectors WHERE active=1 ORDER BY row_idx"):
                yield str(did), str(h)

    def values(self):
        for _did, h in self.items():
            yield h


class DiskMemmapVectorStore(VectorStore):
    """Disk-backed vectors + SQLite metadata, designed for exact full-corpus SAR scoring."""

    def __init__(self, directory: str | Path, dim: int, embedder_name: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.embedder_name = embedder_name
        self.dim = int(dim)
        self.db_path = self.directory / "metadata.sqlite3"
        self.vector_path = self.directory / "vectors.f32"
        self.meta_path = self.directory / "store.json"
        self._init_db()
        self._capacity = 0
        self._mm: np.memmap | None = None
        self._load_meta()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS vectors (doc_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, row_idx INTEGER UNIQUE NOT NULL, active INTEGER NOT NULL DEFAULT 1)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_vectors_hash ON vectors(content_hash, active)")
            con.commit()

    def _load_meta(self):
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("embedder") != self.embedder_name or int(meta.get("dim", -1)) != self.dim:
                raise StoreIncompatibleError(
                    f"disk store {self.directory} was built with embedder={meta.get('embedder')!r}, dim={meta.get('dim')}; "
                    f"current embedder={self.embedder_name!r}, dim={self.dim}"
                )
            self._capacity = int(meta.get("capacity", 0))
        count_bytes = self.vector_path.stat().st_size if self.vector_path.exists() else 0
        if count_bytes and self._capacity == 0:
            self._capacity = count_bytes // (4 * self.dim)
        if self._capacity and count_bytes != self._capacity * self.dim * 4:
            raise EmbeddingCacheCorruptionError(f"vector file size {count_bytes} incompatible with capacity {self._capacity}")
        if self._capacity:
            self._mm = np.memmap(self.vector_path, mode="r+", dtype=np.float32, shape=(self._capacity, self.dim))

    def _write_meta(self):
        active = len(self)
        atomic_write_text(self.meta_path, json.dumps({"embedder": self.embedder_name, "dim": self.dim,
                                                       "capacity": self._capacity, "active": active,
                                                       "updated": utc_now_iso()}, indent=2))

    def _ensure_capacity(self, needed: int):
        needed = int(needed)
        if needed <= self._capacity:
            return
        cap = max(1024, self._capacity)
        while cap < needed:
            cap = int(cap * 1.5) + 1
        if self._mm is not None:
            self._mm.flush()
            del self._mm
            self._mm = None
        with open(self.vector_path, "ab") as fh:
            fh.truncate(cap * self.dim * 4)
        self._capacity = cap
        self._mm = np.memmap(self.vector_path, mode="r+", dtype=np.float32, shape=(cap, self.dim))
        self._write_meta()

    def hashes(self) -> Mapping[str, str]:
        return _SQLiteHashMapping(self.db_path)

    def find_id_by_hash(self, content_hash: str) -> str | None:
        with sqlite3.connect(self.db_path) as con:
            row = con.execute("SELECT doc_id FROM vectors WHERE content_hash=? AND active=1 ORDER BY row_idx LIMIT 1", (content_hash,)).fetchone()
        return str(row[0]) if row else None

    def ids(self) -> list[str]:
        with sqlite3.connect(self.db_path) as con:
            return [str(r[0]) for r in con.execute("SELECT doc_id FROM vectors WHERE active=1 ORDER BY row_idx")]

    def __len__(self) -> int:
        with sqlite3.connect(self.db_path) as con:
            return int(con.execute("SELECT COUNT(*) FROM vectors WHERE active=1").fetchone()[0])

    def upsert(self, ids, vectors, hashes):
        ids = [str(i) for i in ids]
        hashes = [str(h) for h in hashes]
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(ids), self.dim) or len(hashes) != len(ids):
            raise ValueError(f"bad vectors shape {vectors.shape} for {len(ids)} ids and dim={self.dim}")
        if not np.isfinite(vectors).all():
            raise ValueError("non-finite values in embeddings")
        if not ids:
            return
        pending: dict[str, tuple[np.ndarray, str]] = {did: (v, h) for did, v, h in zip(ids, vectors, hashes)}
        with sqlite3.connect(self.db_path) as con:
            # Include inactive rows: a deleted id that comes back must reuse its row, because
            # doc_id is the PRIMARY KEY (a second INSERT would raise IntegrityError).
            existing = {}
            for batch in _batches(list(pending)):
                existing.update({str(did): int(row_idx) for did, row_idx in con.execute(
                    f"SELECT doc_id,row_idx FROM vectors WHERE doc_id IN ({','.join('?' for _ in batch)})", batch)})
            max_row = con.execute("SELECT COALESCE(MAX(row_idx), -1) FROM vectors").fetchone()[0]
            new = [did for did in pending if did not in existing]
            self._ensure_capacity(int(max_row) + 1 + len(new))
            for did, (vec, h) in pending.items():
                if did in existing:
                    row_idx = int(existing[did])
                    self._mm[row_idx] = vec
                    con.execute("UPDATE vectors SET content_hash=?, active=1 WHERE doc_id=?", (h, did))
                else:
                    max_row += 1
                    self._mm[max_row] = vec
                    con.execute("INSERT INTO vectors(doc_id,content_hash,row_idx,active) VALUES(?,?,?,1)", (did, h, max_row))
            con.commit()
        self.flush()

    def delete(self, ids):
        ids = [str(i) for i in ids]
        if not ids:
            return
        with sqlite3.connect(self.db_path) as con:
            for batch in _batches(ids):
                con.execute(f"UPDATE vectors SET active=0 WHERE doc_id IN ({','.join('?' for _ in batch)})", batch)
            con.commit()
        self.flush()

    def get_vectors(self, ids):
        ids = [str(i) for i in ids]
        if not ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: dict[str, int] = {}
        with sqlite3.connect(self.db_path) as con:
            for batch in _batches(ids):
                rows.update({str(did): int(row_idx) for did, row_idx in con.execute(
                    f"SELECT doc_id,row_idx FROM vectors WHERE active=1 AND doc_id IN ({','.join('?' for _ in batch)})", batch)})
        missing = [did for did in ids if did not in rows]
        if missing:
            raise KeyError(missing[0])
        return np.stack([np.asarray(self._mm[int(rows[did])], dtype=np.float32) for did in ids])

    def iter_chunks(self, chunk_size: int):
        if self._mm is None:
            return
        last_row = -1
        while True:
            with sqlite3.connect(self.db_path) as con:
                rows = list(con.execute(
                    "SELECT row_idx,doc_id FROM vectors WHERE active=1 AND row_idx>? ORDER BY row_idx LIMIT ?",
                    (last_row, int(chunk_size)),
                ))
            if not rows:
                break
            row_idx = [int(r[0]) for r in rows]
            dids = [str(r[1]) for r in rows]
            mat = np.stack([np.asarray(self._mm[r], dtype=np.float32) for r in row_idx])
            norms = l2_norms(mat)
            yield row_idx[0], mat, norms, dids
            last_row = row_idx[-1]

    def flush(self) -> None:
        if self._mm is not None:
            self._mm.flush()
        self._write_meta()

    def clear(self) -> None:
        if self._mm is not None:
            self._mm.flush()
            del self._mm
            self._mm = None
        for p in (self.vector_path, self.meta_path):
            if p.exists():
                p.unlink()
        if self.db_path.exists():
            self.db_path.unlink()
        self._capacity = 0
        self._init_db()


def _quarantine(directory: Path) -> None:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for p in directory.glob("*"):
        if p.name.endswith(f".corrupt-{stamp}"):
            continue
        try:
            p.rename(directory / f"{p.name}.corrupt-{stamp}")
        except OSError:
            pass


def _open_local(directory, dim, embedder_name, on_corruption: str = "rebuild", rebuild: bool = False) -> LocalVectorStore:
    directory = Path(directory)
    try:
        return LocalVectorStore(directory, dim, embedder_name)
    except EmbeddingCacheCorruptionError:
        if on_corruption != "rebuild":
            raise
        log.error("embedding cache is corrupt; quarantining %s and rebuilding", directory)
        _quarantine(directory)
        return LocalVectorStore(directory, dim, embedder_name)
    except StoreIncompatibleError:
        if not rebuild:
            raise
        log.warning("incompatible local cache; rebuilding because rebuild=True")
        for p in (directory / "vectors.npy", directory / "manifest.json"):
            if p.exists():
                p.unlink()
        return LocalVectorStore(directory, dim, embedder_name)


def build_vector_store(cfg, embedder: Embedder, cache_root: str | Path = "data/cache", rebuild: bool = False) -> VectorStore:
    root = Path(cache_root)
    if cfg.type == "local":
        return _open_local(root / cfg.local_dir_name(embedder.name), embedder.dim, embedder.name,
                           on_corruption=cfg.on_cache_corruption, rebuild=rebuild)
    if cfg.type == "local_memmap":
        path = root / "memmap" / cfg.local_dir_name(embedder.name)
        try:
            return DiskMemmapVectorStore(path, embedder.dim, embedder.name)
        except (EmbeddingCacheCorruptionError, StoreIncompatibleError):
            if not rebuild and cfg.on_cache_corruption != "rebuild":
                raise
            log.warning("rebuilding disk-backed vector store at %s", path)
            import shutil
            shutil.rmtree(path, ignore_errors=True)
            return DiskMemmapVectorStore(path, embedder.dim, embedder.name)
    raise ValueError(f"unknown vector_store.type {cfg.type!r}")


def sync_corpus_embeddings(store: VectorStore, corpus: Corpus, embedder: Embedder, batch_size: int = 64,
                           remove_missing: bool = True, checkpoint_every: int = 20) -> SyncReport:
    """Incremental maintenance required by methodology Sec. 5.2.2.

    The normal path never materialises a full-corpus vector matrix in RAM.  New/changed
    documents are processed in embedding batches; duplicate-content vectors are copied in
    small batches from the existing store.  ``corpus.diff`` remains available for small callers,
    while this sync path uses its streaming counterpart where available.
    """
    t0 = time.perf_counter()
    # One streaming read of (id, hash) pairs. ``corpus.diff`` probes every id several times, and on a
    # SQLite-backed Mapping each probe is a separate connection + query (~35 s per no-op sync at 100k docs).
    stored = dict(store.hashes().items())
    diff = corpus.diff(stored)
    rep = SyncReport(n_total=len(corpus), n_new=len(diff.new), n_changed=len(diff.changed),
                     n_unchanged=len(diff.unchanged), n_removed=len(diff.removed))

    todo = diff.changed + diff.new
    reuse: list[tuple[str, str]] = []
    embed_ids: list[str] = []
    if hasattr(store, "find_id_by_hash"):
        for did in todo:
            src = store.find_id_by_hash(corpus.content_hashes[did])
            if src is None:
                embed_ids.append(did)
            else:
                reuse.append((did, src))
    else:
        by_hash: dict[str, str] = {}
        for did, h in stored.items():
            by_hash.setdefault(h, did)
        for did in todo:
            src = by_hash.get(corpus.content_hashes[did])
            if src:
                reuse.append((did, src))
            else:
                embed_ids.append(did)

    for bi, s0 in enumerate(range(0, len(reuse), batch_size)):
        batch = reuse[s0:s0 + batch_size]
        vecs = store.get_vectors([src for _, src in batch])
        store.upsert([did for did, _ in batch], vecs,
                     [corpus.content_hashes[did] for did, _ in batch])
        if checkpoint_every and (bi + 1) % checkpoint_every == 0:
            store.flush()
    rep.n_reused_same_content = len(reuse)

    groups: dict[str, list[str]] = {}
    for did in embed_ids:
        groups.setdefault(corpus.content_hashes[did], []).append(did)
    reps = [g[0] for g in groups.values()]
    emb_t = 0.0
    for bi, s0 in enumerate(range(0, len(reps), batch_size)):
        rids = reps[s0:s0 + batch_size]
        t1 = time.perf_counter()
        vecs = embedder.encode(corpus.embed_texts(rids), batch_size=batch_size)
        emb_t += time.perf_counter() - t1
        if vecs.shape != (len(rids), embedder.dim):
            raise ValueError(f"embedder returned {vecs.shape}, expected {(len(rids), embedder.dim)}")
        all_ids: list[str] = []
        all_vecs: list[np.ndarray] = []
        all_h: list[str] = []
        for rid, v in zip(rids, vecs):
            h = corpus.content_hashes[rid]
            for did in groups[h]:
                all_ids.append(did); all_vecs.append(v); all_h.append(h)
        store.upsert(all_ids, np.stack(all_vecs), all_h)
        rep.n_embedded += len(rids)
        if checkpoint_every and (bi + 1) % checkpoint_every == 0:
            store.flush()

    if remove_missing and diff.removed:
        store.delete(diff.removed)
    store.flush()
    rep.embed_seconds = emb_t
    log.info("embedding sync: %s (%.2fs total)", rep.to_dict(), time.perf_counter() - t0)
    return rep
