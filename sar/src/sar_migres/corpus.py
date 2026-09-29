"""Searchable corpus D: loading, validation, hashing, change detection.

Record formats accepted (same as MIGRES' own corpus files):
  * JSONL / JSON list: ``{"id"|"doc_id"|"_id": ..., "title": ..., "text"|"contents": ...}``
  * DPR-style TSV (``id<TAB>text<TAB>title`` with header, e.g. wiki18 ``psgs_w100.tsv``)

Design notes
  * Ranking needs only ids + content hashes; full text is fetched lazily (``lazy=True``
    keeps byte offsets into the JSONL file instead of texts) so big corpora need not sit in RAM.
  * A document's *content hash* covers exactly the text that is embedded, so any change to
    title/text (or to the embed template) triggers re-embedding and nothing else does.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import CorpusFormatError
from .logging_utils import get_logger
from .schemas import Document
from .utils import sha256_text

log = get_logger("corpus")

_ID_KEYS = ("doc_id", "id", "_id")
_TEXT_KEYS = ("text", "contents")
DEFAULT_TEMPLATE = "{title}. {text}"


@dataclass
class CorpusLoadReport:
    n_read: int = 0
    n_valid: int = 0
    n_malformed: int = 0
    n_duplicate_identical: int = 0
    n_same_text_different_id: int = 0
    malformed_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CorpusDiff:
    new: list[str]
    changed: list[str]
    unchanged: list[str]
    removed: list[str]


def record_to_document(rec: dict[str, Any]) -> Document:
    """Validate one raw record. Raises ``CorpusFormatError`` when malformed."""
    if not isinstance(rec, dict):
        raise CorpusFormatError(f"record is {type(rec).__name__}, not an object")
    did = next((rec[k] for k in _ID_KEYS if k in rec and rec[k] not in (None, "")), None)
    if did is None:
        raise CorpusFormatError("missing id (expected one of doc_id/id/_id)")
    text = next((rec[k] for k in _TEXT_KEYS if k in rec and rec[k] is not None), None)
    if not isinstance(text, str) or not text.strip():
        raise CorpusFormatError(f"document {did!r}: empty or non-string text")
    title = rec.get("title") or ""
    if not isinstance(title, str):
        raise CorpusFormatError(f"document {did!r}: title is not a string")
    meta = {k: v for k, v in rec.items() if k not in set(_ID_KEYS) | set(_TEXT_KEYS) | {"title"}}
    return Document(doc_id=str(did), text=text.strip(), title=title.strip(), metadata=meta)


def iter_raw_records(path: str | Path) -> Iterator[tuple[int, Any]]:
    """Yield (byte_offset_or_line_no, record). Malformed JSON lines yield (n, Exception)."""
    path = Path(path)
    if not path.exists():
        raise CorpusFormatError(f"corpus file not found: {path}")
    suf = path.suffix.lower()
    if suf == ".tsv":
        csv.field_size_limit(sys.maxsize)
        with open(path, newline="", encoding="utf-8") as fh:
            for i, row in enumerate(csv.DictReader(fh, delimiter="\t")):
                yield i, row
    elif suf == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if isinstance(data, list):
            for i, r in enumerate(data):
                yield i, r
            return
        yield from _iter_jsonl(path)
    else:
        yield from _iter_jsonl(path)


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Any]]:
    with open(path, "rb") as fh:
        offset = 0
        for raw in fh:
            start, offset = offset, offset + len(raw)
            if not raw.strip():
                continue
            try:
                yield start, json.loads(raw.decode("utf-8"))
            except Exception as e:  # noqa: BLE001 - reported by caller
                yield start, e


class Corpus:
    """Ordered collection of documents with content hashes."""

    def __init__(self, embed_template: str = DEFAULT_TEMPLATE):
        self.embed_template = embed_template
        self._docs: dict[str, Document] = {}
        self._hashes: dict[str, str] = {}
        self._offsets: dict[str, int] = {}
        self._path: Path | None = None
        self.report = CorpusLoadReport()

    # ---- construction -------------------------------------------------
    def _add(self, doc: Document, offset: int | None, keep_text: bool, seen_text: Counter) -> None:
        h = sha256_text(doc.embed_text(self.embed_template))
        if doc.doc_id in self._hashes:
            if self._hashes[doc.doc_id] == h:
                self.report.n_duplicate_identical += 1
                log.warning("duplicate document id %r with identical content skipped", doc.doc_id)
                return
            raise CorpusFormatError(f"document id {doc.doc_id!r} appears twice with DIFFERENT content; "
                                    "ids must be unique (fix the corpus or rename one document)")
        if seen_text[h] > 0:
            self.report.n_same_text_different_id += 1
        seen_text[h] += 1
        self._hashes[doc.doc_id] = h
        if keep_text:
            self._docs[doc.doc_id] = doc
        if offset is not None:
            self._offsets[doc.doc_id] = offset

    @classmethod
    def from_documents(cls, docs: Iterable[Document], embed_template: str = DEFAULT_TEMPLATE) -> "Corpus":
        c = cls(embed_template)
        seen: Counter = Counter()
        for d in docs:
            c.report.n_read += 1
            c._add(d, None, True, seen)
            c.report.n_valid += 1
        return c

    @classmethod
    def from_file(cls, path: str | Path, embed_template: str = DEFAULT_TEMPLATE, lazy: bool = False,
                  max_malformed_fraction: float = 0.01) -> "Corpus":
        c = cls(embed_template)
        c._path = Path(path)
        seen: Counter = Counter()
        keep_text = not lazy or c._path.suffix.lower() not in (".jsonl",)
        for off, rec in iter_raw_records(path):
            c.report.n_read += 1
            try:
                if isinstance(rec, Exception):
                    raise CorpusFormatError(f"invalid JSON: {rec}")
                doc = record_to_document(rec)
            except CorpusFormatError as e:
                c.report.n_malformed += 1
                if len(c.report.malformed_examples) < 20:
                    c.report.malformed_examples.append(f"record@{off}: {e}")
                    log.error("malformed corpus record @%s skipped: %s", off, e)
                continue
            c._add(doc, off if (lazy and keep_text is False) else None, keep_text, seen)
            c.report.n_valid += 1
        r = c.report
        if r.n_read == 0:
            raise CorpusFormatError(f"corpus {path} contains no records")
        frac = r.n_malformed / r.n_read
        if frac > max_malformed_fraction:
            raise CorpusFormatError(
                f"{r.n_malformed}/{r.n_read} malformed records ({frac:.1%}) exceeds "
                f"max_malformed_fraction={max_malformed_fraction:.1%}; first: {r.malformed_examples[:3]}")
        if r.n_malformed:
            log.error("corpus loaded with %d malformed records skipped (see report)", r.n_malformed)
        if r.n_valid == 0:
            raise CorpusFormatError("no valid documents")
        return c

    # ---- access -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._hashes)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._hashes

    @property
    def ids(self) -> list[str]:
        return list(self._hashes)

    @property
    def content_hashes(self) -> dict[str, str]:
        return self._hashes

    def get(self, doc_id: str) -> Document:
        if doc_id in self._docs:
            return self._docs[doc_id]
        if doc_id in self._offsets and self._path is not None:
            with open(self._path, "rb") as fh:
                fh.seek(self._offsets[doc_id])
                return record_to_document(json.loads(fh.readline().decode("utf-8")))
        raise KeyError(doc_id)

    def get_many(self, ids: Iterable[str]) -> list[Document]:
        return [self.get(i) for i in ids]

    def embed_texts(self, ids: Iterable[str]) -> list[str]:
        return [self.get(i).embed_text(self.embed_template) for i in ids]

    # ---- versioning / diff ---------------------------------------------
    @property
    def version(self) -> str:
        """Order-independent hash of (id, content-hash) pairs - the corpus version."""
        lines = sorted(f"{i}\t{h}" for i, h in self._hashes.items())
        return sha256_text("\n".join(lines))

    def diff(self, stored: dict[str, str]) -> CorpusDiff:
        new = [i for i in self._hashes if i not in stored]
        changed = [i for i, h in self._hashes.items() if i in stored and stored[i] != h]
        unchanged = [i for i, h in self._hashes.items() if i in stored and stored[i] == h]
        removed = [i for i in stored if i not in self._hashes]
        return CorpusDiff(new, changed, unchanged, removed)

    # ---- mutation (used for incremental maintenance) ---------------------
    def add_or_update(self, docs: Iterable[Document]) -> tuple[list[str], list[str]]:
        """In-memory corpora only. Returns (added_ids, updated_ids)."""
        added, updated = [], []
        for d in docs:
            h = sha256_text(d.embed_text(self.embed_template))
            if d.doc_id not in self._hashes:
                added.append(d.doc_id)
            elif self._hashes[d.doc_id] != h:
                updated.append(d.doc_id)
            else:
                continue
            self._hashes[d.doc_id] = h
            self._docs[d.doc_id] = d
        return added, updated

    def remove(self, doc_ids: Iterable[str]) -> None:
        for i in doc_ids:
            self._hashes.pop(i, None)
            self._docs.pop(i, None)
            self._offsets.pop(i, None)
