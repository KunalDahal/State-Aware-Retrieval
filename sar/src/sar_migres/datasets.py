"""Dataset loaders and ground-truth helpers for the methodology datasets."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import DatasetNotFoundError
from .logging_utils import get_logger
from .schemas import Document
from .utils import sha256_text

DATASET_HELP = {
    "wikihop": "WikiHop JSON supplied by MIGRES or the official dataset.",
    "hotpotqa": "HotpotQA JSON supplied by MIGRES or the official dataset.",
    "musique": "MuSiQue JSON supplied by MIGRES or the official dataset.",
    "natural_questions": "Natural Questions rows tagged dataset=='nq'.",
    "triviaqa": "TriviaQA rows tagged dataset=='trivia'.",
    "strategyqa": "StrategyQA JSON.",
}
_ODQA_TAG = {"natural_questions": "nq", "triviaqa": "trivia"}
_ID_KEYS = {"wikihop": ("_id",), "hotpotqa": ("_id",), "musique": ("id",), "natural_questions": ("id",),
            "triviaqa": ("id",), "strategyqa": ("qid",), "jsonl": ("id", "_id", "qid"), "sample": ("id",)}
SUPPORTED = tuple(_ID_KEYS)
_log = get_logger("datasets")


def _read_records(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if split:
                key_map = {str(k).lower(): k for k in data.keys()}
                for candidate in (split.lower(), "dev" if split.lower() == "validation" else "", "development" if split.lower() == "validation" else ""):
                    if candidate and candidate in key_map and isinstance(data[key_map[candidate]], list):
                        data = data[key_map[candidate]]
                        break
                else:
                    data = data.get("data", [data])
            else:
                data = data.get("data", [data])
        rows = list(data)
        if split:
            tagged = [r for r in rows if isinstance(r, dict) and any(k in r for k in ("split", "partition", "set"))]
            if tagged:
                target = split.lower()
                aliases = {target, "dev"} if target == "validation" else {target}
                rows = [r for r in rows if str(r.get("split", r.get("partition", r.get("set", "")))).lower() in aliases]
        return rows
    except json.JSONDecodeError:
        out = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DatasetNotFoundError(f"{path}: line {i} is not valid JSON ({e})") from e
        return out


def _maybe_literal(v: Any) -> Any:
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def load_questions(name: str, path: str | Path, limit: int | None = None, split: str | None = None) -> list[dict[str, Any]]:
    if name not in SUPPORTED:
        raise ValueError(f"unknown dataset {name!r}; supported: {SUPPORTED}")
    p = Path(path)
    if not p.exists():
        raise DatasetNotFoundError(f"dataset file not found: {p}\n{DATASET_HELP.get(name, '')}\nSee data/README.md for the expected dataset layout.")
    rows = _read_records(p, split=split)
    tag = _ODQA_TAG.get(name)
    out, seen = [], set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not str(r.get("question", "")).strip():
            raise DatasetNotFoundError(f"{p}: record {i} has no non-empty question")
        if tag and "dataset" in r and r["dataset"] != tag:
            continue
        qid = next((str(r[k]) for k in _ID_KEYS[name] if r.get(k) not in (None, "")), None) or f"{name}-{i}"
        if qid in seen:
            raise DatasetNotFoundError(f"{p}: duplicate question id {qid!r}")
        seen.add(qid)
        rec = dict(r)
        rec.update(id=qid, question=str(r["question"]).strip(), answer=_maybe_literal(r.get("answer")), dataset=name)
        out.append(rec)
        if limit and len(out) >= limit:
            break
    if not out:
        raise DatasetNotFoundError(f"{p}: no usable records for {name!r}")
    return out



def split_holdout_from_train(records: list[dict[str, Any]], fraction: float = 0.1, seed: int = 13) -> list[dict[str, Any]]:
    """Deterministically hold out a validation set from training records.

    The final labeled evaluation uses the public development split by default. This helper keeps
    tuning data separate from that final split when a dataset does not expose a labeled test set.
    Selection is hash-based, so ordering of the source file cannot affect membership.
    """
    if not (0.0 < float(fraction) < 1.0):
        raise ValueError("fraction must be in (0,1)")
    scored = []
    for r in records:
        qid = str(r.get("id", r.get("_id", r.get("qid", ""))))
        h = sha256_text(f"{seed}:{qid}")
        bucket = int(h[:12], 16) / float(16**12)
        scored.append((bucket, r))
    selected = [r for bucket, r in scored if bucket < fraction]
    if not selected and records:
        selected = [min(scored, key=lambda x: x[0])[1]]
    return selected


def load_phase_questions(name: str, path: str | Path, phase: str, *, limit: int | None = None,
                         final_split: str = "dev", validation_fraction: float = 0.1,
                         validation_seed: int = 13) -> list[dict[str, Any]]:
    """Load questions according to the project's two-phase protocol.

    ``tuning`` is a deterministic holdout from the training split. ``final`` uses the labeled
    development split by default because many benchmark test sets do not publish answer labels.
    """
    phase = str(phase).lower()
    if phase == "tuning":
        train = load_questions(name, path, None, split="train")
        tagged = split_holdout_from_train(train, validation_fraction, validation_seed)
        if limit is not None:
            tagged = tagged[:limit]
        return tagged
    if phase == "final":
        return load_questions(name, path, limit, split=final_split)
    raise ValueError("phase must be tuning or final")


def dataset_statistics(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(items)
    with_gold = sum(1 for r in rows if r.get("answer") is not None)
    return {
        "n_questions": len(rows),
        "n_with_gold": with_gold,
        "n_without_gold": len(rows) - with_gold,
    }

def contexts_to_documents(name: str, records: list[dict[str, Any]]) -> Iterator[Document]:
    if name not in ("hotpotqa", "wikihop", "musique"):
        raise ValueError(f"no per-question contexts to build a corpus from for {name!r}")
    seen: set[str] = set(); n_skipped = 0; yielded_records = 0
    for r in records:
        pairs: list[tuple[str, str]] = []
        if name in ("hotpotqa", "wikihop"):
            ctx = _maybe_literal(r.get("context"))
            if not isinstance(ctx, list) or not ctx:
                n_skipped += 1; continue
            for item in ctx:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                title, sents = str(item[0]), item[1]
                pairs.append((title, " ".join(str(x).strip() for x in sents if str(x).strip())))
        else:
            paras = _maybe_literal(r.get("paragraphs"))
            if not isinstance(paras, list) or not paras:
                n_skipped += 1; continue
            for para in paras:
                if isinstance(para, dict):
                    pairs.append((str(para.get("title", "")), str(para.get("paragraph_text", para.get("text", ""))).strip()))
        got = False
        for title, text in pairs:
            if not text:
                continue
            got = True
            h = sha256_text(title + "\n" + text)
            if h in seen:
                continue
            seen.add(h)
            yield Document(doc_id="ctx-" + h[:16], text=text, title=title)
        yielded_records += int(got)
    if n_skipped:
        _log.warning("%s: %d records have no usable context", name, n_skipped)
    if not yielded_records:
        raise DatasetNotFoundError(f"{name}: no records yielded corpus documents; supply a corpus file or provide records with usable context. See data/README.md.")


def gold_answer_aliases(item: dict[str, Any]) -> list[str]:
    from .evaluation import gold_aliases
    return gold_aliases(item.get("answer"))


def relevant_doc_ids(item: dict[str, Any], corpus: Iterable[Document] | Any) -> set[str]:
    """Map supported evidence annotations to corpus document IDs.

    HotpotQA/WikiHop are title-addressable. MuSiQue can be paragraph-addressable via common
    ``is_supporting``/``is_supporting_paragraph`` fields. Other datasets return no retrieval
    IDs unless the record explicitly provides ``supporting_doc_ids``/``relevant_docs``.
    """
    explicit = item.get("supporting_doc_ids") or item.get("relevant_docs") or item.get("positive_doc_ids")
    if isinstance(explicit, (list, tuple, set)):
        return {str(x) for x in explicit}
    name = str(item.get("dataset", "")).lower()
    docs_iter = corpus.iter_documents() if hasattr(corpus, "iter_documents") else iter(corpus)
    out: set[str] = set()
    if name in {"hotpotqa", "wikihop"}:
        sf = _maybe_literal(item.get("supporting_facts")) or []
        titles = {str(x[0]) for x in sf if isinstance(x, (list, tuple)) and x}
        for d in docs_iter:
            if d.title in titles:
                out.add(d.doc_id)
    elif name == "musique":
        paras = _maybe_literal(item.get("paragraphs")) or []
        for i, p in enumerate(paras):
            if not isinstance(p, dict):
                continue
            flag = bool(p.get("is_supporting", p.get("is_supporting_paragraph", False)))
            if flag:
                title = str(p.get("title", ""))
                text = str(p.get("paragraph_text", p.get("text", ""))).strip()
                h = sha256_text(title + "\n" + text)
                out.add("ctx-" + h[:16])
    return out


def load_rgb(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load an RGB file; the evaluator handles the four robustness settings."""
    p = Path(path)
    if not p.exists():
        raise DatasetNotFoundError(f"RGB file not found: {p}")
    out = []
    for i, r in enumerate(_read_records(p)):
        if not isinstance(r, dict):
            continue
        rec = dict(r)
        rec.update(id=str(r.get("id", i)), question=str(r.get("query", r.get("question", ""))),
                   answer=r.get("answer"), dataset="rgb")
        out.append(rec)
        if limit and len(out) >= limit:
            break
    return out
