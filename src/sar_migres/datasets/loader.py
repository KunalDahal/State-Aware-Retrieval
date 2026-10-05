from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from ..utils.errors import DatasetNotFoundError
from ..utils.logging import get_logger
from ..utils.hashing import sha256_text
from .normalization import _maybe_literal

DATASET_HELP = {
    "hotpotqa": "HotpotQA JSON from the distractor setting.",
    "musique": "MuSiQue JSON with paragraphs [{title, paragraph_text, is_supporting}] and answer aliases.",
    "natural_questions": "Natural Questions JSON/JSONL with question_text/document/long_answer/short_answers/yes_no_answer, or normalized question/answer fields.",
}

_ODQA_TAG = {"natural_questions": "nq"}

_ID_KEYS = {"hotpotqa": ("_id",), "musique": ("id",), "natural_questions": ("id", "example_id", "qid"), "jsonl": ("id", "_id", "qid"), "sample": ("id",)}

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
        if not isinstance(r, dict):
            raise DatasetNotFoundError(f"{p}: record {i} is not an object")
        if tag and "dataset" in r and r["dataset"] != tag:
            continue
        question = r.get("question")
        if not str(question or "").strip() and name == "natural_questions":
            question = r.get("question_text")
        if not str(question or "").strip():
            raise DatasetNotFoundError(f"{p}: record {i} has no non-empty question/question_text")
        qid = next((str(r[k]) for k in _ID_KEYS[name] if r.get(k) not in (None, "")), None) or f"{name}-{i}"
        if qid in seen:
            raise DatasetNotFoundError(f"{p}: duplicate question id {qid!r}")
        seen.add(qid)
        rec = dict(r)
        answer = _maybe_literal(r.get("answer"))
        if name == "natural_questions" and answer in (None, ""):
            short = _maybe_literal(r.get("short_answers"))
            yn = r.get("yes_no_answer")
            if isinstance(short, list) and short:
                answer = [str(x.get("text", x)).strip() for x in short if str(x.get("text", x)).strip()]
                if len(answer) == 1:
                    answer = answer[0]
            elif yn not in (None, "", "NONE", "none"):
                answer = yn
        aliases = _maybe_literal(r.get("answer_aliases"))
        if aliases is not None:
            rec["answer_aliases"] = aliases if isinstance(aliases, list) else [aliases]
        rec.update(id=qid, question=str(question).strip(), answer=answer, dataset=name)
        out.append(rec)
        if limit and len(out) >= limit:
            break
    if not out:
        raise DatasetNotFoundError(f"{p}: no usable records for {name!r}")
    return out


def gold_answer_value(item: dict[str, Any]) -> Any:
    """Return the canonical gold answer(s), including dataset-provided aliases."""
    base = _maybe_literal(item.get("answer"))
    if base in (None, "") and str(item.get("dataset", "")).lower() in {"natural_questions", "nq"}:
        short = _maybe_literal(item.get("short_answers"))
        yn = item.get("yes_no_answer")
        if isinstance(short, list) and short:
            vals = [str(x.get("text", x)).strip() for x in short if str(x.get("text", x)).strip()]
            base = vals[0] if len(vals) == 1 else vals
        elif yn not in (None, "", "NONE", "none"):
            base = yn
    aliases = _maybe_literal(item.get("answer_aliases"))
    if aliases is None:
        return base
    alias_list = aliases if isinstance(aliases, (list, tuple, set)) else [aliases]
    base_list = list(base) if isinstance(base, (list, tuple, set)) else ([] if base in (None, "") else [base])
    merged = []
    for value in [*base_list, *alias_list]:
        if value not in merged:
            merged.append(value)
    return merged


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
    target_n = max(1, int(len(records) * float(fraction))) if records else 0
    scored.sort(key=lambda x: (x[0], str(x[1].get("id", x[1].get("_id", x[1].get("qid", ""))))))
    return [r for _, r in scored[:target_n]]


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
