"""Dataset loaders for the datasets named in methodology.tex Sec. 5.4.1 (+ RGB, future).

Acquisition is deliberately separate: nothing here downloads data.  Loaders read LOCAL files
and raise ``DatasetNotFoundError`` with instructions when a file is missing.  Accepted layouts
are the ones verified in the official MIGRES repository (``data/*.json`` are JSON-lines files
despite the extension; plain JSON lists are accepted too).  No preprocessing beyond field
renaming is applied; official split handling is left to the file you provide.

    name                file (MIGRES data/)  fields used
    wikihop             wikihop.json         _id, question, answer, (context, supporting_facts kept)
    hotpotqa            hotpot.json          _id, question, answer, (context, supporting_facts kept)
    musique             musique.json         id, question, answer, (paragraphs, answer_aliases kept)
    natural_questions   odqa.json            id, question, answer, dataset == "nq"
    triviaqa            odqa.json            id, question, answer, dataset == "trivia"
    strategyqa          strategy.json        qid, question, answer, (facts, decomposition kept)
    jsonl               any                  question [+ id, answer]  (generic)
    sample              data/sample/...      bundled synthetic smoke-test questions
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterator

from .errors import DatasetNotFoundError
from .logging_utils import get_logger
from .schemas import Document
from .utils import sha256_text

DATASET_HELP = {
    "wikihop": "WikiHop [Welbl et al. 2018]. The MIGRES repo ships a 200-question sample: MIGRES/data/wikihop.json.",
    "hotpotqa": "HotpotQA [Yang et al. 2018]. MIGRES repo sample: MIGRES/data/hotpot.json (200 questions).",
    "musique": "MuSiQue [Trivedi et al. 2022]. MIGRES repo sample: MIGRES/data/musique.json (200 questions).",
    "natural_questions": "Natural Questions [Kwiatkowski et al. 2019]. MIGRES repo sample: MIGRES/data/odqa.json (rows with dataset=='nq').",
    "triviaqa": "TriviaQA [Joshi et al. 2017]. MIGRES repo sample: MIGRES/data/odqa.json (rows with dataset=='trivia').",
    "strategyqa": "StrategyQA [Geva et al. 2021]. MIGRES repo sample: MIGRES/data/strategy.json (229 questions).",
}
_ODQA_TAG = {"natural_questions": "nq", "triviaqa": "trivia"}
_ID_KEYS = {"wikihop": ("_id",), "hotpotqa": ("_id",), "musique": ("id",), "natural_questions": ("id",),
            "triviaqa": ("id",), "strategyqa": ("qid",), "jsonl": ("id", "_id", "qid"), "sample": ("id",)}
SUPPORTED = tuple(_ID_KEYS)
_log = get_logger("datasets")


def _read_records(path: Path) -> list[dict[str, Any]]:
    txt = path.read_text(encoding="utf-8")
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            data = data.get("data", [data])
        return list(data)
    except json.JSONDecodeError:
        out = []
        for i, line in enumerate(txt.splitlines(), 1):
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


def load_questions(name: str, path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Return ``[{id, question, answer, dataset, ...native extras}]``."""
    if name not in SUPPORTED:
        raise ValueError(f"unknown dataset {name!r}; supported: {SUPPORTED}")
    p = Path(path)
    if not p.exists():
        raise DatasetNotFoundError(f"dataset file not found: {p}\n  {DATASET_HELP.get(name, 'provide a JSON/JSONL file with a `question` field')}\n"
                                   "  See data/README.md for download/preparation instructions.")
    rows = _read_records(p)
    tag = _ODQA_TAG.get(name)
    out, seen = [], set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not str(r.get("question", "")).strip():
            raise DatasetNotFoundError(f"{p}: record {i} has no non-empty `question` field")
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
        raise DatasetNotFoundError(f"{p}: no usable records for dataset {name!r}")
    return out


# -- development corpora built from per-question contexts (NOT the Wikipedia dumps MIGRES uses) ----------
def contexts_to_documents(name: str, records: list[dict[str, Any]]) -> Iterator[Document]:
    """HotpotQA/WikiHop ``context`` = [[title, [sentences]]]; MuSiQue ``paragraphs`` = [{title, paragraph_text}].

    Records without a usable context (e.g. 36/200 rows of the upstream WikiHop sample have none) are
    skipped and counted in a warning; if NO record yields a document a DatasetNotFoundError is raised.
    """
    if name not in ("hotpotqa", "wikihop", "musique"):
        raise ValueError(f"no per-question contexts to build a corpus from for dataset {name!r}")
    seen: set[str] = set()
    n_skipped = n_yielded_any = 0
    for r in records:
        pairs: list[tuple[str, str]] = []
        if name in ("hotpotqa", "wikihop"):
            ctx = _maybe_literal(r.get("context"))
            if not isinstance(ctx, list) or not ctx:
                n_skipped += 1
                continue
            for item in ctx:
                title, sents = item[0], item[1]
                pairs.append((str(title), " ".join(x.strip() for x in sents if x.strip())))
        else:
            paras = _maybe_literal(r.get("paragraphs"))
            if not isinstance(paras, list) or not paras:
                n_skipped += 1
                continue
            for para in paras:
                pairs.append((str(para.get("title", "")), str(para.get("paragraph_text", "")).strip()))
        got = False
        for title, text in pairs:
            if not text:
                continue
            h = sha256_text(title + "\n" + text)
            got = True
            if h in seen:
                continue
            seen.add(h)
            yield Document(doc_id="ctx-" + h[:16], text=text, title=title)
        n_yielded_any += got
    if n_skipped:
        _log.warning("%s: %d/%d records have no usable per-question context and were skipped", name, n_skipped, len(records))
    if not n_yielded_any:
        raise DatasetNotFoundError(f"{name}: none of the {len(records)} records contain per-question contexts; "
                                   "supply a corpus file instead (see data/README.md)")


# -- RGB (FUTURE / October) -------------------------------------------------------------------------------
def load_rgb(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load an RGB JSONL file (github.com/chen700564/RGB, ``data/en.json``: id, query, answer, positive, negative).

    Loader only - the noise-robustness / rejection / integration / counterfactual protocols
    (methodology 5.6.6) are a FUTURE module; see ``evaluation.RGBRobustnessEvaluation``.
    """
    p = Path(path)
    if not p.exists():
        raise DatasetNotFoundError(f"RGB file not found: {p} (clone https://github.com/chen700564/RGB and pass data/en.json)")
    out = []
    for r in _read_records(p):
        if "query" not in r:
            raise DatasetNotFoundError(f"{p}: RGB record without `query`")
        rec = dict(r)
        rec.update(id=str(r.get("id")), question=r["query"], answer=r.get("answer"), dataset="rgb")
        out.append(rec)
        if limit and len(out) >= limit:
            break
    return out
