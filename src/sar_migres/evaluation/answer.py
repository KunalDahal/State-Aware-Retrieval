from __future__ import annotations

from typing import Any, Iterable
from collections import Counter
import re
import string
import numpy as np

_STOP = set("""a an the of in on at to for from by with and or is are was were be been being who what which when where why how""".split())
# Keep binary answers as meaningful tokens for HotpotQA/Natural Questions yes/no scoring.
_STOP.discard("yes")
_STOP.discard("no")

def _norm(s: Any) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t not in _STOP]
    return " ".join(tokens)


def gold_aliases(gold: Any) -> list[str]:
    if gold is None:
        return []
    if isinstance(gold, dict):
        for key in ("answers", "aliases", "answer", "value", "text"):
            if key in gold:
                return gold_aliases(gold[key])
    if isinstance(gold, (list, tuple, set)):
        out = []
        for x in gold:
            out.extend(gold_aliases(x))
        return out
    return [str(gold)]


def _bool_norm(x: Any) -> bool | None:
    s = _norm(x)
    if s in {"yes", "true", "1", "correct"}:
        return True
    if s in {"no", "false", "0", "incorrect"}:
        return False
    return None


def exact_match(pred: Any, golds: Any) -> float:
    if isinstance(golds, (bool, np.bool_)):
        pb = _bool_norm(pred)
        return float(pb is not None and pb == bool(golds))
    aliases = gold_aliases(golds)
    if not aliases:
        return 0.0
    return float(any(_norm(pred) == _norm(g) for g in aliases))


def answer_accuracy(pred: Any, gold: Any) -> float:
    """Accuracy metric from methodology Sec. 5.6.6.

    For extractive QA this is exact-answer correctness; for boolean/decision QA it also accepts
    normalized yes/no or true/false forms.  Multiple gold aliases are treated as equivalent.
    """
    return exact_match(pred, gold)


def token_f1(pred: Any, gold: Any) -> float:
    aliases = gold_aliases(gold)
    if not aliases:
        return 0.0
    p = _norm(pred).split()
    best = 0.0
    for g0 in aliases:
        g = _norm(g0).split()
        common = sum((Counter(p) & Counter(g)).values())
        if common == 0:
            cur = 1.0 if not p and not g else 0.0
        else:
            P, R = common / len(p), common / len(g)
            cur = 2 * P * R / (P + R)
        best = max(best, cur)
    return best


def evaluate_answers(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(records)
    evaluable = [r for r in rows if r.get("gold", r.get("gold_answer")) is not None]
    if not evaluable:
        return {"exact_match": float("nan"), "accuracy": float("nan"), "token_f1": float("nan"), "n": 0, "n_total": len(rows)}
    em = [exact_match(r.get("prediction", r.get("final_answer")), r.get("gold", r.get("gold_answer"))) for r in evaluable]
    acc = [answer_accuracy(r.get("prediction", r.get("final_answer")), r.get("gold", r.get("gold_answer"))) for r in evaluable]
    f1 = [token_f1(r.get("prediction", r.get("final_answer")), r.get("gold", r.get("gold_answer"))) for r in evaluable]
    return {"exact_match": float(np.mean(em)), "accuracy": float(np.mean(acc)), "token_f1": float(np.mean(f1)), "n": len(evaluable), "n_total": len(rows)}
