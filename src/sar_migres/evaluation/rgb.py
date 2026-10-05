from __future__ import annotations

from typing import Any, Sequence
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .answer import answer_accuracy

@dataclass
class RGBCase:
    setting: str
    question: str
    gold: Any
    noise_level: float | None = None
    negative: bool = False
    counterfactual: bool = False
    metadata: dict[str, Any] | None = None


class RGBRobustnessEvaluation:
    """Evaluator for the four RGB settings named in methodology Sec. 5.6.6.

    The class consumes *model outputs* rather than making LLM calls itself.  This keeps retrieval,
    MIGRES execution and RGB evaluation separated and allows SAR and every baseline to use the same
    evaluation protocol.
    """

    SETTINGS = {"noise_robustness", "negative_rejection", "information_integration", "counterfactual_robustness"}

    def __init__(self, cases: Sequence[RGBCase]):
        self.cases = list(cases)

    @staticmethod
    def from_records(records: Sequence[dict[str, Any]], setting: str | None = None) -> "RGBRobustnessEvaluation":
        out = []
        for r in records:
            s = setting or str(r.get("setting") or r.get("type") or r.get("task") or "").lower()
            if "noise" in s or "refine" in s:
                s2 = "noise_robustness"
            elif "negative" in s or "reject" in s or "refuse" in s:
                s2 = "negative_rejection"
            elif "integr" in s:
                s2 = "information_integration"
            elif "fact" in s or "counter" in s:
                s2 = "counterfactual_robustness"
            else:
                s2 = s if s in RGBRobustnessEvaluation.SETTINGS else "information_integration"
            nl = r.get("noise_level", r.get("noise_rate"))
            out.append(RGBCase(s2, str(r.get("question", r.get("query", ""))), r.get("answer"),
                               None if nl is None else float(nl), bool(r.get("negative", False)),
                               bool(r.get("counterfactual", False)), dict(r)))
        return RGBRobustnessEvaluation(out)

    def evaluate(self, outputs: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if len(outputs) != len(self.cases):
            raise ValueError(f"RGB outputs={len(outputs)} does not match cases={len(self.cases)}")
        by = defaultdict(list)
        for c, o in zip(self.cases, outputs):
            row = dict(o)
            pred = row.get("prediction", row.get("final_answer", row.get("answer")))
            correct = answer_accuracy(pred, c.gold)
            row["_correct"] = bool(correct)
            by[c.setting].append((c, row))
        result: dict[str, Any] = {}
        for setting, rows in by.items():
            if setting == "noise_robustness":
                levels = defaultdict(list)
                for c, r in rows:
                    levels[c.noise_level].append(float(r["_correct"]))
                result[setting] = {str(level): float(np.mean(vals)) for level, vals in sorted(levels.items(), key=lambda x: str(x[0]))}
            elif setting == "negative_rejection":
                rejected = []
                for c, r in rows:
                    if c.negative or r.get("is_negative"):
                        pred_reject = bool(r.get("rejected", r.get("predicted_reject", False)))
                        rejected.append(float(pred_reject))
                result[setting] = {"RejectionRate": float(np.mean(rejected)) if rejected else float("nan")}
            elif setting == "counterfactual_robustness":
                ed, cr = [], []
                for c, r in rows:
                    if c.counterfactual or r.get("is_counterfactual"):
                        detected = bool(r.get("detected_error", r.get("error_detected", False)))
                        corrected = bool(r.get("corrected", r.get("counterfactual_corrected", False)))
                        ed.append(float(detected)); cr.append(float(corrected and detected))
                result[setting] = {
                    "ED": float(np.mean(ed)) if ed else float("nan"),
                    "CR": float(np.sum(cr) / max(np.sum(ed), 1)) if ed else float("nan"),
                }
            else:
                result[setting] = {"Accuracy": float(np.mean([r["_correct"] for _, r in rows])) if rows else float("nan")}
        return result
