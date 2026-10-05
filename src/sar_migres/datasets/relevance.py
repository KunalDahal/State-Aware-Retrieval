from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from ..utils.errors import DatasetNotFoundError
from ..utils.hashing import sha256_text
from .loader import _read_records
from .normalization import _maybe_literal

def gold_answer_aliases(item: dict[str, Any]) -> list[str]:
    from ..evaluation.answer import gold_aliases
    vals = list(gold_aliases(item.get("answer")))
    vals.extend(gold_aliases(item.get("answer_aliases")))
    return list(dict.fromkeys(vals))


def _context_title_text(obj: Any) -> tuple[str, str]:
    if isinstance(obj, dict):
        title = str(obj.get("title", obj.get("document_title", obj.get("name", ""))))
        text = obj.get("paragraph_text", obj.get("text", obj.get("contents", obj.get("document_text", ""))))
        return title.strip(), str(text or "").strip()
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return str(obj[0]).strip(), str(obj[1]).strip()
    return "", str(obj or "").strip()

def _parent_id(title: str, text: str) -> str:
    return sha256_text(title.strip() + "\n" + text.strip())

def _match_parent_contexts(corpus: Any, contexts: list[Any]) -> set[str]:
    targets = {_parent_id(*_context_title_text(x)) for x in contexts if _context_title_text(x)[1]}
    if not targets:
        return set()
    out=set()
    docs_iter = corpus.iter_documents() if hasattr(corpus, "iter_documents") else iter(corpus)
    for d in docs_iter:
        parent = str(d.metadata.get("source_parent_id", "")) if getattr(d, "metadata", None) else ""
        if parent in targets:
            out.add(d.doc_id)
    return out

def relevant_doc_ids(item: dict[str, Any], corpus: Iterable[Document] | Any) -> set[str]:
    """Normalize benchmark evidence annotations into the actual corpus chunk IDs."""
    explicit = item.get("supporting_doc_ids") or item.get("relevant_docs") or item.get("positive_doc_ids")
    if isinstance(explicit, (list, tuple, set)) and explicit:
        return {str(x) for x in explicit}
    name = str(item.get("dataset", "")).lower()
    docs_iter = corpus.iter_documents() if hasattr(corpus, "iter_documents") else iter(corpus)
    docs=list(docs_iter)
    if name == "hotpotqa":
        sf = _maybe_literal(item.get("supporting_facts")) or []
        titles = {str(x[0]) for x in sf if isinstance(x, (list, tuple)) and x}
        return {d.doc_id for d in docs if d.title in titles}
    if name == "musique":
        paras = _maybe_literal(item.get("paragraphs")) or []
        positive=[p for p in paras if isinstance(p, dict) and bool(p.get("is_supporting", p.get("is_supporting_paragraph", False)))]
        return _match_parent_contexts(type("C", (), {"iter_documents": lambda self: iter(docs)})(), positive)
    if name in {"natural_questions", "nq"}:
        contexts = []
        for key in ("positive_ctxs", "positive_contexts", "contexts", "long_answer_candidates"):
            v=_maybe_literal(item.get(key))
            if isinstance(v, list): contexts.extend(v)
        positives=[x for x in contexts if not isinstance(x, dict) or x.get("is_positive", x.get("has_answer", True))]
        matched=_match_parent_contexts(type("C", (), {"iter_documents": lambda self: iter(docs)})(), positives)
        if matched: return matched
        # Fallback for pre-existing corpora without parent metadata: exact normalized text match.
        wanted={" ".join(_context_title_text(x)[1].lower().split()) for x in positives if _context_title_text(x)[1]}
        return {d.doc_id for d in docs if " ".join(d.text.lower().split()) in wanted}
    return set()

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
