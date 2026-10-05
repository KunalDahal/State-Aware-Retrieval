from __future__ import annotations

import ast
from typing import Any, Iterator
from ..corpus.models import Document
from ..utils.errors import DatasetNotFoundError
from ..utils.hashing import sha256_text

def _maybe_literal(v: Any) -> Any:
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def contexts_to_documents(name: str, records: list[dict[str, Any]]) -> Iterator[Document]:
    if name not in ("hotpotqa", "musique"):
        raise ValueError(f"no per-question contexts to build a corpus from for {name!r}")
    seen: set[str] = set(); n_skipped = 0; yielded_records = 0
    for r in records:
        pairs: list[tuple[str, str]] = []
        if name == "hotpotqa":
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
