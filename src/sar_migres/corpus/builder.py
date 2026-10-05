from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..datasets.loader import _read_records
from ..datasets.normalization import _maybe_literal
from ..utils.errors import DatasetNotFoundError
from ..utils.hashing import sha256_file, sha256_text
from .chunking import _clean_html, chunk_text, _load_tokenizer

def _iter_passage_pairs(record: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Extract common passage/document structures used by the canonical study datasets."""
    fields = (
        "context", "paragraphs", "passages", "documents", "search_results", "contexts",
        "facts", "positive_ctxs", "negative_ctxs", "retrieved_context", "evidence", "entity_pages", "document"
    )
    for field in fields:
        obj = _maybe_literal(record.get(field))
        if not obj:
            continue
        if isinstance(obj, dict):
            iterable = []
            for key, value in obj.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("title", str(key))
                    iterable.append(item)
                elif isinstance(value, list):
                    iterable.extend((str(key), x) for x in value)
                elif isinstance(value, str):
                    iterable.append({"title": str(key), "text": value})
        else:
            iterable = obj if isinstance(obj, list) else [obj]
        for item in iterable:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (dict, list, str)):
                title, val = str(item[0]), item[1]
                if isinstance(val, dict):
                    text = val.get("text", val.get("contents", val.get("paragraph_text", val.get("document_text", ""))))
                elif isinstance(val, list):
                    text = " ".join(str(x) for x in val)
                else:
                    text = val
                if text:
                    yield title, str(text).strip(), {"source_field": field}
                continue
            if isinstance(item, dict):
                title = str(item.get("title", item.get("document_title", item.get("name", ""))))
                text = item.get("text", item.get("contents", item.get("paragraph_text", item.get("document_text", item.get("html", "")))))
                if isinstance(text, list):
                    text = " ".join(str(x) for x in text)
                text = str(text or "").strip()
                if text:
                    yield title, text, {"source_field": field}
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                text = item[1]
                if isinstance(text, list):
                    text = " ".join(str(x) for x in text)
                text = str(text or "").strip()
                if text:
                    yield title, text, {"source_field": field}
            elif isinstance(item, str) and item.strip():
                yield "", item.strip(), {"source_field": field}
    if record.get("document_text"):
        yield str(record.get("title", "")), str(record["document_text"]), {"source_field": "document_text"}

def build_pooled_corpus(dataset_files: dict[str, str | Path], *, output: str | Path,
                        source_split: str = "train", max_documents: int = 100_000,
                        max_tokens: int = 350, overlap_tokens: int = 50,
                        tokenizer_name: str | None = None, require_all_datasets: bool = True, allow_whitespace_fallback: bool = False) -> dict[str, Any]:
    if not dataset_files:
        raise ValueError("dataset_files must contain at least one dataset")
    if max_documents < len(dataset_files):
        raise ValueError("max_documents must be at least the number of datasets when require_all_datasets=True")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer, tokenizer_label = _load_tokenizer(tokenizer_name, allow_whitespace_fallback=allow_whitespace_fallback)
    # Fixed per-dataset quota avoids filling the whole budget from the first dataset.
    quotas = {name: max(1, max_documents // len(dataset_files)) for name in dataset_files}
    remainder = max_documents - sum(quotas.values())
    for name in list(dataset_files)[:remainder]:
        quotas[name] += 1

    seen: set[str] = set()
    n_source_records = 0
    per_dataset: dict[str, int] = {name: 0 for name in dataset_files}
    source_presence: dict[str, bool] = {name: False for name in dataset_files}
    with out_path.open("w", encoding="utf-8") as fh:
        for dataset, path0 in dataset_files.items():
            path = Path(path0)
            if not path.exists():
                raise DatasetNotFoundError(f"missing corpus source for {dataset}: {path}")
            qcount = 0
            for rec in _read_records(path, split=source_split):
                if not isinstance(rec, dict):
                    continue
                n_source_records += 1
                qid = str(rec.get("id", rec.get("_id", rec.get("qid", n_source_records))))
                yielded_from_record = False
                for title, raw_text, meta in _iter_passage_pairs(rec):
                    text = _clean_html(raw_text)
                    for chunk_no, chunk in enumerate(chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, tokenizer=tokenizer, allow_whitespace_fallback=allow_whitespace_fallback)):
                        if qcount >= quotas[dataset]:
                            break
                        title_clean = title.strip()
                        source_parent_id = sha256_text(title_clean + "\n" + text)
                        content_key = sha256_text(title_clean + "\n" + chunk)
                        if content_key in seen:
                            continue
                        seen.add(content_key)
                        did = f"{dataset}-{qid}-p{source_parent_id[:16]}-c{chunk_no}-{content_key[:12]}"
                        row = {
                            "id": did, "title": title_clean, "text": chunk,
                            "dataset_source": dataset, "source_record_id": qid,
                            "source_field": meta.get("source_field", ""), "source_parent_id": source_parent_id, "chunk_no": chunk_no,
                        }
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        qcount += 1
                        yielded_from_record = True
                    if qcount >= quotas[dataset]:
                        break
                source_presence[dataset] = source_presence[dataset] or yielded_from_record
                if qcount >= quotas[dataset]:
                    break
            per_dataset[dataset] = qcount

    missing = [name for name, n in per_dataset.items() if n == 0]
    if missing and require_all_datasets:
        raise DatasetNotFoundError(
            "pooled corpus construction produced zero passages for: " + ", ".join(missing) +
            ". Check the source split/schema for those datasets or set require_all_datasets=False."
        )
    manifest = {
        "type": "pooled_train_bounded", "source_split": source_split, "max_documents": max_documents,
        "chunk_max_tokens": max_tokens, "chunk_overlap_tokens": overlap_tokens,
        "tokenizer": tokenizer_label, "allow_whitespace_fallback": bool(allow_whitespace_fallback), "n_documents": len(seen), "n_source_records": n_source_records,
        "per_dataset": per_dataset, "quotas": quotas, "missing_datasets": missing,
        "source_files": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in dataset_files.items()},
    }
    manifest["output_sha256"] = sha256_file(out_path)
    (out_path.with_suffix(out_path.suffix + ".manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
