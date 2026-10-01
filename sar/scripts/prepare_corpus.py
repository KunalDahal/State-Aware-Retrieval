
#!/usr/bin/env python3
"""Build the bounded pooled training corpus D used by SAR and all retriever baselines."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sar_migres.corpus_builder import build_pooled_corpus

p = argparse.ArgumentParser()
p.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH", help="repeat for each of the six datasets")
p.add_argument("--output", default="data/processed/pooled_train_corpus.jsonl")
p.add_argument("--source-split", default="train")
p.add_argument("--max-documents", type=int, default=100000)
p.add_argument("--max-tokens", type=int, default=350)
p.add_argument("--overlap-tokens", type=int, default=50)
p.add_argument("--tokenizer", default="sentence-transformers/multi-qa-mpnet-base-cos-v1")
args = p.parse_args()

if not args.dataset:
    p.error("provide --dataset NAME=PATH at least once")
files = {}
for spec in args.dataset:
    if "=" not in spec:
        p.error(f"bad --dataset {spec!r}; use NAME=PATH")
    n, path = spec.split("=", 1)
    files[n] = path

manifest = build_pooled_corpus(files, output=args.output, source_split=args.source_split,
                               max_documents=args.max_documents, max_tokens=args.max_tokens,
                               overlap_tokens=args.overlap_tokens, tokenizer_name=args.tokenizer)
print(manifest)
