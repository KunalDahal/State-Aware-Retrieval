# Data

Nothing large is committed. `data/sample/` holds a tiny **synthetic** corpus (25 documents about
invented entities) and 3 questions that exist only to validate the installation offline.

```
data/
  sample/corpus.jsonl      25 synthetic docs   {"id", "title", "text"}
  sample/questions.jsonl   3 synthetic questions {"id", "question", "answer", "supporting_doc_ids"}
  raw/                     put downloaded benchmark files here      (git-ignored)
  processed/               corpus.jsonl produced by prepare_corpus  (git-ignored)
  cache/                   embedding / vector-store caches            (git-ignored)
```

## Corpus format (what SAR searches)
JSONL (one object per line), JSON list, or DPR-style TSV with header `id  text  title`.
Accepted keys: id = `doc_id|id|_id`; text = `text|contents`; optional `title`. Malformed rows are logged
and skipped; the load fails if more than `data.max_malformed_fraction` (default 1 %) are bad.
Duplicate ids with identical content are dropped with a warning; duplicate ids with *different* content abort.

## Question datasets (methodology Sec. 5.4.1)
Loaders read local files; nothing is downloaded automatically.

| `data.dataset` | expected file | source |
|---|---|---|
| `wikihop` | JSON/JSONL, fields `_id, question, answer, context, supporting_facts` | WikiHop (Welbl et al., 2018) |
| `hotpotqa` | `_id, question, answer, context, supporting_facts` | HotpotQA (Yang et al., 2018) |
| `musique` | `id, question, answer, paragraphs, answer_aliases` | MuSiQue (Trivedi et al., 2022) |
| `natural_questions` | `id, question, answer, dataset=="nq"` | Natural Questions (Kwiatkowski et al., 2019) |
| `triviaqa` | `id, question, answer, dataset=="trivia"` | TriviaQA (Joshi et al., 2017) |
| `strategyqa` | `qid, question, answer, facts, decomposition` | StrategyQA (Geva et al., 2021) |
| `jsonl` | any file with `question` (+ optional `id`, `answer`) | generic |
| `sample` | `data/sample/questions.jsonl` | synthetic |

The official MIGRES repository contains small (200-question) samples in exactly these layouts
(`data/hotpot.json`, `wikihop.json`, `musique.json`, `odqa.json`, `strategy.json`; note they are JSON-lines
files despite the `.json` extension - both forms load). Copy them to `data/raw/` for a first real run.
Use the full official train/dev/test files for the October study; keep validation and test separate.

## Building the benchmark corpus D
The executable study does **not** use a full Wikipedia dump by default. D is a pooled, bounded corpus built from
training-split passages from all six methodology datasets. It is deduplicated, chunked to at most 350 tokenizer
tokens with 50-token overlap, and capped at 100,000 passages with a deterministic per-dataset quota so that one
dataset cannot consume the whole budget. Development/final question records are never used to construct D.

Build it with:

```bash
python scripts/prepare_corpus.py \
  --dataset wikihop=data/raw/wikihop.json \
  --dataset hotpotqa=data/raw/hotpot.json \
  --dataset musique=data/raw/musique.json \
  --dataset natural_questions=data/raw/nq.json \
  --dataset triviaqa=data/raw/triviaqa.json \
  --dataset strategyqa=data/raw/strategyqa.json
```

The output includes a `.manifest.json` containing source hashes, split, tokenizer/chunking settings, quotas,
and per-dataset document counts. A full Wikipedia-scale run remains possible by supplying an appropriate corpus
file, but that is outside the bounded 100k-passage experiment defined for this project.

## RGB robustness
The project now includes an evaluator for the four methodology settings: `noise_robustness`,
`negative_rejection`, `information_integration`, and `counterfactual_robustness`. The evaluator
consumes serialized RGB cases plus the corresponding model outputs; it does not silently synthesize
RGB cases or LLM outputs. Point notebook `06_downstream_rag_robustness.ipynb` at the RGB JSON/JSONL
file and an output file containing the required prediction/rejection/counterfactual fields.
