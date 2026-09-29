# Data

Nothing large is committed. `data/sample/` holds a tiny **synthetic** corpus (25 documents about
invented entities) and 3 questions that exist only to validate the installation offline.

```
data/
  sample/corpus.jsonl      25 synthetic docs   {"id", "title", "text"}
  sample/questions.jsonl   3 synthetic questions {"id", "question", "answer", "supporting_doc_ids"}
  raw/                     put downloaded benchmark files here      (git-ignored)
  processed/               corpus.jsonl produced by prepare_corpus  (git-ignored)
  cache/                   embedding caches / Pinecone mirror        (git-ignored)
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

## Building a corpus
MIGRES retrieves from Wikipedia dumps (wiki-17/18/21 via DPR / Atlas; see the MIGRES README). For a real
evaluation you should embed one of those dumps. That is a very large corpus (wiki-18 is 21M passages);
embedding it with the 768-d model needs a GPU and ~65 GB for float32 vectors, which is why the store keeps
vectors in a memory-mappable matrix and scores in chunks.

For a **development** corpus without downloading a dump, build one from the per-question contexts that
ship with HotpotQA / WikiHop / MuSiQue:

```bash
python scripts/prepare_corpus.py --config configs/default.yaml --from-dataset hotpotqa data/raw/hotpot.json
python scripts/prepare_corpus.py --from-file /path/to/psgs_w100.tsv --out data/processed/corpus.jsonl
```
This is a *different retrieval setting* from MIGRES' Wikipedia setting and results are not comparable to it.

## RGB (future module)
`git clone https://github.com/chen700564/RGB` and pass `data/en.json` to `datasets.load_rgb`. Only the loader
exists; the four robustness protocols are an October task.
