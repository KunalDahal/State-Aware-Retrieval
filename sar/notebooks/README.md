# Experiment notebooks

The seven notebooks intentionally mirror methodology §5.6 one-for-one. They are not an all-in-one notebook.

All model-execution notebooks print the selected device and GPU name when PyTorch/CUDA is available. The selection is performed by the shared retrieval/model layer; analysis-only notebooks do not construct models unnecessarily.

The baseline notebook treats each baseline as a drop-in replacement for the complete SAR retrieval stack. SAR and Sentence Transformer Search use the project Sentence Transformer. BM25, MMR, DPR, Contriever, SPLADE, ColBERTv2 and InfoGain-RAG use native lexical/Transformer/sparse/late-interaction representations instead.

The labeled final protocol defaults to public dev data. Tuning is a deterministic 10% holdout from train. This is explicit because multiple benchmark test splits do not publish answer labels.
