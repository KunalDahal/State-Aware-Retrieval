import json
from pathlib import Path
import subprocess, sys
import pytest
from sar_migres.evaluation.retrieval import recall_at_k
from sar_migres.evaluation.answer import exact_match, token_f1
from sar_migres.experiments.parameter_selection import frozen_setting
from sar_migres.scoring.weights import SARWeights
from sar_migres.utils.config import load_config

def test_answer_metrics():
    assert exact_match('Ada Lovelace', ['ada lovelace']) == 1.0
    assert token_f1('Ada Lovelace wrote notes', 'Ada Lovelace') > 0.5
    assert exact_match('yes', 'yes') == 1.0
    assert exact_match('no', 'no') == 1.0

def test_frozen_parameter_loading(tmp_path):
    p=tmp_path/'selected.json'; p.write_text('{"selected_k":50,"selected_weights":[0.4,0.3,0.3]}')
    k,w=frozen_setting(p); assert k==50; assert isinstance(w,SARWeights)
    with pytest.raises((FileNotFoundError, KeyError)):
        frozen_setting(tmp_path/'missing.json')

def test_configuration_loading_and_override():
    cfg=load_config('configs/local.yaml',overrides={'sar':{'k':4}})
    assert cfg.sar.k == 4 and cfg.embedding.type == 'hashing'

def test_cli_imports_execute():
    cp=subprocess.run([sys.executable,'-m','sar_migres.cli.run_single','--help'],capture_output=True,text=True,check=False)
    assert cp.returncode == 0 and '--question' in cp.stdout

def test_retrieval_evaluation_excludes_unlabeled_questions(tmp_path):
    from sar_migres.experiments.iterative import retrieval_metrics_from_run
    from sar_migres.corpus.models import Corpus, Document
    run=tmp_path/'run'; run.mkdir();
    (run/'traces.jsonl').write_text('{"question_id":"q1","iteration":0,"k":1,"documents":[{"doc_id":"d1"}]}\n')
    items=[{"id":"q1","question":"q","dataset":"unknown"}]
    out=retrieval_metrics_from_run(run,items,Corpus.from_documents([Document('d1','x')]))
    assert out['n_questions_with_relevance_annotations']==0
    assert out['final_mean']['recall_at_k'] != out['final_mean']['recall_at_k']


def test_canonical_ablation_variants():
    from sar_migres.evaluation.aggregation import AblationStudy
    variants = AblationStudy.ablation_weights(SARWeights(0.5, 0.3, 0.2))
    assert list(variants) == [
        "SAR-Full", "SAR-Without-Relevance", "SAR-Without-Gap",
        "SAR-Without-Evidence", "SAR-Without-Tuning", "Question-Only"
    ]
    assert variants["SAR-Without-Tuning"] == (1/3, 1/3, 1/3)
    assert all(abs(sum(v) - 1.0) < 1e-12 for v in variants.values())


def test_rgb_evaluator_runtime_imports():
    from sar_migres.evaluation.rgb import RGBCase, RGBRobustnessEvaluation
    ev=RGBRobustnessEvaluation([RGBCase("information_integration","q","answer")])
    out=ev.evaluate([{"prediction":"answer"}])
    assert out["information_integration"]["Accuracy"] == 1.0


def test_run_output_contract_keeps_results_and_traces_separate(tmp_path):
    from sar_migres.outputs import RunWriter
    from sar_migres.retrieval.common import QuestionResult, TraceRecord
    trace = TraceRecord(
        run_id="r1", question_id="q1", question="q", iteration=0, gaps=[], evidence=[],
        alpha=0.4, beta=0.3, gamma=0.3, k=50, documents=[{
            "doc_id": "d1", "rank": 1, "title": "T", "text": "full passage", "relevance": 0.5,
            "gap_coverage": 0.0, "novelty": 1.0, "score": 0.6, "retrieval_score": 0.6,
        }], n_docs_scored=1, n_similarity_calcs=1, embed_seconds=0.01, score_seconds=0.02,
        total_seconds=0.03,
    )
    result = QuestionResult("r1", "q1", "q", "gold", "answer", [trace], "ok", elapsed_seconds=0.1, backend="standin")
    rd=tmp_path/'runs'/'r1'
    w=RunWriter(rd); w.append_result(result); w.write_tables(w.load_results()); w.write_report({}, w.load_results())
    row=json.loads((rd/'results.jsonl').read_text())
    assert "traces" not in row and row["final_answer"] == "answer"
    assert len((rd/'traces.jsonl').read_text().splitlines()) == 1
    assert (rd/'traces.csv').exists() and (rd/'summary.csv').exists() and (rd/'report.md').exists()
    assert not (tmp_path/'runs'/'tables').exists()
    assert not (rd/'experiment_name.txt').exists()


def test_cost_metrics_exposes_retrieval_overhead(tmp_path):
    from sar_migres.experiments.iterative import cost_metrics_from_run
    rd=tmp_path/'r'; rd.mkdir()
    (rd/'results.jsonl').write_text(json.dumps({"question_id":"q1","status":"ok","llm_calls":1,"llm_estimated_tokens":10})+'\n')
    (rd/'traces.jsonl').write_text(json.dumps({"question_id":"q1","iteration":0,"total_seconds":2.0,"embed_seconds":0.5,"score_seconds":1.0,"n_docs_scored":10,"n_similarity_calcs":20})+'\n')
    (rd/'metadata.json').write_text(json.dumps({"retrieval":{"method":"sar","stats":{}},"embedding_sync_history":[],"peak_rss_mb":42,"comparison":{"baseline_retrieval_latency_total_s":1.0}}))
    out=cost_metrics_from_run(rd)
    assert out["retrieval_overhead_pct"] == 100.0
