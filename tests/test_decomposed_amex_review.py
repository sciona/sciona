"""Reject reduced learner controls and stale or inconsistent full-graph evidence."""
import json
from pathlib import Path
import pytest
from scripts.review_decomposed_amex import ROOT,review


@pytest.fixture
def evidence(monkeypatch):
    path=ROOT/'docs/reviews/decomposed_amex_graph.json'
    document=json.loads(path.read_text());read=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda self,*a,**kw:json.dumps(document) if self==path else read(self,*a,**kw))
    return document


@pytest.mark.parametrize('field,value',[
    ('models',24),('dart_models',14),('rounds_per_dart_model',10),
    ('neural_models',9),('epochs_per_neural_model',1),('actual_runner_nodes',2),
    ('exact_reference_output',False),('exact_source_blend',False),('all_nodes_uncached',False),
    ('input_and_torch_rng_preserved',False),('serialized_graph_sha256','0'*64),
])
def test_incomplete_execution_rejected(evidence,field,value):
    evidence[field]=value
    with pytest.raises(ValueError):review()


def test_training_trace_requires_all_rounds_even_when_retained_trees_are_fewer(evidence):
    for phase in ('reference','graph'):evidence['training_records'][phase][0]['evaluated_rounds']=4499
    with pytest.raises(ValueError,match='boosting iteration trace'):review()


def test_source_and_graph_retained_iterations_must_agree(evidence):
    evidence['training_records']['graph'][0]['retained_iterations']-=1
    with pytest.raises(ValueError,match='Source training trace'):review()


def test_missing_learner_trace_rejected(evidence):
    for phase in ('reference','graph'):evidence['training_records'][phase].pop()
    with pytest.raises(ValueError,match='Source training trace'):review()
