"""Uncached serialized provider calls and exact fitted-state edge routing."""
import asyncio
import numpy as np
import pytest
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from sciona.text_pair_provider_graphs import PROVIDERS, build_provider_graph, build_isotonic_state_graph


def run(graph, inputs, tmp_path, monkeypatch):
    digest, nodes, edges = encode_execution_graph(graph)
    restored = decode_execution_graph(nodes, edges, digest)
    assert encode_execution_graph(restored)[0] == digest
    captured = {}
    monkeypatch.setattr(runner, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(runner, 'save_intermediate_value',
                        lambda directory, node, name, value: captured.__setitem__((node, name), value))
    runner._ensure_atoms_imported()
    result = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-provider', 'qualification').execute(inputs, cdg=restored))
    assert result['status'] == 'completed', result
    assert {t['node_id'] for t in result['trace']} == {n.node_id for n in graph.nodes}
    assert all(not t['cached'] for t in result['trace'])
    return captured


def iso_inputs():
    return dict(X=np.array([.1, .2, .2, .8]), y=np.array([0., 1., 0., 1.]),
                sample_weight=None, y_min=0., y_max=1., increasing=True, out_of_bounds='clip')


@pytest.mark.parametrize('name', ['clean', 'edit', 'bm25', 'isotonic', 'fit', 'transform'])
def test_serialized_single_provider(name, tmp_path, monkeypatch):
    cases = dict(
        clean=dict(text='ＣＡＴ cta!', operations=['unicode', 'lower', 'spell'], vocab_freq={'cat': 8}),
        edit=dict(s1='kitten', s2='sitting'),
        bm25=dict(term_frequencies=[2, 1], document_frequencies=[3, 5], document_length=4,
                  document_count=10, average_length=3., k1=1.2, b=.75),
        isotonic=dict(y=np.array([1., 0., 2.]), sample_weight=None, y_min=0., y_max=1., increasing=True),
        fit=iso_inputs(), transform=dict(T=np.array([0., .15, .7, 1.]), state=PROVIDERS['fit'](**iso_inputs())))
    inputs = cases[name]
    expected = PROVIDERS[name](**inputs)
    actual = run(build_provider_graph(name), inputs, tmp_path, monkeypatch)[name, 'out_result']
    if name == 'fit':
        np.testing.assert_array_equal(actual.x_thresholds, expected.x_thresholds)
        np.testing.assert_array_equal(actual.y_thresholds, expected.y_thresholds)
        assert (actual.x_min, actual.x_max, actual.increasing, actual.out_of_bounds) == (
            expected.x_min, expected.x_max, expected.increasing, expected.out_of_bounds)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize('scores,labels', [([.1,.2,.2,.8],[0.,1.,0.,1.]),
    ([.4,.4,.4,.4],[0.,0.,1.,1.]), ([.1,.3,.5,.9],[1.,1.,0.,0.])])
def test_isotonic_state_routing_and_query_without_refitting(scores, labels, tmp_path, monkeypatch):
    inputs = iso_inputs()
    inputs.update(X=np.array(scores), y=np.array(labels), T=np.array([0., .25, .75, 1.]))
    captured = run(build_isotonic_state_graph(), inputs, tmp_path, monkeypatch)
    state = captured['fit', 'out_result']
    assert captured['transform', 'in_state'] is state
    expected = PROVIDERS['transform'](inputs['T'], state)
    np.testing.assert_array_equal(captured['transform', 'out_result'], expected)
    before = state.x_thresholds.copy(), state.y_thresholds.copy()
    query = np.array([.95, .05, .95])
    query_run = run(build_provider_graph('transform'), dict(T=query, state=state), tmp_path, monkeypatch)
    np.testing.assert_array_equal(query_run['transform', 'out_result'], PROVIDERS['transform'](query, state))
    np.testing.assert_array_equal(state.x_thresholds, before[0])
    np.testing.assert_array_equal(state.y_thresholds, before[1])
