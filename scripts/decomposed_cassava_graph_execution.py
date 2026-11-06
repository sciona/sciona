"""Execute Cassava's serialized draft graph with private, temporary diagnostics.

Only intermediate persistence is intercepted; all seven nodes and the runner
execute normally. Callers retain ownership of returned checkpoint artifacts.
"""
import asyncio
import hashlib
import importlib
import inspect
from pathlib import Path
import tempfile
from unittest.mock import patch

from sciona.atoms.ml import decomposed_cassava_execution as provider
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from sciona.decomposed_cassava_graph import build_decomposed_cassava_graph as build_graph


def execute_through_graph(plan, **kwargs):
    graph = build_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Serialized graph roundtrip changed execution identity')
    paths = [Path(__file__), Path(inspect.getfile(provider)),
             Path(inspect.getfile(build_graph)), Path(inspect.getfile(runner))]
    paths += [Path(inspect.getfile(importlib.import_module(node.matched_primitive.rsplit('.', 1)[0])))
              for node in graph.nodes]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {p: sha(p) for p in paths}
    captured = {}
    def capture(directory, node, name, value):
        if name.startswith('out_'):
            captured[(node, name)] = value
    arguments = dict(sample_keys=list(plan.keys), labels=list(plan.labels),
                     assignments=list(plan.assignments), **kwargs)
    with tempfile.TemporaryDirectory(prefix='cassava-synthetic-graph-') as directory:
        with patch.object(runner, 'RUNS_DIR', Path(directory)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-cassava', 'case').execute(
                arguments, cdg=graph))
    if status['status'] != 'completed':
        raise ValueError('Cassava graph did not complete')
    if set(captured) != {(n.node_id, 'out_' + p.name) for n in graph.nodes for p in n.outputs}:
        raise ValueError('All graph nodes must execute and return their declared outputs')
    if captured[('fold_plan', 'out_plan')] != plan:
        raise ValueError('Graph changed shared fold membership')
    if before != {p: sha(p) for p in paths}:
        raise ValueError('Graph, provider or runner changed during execution')
    result = captured[('result', 'out_result')]
    return tuple(result[name] for name in [
        'predictions', 'family_outputs', 'histories', 'evaluations', 'checkpoints'])
