"""Execute the serialized decomposed APTOS graph with synthetic diagnostics only.

Only intermediate persistence is intercepted. Providers and the execution
runner remain real; private runtime values are kept in memory by this helper.
"""
import asyncio
import hashlib
import inspect
from pathlib import Path
import tempfile
from unittest.mock import patch

from sciona.atoms.ml import decomposed_aptos_execution as provider
from sciona.atoms.ml import aptos_execution as population_provider
from sciona.atoms.numerical.statistics import cut_indices as generic_provider
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from scripts import stage_aptos_corrected_intake as intake


def build_graph():
    plan, providers = intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        return intake.check_candidate(db, plan, providers)


def execute_through_graph(**kwargs):
    graph = build_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Serialized graph roundtrip changed execution identity')
    paths = [Path(__file__), Path(inspect.getfile(provider)),
             Path(inspect.getfile(build_graph)), Path(inspect.getfile(runner)),
             Path(inspect.getfile(population_provider)), Path(inspect.getfile(generic_provider))]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {p: sha(p) for p in paths}
    captured = {}
    def capture(directory, node, name, value):
        if name.startswith('out_'):
            captured[(node, name)] = value
    arguments = dict(kwargs)
    arguments['references'] = arguments.pop('pretrained')
    with tempfile.TemporaryDirectory(prefix='aptos-synthetic-graph-') as directory:
        with patch.object(runner, 'RUNS_DIR', Path(directory)), \
             patch.object(runner, 'save_intermediate_value', side_effect=capture):
            status = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-aptos', 'case').execute(
                arguments, cdg=graph))
    if status['status'] != 'completed':
        raise ValueError('APTOS graph did not complete')
    if set(captured) != {('population','out_plan'),('train_scores','out_training_result'),
            ('cut_inputs','out_values'),('cut_inputs','out_cut_points'),('cut_inputs','out_tie_policy'),
            ('classify','out_ordinals'),('result','out_result')}:
        raise ValueError('All five providers must execute and return declared outputs')
    if before != {p: sha(p) for p in paths}:
        raise ValueError('Graph, provider or runner changed during execution')
    if len(status['trace'])!=5 or any(t['cached'] for t in status['trace']):
        raise ValueError('All five graph nodes must execute uncached')
    result = dict(captured[('result', 'out_result')])
    result['_graph_evidence'] = {'serialized_graph_sha256': digest,
        'provider_sha256': before[Path(inspect.getfile(provider))],
        'actual_runner_nodes': 5, 'edges':7, 'all_nodes_uncached':True, 'codec_roundtrip': True, 'stored_candidate': True,
        'candidate_version_id': intake.qualified_plan()[0]['version_id'],
        'plan_sha256': sha(intake.PLAN),
        'population_provider_sha256':before[Path(inspect.getfile(population_provider))],
        'generic_provider_sha256':before[Path(inspect.getfile(generic_provider))]}
    return result
