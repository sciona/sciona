#!/usr/bin/env python3
"""Read-only integrity audit of Andriy component promotion prerequisites."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import numpy as np
import scipy
from sciona.andriy_execution import build_andriy_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph


def audit(root, reference_dir, library):
    checked = set()
    def check(path, expected):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Evidence drift: '+path.name)
        checked.add(path)
    def read(name):
        return json.loads((root/'docs/reviews'/name).read_text())
    signal = root.parent/'sciona-atoms-signal/src/sciona/atoms/riemannian_bci/signal_processing'
    population = read('andriy_population_parity.json')
    for name, digest in population['provider_sha256'].items():
        check(signal/name, digest)
    for name, digest in population['source_hashes'].items():
        check(reference_dir/name, digest)
    for name, digest in population['validation_sha256'].items():
        check(root/'scripts'/name, digest)
    check(root/'tests/test_andriy_population.py', population['test_sha256'])
    if population['numpy_version'] != np.__version__ or population['scipy_version'] != scipy.__version__:
        raise ValueError('Population numerical runtime drift')
    if not population['exact_labels'] or not population['exact_validity']:
        raise ValueError('Population reference gates failed')
    if any(error > 1e-4 for error in population['maximum_errors'].values()):
        raise ValueError('Population reference error exceeds declared bound')
    source_names = {'xgb': 'Andriy_mod_xgb_7_5.R', 'svm': 'Andriy_mod_svm_5_7.R',
                    'glm': 'Andriy_mod_glmnet_5_3.R'}
    graph = build_andriy_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    r_audit = read('andriy_r_contract_audit.json')
    for node in graph.nodes[1:]:
        report = read(f'andriy_r_{node.node_id}_parity.json')
        module, name = node.matched_primitive.rsplit('.', 1)
        impl = getattr(importlib.import_module(module), name)
        check(Path(inspect.getsourcefile(impl)), report['provider_sha256'])
        check(root/f'scripts/validate_andriy_r_{node.node_id}.py', report['validation_sha256'])
        check(root/f'tests/test_andriy_r_{node.node_id}.py', report['test_sha256'])
        check(reference_dir/source_names[node.node_id], report['source_sha256'])
        for name, sha in report['compiled_library_sha256'].items():
            check(library/name, sha)
        if report['r_version'] != r_audit['r_version'] or report['package_versions'] != r_audit['package_versions']:
            raise ValueError('R runtime evidence mismatch')
        exact = [value for key, value in report.items() if key.startswith('exact_')]
        if not exact or not all(value is True for value in exact) or report['score_spread'] <= .01:
            raise ValueError('Model reference gates failed')
    execution = read('andriy_execution_parity.json')
    for name, sha in execution['validation_files_sha256'].items():
        check(root/name, sha)
    for name, sha in execution['reference_report_sha256'].items():
        check(root/'docs/reviews'/name, sha)
    if execution['full_graph_hash'] != digest or execution['exact_outputs'] != 6 or execution['executed_nodes'] != 3:
        raise ValueError('Model graph reference mismatch')
    pending = []
    raw_path = root/'docs/reviews/andriy_raw_execution_parity.json'
    if not raw_path.exists():
        pending.append('Successful full raw graph execution evidence is missing')
    else:
        raw = json.loads(raw_path.read_text())
        if (raw['graph_digest'] != digest or raw['nodes'] != len(nodes) or raw['edges'] != len(edges)
                or raw['full_runner_cases'] < 1 or raw['synthetic_only'] is not True):
            raise ValueError('Full raw graph evidence mismatch')
        check(root/'scripts/validate_andriy_raw_execution.py', raw['validator_sha256'])
        if set(raw['provider_sha256']) != {node.matched_primitive for node in graph.nodes}:
            raise ValueError('Full raw graph provider set mismatch')
        for runtime, sha in raw['provider_sha256'].items():
            module, name = runtime.rsplit('.', 1)
            check(Path(inspect.getsourcefile(getattr(importlib.import_module(module), name))), sha)
        for name, sha in raw['reference_report_sha256'].items():
            check(root/'docs/reviews'/name, sha)
    return dict(read_only=True, approval_applied=False, checked_files=len(checked),
        graph_digest=digest, component_execution_prerequisites_complete=not pending,
        pending=pending, limitations=[
            'Integrity and prerequisite audit only; does not approve atoms or CDGs.',
            'Provider identity, interfaces, runtime dependency closure and semantic review still require catalog intake checks.',
            'This component does not cover the full eleven-model ensemble or the broader competition/physics backlog.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1], args.reference_dir, args.library)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
