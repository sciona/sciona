#!/usr/bin/env python3
"""Read-only audit of source, adapter and full ensemble execution evidence."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from sciona.ensemble_execution import build_ensemble_execution_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.import_riemannian_execution import validated_graph
from scripts.audit_andriy_promotion_evidence import audit as audit_andriy
from scripts.import_ensemble_adapter_drafts import verify_local_evidence
from scripts.inventory_ensemble_adapters import inventory


def audit(root, reference_dir, library):
    upstream = {}
    for component in ['riemannian', 'relative_power', 'combined_feature', 'feng_knn', 'feng_xgb', 'feng_expanded']:
        _, _, digest, _, _ = validated_graph(root, reference_dir, component)
        upstream[component] = digest
    andriy = audit_andriy(root, reference_dir, library)
    if not andriy['component_execution_prerequisites_complete']:
        raise ValueError('Andriy execution evidence incomplete')
    upstream['andriy'] = andriy['graph_digest']
    verify_local_evidence(root, reference_dir)
    adapters = inventory(root)
    tests = json.loads((root/'docs/reviews/ensemble_test_review.json').read_text())
    if tests['test_results'] != dict(tests=24, failures=0, errors=0, skipped=0):
        raise ValueError('Complete ensemble tests required')
    for name, sha in tests['test_source_sha256'].items():
        if hashlib.sha256((root/'tests'/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Ensemble test source changed')
    if tests['adapter_source_sha256'] != {r['runtime_fqdn']: r['provider_sha256'] for r in adapters['providers']}:
        raise ValueError('Tested adapter source changed')
    graph = build_ensemble_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    pending = []
    path = root/'docs/reviews/ensemble_execution_parity.json'
    if not path.exists():
        pending.append('Successful complete raw ensemble execution evidence missing')
    else:
        report = json.loads(path.read_text())
        if (report['graph_digest'] != digest or report['nodes'] != len(nodes) or report['edges'] != len(edges)
                or report['executed_nodes'] != len(nodes) or report['full_runner_cases'] < 1
                or report['synthetic_only'] is not True or report['exact_final_source_blend'] is not True
                or report['final_score_spread'] <= 0):
            raise ValueError('Full ensemble execution gates differ')
        for relative, field in [('scripts/validate_ensemble_execution.py', 'validator_sha256'),
                                ('sciona/ensemble_execution.py', 'graph_source_sha256')]:
            if hashlib.sha256((root/relative).read_bytes()).hexdigest() != report[field]:
                raise ValueError('Full execution validation source changed')
        expected = {node.matched_primitive for node in graph.nodes}
        if set(report['provider_sha256']) != expected:
            raise ValueError('Full execution provider set differs')
        for runtime, sha in report['provider_sha256'].items():
            module, name = runtime.rsplit('.', 1)
            function = getattr(importlib.import_module(module), name)
            if hashlib.sha256(Path(inspect.getsourcefile(function)).read_bytes()).hexdigest() != sha:
                raise ValueError('Full execution provider source changed')
    return dict(read_only=True, approval_applied=False, upstream_component_digests=upstream,
        graph_digest=digest, nodes=len(nodes), edges=len(edges), adapter_versions_verified=7,
        tests_verified=24, execution_prerequisites_complete=not pending, pending=pending,
        limitations=['Integrity/execution prerequisites only; exact catalog versions, interfaces, dependency closure and semantic review still gate publication.',
            'Current runtime source contracts with declared adaptations; no historical-engine or predictive-quality claim.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(Path(__file__).resolve().parents[1], args.reference_dir, args.library)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
