#!/usr/bin/env python3
"""Intake validated two-branch execution without approving the full source ensemble."""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from uuid import UUID, uuid5
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.riemannian_bci_execution import build_riemannian_branch_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.physics_ingest.pdg_evidence import _digest
from scripts.import_residual_execution_drafts import ensure_row

SOURCE_ID = UUID('a9eed402-7daa-5d35-a268-be73b47d4278')
SOURCE_VERSION = UUID('e2128626-2edb-50fc-bb4f-2bf32456a3f9')
SOURCE_HASH = '5e96b80dd4b205ff643ce3968cd7a2efa4f4c4c1219cda0719fef0e80c6f6b7e'


def validated_graph(root, reference_dir, component='riemannian'):
    if component not in {'riemannian', 'relative_power', 'combined_feature', 'feng_knn', 'feng_xgb', 'feng_expanded'}:
        raise ValueError('unknown execution component')
    filename = component + '_execution_parity.json'
    report = json.loads((root / 'docs/reviews' / filename).read_text())
    coverage_passed = (report['full_runner_cases'] >= 2 and report['branch_comparisons_passed'] >= 4) if component == 'riemannian' else (report['full_runner_cases'] >= 3 and report['minimum_reference_prediction_spread'] >= .01)
    if not coverage_passed or report['maximum_absolute_error'] > 1e-11 or report['synthetic_only'] is not True:
        raise ValueError('execution evidence does not pass')
    for name, digest in report['source_hashes'].items():
        if hashlib.sha256((reference_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError('public source drift: ' + name)
    for name, digest in report['validation_script_hashes'].items():
        if hashlib.sha256((root / 'scripts' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('validation implementation drift')
    modules = dict(report['execution_module_hashes'])
    for runtime_name, digest in report['provider_hashes'].items():
        module, name = runtime_name.rsplit('.', 1)
        function = getattr(importlib.import_module(module), name)
        if function.__module__ + '.' + function.__name__ != runtime_name:
            raise ValueError('runtime identity differs')
        modules[module] = digest
    for module, digest in modules.items():
        if hashlib.sha256(Path(inspect.getfile(importlib.import_module(module))).read_bytes()).hexdigest() != digest:
            raise ValueError('execution/provider implementation drift: ' + module)
    for package, version in report['library_versions'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('validated library version changed')
    if component == 'riemannian':
        graph = build_riemannian_branch_graph()
    elif component == 'relative_power':
        from sciona.relative_power_execution import build_relative_power_execution_graph
        graph = build_relative_power_execution_graph()
    elif component == 'feng_expanded':
        from sciona.feng_expanded_execution import build_feng_expanded_execution_graph
        graph = build_feng_expanded_execution_graph()
    elif component == 'feng_xgb':
        from sciona.feng_xgb_execution import build_feng_xgb_execution_graph
        graph = build_feng_xgb_execution_graph()
    elif component == 'feng_knn':
        from sciona.feng_knn_execution import build_feng_knn_execution_graph
        graph = build_feng_knn_execution_graph()
    else:
        from sciona.combined_feature_execution import build_combined_feature_execution_graph
        graph = build_combined_feature_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    if digest != report['graph_digest'] or len(nodes) != report['nodes'] or len(edges) != report['edges']:
        raise ValueError('validated graph differs')
    return report, graph, digest, nodes, edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--component', choices=['riemannian', 'relative_power', 'combined_feature', 'feng_knn', 'feng_xgb', 'feng_expanded'], default='riemannian')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report, graph, digest, nodes, edges = validated_graph(root, args.reference_dir, args.component)
    relative = args.component == 'relative_power'
    combined = args.component == 'combined_feature'
    feng = args.component in {'feng_knn', 'feng_xgb', 'feng_expanded'}
    runner_version = args.component.replace('_', '-') + '-execution-intake.v1'
    identity_seed = args.component.replace('_', '-') + '-execution.v1' if relative or combined or feng else 'riemannian-branches-execution.v1'
    artifact_id = uuid5(SOURCE_ID, identity_seed)
    version_id = uuid5(artifact_id, digest)
    fqdn = 'cdg.competition.solution.kaggle.barachant_seizure_1st.' + (args.component + '_execution' if relative or combined or feng else 'riemannian_branches_execution')
    description = ('Train and execute the source relative-log-power model from raw signal segments through one aligned segment-score output. One source branch only; full ensemble remains outside this component. Modern-library synthetic parity only.' if relative else 'Train and execute the source autocorrelation and frequency-coherence model branches from raw signal segments. Two aligned score outputs; other nine source ensemble models remain outside this component. Modern-library synthetic parity only.')
    if combined:
        description = 'Train and execute the source combined-feature model using relative power, AR coefficient standard errors, basic statistics and PFD/HFD/Hurst. One aligned segment-score output; full ensemble remains outside this component. Modern-library synthetic parity only.'
    if feng:
        description = 'Train and execute the source Feng base KNN model from ordered 600-second signal segments using causal filtering, mean log FFT bands, separate partition scalers and mean window probabilities. One population/model branch; full ensemble outside scope. Modern-library synthetic parity only.'
    if args.component == 'feng_xgb':
        description = 'Train and execute source Feng XGBoost on ordered 600-second segments using causal filtering, mean log FFT bands, unscaled features and 500 boosting rounds. One population/model branch; full ensemble outside scope. Modern-library synthetic parity only.'
    if args.component == 'feng_expanded':
        description = 'Train and execute source Feng expanded-feature KNN and GLM from ordered 600-second 16-channel segments using shared causal filtering and 384 spectral/correlation features. Two model outputs; full ensemble outside scope. Current-library synthetic parity only.'
    created = 0
    with psycopg.connect(dotenv_values(root / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (runner_version,))
        source = db.execute("SELECT a.fqdn,v.content_hash,e.details FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e USING(version_id) WHERE a.artifact_id=%s AND v.version_id=%s AND e.runner_version='competition-intake.v1' AND e.passed FOR SHARE OF a,v,e", (SOURCE_ID, SOURCE_VERSION)).fetchone()
        if not source or source['content_hash'] != SOURCE_HASH:
            raise ValueError('source intake version mismatch')
        created += ensure_row(db, 'artifacts', {'artifact_id': artifact_id}, {
            'artifact_id': artifact_id, 'artifact_kind': 'cdg', 'fqdn': fqdn,
            'status': 'draft', 'is_publishable': False, 'verified_leaf_coverage': 0., 'leaf_count': len(nodes),
            'description': description})
        created += ensure_row(db, 'artifact_versions', {'version_id': version_id}, {
            'version_id': version_id, 'artifact_id': artifact_id, 'content_hash': digest,
            'semver': '0.0.0+execution.' + digest[:12], 'is_latest': False, 'trust_tier': 3})
        for table, items, fields, keys in [
            ('artifact_cdg_nodes', nodes, ['node_id', 'parent_node_id', 'name', 'description', 'concept_type', 'status', 'matched_primitive', 'type_signature'], ['node_id']),
            ('artifact_cdg_edges', edges, ['source_id', 'target_id', 'output_name', 'input_name'], ['source_id', 'target_id', 'output_name', 'input_name']),
        ]:
            for item in items:
                row = {field: item[field] for field in fields}
                row['version_id'] = version_id
                created += ensure_row(db, table, {k: row[k] for k in ['version_id', *keys]}, row)
        consumed = {(e.target_id, e.input_name) for e in graph.edges}
        producers = {e.source_id for e in graph.edges}
        for direction, ports in [
            ('input', [p for n in graph.nodes for p in n.inputs if (n.node_id, p.name) not in consumed]),
            ('output', [p for n in graph.nodes if n.node_id not in producers for p in n.outputs]),
        ]:
            unique = {}
            for port in ports:
                if port.name in unique and (direction == 'output' or unique[port.name] != port):
                    raise ValueError('ambiguous graph boundary')
                unique[port.name] = port
            for ordinal, port in enumerate(unique.values()):
                row = dict(artifact_id=artifact_id, version_id=version_id, direction=direction, name=port.name,
                           ordinal=ordinal, type_desc=port.type_desc, constraints=port.constraints,
                           required=port.required, default_value_repr=port.default_value_repr)
                created += ensure_row(db, 'artifact_io_specs', {k: row[k] for k in ['artifact_id', 'version_id', 'direction', 'name']}, row)
        dependency = dict(dependent_version_id=version_id, dependency_artifact_fqdn=source['fqdn'], dependency_content_hash=SOURCE_HASH, port_name='')
        created += ensure_row(db, 'artifact_dependencies', dependency, {**dependency, 'dependency_role': 'cdg', 'optional': False,
            'binding_metadata': Jsonb({'scope': 'mandatory conceptual source provenance, not an invoked runtime dependency'})})
        evidence_id = uuid5(version_id, runner_version)
        details = {'execution_report': report, 'execution_report_sha256': _digest(report),
                   'execution_graph': graph.model_dump(mode='json'), 'source_intake_sha256': _digest(source['details']),
                   'source_version_id': str(SOURCE_VERSION),
                   'scope': ('Validated relative-power component intake; provider bindings and publication review pending. Full eleven-model source ensemble not approved by this evidence.' if relative else 'Validated two-branch component intake; provider bindings and publication review pending. Full eleven-model source ensemble not approved by this evidence.')}
        if combined:
            details['scope'] = 'Validated combined-feature component intake; provider bindings and publication review pending. Full eleven-model source ensemble not approved by this evidence.'
        if feng:
            details['scope'] = 'Validated Feng base KNN component intake; provider bindings and publication review pending. Full eleven-model source ensemble not approved by this evidence.'
        if args.component == 'feng_xgb':
            details['scope'] = 'Validated Feng XGBoost component intake; provider bindings and publication review pending. Full eleven-model source ensemble not approved by this evidence.'
        if args.component == 'feng_expanded':
            details['scope'] = 'Validated Feng expanded KNN/GLM component intake; provider bindings and publication review pending. Full eleven-model source ensemble not approved by this evidence.'
        created += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence_id}, {
            'evidence_id': evidence_id, 'artifact_id': artifact_id, 'version_id': version_id,
            'audit_type': 'regression_test', 'passed': True, 'status': 'completed', 'source_kind': 'automated',
            'runner_version': runner_version, 'details': Jsonb(details)})
        document = db.execute('SELECT get_artifact_document(%s) AS d', (fqdn,)).fetchone()['d']
        if _artifact_document_to_cdg(document, version_id=str(version_id), content_hash=digest, require_execution_envelope=True) != graph:
            raise ValueError('catalog roundtrip differs')
        if not args.apply:
            db.rollback()
    print(json.dumps({'applied': args.apply, 'rows_created': created, 'nodes': len(nodes), 'edges': len(edges), 'catalog_roundtrip': 'passed', 'publication': 'draft'}, indent=2))


if __name__ == '__main__':
    main()
