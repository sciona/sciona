"""Qualify immutable complete execution; catalog publication is separate."""
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import subprocess

from sciona.decomposed_openvaccine_graph import build_decomposed_openvaccine_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.review_decomposed_openvaccine_interfaces import review as review_interfaces
from scripts.audit_decomposed_openvaccine_packages import audit as audit_packages
from scripts.provision_openvaccine_reviewed_software import provision

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '75656787a7703401dc34ce3cfaeb21ef700a77bb'
EVIDENCE = 'docs/reviews/decomposed_openvaccine_graph_execution.json'
BASELINE = 'docs/reviews/decomposed_openvaccine_baseline_execution.json'
CHECKS = dict(actual_runner_nodes=6, actual_runner_edges=7, models=20, refinement_rounds=1,
    pretraining_steps=1, initial_supervised_steps=20, refinement_supervised_steps=20,
    pseudo_label_steps=20, full_reference_retrained=True, exact_reconstruction_result=True,
    prediction_and_teacher_bytes_equal=True, strict_json_output=True, graph_codec_roundtrip=True)
RESOURCE_POLICY = dict(native_threads=1, tensorflow_intra_threads=1,
                       tensorflow_inter_threads=1, serial_models=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed(path):
    raw = (ROOT/path).read_bytes()
    require(raw == subprocess.check_output(['git', 'show', COMMIT+':'+path], cwd=ROOT),
            'Immutable evidence differs: '+path)
    return json.loads(raw)


def check_hashes(hashes):
    require(bool(hashes), 'Nonempty evidence hashes required')
    for path, digest in hashes.items():
        resolved = (ROOT/path).resolve()
        require(resolved.is_relative_to(ROOT) and resolved.is_file() and sha(resolved) == digest,
                'Evidence source changed: '+path)


def validate_execution(full):
    headers = dict(format='decomposed-openvaccine-execution.v1', status='passed', approved=False,
                   catalog_mutations=0, synthetic_only=True, all_nodes_uncached=True,
                   actual_imports_verified=True, active_dependency_constraints_satisfied=True)
    require(type(full) is dict and set(full) == set(headers) | {
        'checks', 'resource_policy', 'code_sha256', 'runtime', 'provisioned_software',
        'serialized_graph_sha256', 'bound_provider_sha256', 'limits'}, 'Exact execution evidence fields required')
    require(all(full.get(k) == v and type(full.get(k)) is type(v) for k, v in headers.items()),
            'Complete execution headers required')
    require(full['checks'] == CHECKS and all(type(full['checks'][k]) is type(v) for k, v in CHECKS.items()),
            'Complete lifecycle coverage required')
    require(full['resource_policy'] == RESOURCE_POLICY
            and all(type(full['resource_policy'][k]) is type(v) for k, v in RESOURCE_POLICY.items()),
            'Resource policy differs')
    baseline = committed(BASELINE)
    expected_paths = set(baseline['code_sha256']) | {
        'sciona/decomposed_openvaccine_runtime.py', 'sciona/decomposed_openvaccine_ensemble.py',
        'sciona/decomposed_openvaccine_graph.py', 'scripts/validate_decomposed_openvaccine_graph_execution.py',
        'tests/test_decomposed_openvaccine_runtime.py', 'tests/test_decomposed_openvaccine_ensemble.py'}
    require(set(full['code_sha256']) == expected_paths, 'Complete implementation hash coverage required')
    check_hashes(full['code_sha256'])
    require(all(full['code_sha256'][p] == h for p, h in baseline['code_sha256'].items()),
            'Baseline reference changed')
    require(full['runtime'] == baseline['runtime'], 'Execution differs from baseline runtime')
    require(all(metadata.version(k) == v for k, v in full['runtime'].items()), 'Installed runtime differs')
    require(full['provisioned_software'] == baseline['provisioned_software'], 'Reviewed software differs')
    graph = build_decomposed_openvaccine_graph()
    require(encode_execution_graph(graph)[0] == full['serialized_graph_sha256'], 'Executed graph differs')
    wanted = {n.matched_primitive for n in graph.nodes}
    require(set(full['bound_provider_sha256']) == wanted, 'Complete bound-provider coverage required')
    for name, digest in full['bound_provider_sha256'].items():
        module = importlib.import_module(name.rsplit('.', 1)[0])
        require(sha(inspect.getfile(module)) == digest, 'Provider implementation differs')
    return graph


def review():
    full = committed(EVIDENCE)
    graph = validate_execution(full)
    require(provision() == full['provisioned_software'], 'Provisioned software changed')
    test_paths = ['docs/reviews/decomposed_openvaccine_lifecycle_tests.json',
                  'docs/reviews/decomposed_openvaccine_interface_tests.json']
    for path, count, hash_key in zip(test_paths, (157, 13), ('code_sha256', 'sha256')):
        report = committed(path)
        require(report['passed'] is True and report['approved'] is False and report['catalog_mutations'] == 0,
                'Passing unapproved test evidence required')
        require(report['tests'] == dict(tests=count, failures=0, errors=0, skipped=0),
                'Complete test coverage required')
        check_hashes(report[hash_key])
    interface_path = 'docs/reviews/decomposed_openvaccine_interfaces.json'
    interfaces = committed(interface_path)
    require(review_interfaces() == interfaces, 'Current provider interfaces or contracts differ')
    require(interfaces['graph_digest'] == full['serialized_graph_sha256'], 'Interface graph differs')
    require({r['runtime_fqdn']: r['provider_sha256'] for r in interfaces['providers']}
            == full['bound_provider_sha256'], 'Interface provider implementations differ')
    package_path = 'docs/reviews/decomposed_openvaccine_package_files.json'
    packages = committed(package_path)
    require(packages['direct_versions'] == full['runtime'] and audit_packages() == packages,
            'Installed package closure or files changed')
    evidence_paths = [EVIDENCE, BASELINE, interface_path, package_path, *test_paths]
    return dict(format='decomposed-openvaccine-implementation-review.v1', passed=True, approved=False,
        catalog_mutations=0, source_qualification_complete=False, execution_evidence_commit=COMMIT,
        graph_nodes=len(graph.nodes), graph_edges=len(graph.edges), graph_digest=full['serialized_graph_sha256'],
        providers=interfaces['providers'], root_inputs=['payload'], root_outputs=['result'], runtime=full['runtime'],
        evidence_sha256={p: sha(ROOT/p) for p in evidence_paths}, reviewer_sha256=sha(__file__),
        checks=dict(exact_full_result=True, prediction_and_teacher_bytes_equal=True,
            original_reference_unchanged=True, registered_provider_interfaces=True,
            package_files_verified=sum(p['hashed_files_verified'] for p in packages['packages'].values()),
            default_dependency_packages=len(packages['packages'])),
        limitations=['Current full synthetic execution and interfaces are qualified; catalog publication and corrected original source scopes are separate.',
                    'One native refinement round was compared; mocked multi-round tests preserve configured loop forwarding without certifying historical training schedules.',
                    'Installed files match their own RECORD; publisher authentication, optional extras, clean deployment and other platforms are not certified.',
                    'The fixed dense-feature contract supports compatible caller inputs; RNA folding and recipe policy remain explicit.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/decomposed_openvaccine_implementation_review.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], providers=len(result['providers']), approved=False)))
