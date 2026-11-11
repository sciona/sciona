"""Qualify current Santander implementation evidence; source/publication are separate."""
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import subprocess
from typing import get_args, get_origin, get_type_hints
import numpy as np

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.decomposed_santander_graph import build_decomposed_santander_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.promote_connectomics_execution import boundary_ports, provider_ports

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '1c3179d1c41ea5d5574b0be9e39a1c62c3eeff4e'
EVIDENCE = 'docs/reviews/decomposed_santander_graph_execution.json'
GENERIC = 'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_execution(full):
    expected = dict(format='decomposed-santander-execution.v1',status='passed',approved=False,
                    catalog_mutations=0,synthetic_only=True,all_nodes_uncached=True)
    require(all(full.get(k) == v for k,v in expected.items()), 'Complete execution headers required')
    require(full['checks'] == dict(actual_runner_nodes=7,actual_runner_edges=13,models_fitted=30,
        synthetic_query_rows=8500,full_default_pseudo_counts=True,
        exact_reconstruction_result=True,strict_json_output=True,graph_codec_roundtrip=True),
        'Complete lifecycle coverage required')
    require(full['resource_policy'] == dict(native_threads=1,lightgbm_threads=1,torch_threads=1,torch_interop_threads=1,serial_models=True),
            'Resource policy differs')
    expected_paths = {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('santander_*.py')}
    expected_paths |= {'sciona/decomposed_santander_graph.py','sciona/decomposed_santander_runtime.py',
        'scripts/santander_synthetic.py','scripts/validate_decomposed_santander_graph_execution.py',
        'sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py'}
    require(set(full['code_sha256']) == expected_paths, 'Implementation hash coverage differs')
    require(all(sha(ROOT/p) == digest for p,digest in full['code_sha256'].items()), 'Implementation changed')
    packages = {line.split('==')[0] for line in (ROOT/'requirements/santander-execution.txt').read_text().splitlines() if line}
    require(set(full['runtime']) == packages and all(metadata.version(k) == v for k,v in full['runtime'].items()),
            'Runtime differs')
    graph = build_decomposed_santander_graph()
    require(encode_execution_graph(graph)[0] == full['serialized_graph_sha256'], 'Graph differs')
    wanted = {n.matched_primitive for n in graph.nodes}
    require(set(full['bound_provider_sha256']) == wanted, 'Provider hash coverage differs')
    for name,digest in full['bound_provider_sha256'].items():
        function = getattr(importlib.import_module(name.rsplit('.',1)[0]),name.rsplit('.',1)[1])
        require(sha(inspect.getfile(function)) == digest, 'Provider implementation differs')
    # Preserve the full learners and both pseudo-label expansions by identity.
    from sciona import santander_execution as original, decomposed_santander_runtime as candidate
    from sciona import santander_lifecycle, santander_retraining
    require(candidate.prepare is original.prepare
            and candidate.train_neural_cv is santander_lifecycle.train_neural_cv
            and candidate.train_tree_cv is santander_retraining.train_tree_cv
            and candidate.expanded_population is santander_retraining.expanded_population,
            'Complete reconstruction dependencies changed')
    return graph


def interfaces(function, node):
    kinds = {object:'object',dict:'dict',np.ndarray:'ndarray'}
    signature, hints = inspect.signature(function), get_type_hints(function)
    require(list(signature.parameters) == [p.name for p in node.inputs], 'Input order differs')
    ports = []
    for ordinal,(name,param) in enumerate(signature.parameters.items()):
        require(param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and param.default is inspect.Parameter.empty,
                'Required keyword input expected')
        port = node.inputs[ordinal]
        require(hints.get(name) in kinds and port.type_desc == kinds[hints[name]]
                and port.required and port.default_value_repr in ('',None), 'Input annotation differs')
        ports.append(dict(direction='input',name=name,ordinal=ordinal,type_desc=port.type_desc,
                          required=True,default_value_repr='',constraints=port.constraints))
    returns = get_args(hints['return']) if get_origin(hints['return']) is tuple else (hints['return'],)
    require(all(t in kinds for t in returns) and [kinds[t] for t in returns] == [p.type_desc for p in node.outputs],
            'Return annotation differs')
    ports.extend(dict(direction='output',name=p.name,ordinal=i,type_desc=p.type_desc,required=True,
                      default_value_repr='',constraints=p.constraints) for i,p in enumerate(node.outputs))
    return ports


def review():
    raw = (ROOT/EVIDENCE).read_bytes()
    require(raw == subprocess.check_output(['git','show',COMMIT+':'+EVIDENCE],cwd=ROOT),
            'Immutable execution evidence differs')
    full = json.loads(raw)
    graph = validate_execution(full)
    tests_path = ROOT/'docs/reviews/decomposed_santander_runtime_tests.json'
    require(tests_path.read_bytes() == subprocess.check_output(['git','show',COMMIT+':'+str(tests_path.relative_to(ROOT))],cwd=ROOT),
            'Immutable test evidence differs')
    tests = json.loads(tests_path.read_text())
    require(tests['status']=='passed' and tests['tests_passed']==121 and tests['synthetic_only'] is True,
            'Complete test evidence required')
    require(all(sha(ROOT/p)==digest for p,digest in tests['sha256'].items()), 'Test evidence changed')
    from scripts.audit_decomposed_santander_packages import audit
    package_path = ROOT/'docs/reviews/decomposed_santander_package_files.json'
    require(audit() == json.loads(package_path.read_text()), 'Package evidence changed')
    wanted = {n.matched_primitive for n in graph.nodes}
    specs = {}
    for repo in ('sciona-atoms','sciona-atoms-ml'):
        directory = ROOT.parent/repo
        for spec in _parse_registered_atoms(repo=ProviderRepo(repo,directory),artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime in wanted:
                require(runtime not in specs, 'Ambiguous provider registration')
                specs[runtime] = spec
    require(set(specs)==wanted, 'Complete registered providers required')
    records, ports = [], {}
    for node in graph.nodes:
        spec = specs[node.matched_primitive]
        function = getattr(importlib.import_module(spec.import_module),spec.source_symbol)
        require(Path(inspect.getfile(function)).resolve()==spec.file_path.resolve(), 'Provider origin differs')
        contract = interfaces(function,node)
        ports[node.node_id] = provider_ports(node,function)
        records.append(dict(runtime_fqdn=node.matched_primitive,catalog_fqdn=spec.fqdn,
            version_id=str(spec.version_id),content_hash=spec.content_hash,interfaces=contract,
            provider_sha256=sha(spec.file_path),classification='generic_operation' if node.matched_primitive==GENERIC else 'source_recipe_adapter'))
    boundary = boundary_ports(graph,ports)
    require([p.name for p in boundary['input']]==['payload'] and [p.name for p in boundary['output']]==['result'],
            'Root invocation boundary differs')
    return dict(format='decomposed-santander-implementation-review.v1',passed=True,approved=False,
        catalog_mutations=0,source_qualification_complete=False,graph_nodes=7,graph_edges=13,
        graph_digest=full['serialized_graph_sha256'],providers=records,execution_evidence_commit=COMMIT,
        root_inputs=['payload'],root_outputs=['result'],runtime=full['runtime'],
        evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/EVIDENCE,tests_path,package_path)},
        reviewer_sha256=sha(__file__),limitations=[
            'This review establishes current implementation parity and interfaces, not publication eligibility.',
            'Exact source correction and provider semantic contracts still require review before promotion.',
            'Generic ordered combination supports negative and non-unit-sum coefficients; recipe controls remain separate.',
            'Intermediate shape checks do not authenticate provenance; inputs must come from one graph invocation.'])


if __name__=='__main__':
    result = review()
    (ROOT/'docs/reviews/decomposed_santander_implementation_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],approved=False,providers=len(result['providers']))))
