"""Atomically publish the source-compared HuBMAP realization at automated Tier 3.

Default execution rolls back. Existing conflicting identities or evidence fail
closed; the original competition intake and its proposed bindings are untouched.
"""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from uuid import UUID, NAMESPACE_URL, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import AtomSeedRow, _parse_registered_atoms
from sciona.cdg_projection import build_published_cdg_projection
from sciona.competition_graph import load_competition_graph
from sciona.hubmap_execution import build_hubmap_lifecycle_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.import_residual_execution_drafts import ensure_row

SOURCE_VERSION = '325c30ff-ea2a-515c-8d40-110a5fd6991d'
SOURCE_HASH = '6f290e9ea9794b85f83ff3e52034a41a1c400f84d2f3f5182beb65d837664468'
COMMIT = '615444e86f4fb5ab916c0b66d311a307f4e4dd27'
RUNNER = 'hubmap-execution-community.v1'
LIMITATIONS = [
    'Automated Tier 3 computational review; no human-reviewed Tier 1 designation.',
    'Decoded RGB/binary arrays, runtime encoder states, ordered fold/checkpoint selectors and explicit RNGs required; no data or weights embedded.',
    'Independent connected source comparison covers one fold and one epoch per training stage with synthetic inputs, not learned competition accuracy or clinical usefulness.',
    'Full 134133707-parameter network retained; source-sized 1024/320 inference and 320 network evaluation have separate comparisons.',
    'Current CPU float32 and explicit zero-worker generators; historical GPU/mixed precision/worker scheduling and TIFF/subdataset codecs excluded.',
    'Initial checkpoints generate four-view pseudo labels; retraining starts fresh from explicit encoder state; final notebook uses three-view TTA.',
    'Ordered multi-fold routing has synthetic orchestration witnesses, not a full multi-fold training source comparison.',
    'Source bin-positive pseudo filtering and batch-wide classifier shortcut preserved, including empty final masks.',
    'Original intake stain-normalization provenance is unsupported and excluded; original intake stays draft.',
    'Immutable graph metadata records validation-time draft state; version-bound catalog evidence records Tier 3 publication.',
    'Runtime requires exact provider source and transitive matcher implementations bound by this review.',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base, hashes):
    if not hashes:
        raise ValueError('Missing evidence hashes')
    for name, digest in hashes.items():
        path = (base / name).resolve()
        if not path.is_relative_to(base.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError('Evidence drift: ' + name)


def provider_ports(node, fn):
    """Check exact callable order/requiredness and supply reviewed defaults."""
    signature = inspect.signature(fn)
    if list(signature.parameters) != [p.name for p in node.inputs]:
        raise ValueError('Graph/callable parameter order differs')
    result = {'input': [], 'output': []}
    for port, param in zip(node.inputs, signature.parameters.values()):
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            raise ValueError('Unsupported callable parameter')
        required = param.default is inspect.Parameter.empty
        default = '' if required else repr(param.default)
        if port.required != required or port.default_value_repr not in (None, '', default):
            raise ValueError('Graph/callable default differs')
        result['input'].append(port.model_copy(update={'default_value_repr': default}))
    result['output'] = list(node.outputs)
    return result


def boundary_ports(graph, interfaces):
    connected = {(e.target_id, e.input_name) for e in graph.edges}
    consumed = {(e.source_id, e.output_name) for e in graph.edges}
    result = {'input': {}, 'output': {}}
    for node in graph.nodes:
        for direction, excluded in [('input', connected), ('output', consumed)]:
            for port in interfaces[node.node_id][direction]:
                if (node.node_id, port.name) in excluded:
                    continue
                previous = result[direction].get(port.name)
                if previous is not None and (direction == 'output' or previous != port):
                    raise ValueError('Ambiguous graph boundary: ' + port.name)
                result[direction][port.name] = port
    return {direction: list(ports.values()) for direction, ports in result.items()}


def review(root):
    from sciona.visualizer.runner import _ensure_atoms_imported
    _ensure_atoms_imported()
    graph = build_hubmap_lifecycle_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    reviews = root / 'docs/reviews'
    full = json.loads((reviews / 'competition_hubmap_serialized.json').read_text())
    if full['serialized_graph_sha256'] != digest or not full['synthetic_only']:
        raise ValueError('Serialized graph identity differs')
    for key in ['independent_raw_lifecycle_reference','exact_both_stage_metrics_checkpoints_rng','exact_generated_pseudo_and_final_masks']:
        if full.get(key) is not True:raise ValueError('Missing full lifecycle comparison')
    if len(nodes)!=6 or len(edges)!=5 or (full['graph_nodes'],full['graph_edges'])!=(6,5):
        raise ValueError('Lifecycle coverage differs')
    if [c['tta'] for c in full['inference_calls']] != [4,3] or full['inference_calls'][0]['foreground_pixels']<=0:
        raise ValueError('Nonempty pseudo handoff and distinct TTA evidence required')
    suffixes=['serialized','lifecycle_source','final_inference','final_network','raw_inference','inference_network',
              'inference_tiles','inference','retraining_range','retraining','pseudo_orchestration','raw_training',
              'raw_population','partition','preparation','tiling','range','epoch','training','validation','losses',
              'augmentation','loader','batches','sampling_schedule','checkpoint_policy','network_eval_heads_320',
              'network_train_heads_32','network_eval_heads_32','network_eval_plain_32']
    evidence={}
    for suffix in suffixes:
        name='competition_hubmap_'+suffix+'.json'
        document=json.loads((reviews/name).read_text())
        if not document.get('synthetic_only'):raise ValueError('Synthetic source evidence required')
        check_hashes(root,document['implementation_sha256'])
        evidence[name]=sha(reviews/name)
    for path in reviews.glob('competition_hubmap_*pins.json'):
        evidence[path.name]=sha(path)
    pins=json.loads((reviews/'competition_hubmap_source_pins.json').read_text())
    if pins['commit']!=COMMIT or sha(root/'docs/licenses/HuBMAP-MIT.txt')!=pins['files']['LICENSE']:
        raise ValueError('Source pin/license differs')
    if sha(root/'docs/licenses/HuBMAP-pretrainedmodels-LICENSE.txt')!=pins['files']['src/pretrained-models.pytorch-master/LICENSE.txt']:
        raise ValueError('Encoder license differs')
    directory = root.parent / 'sciona-atoms-dl'
    parsed = _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-dl', directory), artifact_root=directory / 'src/sciona/atoms')
    specs = {}; interfaces = {}; signatures = {}
    for node in graph.nodes:
        matches = [s for s in parsed if s.import_module + '.' + s.source_symbol == node.matched_primitive]
        if len(matches) != 1:
            raise ValueError('Unique provider identity required')
        spec = matches[0]
        fn = getattr(importlib.import_module(spec.import_module), spec.source_symbol)
        if Path(inspect.getfile(inspect.unwrap(fn))).resolve() != spec.file_path.resolve():
            raise ValueError('Runtime provider source differs')
        if sha(spec.file_path) != full['provider_sha256']:
            raise ValueError('Provider comparison is stale')
        specs[node.node_id] = spec
        interfaces[node.node_id] = provider_ports(node, fn)
        signatures[node.node_id] = str(inspect.signature(fn))
    symbolic={}
    for node in graph.nodes:
        spec=specs[node.node_id]
        witness=getattr(importlib.import_module(spec.import_module),'witness_'+spec.source_symbol)
        if list(inspect.signature(witness).parameters)!=[port.name for port in node.inputs]:
            raise ValueError('Witness callable ports differ')
        values={}
        for port in node.inputs:
            incoming=[edge for edge in graph.edges if edge.target_id==node.node_id and edge.input_name==port.name]
            if incoming:
                edge=incoming[0];values[port.name]=symbolic[(edge.source_id,edge.output_name)]
            elif port.required:values[port.name]=object()
        result=witness(**values)
        outputs=result if len(node.outputs)>1 else (result,)
        if len(outputs)!=len(node.outputs):raise ValueError('Witness output arity differs')
        symbolic.update({(node.node_id,port.name):value for port,value in zip(node.outputs,outputs)})
    boundary = boundary_ports(graph, interfaces)
    if {p.name for p in boundary['output']} != {'initial_metrics', 'retraining_metrics', 'predictions'}:
        raise ValueError('Incomplete output boundary')
    evidence['competition_hubmap_source_pins.json'] = sha(reviews / 'competition_hubmap_source_pins.json')
    tests = json.loads((reviews / 'competition_hubmap_publication_tests.json').read_text())
    if tests['results'] != {'passed': 49, 'failed': 0, 'errors': 0, 'skipped': 0}:
        raise ValueError('Complete runtime and publication tests required')
    check_hashes(root, tests['implementation_sha256'])
    evidence['competition_hubmap_publication_tests.json'] = sha(reviews / 'competition_hubmap_publication_tests.json')
    return graph, digest, nodes, edges, specs, interfaces, boundary, signatures, evidence


def promote(root, apply=False):
    graph, digest, nodes, edges, specs, interfaces, boundary, signatures, evidence = review(root)
    created = 0
    rollup = dict(overall_verdict='acceptable_with_limits', structural_status='pass', runtime_status='pass',
        semantic_status='pass', developer_semantics_status='pass', review_status='approved',
        review_semantic_verdict='pass', review_developer_semantics_verdict='pass', trust_readiness='ready',
        review_limitations=LIMITATIONS, review_required_actions=[], trust_blockers=[],
        acceptability_band='acceptable_with_limits', parity_coverage_level='positive_and_negative', parity_test_status='pass')
    with psycopg.connect(dotenv_values(root / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        source = db.execute('SELECT a.artifact_id,a.fqdn,a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s FOR SHARE OF a,v', (SOURCE_VERSION,)).fetchone()
        if not source or source['content_hash'] != SOURCE_HASH or source['status'] != 'draft' or source['is_publishable']:
            raise ValueError('Original source identity/state differs')
        original = load_competition_graph(db, version_id=SOURCE_VERSION)
        if len(original.nodes) != 6:
            raise ValueError('Original source stage inventory differs')
        graph_id = uuid5(source['artifact_id'], RUNNER)
        version_id = uuid5(graph_id, digest)
        fqdn = source['fqdn'] + '.execution'
        owners = db.execute("SELECT DISTINCT a.owner_id,a.source_repo_id FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-dl'").fetchall()
        if len(owners) != 1:
            raise ValueError('Ambiguous provider ownership')
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}

        def ensure(table, identity, row):
            nonlocal created
            created += ensure_row(db, table, identity, row)

        def write_ports(identity, selected, ports, legacy=False):
            tables = [('artifact_io_specs', 'artifact_id')]
            if legacy:
                tables.append(('atom_io_specs', 'atom_id'))
            for table, key in tables:
                expected_count = 0
                for direction, selected_ports in ports.items():
                    for ordinal, port in enumerate(selected_ports):
                        row = {key: identity, 'version_id': selected, 'direction': direction, 'name': port.name,
                            'ordinal': ordinal, 'type_desc': port.type_desc, 'constraints': port.constraints,
                            'required': port.required, 'default_value_repr': port.default_value_repr}
                        if not table.startswith('atom_'):
                            row['dim_signature'] = port.dim_signature
                        ensure(table, {k: row[k] for k in [key, 'version_id', 'direction', 'name']}, row)
                        expected_count += 1
                if db.execute('SELECT count(*) AS n FROM ' + table + ' WHERE version_id=%s', (selected,)).fetchone()['n'] != expected_count:
                    raise ValueError('Unexpected additional catalog ports')

        provider_ids = {}; written_providers=set()
        for node_id, spec in specs.items():
            identity = uuid5(NAMESPACE_URL, 'sciona-provider-draft:' + spec.fqdn)
            provider_ids[node_id] = identity
            if identity in written_providers:continue
            written_providers.add(identity)
            existing = db.execute('SELECT atom_id,status,is_publishable FROM atoms WHERE fqdn=%s FOR UPDATE', (spec.fqdn,)).fetchone()
            if existing and (existing['atom_id'] != identity or existing['status'] != 'approved' or not existing['is_publishable']):
                raise ValueError('Conflicting existing provider identity/state')
            fields = {name: getattr(spec, name) for name in ['fqdn', 'namespace_root', 'namespace_path', 'repo_name', 'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi']}
            fields.update(source_package=spec.namespace_root, status='approved' if existing else 'flagged', is_publishable=bool(existing))
            atom = AtomSeedRow(**fields).as_dict(owner_id=str(owners[0]['owner_id']), source_repo_id=str(owners[0]['source_repo_id']))
            atom['atom_id'] = identity
            ensure('atoms', {'atom_id': identity}, atom)
            canonical = {k: v for k, v in atom.items() if k in columns and k not in {'created_at', 'updated_at'}}
            canonical.update(artifact_id=identity, artifact_kind='atom', status='approved' if existing else 'draft')
            ensure('artifacts', {'artifact_id': identity}, canonical)
            for table, key in [('atom_versions', 'atom_id'), ('artifact_versions', 'artifact_id')]:
                ensure(table, {'version_id': spec.version_id}, dict(version_id=spec.version_id, **{key: identity}, content_hash=spec.content_hash, semver=spec.semver, is_latest=True, trust_tier=3, s3_key='', fingerprint=spec.fingerprint))
                if db.execute('SELECT count(*) AS n FROM ' + table + ' WHERE ' + key + '=%s AND is_latest', (identity,)).fetchone()['n'] != 1:
                    raise ValueError('Ambiguous latest provider version')
            write_ports(identity, spec.version_id, interfaces[node_id], legacy=True)

        existing_graph = db.execute('SELECT artifact_id,status,is_publishable FROM artifacts WHERE fqdn=%s FOR UPDATE', (fqdn,)).fetchone()
        if existing_graph and (existing_graph['artifact_id'] != graph_id or existing_graph['status'] != 'approved' or not existing_graph['is_publishable']):
            raise ValueError('Conflicting existing graph identity/state')
        ensure('artifacts', {'artifact_id': graph_id}, dict(artifact_id=graph_id, artifact_kind='cdg', fqdn=fqdn,
            description='HuBMAP source-compatible initial fold training, four-view pseudo labels, fresh pseudo-augmented retraining and three-view final inference from runtime arrays and encoder states. Synthetic computational parity; no accuracy claim.',
            status='approved' if existing_graph else 'draft', is_publishable=bool(existing_graph)))
        ensure('artifact_versions', {'version_id': version_id}, dict(version_id=version_id, artifact_id=graph_id, content_hash=digest, semver='0.0.0+execution.' + digest[:12], is_latest=bool(existing_graph), trust_tier=3))
        for table, rows, keys in [('artifact_cdg_nodes', nodes, ['node_id']), ('artifact_cdg_edges', edges, ['source_id', 'target_id', 'output_name', 'input_name'])]:
            for item in rows:
                row = {'version_id': version_id, **item}
                ensure(table, {k: row[k] for k in ['version_id', *keys]}, row)
        write_ports(graph_id, version_id, boundary)
        for node_id, spec in specs.items():
            ensure('artifact_cdg_bindings', {'version_id': version_id, 'node_id': node_id}, dict(version_id=version_id, node_id=node_id,
                bound_artifact_fqdn=spec.fqdn, bound_version_content_hash=spec.content_hash, binding_confidence=1.,
                binding_source=RUNNER, status='active', alternatives=Jsonb([]), evidence_summary=Jsonb(dict(
                    runtime_fqdn=spec.import_module + '.' + spec.source_symbol, provider_version_id=str(spec.version_id),
                    source_sha256=sha(spec.file_path), output_aliases_by_ordinal=[p.name for p in interfaces[node_id]['output']]))))
        dependency = dict(dependent_version_id=version_id, dependency_artifact_fqdn=source['fqdn'], dependency_content_hash=SOURCE_HASH, port_name='')
        ensure('artifact_dependencies', dependency, {**dependency, 'dependency_role': 'cdg', 'optional': False,
            'binding_metadata': Jsonb(dict(scope='Mandatory original conceptual source provenance; not an invoked numerical dependency. Original intake and proposed bindings remain draft.'))})

        reference = 'tikutikutiku-hubmap-' + COMMIT[:12]
        url = 'https://github.com/tikutikutiku/kaggle-hubmap/tree/' + COMMIT
        title = 'Pinned HuBMAP source implementation (MIT)'
        ensure('references_registry', {'ref_id': reference}, dict(ref_id=reference, ref_type='repository', title=title, url=url))
        notebook_ref='kaggle-hubmap-inference-59475269'
        notebook_url='https://www.kaggle.com/kernels/scriptcontent/59475269/download'
        ensure('references_registry',{'ref_id':notebook_ref},dict(ref_id=notebook_ref,ref_type='repository',title='Pinned final HuBMAP inference notebook',url=notebook_url))
        targets=[]; seen_targets=set()
        for n,s in specs.items():
            if s.version_id not in seen_targets:
                targets.append((provider_ids[n],UUID(str(s.version_id)),n));seen_targets.add(s.version_id)
        targets.append((graph_id,version_id,None))
        for identity, selected, node_id in targets:
            if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (selected,)).fetchone():
                raise ValueError('Unresolved failed target audit')
            ref_tables = [('artifact_references', 'artifact_id')]
            rollup_tables = [('artifact_audit_rollups', 'artifact_id')]
            if node_id is not None:
                ref_tables.append(('atom_references', 'atom_id'))
                rollup_tables.append(('atom_audit_rollups', 'atom_id'))
                if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (selected,)).fetchone():
                    raise ValueError('Unexpected provider dependency')
            for table, key in ref_tables:
                ensure(table, {key: identity, 'ref_key': reference}, {key: identity, 'ref_id': reference, 'ref_key': reference,
                    'title': title, 'url': url, 'source': 'llm_extracted', 'verified': True, 'confidence': 'high',
                    'relevance_note': 'Pinned public software source; automated synthetic computation parity under documented adaptations and runtime requirements.'})
                ensure(table,{key:identity,'ref_key':notebook_ref},{key:identity,'ref_id':notebook_ref,'ref_key':notebook_ref,
                    'title':'Pinned final HuBMAP inference notebook','url':notebook_url,'source':'llm_extracted',
                    'verified':True,'confidence':'high','relevance_note':'Exact script version 59475269; full synthetic notebook inference comparison and source hash bound in review.'})
            for table, key in rollup_tables:
                ensure(table, {key: identity}, {key: identity, **rollup})
            details = dict(kind='provider' if node_id is not None else 'execution', publication_tier=3, review_source='automated',
                execution_graph_sha256=digest, source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_commit=COMMIT, final_notebook_version=59475269, evidence_sha256=evidence, limitations=LIMITATIONS,
                implementation_sha256=json.loads((root / 'docs/reviews/competition_hubmap_serialized.json').read_text())['implementation_sha256'],
                provider_versions={n: dict(version_id=str(s.version_id), content_hash=s.content_hash, source_sha256=sha(s.file_path)) for n, s in specs.items()})
            if node_id is not None:
                spec = specs[node_id]
                details.update(runtime_fqdn=spec.import_module + '.' + spec.source_symbol, signature=signatures[node_id],
                    output_aliases_by_ordinal=[p.name for p in interfaces[node_id]['output']])
            evidence_id = uuid5(selected, RUNNER)
            ensure('artifact_audit_evidence', {'evidence_id': evidence_id}, dict(evidence_id=evidence_id, artifact_id=identity,
                version_id=selected, audit_type='semantic_audit', passed=True, status='completed', source_kind='automated',
                runner_version=RUNNER, details=Jsonb(details)))

        for table, field, selected, expected in [('artifact_cdg_bindings', 'version_id', version_id, 6),
                ('artifact_dependencies', 'dependent_version_id', version_id, 1)]:
            if db.execute('SELECT count(*) AS n FROM ' + table + ' WHERE ' + field + '=%s', (selected,)).fetchone()['n'] != expected:
                raise ValueError('Unexpected catalog relationships')
        doc = db.execute('SELECT get_artifact_document(%s) AS d', (fqdn,)).fetchone()['d']
        if _artifact_document_to_cdg(doc, version_id=str(version_id), content_hash=digest, require_execution_envelope=True) != graph:
            raise ValueError('Catalog graph differs from validated graph')
        topo = build_published_cdg_projection(artifact={'artifact_id': str(graph_id), 'fqdn': fqdn}, version={'version_id': str(version_id), 'content_hash': digest}, cdg=graph).topo_hash
        for identity in provider_ids.values():
            db.execute("UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s", (identity,))
            db.execute("UPDATE atoms SET status='approved',is_publishable=true WHERE atom_id=%s", (identity,))
            if not db.execute('SELECT 1 FROM catalog_atoms_served WHERE atom_id=%s', (identity,)).fetchone():
                raise ValueError('Provider not served')
        db.execute("UPDATE artifacts SET status='approved',is_publishable=true,verified_leaf_coverage=1,leaf_count=6,top_level_input_arity=%s,top_level_output_arity=%s,topo_hash=%s WHERE artifact_id=%s", (len(boundary['input']), len(boundary['output']), topo, graph_id))
        db.execute('UPDATE artifact_versions SET is_latest=(version_id=%s) WHERE artifact_id=%s', (version_id, graph_id))
        if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (graph_id,)).fetchone():
            raise ValueError('Graph not served')
        # Recheck all immutable evidence immediately before the transaction decision.
        if review(root)[1:] != (digest, nodes, edges, specs, interfaces, boundary, signatures, evidence):
            raise ValueError('Publication evidence changed during transaction')
        if not apply:
            db.rollback()
    return dict(applied=apply, rows_created=created, trust_tier=3, served_provider_versions=5, served_execution_cdgs=1,
        input_arity=len(boundary['input']), output_arity=len(boundary['output']), graph_sha256=digest, version_id=str(version_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(promote(Path(__file__).resolve().parents[1], apply=args.apply), indent=2))
