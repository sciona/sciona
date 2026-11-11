"""Freeze explicit image population routing and its qualified graph evidence."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import NAMESPACE_URL, uuid5

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_REVISION = '36119b9'
EVIDENCE_COMMIT = 'ba481f9b'
MODULE = 'sciona.atoms.ml.domain_adapters.osic_image_routing'
PROVIDER_PATH = 'src/sciona/atoms/ml/domain_adapters/osic_image_routing.py'
EVIDENCE_FILES = ('docs/reviews/osic_image_routing.json','docs/reviews/osic_image_population_graph.json')
CONSTRAINTS = {
    'baselines': ('pandas.DataFrame', 'Nonempty exact mapped columns entity,time,target,numeric,category_a,category_b,uncertainty with unique columns and unique nonempty string entities. Time,target,numeric,uncertainty must be finite int64 or float64. Preserve baseline entity order. Caller establishes category meanings, compatible time/response coordinates and availability; no source schema inference.'),
    'queries': ('pandas.DataFrame', 'Nonempty exact mapped columns key,entity,time with unique columns, unique nonempty string keys, nonempty string entities covered by baselines and int64 times. Preserve query order, including interleaved entities. Keys are opaque caller identities; no query ordering or availability inference.'),
    'planes': ('dict', 'Exact baseline entity key coverage. Values are source-ordered encoded image records accepted by the injected preparation function. This adapter does not decode or inspect image payloads; caller establishes ordering and provenance.'),
    'model': ('object', 'Caller-trusted model accepted by predict_function, shared across entities. No construction, training, state loading, purity or concurrency guarantee.'),
    'batch_size': ('int', 'Explicit Python int exactly 32, excluding bool, to preserve this source policy. Generic bounded prediction is a separate callable and supports broader batch sizes.'),
    'prepare_function': ('object', 'Trusted callable invoked once per baseline entity as prepare_function(records,numeric,category_a,category_b). Return mapping containing images,metadata,selected_positions. Positions must be strictly increasing int64 rank-one array, at least two entries, within record bounds. Caller provides compatible selected image/metadata populations; source preparation policy is separately bound in the graph.'),
    'select_function': ('object', 'Trusted callable invoked as select_function(prepared,["images","metadata"]) once per entity, returning ordered predictor inputs. Caller binds explicit field selection semantics; no implicit conversion or schema inference.'),
    'predict_function': ('object', 'Trusted callable invoked as predict_function(model,inputs,batch_size) once per entity. Return finite float32 or float64 ndarray of shape (selected_count,1), with consistent dtype across entities. Caller owns model compatibility, inference semantics, units, native thread limits and prediction dataset-worker allocation. Callbacks may retain references; no global memory bound.'),
}
OUTPUTS = {'route_and_predict': 'Eight-item tuple in exact order: list of scalar slice prediction arrays; baseline time vector; baseline target vector; baseline uncertainty vector; int64 query entity-index vector into baseline order; query time vector; query-key list; list of selected-position arrays. Preserve prediction dtype and original baseline/query order. Vectors and prediction arrays may borrow references. Local prepared-image/input references are released between entities, but callbacks or returned views may retain data. No trajectory reduction or branch blending.'}
COMMON = ' Concrete in-process execution only; abstract witness rejects. Source-specific orchestration with separately injected computations. Synthetic evidence establishes routing and graph wiring over intermediate numeric state, not completed training, empirical effectiveness, HTTP transport or original-CDG approval.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    reports={}
    for name in EVIDENCE_FILES:
        raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+name],cwd=ROOT)
        if raw!=(ROOT/name).read_bytes():raise ValueError('Immutable qualification evidence changed')
        reports[name]=json.loads(raw)
    analytic,graph=[reports[name] for name in EVIDENCE_FILES]
    if (not all(r.get('passed') is True and r.get('synthetic_only') is True for r in reports.values())
            or {c['dtype'] for c in analytic['cases']} != {'float32','float64'}
            or len(analytic['cases'])!=2 or len(analytic['rejected_contracts'])!=9
            or not all(c['exact_source_routing'] and c['query_order_preserved'] and c['prepared_arrays_released']
                       and c['calls']==dict(prepare=2,select=2,predict=2) for c in analytic['cases'])):
        raise ValueError('Complete routing qualification required')
    execution=graph['execution']
    if (execution['nodes']!=6 or execution['edges']!=10 or execution['entities']!=2 or execution['queries']!=4
            or not all(execution[key] for key in ('exact_direct_population_predictions','query_order_preserved',
                'explicit_callable_operations','private_state_unchanged','fitting_disabled','hdf5_loading_disabled'))):
        raise ValueError('Qualified population graph-wiring evidence required')
    provider=ROOT.parent/'sciona-atoms-ml'/PROVIDER_PATH
    if sha(provider)!=analytic['provider_sha256'] or sha(provider)!=execution['provider_sha256']['routing']:
        raise ValueError('Qualified routing provider changed')
    for report in reports.values():
        for name,digest in report['software_sha256'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Qualified routing implementation changed')
    for key,name in [('preparation','domain_adapters/osic_image.py'),
                     ('lifecycle','domain_adapters/osic_image_lifecycle.py'),
                     ('prediction','keras_bounded_prediction.py'),('ordered_fields','pipeline/ordered_fields.py')]:
        if sha(ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml'/name)!=execution['provider_sha256'][key]:
            raise ValueError('Qualified graph provider changed')
    return json.dumps(reports,sort_keys=True,separators=(',',':')).encode()


def plan():
    raw = evidence()
    repo = ROOT.parent/'sciona-atoms-ml'
    committed = subprocess.check_output(['git', 'show', PROVIDER_REVISION+':'+PROVIDER_PATH], cwd=repo)
    if committed != (repo/PROVIDER_PATH).read_bytes():
        raise ValueError('Immutable provider changed')
    specs = [s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name, repo),
             artifact_root=repo/'src/sciona/atoms') if s.import_module == MODULE]
    if len(specs) != 1 or {s.source_symbol for s in specs} != set(OUTPUTS):
        raise ValueError('Exactly one image population routing provider required')
    nodes = {n.name: n for n in ast.parse(committed).body if isinstance(n, ast.FunctionDef)}
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        owners = db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms-ml'").fetchall()
        if len(owners) != 1:
            raise ValueError('Unique provider ownership required')
        for spec in sorted(specs, key=lambda s: s.source_symbol):
            identity = uuid5(NAMESPACE_URL, 'sciona-provider-draft:'+spec.fqdn)
            for table, key in [('atoms', 'atom_id'), ('artifacts', 'artifact_id')]:
                if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s OR {key}=%s', (spec.fqdn, identity)).fetchone():
                    raise ValueError('Provider identity already exists')
            for table in ('atom_versions', 'artifact_versions'):
                if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s', (spec.version_id,)).fetchone():
                    raise ValueError('Version identity collision')
            ports = [dict(direction='input', ordinal=i, name=a.arg, type_desc=CONSTRAINTS[a.arg][0],
                required=True, default_value_repr='', constraints=CONSTRAINTS[a.arg][1])
                for i, a in enumerate(nodes[spec.source_symbol].args.args)]
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc='tuple',
                required=True, default_value_repr='', constraints=OUTPUTS[spec.source_symbol]+COMMON))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+spec.source_symbol, atom_id=str(identity),
                version_id=str(spec.version_id), content_hash=spec.content_hash, provider_sha256=sha(repo/PROVIDER_PATH),
                semver=spec.semver, fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='osic-image-routing-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records, registered_provider_dependencies=[],
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(__file__),
        implementation_sha256=sha(repo/PROVIDER_PATH),
        execution_evidence_files=list(EVIDENCE_FILES),
        limitations=['Draft plan only; transaction, semantic review and publication gates remain required.',
            'The dispatcher accepts three explicit callable inputs and imports no fixed registered computation. The graph binds preparation, ordered selection and prediction separately; these are caller-injected composition dependencies, not hidden registered dependencies of the adapter.',
            'Caller-supplied mappings, category semantics, time and response coordinates, availability and trusted callback behavior remain explicit. Local reference release does not bound memory retained by callbacks or returned views.',
            'The graph remains unapproved; original workflow execution and empirical transfer are separate.'])



if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/osic_image_routing_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), applied=False, approved=False)))
