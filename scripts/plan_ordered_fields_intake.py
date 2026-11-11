"""Freeze explicit ordered field routing and its synthetic reuse evidence."""
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
PROVIDER_REVISION = '51fd789'
EVIDENCE_COMMIT = 'fa541de0'
MODULE = 'sciona.atoms.ml.pipeline.ordered_fields'
PROVIDER_PATH = 'src/sciona/atoms/ml/pipeline/ordered_fields.py'
EVIDENCE_FILES = ('docs/reviews/ordered_fields_provider.json','docs/reviews/osic_image_slice_graph.json')
CONSTRAINTS = {
    'mapping': ('dict', 'Plain dict only, with nonblank string keys. Extra fields are permitted and ignored. Values may be arbitrary caller-owned objects; this operation does not inspect, copy, convert or validate values. Caller establishes schema meanings, alignment, units and downstream value contracts.'),
    'fields': ('list[str]', 'Nonempty plain list or tuple of unique nonblank string field names, all present in the mapping. Preserve supplied order and exact names without trimming. Reject missing, duplicate, unordered or invalid field names and container subclasses. No field order or application schema is inferred.'),
}
OUTPUTS = {'select_values': 'New outer Python list of selected borrowed values in explicit field order. Preserve value identity, aliasing, dtype and shape without mutating inputs. Downstream mutations of borrowed objects affect caller-owned values. No deep copy, value validation, model execution or unit conversion.'}
COMMON = ' Concrete in-process execution only; abstract witness rejects. Synthetic motion and electrical-power examples demonstrate constructed ordering reuse, not empirical effectiveness. No HTTP transport or whole-CDG approval claim.'


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
            or len(analytic['cases'])!=8 or len(analytic['rejected_contracts'])!=15
            or not all(c['analytic_result_exact'] and c['explicit_order_preserved'] and c['borrowed_values_unchanged'] for c in analytic['cases'])
            or not all(analytic[key] for key in ('borrowed_aliasing_preserved','outer_list_independent','exact_names_preserved'))):
        raise ValueError('Complete field ordering and identity qualification required')
    execution=graph['execution']
    if (execution['nodes']!=4 or execution['edges']!=3
            or not all(execution[key] for key in ('exact_direct_provider_predictions','exact_prepared_arrays','borrowed_graph_inputs_verified','private_state_unchanged'))):
        raise ValueError('Qualified graph-wiring evidence required')
    provider=ROOT.parent/'sciona-atoms-ml'/PROVIDER_PATH
    if (sha(provider)!=analytic['provider_sha256'] or sha(provider)!=execution['provider_sha256']['ordered_fields']
            or sha(ROOT/'scripts/validate_ordered_fields_provider.py')!=analytic['validator_sha256']):
        raise ValueError('Qualified field provider or validator changed')
    for name,digest in graph['software_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified graph implementation changed')
    for key,name in [('preparation','domain_adapters/osic_image.py'),
                     ('lifecycle','domain_adapters/osic_image_lifecycle.py'),
                     ('prediction','keras_bounded_prediction.py')]:
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
        raise ValueError('Exactly one ordered field provider required')
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
            ports.append(dict(direction='output', ordinal=0, name='result', type_desc='list',
                required=True, default_value_repr='', constraints=OUTPUTS[spec.source_symbol]+COMMON))
            fields = {k: getattr(spec, k) for k in ('fqdn', 'namespace_root', 'namespace_path', 'repo_name',
                'source_module_path', 'import_module', 'source_symbol', 'description', 'domain_tags', 'source_kind', 'is_ffi')}
            records.append(dict(runtime_fqdn=MODULE+'.'+spec.source_symbol, atom_id=str(identity),
                version_id=str(spec.version_id), content_hash=spec.content_hash, provider_sha256=sha(repo/PROVIDER_PATH),
                semver=spec.semver, fingerprint=comparable(spec.fingerprint), seed_fields=comparable(fields), interfaces=ports))
    return dict(format='ordered-fields-intake-plan.v1', passed=True, applied=False, approved=False,
        catalog_mutations=0, owner=owners[0], new_providers=records, registered_provider_dependencies=[],
        execution_evidence_commit=EVIDENCE_COMMIT, execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION, planner_sha256=sha(__file__),
        implementation_sha256=sha(repo/PROVIDER_PATH),
        execution_evidence_files=list(EVIDENCE_FILES),
        limitations=['Draft plan only; transaction, semantic review and publication gates remain required.',
            'The provider uses Python container operations and has no registered atom dependencies. Image-graph evidence is a composed use, not a dependency of field selection.',
            'Borrowed values are not validated or copied; callers retain responsibility for downstream mutation, alignment and units.',
            'The graph remains unapproved; original workflow execution and empirical transfer are separate.'])



if __name__ == '__main__':
    result = plan()
    (ROOT/'docs/reviews/ordered_fields_intake_plan.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['new_providers']), applied=False, approved=False)))
