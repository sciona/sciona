"""Freeze reusable affine provider identity, concrete ports and numerical evidence."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import NAMESPACE_URL,uuid5

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable

ROOT=Path(__file__).resolve().parents[1]
PROVIDER_REVISION='73557bd'
EVIDENCE_COMMIT='8a663cbe'
MODULE='sciona.atoms.numerical.statistics.affine_fit'
PROVIDER_PATH='src/sciona/atoms/numerical/statistics/affine_fit.py'
EVIDENCE='docs/reviews/affine_provider_qualification.json'
CONSTRAINTS={
    'coordinates':'Nonempty finite real vector, at most 64-bit elements, aligned row-for-row with values. No sorting or centering. Caller establishes coordinate units and observation availability. Converted through an uncentered float64 [coordinate,1] design; large integers may lose precision.',
    'values':'Finite real vector with exactly the coordinate shape, at most 64-bit elements. Caller establishes value units and row alignment. No weighting, imputation or availability inference.',
    'rcond':'Required explicit None, -1 or finite ratio in [0,1); booleans rejected. None uses installed NumPy size-dependent cutoff, -1 machine precision, otherwise relative singular-value threshold. Changes rank and minimum-norm coefficients; not regularization.'}
OUTPUT='Dictionary with float64 coefficients of shape (2,) ordered slope/intercept, integer rank, float64 singular_values of shape (min(N,2),), and float64 residual_sums_of_squares shape (1,) only when N>2 and rank=2, otherwise (0,). Empty residuals do not imply zero error. Rank below two means coefficients are not uniquely identifiable; the minimum-norm solution is retained. Slope units are value/coordinate; intercept uses value units. Inputs unchanged. Runtime LAPACK dependent. Concrete provisioned execution only; abstract witness intentionally rejects.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    raw=subprocess.check_output(['git','show',EVIDENCE_COMMIT+':'+EVIDENCE],cwd=ROOT)
    if raw!=(ROOT/EVIDENCE).read_bytes():
        raise ValueError('Immutable numerical evidence changed')
    report=json.loads(raw)
    if (report.get('passed') is not True or report.get('synthetic_only') is not True
            or sum(r['constructed_cases'] for r in report['cases'])!=128
            or len(report['cases'])!=2 or report['invalid_cases_rejected']!=8
            or not report['minimum_norm_single_observation']
            or not all(r['independent_decimal_precision']==60 and r['coefficients_and_residuals_verified']
                       and r['inputs_unchanged'] for r in report['cases'])
            or report['validator_sha256']!=sha(ROOT/'scripts/validate_affine_provider.py')):
        raise ValueError('Complete independent numerical evidence required')
    if report['provider_sha256']!=sha(ROOT.parent/'sciona-atoms'/PROVIDER_PATH):
        raise ValueError('Qualified provider changed')
    return raw


def plan():
    raw=evidence()
    repo=ROOT.parent/'sciona-atoms'
    committed=subprocess.check_output(['git','show',PROVIDER_REVISION+':'+PROVIDER_PATH],cwd=repo)
    if committed!=(repo/PROVIDER_PATH).read_bytes():
        raise ValueError('Immutable provider changed')
    specs=[s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name,repo),artifact_root=repo/'src/sciona/atoms')
           if s.import_module==MODULE]
    if len(specs)!=1 or specs[0].source_symbol!='affine_least_squares':
        raise ValueError('Exactly one affine provider required')
    spec=specs[0]
    identity=uuid5(NAMESPACE_URL,'sciona-provider-draft:'+spec.fqdn)
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        owners=db.execute("SELECT DISTINCT a.owner_id::text,a.source_repo_id::text FROM atoms a JOIN atom_source_repositories r USING(source_repo_id) WHERE r.repo_name='sciona-atoms'").fetchall()
        if len(owners)!=1:
            raise ValueError('Unique provider ownership required')
        for table,key in [('atoms','atom_id'),('artifacts','artifact_id')]:
            if db.execute(f'SELECT 1 FROM {table} WHERE fqdn=%s OR {key}=%s',(spec.fqdn,identity)).fetchone():
                raise ValueError('Provider identity already exists')
        for table in ('atom_versions','artifact_versions'):
            if db.execute(f'SELECT 1 FROM {table} WHERE version_id=%s',(spec.version_id,)).fetchone():
                raise ValueError('Version identity collision')
    node=next(n for n in ast.parse(committed).body if isinstance(n,ast.FunctionDef) and n.name==spec.source_symbol)
    ports=[dict(direction='input',ordinal=i,name=a.arg,type_desc=ast.unparse(a.annotation),required=True,
        default_value_repr='',constraints=CONSTRAINTS[a.arg]) for i,a in enumerate(node.args.args)]
    ports.append(dict(direction='output',ordinal=0,name='result',type_desc='dict',required=True,
        default_value_repr='',constraints=OUTPUT))
    fields={k:getattr(spec,k) for k in ('fqdn','namespace_root','namespace_path','repo_name','source_module_path',
        'import_module','source_symbol','description','domain_tags','source_kind','is_ffi')}
    record=dict(runtime_fqdn=MODULE+'.'+spec.source_symbol,atom_id=str(identity),version_id=str(spec.version_id),
        content_hash=spec.content_hash,provider_sha256=sha(repo/PROVIDER_PATH),semver=spec.semver,
        fingerprint=comparable(spec.fingerprint),seed_fields=comparable(fields),interfaces=ports)
    return dict(format='affine-intake-plan.v1',passed=True,applied=False,approved=False,catalog_mutations=0,
        owner=owners[0],new_providers=[record],registered_provider_dependencies=[],
        execution_evidence_commit=EVIDENCE_COMMIT,execution_evidence_sha256=hashlib.sha256(raw).hexdigest(),
        provider_revision=PROVIDER_REVISION,planner_sha256=sha(__file__),
        limitations=['Draft intake only; metadata, runtime dependency review, rollback and served publication gates remain required.',
            'NumPy/LAPACK is a software dependency, not a registered atom dependency.',
            'This reusable numerical provider does not approve the original OSIC workflow.'])


if __name__=='__main__':
    result=plan()
    (ROOT/'docs/reviews/affine_intake_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(providers=1,ports_per_catalog=4,applied=False)))
