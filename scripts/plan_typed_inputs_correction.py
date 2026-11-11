"""Plan a history-preserving correction of non-serving conversion drafts."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import UUID

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import comparable
from scripts.promote_openvaccine_decomposed_providers import snapshot
from scripts.validate_typed_inputs_unicode_correction import validate

ROOT=Path(__file__).resolve().parents[1]
PROVIDER_REVISION='9512aa6'
EVIDENCE_COMMIT='c529089b'
ORIGINAL_PLAN_COMMIT='ed5eb5b1'
PROVIDER_PATH='src/sciona/atoms/ml/pipeline/typed_inputs.py'
REPORTS=('docs/reviews/typed_inputs_unicode_correction.json',
         'docs/reviews/osic_quantile_replay_graph_unicode_correction.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable(name,revision):
    raw=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
    if raw!=(ROOT/name).read_bytes():raise ValueError('Immutable correction input changed')
    return json.loads(raw)


def evidence():
    reports=[immutable(name,EVIDENCE_COMMIT) for name in REPORTS]
    if validate()!=reports[0]:raise ValueError('Fresh Unicode correction differs')
    graph=reports[1]
    if not graph['passed'] or graph['graph_execution']['nodes']!=6 or graph['graph_execution']['edges']!=8:
        raise ValueError('Complete corrected graph evidence required')
    for name,digest in graph['software_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified graph software changed')
    repo=ROOT.parent/'sciona-atoms-ml'
    for module,digest in graph['graph_execution']['provider_sha256'].items():
        path=repo/'src/sciona/atoms/ml'/(module.removeprefix('sciona.atoms.ml.').replace('.','/')+'.py')
        if sha(path)!=digest:raise ValueError('Qualified graph provider changed')
    if subprocess.check_output(['git','show',PROVIDER_REVISION+':'+PROVIDER_PATH],cwd=repo)!=(repo/PROVIDER_PATH).read_bytes():
        raise ValueError('Corrected provider revision changed')
    return reports


def plan():
    reports=evidence()
    old=immutable('docs/reviews/typed_inputs_intake_plan.json',ORIGINAL_PLAN_COMMIT)
    repo=ROOT.parent/'sciona-atoms-ml'
    specs={s.source_symbol:s for s in _parse_registered_atoms(repo=ProviderRepo(repo.name,repo),
        artifact_root=repo/'src/sciona/atoms') if s.import_module=='sciona.atoms.ml.pipeline.typed_inputs'}
    if set(specs)!={'cast_finite_array','encode_feature_descriptors'}:raise ValueError('Exact provider inventory required')
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        ids=[UUID(r['atom_id']) for r in old['new_providers']]
        before=snapshot(db,ids)
        for original in old['new_providers']:
            symbol=original['seed_fields']['source_symbol'];spec=specs[symbol]
            record=copy.deepcopy(original)
            record.update(version_id=str(spec.version_id),content_hash=spec.content_hash,semver=spec.semver,
                          fingerprint=comparable(spec.fingerprint),provider_sha256=sha(repo/PROVIDER_PATH))
            record['seed_fields']={k:comparable(getattr(spec,k)) for k in original['seed_fields']}
            changed=symbol=='encode_feature_descriptors'
            if (record['version_id']!=original['version_id'])!=changed:
                raise ValueError('Only the encoder version must change')
            if changed:
                record['interfaces'][0]['constraints']+=' Require Unicode scalar-value strings; reject every surrogate code point, including explicit UTF-16 pairs, to preserve exact JSON round trips.'
                record['interfaces'][1]['constraints']+=' Exact round trips require the declared Unicode scalar-value string input contract.'
            old_versions={}
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                base=db.execute(f'SELECT fqdn,status,is_publishable,description FROM {prefix}s WHERE {key}=%s',
                    (original['atom_id'],)).fetchone()
                expected=dict(fqdn=original['runtime_fqdn'],status='flagged' if prefix=='atom' else 'draft',
                              is_publishable=False,description=original['seed_fields']['description'])
                if base!=expected:raise ValueError('Existing draft identity or state changed')
                versions=db.execute(f'SELECT to_jsonb(v) AS row FROM {prefix}_versions v WHERE {key}=%s',
                    (original['atom_id'],)).fetchall()
                if len(versions)!=1:raise ValueError('Expected one immutable starting version')
                version=versions[0]['row']
                if (version['version_id']!=original['version_id'] or version['content_hash']!=original['content_hash']
                    or version['trust_tier']!=3 or version['is_latest'] is not True):
                    raise ValueError('Starting version differs')
                old_versions[prefix]=comparable(version)
                ports=db.execute(f'''SELECT direction,ordinal,name,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''',(original['version_id'],)).fetchall()
                if ports!=sorted(original['interfaces'],key=lambda p:(p['direction'],p['ordinal'])):
                    raise ValueError('Starting contracts changed')
                if changed and db.execute(f'SELECT 1 FROM {prefix}_versions WHERE version_id=%s',(record['version_id'],)).fetchone():
                    raise ValueError('Corrected version already exists')
                if db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s',(original['atom_id'],)).fetchone():
                    raise ValueError('Correction must start with non-serving drafts')
            record.update(previous=original,previous_versions=old_versions,new_version_required=changed)
            records.append(record)
        if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=ANY(%s)',
            ([UUID(r['version_id']) for r in old['new_providers']],)).fetchone():
            raise ValueError('Unexpected draft dependencies')
    return dict(format='typed-inputs-correction-plan.v1',passed=True,applied=False,approved=False,catalog_mutations=0,
        providers=records,owner=old['owner'],before=before,provider_revision=PROVIDER_REVISION,
        evidence_commit=EVIDENCE_COMMIT,evidence_files=list(REPORTS),
        evidence_sha256=hashlib.sha256(json.dumps(reports,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        original_plan_commit=ORIGINAL_PLAN_COMMIT,planner_sha256=sha(__file__),
        limitations=['Plan only; transaction rollback gates and reviewed staging are required before mutation.',
            'Preserve old encoder version, ports and audit evidence; change only its latest flag when adding the corrected version.',
            'Record the demonstrated old-encoder semantic failure against its old version, not the unchanged casting version.',
            'The cast version stays unchanged; both providers receive corrected module evidence and remain non-serving until publication.'])


if __name__=='__main__':
    result=plan()
    (ROOT/'docs/reviews/typed_inputs_correction_plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=2,new_versions=sum(r['new_version_required'] for r in result['providers']),applied=False)))
