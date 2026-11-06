"""Complete missing metadata on the reused approved numerical mean provider."""
import argparse
import inspect
import json
from uuid import UUID
from psycopg.types.json import Jsonb

from scripts.audit_decomposed_trackml_readiness import ROOT, sha
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import ensure_row
from scripts.promote_video_numeric_providers import qualify_plan
from scripts.promote_openvaccine_decomposed_providers import snapshot
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

PLAN = ROOT/'docs/reviews/video_mean_metadata_plan.json'
GATES = ROOT/'docs/reviews/video_mean_metadata_gates.json'
RUNNER = 'video-mean-metadata.v1'
IDENTITY='3e054efb-38d6-56c1-a7c8-26da81e7a310'
VERSION='fbea5906-9987-5e67-88c4-5ffe3d971e66'
CONTENT='280b371b126f6d7f580dd169289280af76fc50d64350b784c3aefd346df18af5'
PLAIN='Average equally sized lists of scores in the supplied order.'
TECHNICAL='Equal-weight ordered mean of finite equally shaped one-dimensional vectors, converted to float64. Requires a nonempty list or tuple of vectors; individual empty vectors are permitted. Sum in caller order then divide by count; reject overflow. No broadcasting, calibration, class ordering or split policy. Returns a new float64 vector.'


def check_mean(db):
    for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
        rows=db.execute(f"""SELECT a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
            FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest""",(IDENTITY,)).fetchall()
        if rows!=[dict(status='approved',is_publishable=True,version_id=VERSION,content_hash=CONTENT,trust_tier=3)]:
            raise ValueError('Exact approved mean source version required')


def plan():
    qualify_plan()  # Rechecks executed mean source and installed numeric runtime.
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        check_mean(db)
        for identity,version in [(IDENTITY,VERSION)]:
            plain, technical = PLAIN, TECHNICAL
            inserts = []
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                for suffix in ('parameters', 'descriptions'):
                    if db.execute(f'SELECT 1 FROM {prefix}_{suffix} WHERE {key}=%s', (identity,)).fetchone():
                        raise ValueError('Repair only missing metadata')
                ports = db.execute(f"SELECT name,ordinal,type_desc,required,default_value_repr FROM {prefix}_io_specs WHERE version_id=%s AND direction='input' ORDER BY ordinal", (version,)).fetchall()
                if len(ports) != 1:
                    raise ValueError('Single boundary input required')
                p = ports[0]
                row = {key:identity, 'version_id':version, 'name':p['name'], 'position':p['ordinal'],
                    'kind':'positional_or_keyword', 'type_desc':p['type_desc'], 'required':p['required'],
                    'default_value_repr':p['default_value_repr'], 'technical_description':technical,
                    'dejargonized_description':plain, 'constraints_json':dict(scope=technical, synthetic_qualification_only=True)}
                inserts.append(dict(table=prefix+'_parameters', keys={k:row[k] for k in (key,'version_id','name')}, row=row))
                for kind, content in [('technical', technical), ('dejargonized', plain)]:
                    score = round(estimate_jargon_score(content), 6)
                    if kind == 'dejargonized' and score >= .4:
                        raise ValueError('Plain description requires revision')
                    keys = {key:identity, 'kind':kind, 'language':'en'}
                    inserts.append(dict(table=prefix+'_descriptions', keys=keys,
                        row=dict(keys, content=content, generated_by=RUNNER, reviewed=False, jargon_score=score)))
            records.append(dict(node='mean', identity=identity, version_id=version, inserts=inserts))
        if len(records) != 1:
            raise ValueError('One existing mean provider required')
        before = snapshot(db, [UUID(r['identity']) for r in records])
    return dict(records=records, before=before, repairer_sha256=sha(__file__),
        qualification_sha256=sha(ROOT/'docs/reviews/decomposed_video_multilabel_graph_execution_v4.json'),
        scope='Missing parameter and description metadata only; existing approved versions, interfaces and audit histories retained.')


class InjectedFailure(RuntimeError):
    pass


def qualified():
    p = json.loads(PLAN.read_text())
    if p['repairer_sha256'] != sha(__file__) or p['qualification_sha256'] != sha(ROOT/'docs/reviews/decomposed_video_multilabel_graph_execution_v4.json'):
        raise ValueError('Repair qualification changed')
    qualify_plan()  # Includes exact executed provider sources and installed packages.
    return p


def repair(apply=False, fail_after=None):
    p = qualified()
    if apply:
        gates = json.loads(GATES.read_text())
        if not gates['passed'] or gates['plan_sha256'] != sha(PLAN) or gates['repairer_sha256'] != sha(__file__):
            raise ValueError('Matching rollback gates required')
    writes = 0
    ids = [UUID(r['identity']) for r in p['records']]
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        db.execute('SELECT artifact_id FROM artifacts WHERE artifact_id=ANY(%s) FOR UPDATE', (ids,)).fetchall()
        check_mean(db)
        before = snapshot(db, ids)
        for record in p['records']:
            for item in record['inserts']:
                row = dict(item['row'])
                if 'constraints_json' in row:
                    row['constraints_json'] = Jsonb(row['constraints_json'])
                writes += int(ensure_row(db, item['table'], item['keys'], row))
                if writes == fail_after:
                    raise InjectedFailure('Injected metadata repair failure')
        after = snapshot(db, ids)
        for table, digest in before.items():
            if table not in ('atoms', 'artifacts', 'atom_parameters', 'artifact_parameters', 'atom_descriptions', 'artifact_descriptions') and after[table] != digest:
                raise ValueError('Existing provider history changed')
        for record in p['records']:
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                row = db.execute(f'SELECT status,is_publishable,{prefix}_is_publishable({key}) AS eligible FROM {prefix}s WHERE {key}=%s', (record['identity'],)).fetchone()
                if row != dict(status='approved', is_publishable=True, eligible=True):
                    raise ValueError('Complete native eligibility required')
        check_mean(db)
        if fail_after is not None:
            raise ValueError('Fault point not reached')
        if not apply:
            db.rollback()
    return dict(applied=apply, rows_created=writes, native_eligible_boundaries=1, source_and_audit_history_preserved=True)


def validate():
    p = qualified()
    def unchanged():
        with connect() as db:
            db.execute('SET TRANSACTION READ ONLY')
            if snapshot(db, [UUID(r['identity']) for r in p['records']]) != p['before']:
                raise ValueError('Rollback left metadata changes')
    unchanged()
    dry = repair()
    unchanged()
    points = [1, dry['rows_created']//2, dry['rows_created']]
    for point in points:
        try:
            repair(fail_after=point)
        except InjectedFailure:
            pass
        else:
            raise ValueError('Expected rollback failure')
        unchanged()
    return dict(passed=True, dry_run=dry, rollback_points=points, all_rollbacks_complete=True,
        plan_sha256=sha(PLAN), repairer_sha256=sha(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'validate', 'apply', 'repeat'])
    action = parser.parse_args().action
    if action == 'plan':
        result, path = plan(), PLAN
    elif action == 'validate':
        result, path = validate(), GATES
    else:
        result, path = repair(apply=True), ROOT/f'docs/reviews/video_mean_metadata_{action}.json'
        if action == 'repeat' and result['rows_created']:
            raise ValueError('Repeat created rows')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(planned=True, boundaries=len(result['records'])) if action == 'plan' else result))
