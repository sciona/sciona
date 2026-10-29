"""Atomically qualify an existing legacy window provider in the canonical catalog."""
import argparse
import json
from uuid import UUID, uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import dotenv_values
from scripts.import_residual_execution_drafts import ensure_row
from scripts.validate_audio_windows_canonical import (
    ROOT, ARTIFACT, VERSION, CONTENT_HASH, FQDN, RUNTIME, DESCRIPTION, BOUND, PORTS,
    validate, sha, require,
)

RUNNER = 'audio-windows-canonical-community.v1'
REFERENCE = 'numpy-sliding-window-2.4'
URL = 'https://numpy.org/doc/2.4/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html'
EVIDENCE = str(uuid5(UUID(VERSION), RUNNER))
BOUND_ID = str(uuid5(UUID(VERSION), RUNNER+':regime'))


def connect(readonly=False):
    return psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
        options='-c statement_timeout=30000 -c lock_timeout=10000'+(' -c default_transaction_read_only=on' if readonly else ''))


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def legacy_identity(db):
    return {table:digest(db.execute(f"SELECT to_jsonb(t)-'updated_at' AS row FROM {table} t WHERE atom_id=%s ORDER BY (to_jsonb(t)-'updated_at')::text", (ARTIFACT,)).fetchall())
            for table in ['atoms', 'atom_versions']}


def qualification():
    retained = json.loads((ROOT/'docs/reviews/audio_windows_canonical_validation.json').read_text())
    require(validate() == retained, 'Current numerical qualification differs')
    return retained


def check(db, report, legacy):
    require(legacy_identity(db) == legacy, 'Legacy provider identity changed')
    artifact = db.execute('SELECT artifact_kind,fqdn,status,is_publishable,description FROM artifacts WHERE artifact_id=%s', (ARTIFACT,)).fetchone()
    require(artifact == dict(artifact_kind='atom',fqdn=FQDN,status='approved',is_publishable=True,description=DESCRIPTION), 'Canonical approval differs')
    version = db.execute('SELECT version_id::text,content_hash,trust_tier,is_latest FROM artifact_versions WHERE artifact_id=%s', (ARTIFACT,)).fetchall()
    require(version == [dict(version_id=VERSION,content_hash=CONTENT_HASH,trust_tier=3,is_latest=True)], 'Canonical version differs')
    for table in ['artifact_io_specs','atom_io_specs']:
        actual = db.execute(f'SELECT direction,name,ordinal,type_desc,constraints,required,default_value_repr FROM {table} WHERE version_id=%s ORDER BY direction,ordinal', (VERSION,)).fetchall()
        require(actual == PORTS, 'Provider ports differ')
    evidence = db.execute('SELECT audit_type,passed,status,source_kind,runner_version,details FROM artifact_audit_evidence WHERE version_id=%s', (VERSION,)).fetchall()
    require(evidence == [dict(audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
        details=dict(publication_tier=3,qualification=report,legacy_identity_sha256=legacy,publisher_sha256=sha(__file__)))], 'Canonical evidence differs')
    bounds = db.execute('SELECT scope,bound_kind,validity_statement,evidence_ref_key,review_status FROM artifact_validity_bounds WHERE version_id=%s', (VERSION,)).fetchall()
    require(bounds == [dict(scope='version',bound_kind='regime',validity_statement=BOUND,evidence_ref_key=REFERENCE,review_status='automated_pass')], 'Canonical regime differs')
    descriptions = db.execute('SELECT kind,content FROM artifact_descriptions WHERE artifact_id=%s ORDER BY kind', (ARTIFACT,)).fetchall()
    require(descriptions == [dict(kind='dejargonized',content=BOUND),dict(kind='technical',content=DESCRIPTION)], 'Canonical descriptions differ')
    refs = db.execute('SELECT ref_key,url,verified FROM artifact_references WHERE artifact_id=%s', (ARTIFACT,)).fetchall()
    require(refs == [dict(ref_key=REFERENCE,url=URL,verified=True)], 'Canonical reference differs')
    require(not db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (VERSION,)).fetchone(), 'Unexpected registered dependency')
    require(db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_id=%s', (ARTIFACT,)).fetchone()['n'] == 1, 'Canonical provider not served')
    require(db.execute('SELECT count(*) AS n FROM catalog_atoms_served a JOIN atom_versions v USING(atom_id) WHERE a.atom_id=%s AND v.version_id=%s AND v.is_latest', (ARTIFACT, VERSION)).fetchone()['n'] == 1, 'Legacy provider not served')


def faults():
    return [
        ('approval', "UPDATE artifacts SET is_publishable=false WHERE artifact_id=%s", (ARTIFACT,), 'Canonical approval differs'),
        ('hash', 'UPDATE artifact_versions SET content_hash=%s WHERE version_id=%s', ('0'*64,VERSION), 'Canonical version differs'),
        ('tier', 'UPDATE artifact_versions SET trust_tier=1 WHERE version_id=%s', (VERSION,), 'Canonical version differs'),
        ('latest', 'UPDATE artifact_versions SET is_latest=false WHERE version_id=%s', (VERSION,), 'Canonical version differs'),
        ('canonical_ports', 'DELETE FROM artifact_io_specs WHERE version_id=%s', (VERSION,), 'Provider ports differ'),
        ('legacy_ports', 'DELETE FROM atom_io_specs WHERE version_id=%s', (VERSION,), 'Provider ports differ'),
        ('evidence', "UPDATE artifact_audit_evidence SET details='{}'::jsonb WHERE version_id=%s", (VERSION,), 'Canonical evidence differs'),
        ('failed_audit', 'UPDATE artifact_audit_evidence SET passed=false WHERE version_id=%s', (VERSION,), 'Canonical evidence differs'),
        ('regime', "UPDATE artifact_validity_bounds SET validity_statement='corrupt' WHERE version_id=%s", (VERSION,), 'Canonical regime differs'),
        ('description', "UPDATE artifact_descriptions SET content='corrupt' WHERE artifact_id=%s", (ARTIFACT,), 'Canonical descriptions differ'),
        ('reference', 'UPDATE artifact_references SET verified=false WHERE artifact_id=%s', (ARTIFACT,), 'Canonical reference differs'),
        ('legacy_identity', 'UPDATE atom_versions SET content_hash=%s WHERE version_id=%s', ('0'*64,VERSION), 'Legacy provider identity changed'),
    ]


def promote(apply=False, fault_checks=False, fail_after=None):
    report = qualification()
    if apply:
        gates = json.loads((ROOT/'docs/reviews/audio_windows_canonical_transaction.json').read_text())
        require(gates['passed'] and gates['full_rollback'] and gates['partial_write_rollback'] and gates['post_activation_rollback'] and gates['publisher_sha256'] == sha(__file__) and gates['rejected_faults'] == [f[0] for f in faults()], 'Current transaction qualification required')
    created = 0
    rejected = []
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (RUNNER,))
        legacy = legacy_identity(db)
        atom = db.execute('SELECT * FROM atoms WHERE atom_id=%s FOR UPDATE', (ARTIFACT,)).fetchone()
        version = db.execute('SELECT * FROM atom_versions WHERE version_id=%s FOR UPDATE', (VERSION,)).fetchone()
        require(atom and atom['fqdn'] == FQDN and atom['status'] == 'approved' and atom['is_publishable'] and atom['import_module']+'.'+atom['source_symbol'] == RUNTIME, 'Exact legacy provider required')
        require(version and str(version['atom_id']) == ARTIFACT and version['content_hash'] == CONTENT_HASH and version['trust_tier'] == 3 and version['is_latest'], 'Exact legacy version required')
        require(db.execute("SELECT 1 FROM references_registry WHERE ref_id='numpy_docs'").fetchone(), 'NumPy reference missing')
        def ensure(table, keys, row):
            nonlocal created
            created += int(ensure_row(db,table,keys,row))
            if fail_after is not None and created == fail_after:
                raise RuntimeError('injected canonical publication failure')
        columns = {r['column_name'] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='artifacts'")}
        fields = {k:v for k,v in atom.items() if k in columns and k not in ['created_at','updated_at']}
        fields.update(artifact_id=UUID(ARTIFACT), artifact_kind='atom', description=DESCRIPTION, status='approved', is_publishable=True)
        ensure('artifacts',dict(artifact_id=UUID(ARTIFACT)),fields)
        fields = {k:v for k,v in version.items() if k not in ['atom_id','created_at']}
        fields.update(artifact_id=UUID(ARTIFACT))
        ensure('artifact_versions',dict(version_id=UUID(VERSION)),fields)
        for table, key in [('artifact_io_specs','artifact_id'),('atom_io_specs','atom_id')]:
            for port in PORTS:
                row = dict(port, **{key:UUID(ARTIFACT)},version_id=UUID(VERSION))
                if table == 'artifact_io_specs':
                    row['dim_signature'] = ''
                ensure(table,dict(version_id=UUID(VERSION),direction=port['direction'],name=port['name']),row)
        for kind, content in [('technical',DESCRIPTION),('dejargonized',BOUND)]:
            ensure('artifact_descriptions',dict(artifact_id=UUID(ARTIFACT),kind=kind,language='en'),dict(description_id=uuid5(UUID(VERSION),RUNNER+kind),artifact_id=UUID(ARTIFACT),kind=kind,language='en',content=content,generated_by=RUNNER,reviewed=False))
        ensure('artifact_references',dict(artifact_id=UUID(ARTIFACT),ref_key=REFERENCE),dict(artifact_id=UUID(ARTIFACT),ref_key=REFERENCE,ref_id='numpy_docs',title='NumPy 2.4 sliding_window_view',url=URL,source='llm_extracted',verified=True,confidence='high',relevance_note='Window shape, stepped slicing and read-only shared-memory behavior; verified independently on synthetic arrays.'))
        ensure('artifact_validity_bounds',dict(bound_id=UUID(BOUND_ID)),dict(bound_id=UUID(BOUND_ID),artifact_id=UUID(ARTIFACT),version_id=UUID(VERSION),scope='version',bound_kind='regime',validity_statement=BOUND,evidence_ref_key=REFERENCE,review_status='automated_pass',metadata=Jsonb(dict(review_source='automated',approval_evidence_id=EVIDENCE))))
        rollup = dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',semantic_status='pass',developer_semantics_status='pass',review_status='approved',review_semantic_verdict='pass',review_developer_semantics_verdict='pass',trust_readiness='ready',review_limitations=report['limitations'],review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits',parity_coverage_level='positive_and_negative',parity_test_status='pass')
        ensure('artifact_audit_rollups',dict(artifact_id=UUID(ARTIFACT)),dict(artifact_id=UUID(ARTIFACT),**rollup))
        ensure('artifact_audit_evidence',dict(evidence_id=UUID(EVIDENCE)),dict(evidence_id=UUID(EVIDENCE),artifact_id=UUID(ARTIFACT),version_id=UUID(VERSION),audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,details=Jsonb(dict(publication_tier=3,qualification=report,legacy_identity_sha256=legacy,publisher_sha256=sha(__file__)))))
        check(db,report,legacy)
        if fail_after == 'activation':
            raise RuntimeError('injected canonical activation failure')
        if fault_checks:
            for name, statement, params, expected in faults():
                db.execute('SAVEPOINT fault')
                try:
                    require(db.execute(statement,params).rowcount > 0, 'Fault changed no rows')
                    try:
                        check(db,report,legacy)
                    except ValueError as error:
                        require(str(error) == expected, 'Unexpected fault rejection: '+str(error))
                        rejected.append(name)
                    else:
                        raise ValueError('Fault accepted: '+name)
                finally:
                    db.execute('ROLLBACK TO SAVEPOINT fault')
                    db.execute('RELEASE SAVEPOINT fault')
                check(db,report,legacy)
        if not apply:
            db.rollback()
    return dict(applied=apply,rows_created=created,trust_tier=3,legacy_identity_preserved=True,rejected_faults=rejected,publisher_sha256=sha(__file__))


def snapshot():
    result = {}
    with connect(True) as db:
        for table in ['artifacts','artifact_versions','artifact_io_specs','artifact_descriptions','artifact_references','artifact_validity_bounds','artifact_audit_rollups','artifact_audit_evidence','atoms','atom_versions','atom_io_specs']:
            key = 'atom_id' if table in ['atoms','atom_versions','atom_io_specs'] else 'artifact_id'
            result[table] = digest(db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE {key}=%s ORDER BY to_jsonb(t)::text', (ARTIFACT,)).fetchall())
    return result


def transaction():
    before = snapshot()
    dry = promote(fault_checks=True)
    require(dry['rows_created'] > 0 and snapshot() == before, 'Absent-candidate rollback qualification required')
    for failure in [5, 'activation']:
        try:
            promote(fail_after=failure)
        except RuntimeError as error:
            require(str(error) == ('injected canonical activation failure' if failure == 'activation' else 'injected canonical publication failure'), 'Unexpected rollback failure')
        else:
            raise ValueError('Failure injection did not fire')
        require(snapshot() == before, 'Publication failure left mutations')
    return dict(passed=True,catalog_mutations=0,full_rollback=True,partial_write_rollback=True,post_activation_rollback=True,
        rejected_faults=dry['rejected_faults'],planned_rows=dry['rows_created'],publisher_sha256=sha(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-transaction',action='store_true')
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    require(not (args.qualify_transaction and args.apply), 'Choose transaction qualification or apply')
    if args.qualify_transaction:
        report = transaction();name = 'transaction'
    else:
        report = promote(args.apply);name = 'publication_repeat' if report['rows_created'] == 0 else 'publication'
    if args.qualify_transaction or args.apply:
        (ROOT/f'docs/reviews/audio_windows_canonical_{name}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
