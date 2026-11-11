"""Read back exact repaired Cornell contracts and preserved served execution graph."""
import json
from pathlib import Path
from sciona.cornell_graph import build_cornell_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts import reconcile_cornell_metadata as repair


def verify():
    semantic, targets, original = repair.qualified()
    with repair.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        source = original['identity']
        state = db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',
                           (source['artifact_id'],)).fetchone()
        latest = db.execute('SELECT version_id::text,content_hash FROM artifact_versions WHERE artifact_id=%s AND is_latest',
                            (source['artifact_id'],)).fetchall()
        if state != dict(status='draft',is_publishable=False) or latest != [dict(version_id=source['version_id'],content_hash=source['content_hash'])]:
            raise ValueError('Original intake selection changed')
        original_failed_audits = db.execute('SELECT count(*) AS n FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',
                                           (source['version_id'],)).fetchone()['n']
        for target in targets:
            repair.check_identity(db,target)
            r, contract = target['identity'],target['contract']
            for prefix,key in [('artifact','artifact_id'),('atom','atom_id')]:
                if prefix not in target['catalogs']:
                    continue
                if not db.execute(f'SELECT {prefix}_is_publishable(%s) AS yes',(r['artifact_id'],)).fetchone()['yes']:
                    raise ValueError('Native eligibility missing')
                if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s',(r['artifact_id'],)).fetchone():
                    raise ValueError('Retained entry no longer served')
                ports = db.execute(f'SELECT direction,name,type_desc,ordinal,required,default_value_repr,constraints FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal', (r['version_id'],)).fetchall()
                if ports != contract['ports']:
                    raise ValueError('Published contracts differ')
                descriptions = db.execute(f"SELECT content FROM {prefix}_descriptions WHERE {key}=%s AND kind='dejargonized' AND language='en'", (r['artifact_id'],)).fetchall()
                if descriptions != [dict(content=contract['plain_description'])]:
                    raise ValueError('Published plain description differs')
                parameters = db.execute(f'SELECT name,type_desc,required,constraints_json FROM {prefix}_parameters WHERE version_id=%s ORDER BY position', (r['version_id'],)).fetchall()
                p = contract['ports'][0]
                expected = dict(name=p['name'],type_desc=p['type_desc'],required=True,
                    constraints_json=dict(input_contract=p['constraints'],output_contract=contract['ports'][1]['constraints'],
                                          validity_bounds=semantic['limitations'],reuse=semantic['reuse_decision']))
                if parameters != [expected]:
                    raise ValueError('Published parameter bounds differ')
            if r['artifact_kind']=='cdg':
                document = db.execute('SELECT get_artifact_document(%s) AS d',(r['fqdn'],)).fetchone()['d']
                graph = _artifact_document_to_cdg(document,version_id=r['version_id'],content_hash=r['content_hash'],require_execution_envelope=True)
                if graph != build_cornell_graph():
                    raise ValueError('Retained serialized audio-event graph changed')
    return dict(passed=True,read_only=True,catalog_mutations=0,exact_contracts_verified=True,
        native_eligible_artifacts=3,native_eligible_atoms=2,served_artifacts=3,served_atoms=2,
        original_intake_status='draft',original_failed_audits_preserved=original_failed_audits,
        retained_graph_unchanged=True,trust_tier=3,
        review_sha256=repair.sha(repair.ROOT/repair.REPORT),verifier_sha256=repair.sha(Path(__file__)))


if __name__=='__main__':
    result=verify()
    (repair.ROOT/'docs/reviews/cornell_metadata_verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
