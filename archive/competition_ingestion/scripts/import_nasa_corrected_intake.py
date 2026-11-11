"""Stage a corrected execution version without changing the original snapshot or approval."""
import argparse
import json
from pathlib import Path
from uuid import UUID,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from sciona.nasa_complete_workflow_graph import build_complete_workflow_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.import_residual_execution_drafts import ensure_row
from scripts.plan_nasa_corrected_intake import ROOT,plan,check_source_anchor,check_atoms,check_parents,origin_rows,sha,require

RUNNER='nasa-corrected-intake-draft.v1'


def stage(apply=False):
    if apply:
        gate=json.loads((ROOT/'docs/reviews/nasa_corrected_intake_draft_transaction.json').read_text())
        require(gate['passed'] and gate['failure_rollback_verified'] and gate['importer_sha256']==sha(Path(__file__))
            and gate['validator_sha256']==sha(ROOT/'scripts/validate_nasa_corrected_intake_draft_transaction.py'),'Draft transaction qualification missing or stale')
    proposed=plan();graph=build_complete_workflow_graph(10)
    digest,nodes,edges=encode_execution_graph(graph)
    require(digest==proposed['graph_sha256'],'Corrected graph changed')
    identity,version=UUID(proposed['artifact_id']),UUID(proposed['version_id']);created=0
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        for lock in ['residual-classifier-draft.v1','nasa-domain-draft.v1','nasa-lifecycle-draft.v1','nasa-population-draft.v1',RUNNER]:
            db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(lock,))
        row=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s FOR UPDATE',(identity,)).fetchone()
        require(row==dict(status='draft',is_publishable=False),'Original must remain unapproved during staging')
        check_source_anchor(db);check_atoms(db,proposed['atoms']);check_parents(db,proposed['parent_graphs'])
        require(origin_rows(db)==proposed['original_rows_sha256'],'Original records changed before staging')
        def ensure(table,keys,row):
            nonlocal created
            created+=ensure_row(db,table,keys,row)
        ensure('artifact_versions',{'version_id':version},dict(version_id=version,artifact_id=identity,content_hash=digest,
            semver='0.1.0+corrected.'+digest[:12],is_latest=False,trust_tier=3))
        for table,rows,keys in [('artifact_cdg_nodes',nodes,['node_id']),
            ('artifact_cdg_edges',edges,['source_id','target_id','output_name','input_name'])]:
            for item in rows:
                row=dict(version_id=version,**item)
                ensure(table,{key:row[key] for key in ['version_id',*keys]},row)
        for direction,ports in [('input',proposed['inputs']),('output',proposed['outputs'])]:
            for ordinal,port in enumerate(ports):
                fields=['name','type_desc','constraints','required','default_value_repr','dim_signature']
                require(not any(port.get(name) for name in ['data_kind','time_basis','provenance']),
                    'Catalog root schema cannot silently discard populated metadata')
                row=dict(artifact_id=identity,version_id=version,direction=direction,ordinal=ordinal,
                    **{name:port[name] for name in fields})
                ensure('artifact_io_specs',{key:row[key] for key in ['artifact_id','version_id','direction','name']},row)
        for binding in proposed['bindings']:
            row=dict(version_id=version,node_id=binding['node_id'],bound_artifact_fqdn=binding['fqdn'],
                bound_version_content_hash=binding['content_hash'],binding_confidence=1.,binding_source=RUNNER,status='active',
                alternatives=Jsonb([]),evidence_summary=Jsonb(dict(runtime_fqdn=binding['runtime_fqdn'],
                    provider_version_id=binding['version_id'],output_aliases_by_ordinal=[port['name'] for port in binding['outputs']])))
            ensure('artifact_cdg_bindings',dict(version_id=version,node_id=binding['node_id']),row)
        sources=[dict(fqdn=proposed['fqdn'],graph_sha256=proposed['original_content_hash'],version_id=proposed['original_version_id'])]
        sources+=list(proposed['parent_graphs'].values())
        for source in sources:
            key=dict(dependent_version_id=version,dependency_artifact_fqdn=source['fqdn'],dependency_content_hash=source['graph_sha256'],port_name='')
            ensure('artifact_dependencies',key,dict(**key,dependency_role='cdg',optional=False,
                binding_metadata=Jsonb(dict(source_version_id=source['version_id'],scope='Immutable source/version provenance; runtime computation is expressed by direct atom bindings.'))))
        evidence=uuid5(version,RUNNER)
        ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=identity,version_id=version,
            audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
            details=Jsonb(dict(scope='Corrected execution version staged without approving or rewriting the original snapshot.',
                original_rows_sha256=proposed['original_rows_sha256'],evidence_sha256=proposed['evidence_sha256'],planner_sha256=proposed['planner_sha256']))))
        document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
        restored=_artifact_document_to_cdg(document,version_id=str(version),content_hash=digest,require_execution_envelope=True)
        require(restored==graph,'Version-selected catalog round-trip differs')
        require(not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(identity,)).fetchone(),'Draft intake unexpectedly served')
        require(origin_rows(db)==proposed['original_rows_sha256'],'Original records changed during staging')
        require(db.execute('SELECT is_latest FROM artifact_versions WHERE version_id=%s',(proposed['original_version_id'],)).fetchone()['is_latest'],
            'Original version must stay latest until promotion')
        check_atoms(db,proposed['atoms']);check_parents(db,proposed['parent_graphs'])
        require(plan()==proposed,'Qualification changed during staging')
        if not apply:db.rollback()
    return dict(applied=apply,rows_created=created,version_id=str(version),artifact_id=str(identity),graph_sha256=digest,
        new_atoms=0,reused_atoms=len(proposed['atoms']),nodes=len(nodes),edges=len(edges),original_records_preserved=True,
        version_selected_roundtrip=True,approved=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    report=stage(args.apply)
    name='nasa_corrected_intake_draft_rollback.json' if not args.apply else ('nasa_corrected_intake_draft_import.json' if report['rows_created'] else 'nasa_corrected_intake_draft_repeat.json')
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
