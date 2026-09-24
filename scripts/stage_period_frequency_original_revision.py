"""Stage a version of the original physics artifact using its qualified execution graph."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from scripts.audit_period_frequency_original_scope import ROOT,ARTIFACT,ORIGINAL,REPLAY,EXECUTION,audit,require
from scripts.import_residual_execution_drafts import ensure_row
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph

RUNNER='period-frequency-original-revision-draft.v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history(db):
    result={}
    for table in ['artifact_versions','artifact_cdg_nodes','artifact_cdg_edges','artifact_cdg_bindings',
            'artifact_io_specs','artifact_audit_evidence','artifact_validity_bounds','artifact_dependencies']:
        key='dependent_version_id' if table=='artifact_dependencies' else 'version_id'
        rows=db.execute(f"SELECT to_jsonb(t)-'updated_at'-'is_latest' AS row FROM {table} t WHERE {key}=ANY(%s::uuid[]) ORDER BY (to_jsonb(t)-'updated_at'-'is_latest')::text",([ORIGINAL,REPLAY],)).fetchall()
        result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    return result


def stage(source,apply=False,fail_after_writes=None):
    if apply:
        gates=json.loads((ROOT/'docs/reviews/period_frequency_original_revision_transaction.json').read_text())
        require(gates['passed'] and gates['rollback_verified'] and gates['injected_failure_rollback_verified']
            and gates['stager_sha256']==sha(__file__)
            and gates['validator_sha256']==sha(ROOT/'scripts/validate_period_frequency_revision_transaction.py'),'Current draft transaction gates required')
    scope=audit(source)
    require(scope==json.loads((ROOT/'docs/reviews/physics_period_frequency_original_scope.json').read_text()),'Frozen scope evidence differs')
    created=0
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        state=db.execute('SELECT fqdn,status,is_publishable FROM artifacts WHERE artifact_id=%s FOR UPDATE',(ARTIFACT,)).fetchone()
        require(state and state['status']=='draft' and not state['is_publishable'],'Original artifact must remain draft while staging')
        original_history=history(db)
        parent=db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest,v.trust_tier FROM artifacts a '
            'JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s FOR SHARE OF a,v',(EXECUTION,)).fetchone()
        require(parent['status']=='approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier']==3,'Qualified parent changed')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(parent['fqdn'],)).fetchone()['d']
        graph=_artifact_document_to_cdg(document,version_id=EXECUTION,content_hash=parent['content_hash'],require_execution_envelope=True)
        digest,nodes,edges=encode_execution_graph(graph)
        require(digest==scope['execution_graph_sha256'] and len(nodes)==1 and not edges,'Qualified graph differs')
        graph=graph.model_copy(deep=True)
        graph.metadata.update(original_artifact_id=ARTIFACT,original_version_id=ORIGINAL,
            qualified_numerical_parent_version_id=EXECUTION,
            revision_scope='Executable terminal relation under original identity; original algebraic derivation remains immutable provenance.')
        digest,nodes,edges=encode_execution_graph(graph)
        version=uuid5(UUID(ARTIFACT),'corrected-execution:'+digest)
        bindings=db.execute('SELECT bound_artifact_fqdn,bound_version_content_hash,status FROM artifact_cdg_bindings WHERE version_id=%s',(EXECUTION,)).fetchall()
        require(len(bindings)==1 and bindings[0]['status']=='active' and bindings[0]['bound_artifact_fqdn']==graph.nodes[0].matched_primitive,'Parent binding differs')
        binding=bindings[0]
        providers=db.execute("SELECT a.artifact_id,v.version_id,a.fqdn,l.import_module,l.source_symbol FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN atoms l ON l.atom_id=a.artifact_id JOIN atom_versions lv ON lv.version_id=v.version_id WHERE a.fqdn=%s AND v.content_hash=%s AND lv.content_hash=v.content_hash AND a.status='approved' AND a.is_publishable AND l.status='approved' AND l.is_publishable AND v.is_latest AND lv.is_latest AND v.trust_tier=3 AND lv.trust_tier=3 FOR SHARE OF a,v,l,lv",(binding['bound_artifact_fqdn'],binding['bound_version_content_hash'])).fetchall()
        require(len(providers)==1 and providers[0]['import_module']+'.'+providers[0]['source_symbol']==graph.nodes[0].matched_primitive,'Exact approved runtime provider required')
        def ensure(table,keys,row):
            nonlocal created
            created+=int(ensure_row(db,table,keys,row))
            if fail_after_writes is not None and created==fail_after_writes:raise RuntimeError('injected revision staging failure')
        ensure('artifact_versions',dict(version_id=version),dict(version_id=version,artifact_id=UUID(ARTIFACT),
            content_hash=digest,semver='0.0.0+execution.'+digest[:12],is_latest=False,trust_tier=3,derives_from=UUID(REPLAY)))
        for node in nodes:
            row=dict(version_id=version,**node)
            ensure('artifact_cdg_nodes',dict(version_id=version,node_id=node['node_id']),row)
        for direction,ports in [('input',graph.nodes[0].inputs),('output',graph.nodes[0].outputs)]:
            for ordinal,port in enumerate(ports):
                row=dict(artifact_id=UUID(ARTIFACT),version_id=version,direction=direction,ordinal=ordinal,
                    **{key:getattr(port,key) for key in ['name','type_desc','constraints','required','default_value_repr','dim_signature']})
                ensure('artifact_io_specs',dict(version_id=version,direction=direction,name=port.name),row)
        ensure('artifact_cdg_bindings',dict(version_id=version,node_id='frequency'),dict(version_id=version,node_id='frequency',
            **binding,binding_confidence=1.0,binding_source=RUNNER,alternatives=Jsonb([]),
            evidence_summary=Jsonb(dict(provider_version_id=str(providers[0]['version_id']),scope_audit_sha256=sha(ROOT/'docs/reviews/physics_period_frequency_original_scope.json')))))
        for selected in [ORIGINAL,REPLAY,EXECUTION]:
            target=db.execute('SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(selected,)).fetchone()
            keys=dict(dependent_version_id=version,dependency_artifact_fqdn=target['fqdn'],dependency_content_hash=target['content_hash'],port_name='')
            ensure('artifact_dependencies',keys,dict(**keys,dependency_role='cdg',optional=False,
                binding_metadata=Jsonb(dict(source_version_id=selected,scope='Immutable source proof and qualified numerical provenance; runtime uses direct provider binding.'))))
        evidence=uuid5(version,RUNNER)
        ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=UUID(ARTIFACT),version_id=version,
            audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
            details=Jsonb(dict(scope_audit=scope,original_history_sha256=original_history,stager_sha256=sha(__file__),approved=False))))
        current=db.execute('SELECT get_artifact_document(%s) AS d',(state['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(current,version_id=str(version),content_hash=digest,require_execution_envelope=True)==graph,'Stored corrected graph differs')
        require(history(db)==original_history,'Original history changed')
        require([str(r['version_id']) for r in db.execute('SELECT version_id FROM artifact_versions WHERE artifact_id=%s AND is_latest',(ARTIFACT,))]==[ORIGINAL],'Staging changed selection')
        require(not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(ARTIFACT,)).fetchone(),'Draft unexpectedly served')
        if not apply:db.rollback()
    return dict(applied=apply,approved=False,rows_created=created,artifact_id=ARTIFACT,version_id=str(version),
        graph_sha256=digest,reused_atoms=1,new_atoms=0,original_history_sha256=original_history,catalog_roundtrip=True,
        stager_sha256=sha(__file__),scope_audit_sha256=sha(ROOT/'docs/reviews/physics_period_frequency_original_scope.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();report=stage(args.source_directory,args.apply)
    if args.apply:
        name='period_frequency_original_revision_repeat.json' if report['rows_created']==0 else 'period_frequency_original_revision_import.json'
        (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
