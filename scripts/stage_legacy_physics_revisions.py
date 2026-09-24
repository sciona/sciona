"""Stage five legacy identities using qualified execution graphs, atomically."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
from uuid import UUID,uuid5
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from scripts.audit_legacy_physics_execution_scope import ROOT,audit,require
from scripts.import_residual_execution_drafts import ensure_row
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg

RUNNER='legacy-physics-execution-draft.v1'
SCOPE='Executable realization of complete legacy source scope; approved family corrections and interpretation limits apply; legacy split-step graph remains immutable provenance.'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history(db,version):
    result={}
    for table in ['artifact_versions','artifact_cdg_nodes','artifact_cdg_edges','artifact_cdg_bindings','artifact_io_specs','artifact_audit_evidence','artifact_validity_bounds','artifact_dependencies']:
        key='dependent_version_id' if table=='artifact_dependencies' else 'version_id'
        rows=db.execute(f"SELECT to_jsonb(t)-'updated_at'-'is_latest' AS row FROM {table} t WHERE {key}=%s ORDER BY (to_jsonb(t)-'updated_at'-'is_latest')::text",(version,)).fetchall()
        result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    return result


def revised_graph(graph,item):
    graph=graph.model_copy(deep=True)
    graph.metadata.update(legacy_source_artifact_id=item['legacy_artifact_id'],legacy_source_version_id=item['legacy_version_id'],
        projected_source_version_id=item['projected_source_version_id'],qualified_family_execution_version_id=item['approved_execution_version_id'],legacy_revision_scope=SCOPE)
    return graph


def stage(source,apply=False,fail_after_writes=None):
    if apply:
        gates=json.loads((ROOT/'docs/reviews/legacy_physics_revision_transaction.json').read_text())
        require(gates['passed'] and gates['rollback_verified'] and gates['injected_failure_rollback_verified']
            and gates['stager_sha256']==sha(__file__) and gates['validator_sha256']==sha(ROOT/'scripts/validate_legacy_physics_revision_transaction.py'),'Current transaction qualification required')
    scope=audit(source)
    require(scope==json.loads((ROOT/'docs/reviews/legacy_physics_execution_scope.json').read_text()),'Qualified legacy scopes changed')
    created=0;reports=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        def ensure(table,keys,row):
            nonlocal created
            created+=int(ensure_row(db,table,keys,row))
            if fail_after_writes is not None and created==fail_after_writes:raise RuntimeError('injected legacy staging failure')
        for item in sorted(scope['graphs'],key=lambda r:r['legacy_artifact_id']):
            prior=created;artifact=UUID(item['legacy_artifact_id']);original=item['legacy_version_id']
            state=db.execute('SELECT fqdn,status,is_publishable FROM artifacts WHERE artifact_id=%s FOR UPDATE',(artifact,)).fetchone()
            require(state and state['status']=='draft' and not state['is_publishable'],'Legacy identity must remain draft')
            before=history(db,original)
            publisher=importlib.import_module('scripts.promote_'+item['family']+'_revision')
            parent,qualified=publisher.check_publication(db,item['qualification'])
            require(qualified['version_id']==item['approved_execution_version_id'] and qualified['graph_sha256']==item['approved_execution_graph_sha256'],'Approved family version changed')
            graph=revised_graph(parent,item);digest,nodes,edges=encode_execution_graph(graph)
            require(len(nodes)==1 and not edges,'Expected qualified single-provider graph')
            version=uuid5(artifact,'legacy-execution:'+digest)
            bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s',(qualified['version_id'],)).fetchall()
            require(len(bindings)==1 and bindings[0]['status']=='active' and bindings[0]['bound_artifact_fqdn']==graph.nodes[0].matched_primitive,'Qualified binding differs')
            binding=bindings[0];node_id=binding.pop('node_id');parent_evidence=binding.pop('evidence_summary')
            ensure('artifact_versions',dict(version_id=version),dict(version_id=version,artifact_id=artifact,content_hash=digest,
                semver='0.0.0+execution.'+digest[:12],is_latest=False,trust_tier=3,derives_from=UUID(original)))
            for node in nodes:ensure('artifact_cdg_nodes',dict(version_id=version,node_id=node['node_id']),dict(version_id=version,**node))
            for direction,ports in [('input',graph.nodes[0].inputs),('output',graph.nodes[0].outputs)]:
                for ordinal,port in enumerate(ports):
                    row=dict(artifact_id=artifact,version_id=version,direction=direction,ordinal=ordinal,
                        **{k:getattr(port,k) for k in ['name','type_desc','constraints','required','default_value_repr','dim_signature']})
                    ensure('artifact_io_specs',dict(version_id=version,direction=direction,name=port.name),row)
            ensure('artifact_cdg_bindings',dict(version_id=version,node_id=node_id),dict(version_id=version,node_id=node_id,**binding,
                binding_confidence=1.,binding_source=RUNNER,alternatives=Jsonb([]),evidence_summary=Jsonb(dict(
                    qualified_parent_evidence=parent_evidence,qualified_parent_version_id=qualified['version_id'],legacy_scope_sha256=sha(ROOT/'docs/reviews/legacy_physics_execution_scope.json')))))
            for selected in [original,item['projected_source_version_id'],qualified['version_id']]:
                target=db.execute('SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(selected,)).fetchone()
                keys=dict(dependent_version_id=version,dependency_artifact_fqdn=target['fqdn'],dependency_content_hash=target['content_hash'],port_name='')
                ensure('artifact_dependencies',keys,dict(**keys,dependency_role='cdg',optional=False,
                    binding_metadata=Jsonb(dict(source_version_id=selected,scope='Immutable legacy source, complete source-step projection and approved execution-family provenance; runtime uses direct provider binding.'))))
            evidence=uuid5(version,RUNNER)
            ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=artifact,version_id=version,
                audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
                details=Jsonb(dict(scope=item,original_history_sha256=before,stager_sha256=sha(__file__),approved=False))))
            document=db.execute('SELECT get_artifact_document(%s) AS d',(state['fqdn'],)).fetchone()['d']
            require(_artifact_document_to_cdg(document,version_id=str(version),content_hash=digest,require_execution_envelope=True)==graph,'Stored legacy execution differs')
            require(history(db,original)==before,'Legacy history changed')
            require([r['version_id'] for r in db.execute('SELECT version_id::text FROM artifact_versions WHERE artifact_id=%s AND is_latest',(artifact,))]==[original],'Staging changed latest selection')
            require(not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(artifact,)).fetchone(),'Legacy draft unexpectedly served')
            reports.append(dict(family=item['family'],artifact_id=str(artifact),version_id=str(version),graph_sha256=digest,
                original_version_id=original,original_history_sha256=before,qualified_family_version_id=qualified['version_id'],
                reused_atoms=1,new_atoms=0,rows_created=created-prior,catalog_roundtrip=True))
        if not apply:db.rollback()
    return dict(applied=apply,approved=False,rows_created=created,graphs=reports,stager_sha256=sha(__file__),
        scope_audit_sha256=sha(ROOT/'docs/reviews/legacy_physics_execution_scope.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();report=stage(args.source_directory,args.apply)
    if args.apply:
        name='legacy_physics_revision_repeat.json' if report['rows_created']==0 else 'legacy_physics_revision_import.json'
        (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(applied=args.apply,approved=False,graphs=len(report['graphs']),rows_created=report['rows_created'])))
