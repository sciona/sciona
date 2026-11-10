"""Stage the qualified full Cornell ensemble on its original identity, preserving history."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from uuid import UUID, uuid5
from psycopg.types.json import Jsonb
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.import_residual_execution_drafts import ensure_row
from scripts.stage_remaining_legacy_physics_revisions import history
from scripts import reconcile_cornell_metadata as metadata
from scripts.audit_cornell_original_readiness import sha
from sciona.catalog_cornell_graph import build_catalog_cornell_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.competition_import import _canonical

IDENTITY='cce1e54a-66f6-56ee-8819-44594773acf0'
FQDN='cdg.competition.solution.kaggle.cornell_birdcall_1st'
ORIGINAL='5326bafb-d26f-5202-85c9-6f903df0fc77'
PARENT='20e9c982-61ed-5080-a9cb-87cf637f3537'
RUNNER='cornell-corrected-intake-draft.v1'
PLAN=ROOT/'docs/reviews/cornell_corrected_intake_plan.json'
GATES=ROOT/'docs/reviews/cornell_corrected_intake_gates.json'


def sources():
    from scripts.review_cornell_original_scope import STAGES, anchor
    path='docs/reviews/cornell_original_scope.json'
    raw=subprocess.check_output(['git','show','1b50a8e6:'+path],cwd=ROOT)
    if raw!=(ROOT/path).read_bytes():
        raise ValueError('Immutable five-stage source qualification changed')
    scope=json.loads(raw)
    if not scope['passed'] or {r['stage_id'] for r in scope['source_stage_review']}!=set(STAGES):
        raise ValueError('Complete source review required')
    for item in scope['source_stage_review']:
        stage=STAGES[item['stage_id']]
        if item['source_anchors']!=[anchor(*a) for a in stage['anchors']] or any(item[k]!=stage[k] for k in ('realization','limits')):
            raise ValueError('Reviewed implementation changed')
    for path,key in [('docs/reviews/cornell_provider_contracts.json','provider_contracts_sha256'),
                     ('docs/reviews/cornell_current_graph.json','current_execution_sha256'),
                     ('docs/reviews/competition_cornell_semantic_review.json','source_review_sha256')]:
        if sha(ROOT/path)!=scope[key]:
            raise ValueError('Source review evidence changed')
    metadata.qualified()
    graph=build_catalog_cornell_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    from sciona.services.execution_graph_codec import decode_execution_graph
    if decode_execution_graph(nodes,edges,digest)!=graph or (len(nodes),len(edges))!=(2,1):
        raise ValueError('Complete corrected graph roundtrip differs')
    return graph,digest,nodes,edges


def provider_state(db):
    _,targets,_=metadata.qualified()
    for target in targets:
        metadata.check_identity(db,target)
        r=target['identity']
        for prefix,key in [('artifact','artifact_id'),('atom','atom_id')]:
            if prefix not in target['catalogs']:
                continue
            if not db.execute(f'SELECT {prefix}_is_publishable(%s) AS yes',(r['artifact_id'],)).fetchone()['yes']:
                raise ValueError('Native eligibility missing')
            if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s',(r['artifact_id'],)).fetchone():
                raise ValueError('Retained provider or derivative not served')
            ports=db.execute(f'SELECT direction,name,type_desc,ordinal,required,default_value_repr,constraints FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(r['version_id'],)).fetchall()
            if ports!=target['contract']['ports']:
                raise ValueError('Exact catalog contracts differ')
    frozen=metadata.snapshot(db,targets)
    return hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()


def make_plan():
    graph,digest,nodes,edges=sources()
    semantic,targets,original=metadata.qualified()
    providers={r['runtime_fqdn']:r for r in semantic['providers']}
    version=str(uuid5(UUID(IDENTITY),'corrected-complete-execution:'+digest))
    bindings=[dict(node_id=n.node_id,bound_artifact_fqdn=n.matched_primitive,
        bound_version_content_hash=providers[n.matched_primitive]['content_hash'],status='active',
        evidence_summary=dict(runtime_fqdn=n.matched_primitive,provider_version_id=providers[n.matched_primitive]['version_id'],
                              output_aliases_by_ordinal=[p.name for p in n.outputs])) for n in graph.nodes]
    dependencies=[dict(dependency_artifact_fqdn=r['runtime_fqdn'],dependency_content_hash=r['content_hash'],
        dependency_role='logic_atom',optional=False,port_name='',binding_metadata=dict(provider_version_id=r['version_id'])) for r in semantic['providers']]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(IDENTITY,)).fetchone()!=dict(status='draft',is_publishable=False):
            raise ValueError('Original must remain draft')
        for v in (ORIGINAL,PARENT):
            row=db.execute('SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(v,)).fetchone()
            expected=original['identity'] if v==ORIGINAL else next(t['identity'] for t in targets if t['identity']['version_id']==PARENT)
            if row!={k:expected[k] for k in ('fqdn','content_hash')}:
                raise ValueError('Reviewed source lineage differs')
            dependencies.append(dict(dependency_artifact_fqdn=row['fqdn'],dependency_content_hash=row['content_hash'],
                dependency_role='cdg',optional=False,port_name='',binding_metadata=dict(source_version_id=v,scope='Immutable source lineage; computation uses approved atom bindings.')))
        intake=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(ORIGINAL,)).fetchall()
        if len(intake)!=1 or hashlib.sha256(_canonical(intake[0]['details']['snapshot']).encode()).hexdigest()!=original['identity']['content_hash']:
            raise ValueError('Original stored snapshot differs')
        return dict(passed=True,approved=False,artifact_id=IDENTITY,fqdn=FQDN,version_id=version,graph_sha256=digest,
            bindings=bindings,dependencies=dependencies,original_history=history(db,ORIGINAL),parent_history=history(db,PARENT),
            provider_state=provider_state(db),builder_sha256=sha(Path(__file__)),
            graph_builder_sha256=sha(ROOT/'sciona/catalog_cornell_graph.py'),nodes=len(nodes),edges=len(edges),
            ports=semantic['derivative']['ports'])


def qualified_plan():
    plan=json.loads(PLAN.read_text())
    _,digest,_,_=sources()
    if (plan['builder_sha256']!=sha(Path(__file__)) or plan['graph_sha256']!=digest
            or plan['graph_builder_sha256']!=sha(ROOT/'sciona/catalog_cornell_graph.py')):
        raise ValueError('Staging plan changed')
    return plan


def check(db,plan,approved=False):
    if provider_state(db)!=plan['provider_state']:
        raise ValueError('Frozen retained provider metadata changed')
    if history(db,ORIGINAL)!=plan['original_history'] or history(db,PARENT)!=plan['parent_history']:
        raise ValueError('Original or derivative history changed')
    state=db.execute('SELECT fqdn,status,is_publishable FROM artifacts WHERE artifact_id=%s',(IDENTITY,)).fetchone()
    if state!=dict(fqdn=FQDN,status='approved' if approved else 'draft',is_publishable=approved):
        raise ValueError('Original publication state differs')
    latest=db.execute('SELECT version_id::text FROM artifact_versions WHERE artifact_id=%s AND is_latest',(IDENTITY,)).fetchall()
    if latest!=[dict(version_id=plan['version_id'] if approved else ORIGINAL)]:
        raise ValueError('Latest selection differs')
    candidate=db.execute('SELECT artifact_id::text,content_hash,trust_tier,is_latest,derives_from::text FROM artifact_versions WHERE version_id=%s',(plan['version_id'],)).fetchone()
    if candidate!=dict(artifact_id=IDENTITY,content_hash=plan['graph_sha256'],trust_tier=3,is_latest=approved,derives_from=ORIGINAL):
        raise ValueError('Candidate identity differs')
    for table,fields,expected,key in [
        ('artifact_cdg_bindings','node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary',plan['bindings'],'version_id'),
        ('artifact_dependencies','dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional,port_name,binding_metadata',plan['dependencies'],'dependent_version_id'),
        ('artifact_io_specs','direction,name,type_desc,ordinal,required,default_value_repr,constraints',plan['ports'],'version_id')]:
        rows=db.execute(f'SELECT {fields} FROM {table} WHERE {key}=%s',(plan['version_id'],)).fetchall()
        normalize=lambda x: sorted(json.dumps(r,sort_keys=True) for r in x)
        if normalize(rows)!=normalize(expected):
            raise ValueError('Exact candidate rows differ: '+table)
    if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(plan['version_id'],)).fetchone():
        raise ValueError('Failed candidate evidence')
    document=db.execute('SELECT get_artifact_document(%s) AS d',(FQDN,)).fetchone()['d']
    graph=_artifact_document_to_cdg(document,version_id=plan['version_id'],content_hash=plan['graph_sha256'],require_execution_envelope=True)
    if graph!=build_catalog_cornell_graph():
        raise ValueError('Stored graph differs')
    if bool(db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(IDENTITY,)).fetchone())!=approved:
        raise ValueError('Serving state differs')
    return graph


class InjectedFailure(RuntimeError):
    pass


def stage(apply=False,fail_after=None):
    plan=qualified_plan()
    if apply:
        gates=json.loads(GATES.read_text())
        if gates.get('passed') is not True or gates['builder_sha256']!=sha(Path(__file__)) or gates['plan_sha256']!=sha(PLAN):
            raise ValueError('Current staging gates required')
    _,digest,nodes,edges=sources()
    version=UUID(plan['version_id']);writes=0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        db.execute('SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE',(IDENTITY,)).fetchone()
        def ensure(table,keys,row):
            nonlocal writes
            writes+=int(ensure_row(db,table,keys,row))
            if fail_after is not None and writes>=fail_after:
                raise InjectedFailure('Injected staging failure')
        ensure('artifact_versions',dict(version_id=version),dict(version_id=version,artifact_id=UUID(IDENTITY),content_hash=digest,
            semver='0.1.0+corrected.'+digest[:12],trust_tier=3,is_latest=False,derives_from=UUID(ORIGINAL)))
        for table,rows,keys in [('artifact_cdg_nodes',nodes,['node_id']),('artifact_cdg_edges',edges,['source_id','target_id','output_name','input_name'])]:
            for item in rows:
                row=dict(version_id=version,**item)
                ensure(table,{k:row[k] for k in ['version_id',*keys]},row)
        for port in plan['ports']:
            row=dict(artifact_id=UUID(IDENTITY),version_id=version,**port)
            ensure('artifact_io_specs',{k:row[k] for k in ('artifact_id','version_id','direction','name')},row)
        for item in plan['bindings']:
            row=dict(item,version_id=version,binding_confidence=1.,binding_source=RUNNER,alternatives=Jsonb([]),evidence_summary=Jsonb(item['evidence_summary']))
            ensure('artifact_cdg_bindings',dict(version_id=version,node_id=item['node_id']),row)
        for item in plan['dependencies']:
            row=dict(item,dependent_version_id=version,binding_metadata=Jsonb(item['binding_metadata']))
            ensure('artifact_dependencies',{k:row[k] for k in ('dependent_version_id','dependency_artifact_fqdn','dependency_content_hash','port_name')},row)
        check(db,plan)
        if fail_after is not None:
            raise ValueError('Fault point not reached')
        if not apply:
            db.rollback()
    return dict(applied=apply,approved=False,writes=writes,version_id=str(version),graph_sha256=digest,source_histories_preserved=True)


def validate():
    plan=qualified_plan()
    def absent():
        with connect() as db:
            if db.execute('SELECT 1 FROM artifact_versions WHERE version_id=%s',(plan['version_id'],)).fetchone():
                raise ValueError('Candidate survived rollback')
            if history(db,ORIGINAL)!=plan['original_history'] or history(db,PARENT)!=plan['parent_history'] or provider_state(db)!=plan['provider_state']:
                raise ValueError('Preserved state changed')
    absent();dry=stage();absent()
    points=[1,dry['writes']//2,dry['writes']]
    for point in points:
        try:stage(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected injected failure')
        absent()
    return dict(passed=True,dry_run=dry,rollback_points=points,catalog_mutations=0,builder_sha256=sha(Path(__file__)),plan_sha256=sha(PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','validate','apply','repeat'])
    action=parser.parse_args().action
    result=make_plan() if action=='plan' else validate() if action=='validate' else stage(apply=True)
    if action=='repeat' and result['writes']:raise ValueError('Repeat changed rows')
    path=PLAN if action=='plan' else GATES if action=='validate' else ROOT/('docs/reviews/cornell_corrected_intake_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(planned=True,nodes=result['nodes'],edges=result['edges']) if action=='plan' else result))
