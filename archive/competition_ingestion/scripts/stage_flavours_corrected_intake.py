"""Stage the complete decomposed original CDG with immutable source lineage."""
import argparse
import hashlib
import json
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb

from sciona.flavours_decomposed_evaluation import build_evaluated_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.competition_import import _canonical
from scripts.import_flavours_provider_drafts import ROOT,connect
from scripts.import_residual_execution_drafts import ensure_row
from scripts.stage_remaining_legacy_physics_revisions import history
from scripts.audit_flavours_original_intake_scope import ORIGINAL,ORIGINAL_HASH,PARENT
from scripts.verify_flavours_provider_publication import verify as verify_providers
from scripts import promote_flavours_providers as provider_publication
from scripts.plan_flavours_provider_intake import sha

IDENTITY='b9a53c31-7059-5909-8d31-970abf965fcf'
FQDN='cdg.competition.solution.kaggle.flavours_physics_1st'
PLAN=ROOT/'docs/reviews/flavours_corrected_intake_plan.json'
RUNNER='flavours-corrected-intake-draft.v1'
IO_FIELDS=('name','type_desc','constraints','required','default_value_repr','dim_signature')


def build_graph():
    graph=build_evaluated_graph()
    producers={e.source_id for e in graph.edges}
    output_sources={('result' if n.node_id=='report' else n.node_id+'__'+p.name):
        dict(node_id=n.node_id,output_name=p.name) for n in graph.nodes if n.node_id not in producers for p in n.outputs}
    graph.metadata.update(original_source_version_id=ORIGINAL,original_source_content_hash=ORIGINAL_HASH,
        qualified_reference_version_id=PARENT,catalog_output_sources=output_sources,
        artifact_source='corrected_original_competition_execution')
    return graph


def boundaries(graph):
    consumed={(e.target_id,e.input_name) for e in graph.edges};inputs={}
    for node in graph.nodes:
        for port in node.inputs:
            if (node.node_id,port.name) in consumed:continue
            value=port.model_dump()
            if port.name in inputs and inputs[port.name]!=value:raise ValueError('Repeated root input declarations differ')
            inputs[port.name]=value
    by_node={n.node_id:n for n in graph.nodes};outputs=[]
    for alias,source in graph.metadata['catalog_output_sources'].items():
        port=next(p for p in by_node[source['node_id']].outputs if p.name==source['output_name']).model_dump()
        port.update(name=alias,constraints='Graph output '+source['node_id']+'/'+source['output_name']+'. Diagnostic loss outputs never feed training.')
        outputs.append(port)
    if set(inputs)!={'training_inputs','labels','mass','excluded_column','query_inputs','controls'} or len(outputs)!=51:
        raise ValueError('Complete root boundary differs')
    return list(inputs.values()),outputs


def check_anchor(db,plan):
    anchor=db.execute('SELECT a.artifact_id::text,a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(ORIGINAL,)).fetchone()
    if anchor!=dict(artifact_id=IDENTITY,fqdn=FQDN,content_hash=ORIGINAL_HASH):raise ValueError('Original identity differs')
    evidence=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(ORIGINAL,)).fetchall()
    if len(evidence)!=1 or hashlib.sha256(_canonical(evidence[0]['details']['snapshot']).encode()).hexdigest()!=ORIGINAL_HASH:
        raise ValueError('Original snapshot differs')
    for version,key in [(ORIGINAL,'original_history'),(PARENT,'parent_history')]:
        if history(db,version)!=plan[key]:raise ValueError('Preserved source history differs')


def make_plan():
    verification=verify_providers()
    providers=json.loads(provider_publication.PLAN.read_text())
    graph=build_graph();digest,nodes,edges=encode_execution_graph(graph)
    inputs,outputs=boundaries(graph)
    version=str(uuid5(UUID(IDENTITY),'corrected-complete-execution:'+digest))
    by_runtime={r['runtime']:r for r in providers['records']}
    bindings=[dict(node_id=n.node_id,bound_artifact_fqdn=by_runtime[n.matched_primitive]['catalog_fqdn'],
        bound_version_content_hash=by_runtime[n.matched_primitive]['content_hash'],status='active',
        evidence_summary=dict(runtime_fqdn=n.matched_primitive,provider_version_id=by_runtime[n.matched_primitive]['version_id'],
            output_aliases_by_ordinal=[p.name for p in n.outputs])) for n in graph.nodes]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        parent=db.execute('SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(PARENT,)).fetchone()
        original_history=history(db,ORIGINAL);parent_history=history(db,PARENT)
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(IDENTITY,)).fetchone()
        if state!=dict(status='draft',is_publishable=False):raise ValueError('Original must be draft')
    dependencies=[dict(dependency_artifact_fqdn=FQDN,dependency_content_hash=ORIGINAL_HASH,dependency_role='cdg',
                      optional=False,port_name='',binding_metadata=dict(source_version_id=ORIGINAL,scope='Immutable original intake provenance.')),
                  dict(dependency_artifact_fqdn=parent['fqdn'],dependency_content_hash=parent['content_hash'],dependency_role='cdg',
                      optional=False,port_name='',binding_metadata=dict(source_version_id=PARENT,scope='Preserved executable reference provenance; computation uses explicit atom bindings.'))]
    for runtime in sorted({n.matched_primitive for n in graph.nodes}):
        record=by_runtime[runtime]
        dependencies.append(dict(dependency_artifact_fqdn=record['catalog_fqdn'],dependency_content_hash=record['content_hash'],
            dependency_role='logic_atom',optional=False,port_name='',binding_metadata=dict(runtime_fqdn=runtime,provider_version_id=record['version_id'])))
    return dict(passed=True,approved=False,artifact_id=IDENTITY,fqdn=FQDN,version_id=version,graph_sha256=digest,
        original_version_id=ORIGINAL,original_content_hash=ORIGINAL_HASH,original_history=original_history,parent_history=parent_history,
        nodes=len(nodes),edges=len(edges),inputs=inputs,outputs=outputs,bindings=bindings,dependencies=dependencies,
        provider_publication_plan_sha256=sha(provider_publication.PLAN),provider_verification=verification,
        builder_sha256=sha(__file__),qualified_base_execution_sha256=sha(ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json'))


def qualified_plan():
    plan=json.loads(PLAN.read_text())
    provider_plan=provider_publication.qualify_plan()
    if sha(provider_publication.PLAN)!=plan['provider_publication_plan_sha256'] or sha(__file__)!=plan['builder_sha256']:
        raise ValueError('Qualified plan or builder changed')
    if sha(ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json')!=plan['qualified_base_execution_sha256']:
        raise ValueError('Base execution evidence changed')
    graph=build_graph();digest,nodes,edges=encode_execution_graph(graph)
    if (digest,len(nodes),len(edges))!=(plan['graph_sha256'],841,3113) or boundaries(graph)!=(plan['inputs'],plan['outputs']):
        raise ValueError('Complete corrected graph differs')
    return plan,provider_plan


def check_candidate(db,plan,provider_plan,approved=False):
    check_anchor(db,plan);provider_publication.check(db,provider_plan,published=True)
    state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(IDENTITY,)).fetchone()
    if state!=dict(status='approved' if approved else 'draft',is_publishable=approved):raise ValueError('Original publication state differs')
    version=db.execute('SELECT artifact_id::text,content_hash,is_latest,trust_tier,derives_from::text FROM artifact_versions WHERE version_id=%s',(plan['version_id'],)).fetchone()
    if version!=dict(artifact_id=IDENTITY,content_hash=plan['graph_sha256'],is_latest=approved,trust_tier=3,derives_from=ORIGINAL):raise ValueError('Corrected version differs')
    latest=db.execute('SELECT version_id::text FROM artifact_versions WHERE artifact_id=%s AND is_latest',(IDENTITY,)).fetchall()
    if latest!=[{'version_id':plan['version_id'] if approved else ORIGINAL}]:raise ValueError('Latest selection differs')
    bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(plan['version_id'],)).fetchall()
    if bindings!=sorted(plan['bindings'],key=lambda b:b['node_id']):raise ValueError('Exact graph bindings differ')
    dependencies=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional,port_name,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn',(plan['version_id'],)).fetchall()
    if dependencies!=sorted(plan['dependencies'],key=lambda d:d['dependency_artifact_fqdn']):raise ValueError('Exact graph dependencies differ')
    expected=[]
    for direction in ('input','output'):
        for ordinal,port in enumerate(plan[direction+'s']):expected.append(dict(direction=direction,ordinal=ordinal,**{k:port[k] for k in IO_FIELDS}))
    actual=db.execute('SELECT direction,ordinal,'+','.join(IO_FIELDS)+' FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(plan['version_id'],)).fetchall()
    if actual!=expected:raise ValueError('Graph root contracts differ')
    if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(plan['version_id'],)).fetchone():raise ValueError('Failed candidate audit')
    evidence=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s AND passed AND status=%s',(uuid5(UUID(plan['version_id']),RUNNER),'completed')).fetchone()
    if not evidence or evidence['details']!=dict(plan_sha256=sha(PLAN),scope='Complete decomposed candidate; not publication approval.'):
        raise ValueError('Staging evidence differs')
    document=db.execute('SELECT get_artifact_document(%s) AS d',(FQDN,)).fetchone()['d']
    graph=_artifact_document_to_cdg(document,version_id=plan['version_id'],content_hash=plan['graph_sha256'],require_execution_envelope=True)
    if graph!=build_graph():raise ValueError('Stored graph differs')
    served=db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(IDENTITY,)).fetchone()
    if bool(served)!=approved:raise ValueError('Serving state differs')
    return graph


class InjectedFailure(RuntimeError):pass


def stage(apply=False,fail_after=None):
    plan,provider_plan=qualified_plan()
    if apply:
        gates=json.loads((ROOT/'docs/reviews/flavours_corrected_intake_transaction.json').read_text())
        if not gates['passed'] or gates['importer_sha256']!=sha(__file__) or gates['plan_sha256']!=sha(PLAN):raise ValueError('Matching staging gates required')
    identity,version=UUID(IDENTITY),UUID(plan['version_id']);writes=0
    digest,nodes,edges=encode_execution_graph(build_graph())
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        db.execute('SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE',(identity,)).fetchone()
        check_anchor(db,plan);provider_publication.check(db,provider_plan,published=True)
        def ensure(table,keys,row):
            nonlocal writes
            writes+=int(ensure_row(db,table,keys,row))
            if writes==fail_after:raise InjectedFailure('Injected corrected intake failure')
        ensure('artifact_versions',{'version_id':version},dict(version_id=version,artifact_id=identity,content_hash=digest,
            semver='0.1.0+corrected.'+digest[:12],is_latest=False,trust_tier=3,derives_from=UUID(ORIGINAL)))
        for table,rows,keys in [('artifact_cdg_nodes',nodes,['node_id']),('artifact_cdg_edges',edges,['source_id','target_id','output_name','input_name'])]:
            for item in rows:
                row=dict(version_id=version,**item);ensure(table,{k:row[k] for k in ['version_id',*keys]},row)
        for direction in ('input','output'):
            for ordinal,port in enumerate(plan[direction+'s']):
                if any(port.get(k) for k in ('data_kind','time_basis','provenance')):raise ValueError('Unsupported populated root metadata')
                row=dict(artifact_id=identity,version_id=version,direction=direction,ordinal=ordinal,**{k:port[k] for k in IO_FIELDS})
                ensure('artifact_io_specs',{k:row[k] for k in ('artifact_id','version_id','direction','name')},row)
        for binding in plan['bindings']:
            row=dict(binding,version_id=version,binding_confidence=1.,binding_source=RUNNER,
                     alternatives=Jsonb([]),evidence_summary=Jsonb(binding['evidence_summary']))
            ensure('artifact_cdg_bindings',dict(version_id=version,node_id=binding['node_id']),row)
        for dependency in plan['dependencies']:
            row=dict(dependency,dependent_version_id=version,binding_metadata=Jsonb(dependency['binding_metadata']))
            ensure('artifact_dependencies',{k:row[k] for k in ('dependent_version_id','dependency_artifact_fqdn','dependency_content_hash','port_name')},row)
        evidence=uuid5(version,RUNNER)
        ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=identity,version_id=version,
            audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
            details=Jsonb(dict(plan_sha256=sha(PLAN),scope='Complete decomposed candidate; not publication approval.'))))
        check_candidate(db,plan,provider_plan)
        if fail_after is not None:raise ValueError('Fault point not reached')
        if not apply:db.rollback()
    return dict(applied=apply,approved=False,rows_created=writes,version_id=str(version),graph_sha256=digest,
        nodes=len(nodes),edges=len(edges),reused_providers=39,new_providers=0,root_inputs=6,root_outputs=51,
        source_histories_preserved=True,stored_roundtrip_verified=True)


def validate_transaction():
    plan,_=qualified_plan()
    def unchanged():
        with connect() as db:
            check_anchor(db,plan)
            if db.execute('SELECT 1 FROM artifact_versions WHERE version_id=%s',(plan['version_id'],)).fetchone():raise ValueError('Candidate survived rollback')
            state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(IDENTITY,)).fetchone()
            if state!=dict(status='draft',is_publishable=False):raise ValueError('Publication state changed')
    unchanged();dry=stage();unchanged();points=[3,dry['rows_created']//2,dry['rows_created']]
    for point in points:
        try:stage(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected injected rollback')
        unchanged()
    return dict(passed=True,catalog_mutations=0,dry_run=dry,injected_failure_points=points,
        all_rollbacks_complete=True,importer_sha256=sha(__file__),plan_sha256=sha(PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','validate','apply','repeat'])
    action=parser.parse_args().action
    if action=='plan':result=make_plan();path=PLAN
    elif action=='validate':result=validate_transaction();path=ROOT/'docs/reviews/flavours_corrected_intake_transaction.json'
    else:
        result=stage(apply=True)
        if action=='repeat' and result['rows_created']:raise ValueError('Repeat changed rows')
        path=ROOT/('docs/reviews/flavours_corrected_intake_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(planned=True,nodes=result['nodes'],edges=result['edges']) if action=='plan' else result))
