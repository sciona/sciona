"""Stage the complete decomposed original CDG with immutable source lineage."""
import argparse
import hashlib
import json
import subprocess
from uuid import UUID,uuid5
from psycopg.types.json import Jsonb

from sciona.decomposed_vsb_graph import build_decomposed_vsb_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.architect.models import EdgeKind
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.competition_import import _canonical
from scripts.import_flavours_provider_drafts import ROOT,connect
from scripts.import_residual_execution_drafts import ensure_row
from scripts.stage_remaining_legacy_physics_revisions import history
ORIGINAL='d79e40e4-45b0-5e74-a83c-c68c91f12afb'
ORIGINAL_HASH='565e511ee8d20ce0b32adf5c8cc726150f235b1f97d5502d62c96cfe339f5ac1'
PARENT='feab5779-925c-5fe0-850b-c865942833ba'
from scripts import decomposed_vsb_approved_provider_state as provider_publication
from scripts.plan_flavours_provider_intake import sha

IDENTITY='905df5de-530b-5394-b8e0-a355c8f1c3ce'
FQDN='cdg.competition.solution.kaggle.vsb_power_line_1st'
PLAN=ROOT/'docs/reviews/vsb_corrected_intake_plan.json'
RUNNER='vsb-corrected-intake-draft.v1'
IO_FIELDS=('name','type_desc','constraints','required','default_value_repr','dim_signature')
CORRECTIONS={'dl_model': 'Remove unsupported deep branch; all 125 models are LightGBM.',
 'peak_detection': 'Bidirectional directional scan intersection, sorted amplitude gradient and cumulative '
                   'threshold knee with source negative slicing.',
 'peak_feature_extraction': 'Four normalized waveform descriptors, clipped windows, source offset convention '
                            'and aligned sawtooth MSE.',
 'signal_to_matrix_transform': 'Correct to three-signal fundamental phase and nine measurement aggregates; '
                               'no deep image transform established.',
 'tabular_model': 'Full feature permutation and 25 repetitions of five LightGBM folds; both heldouts affect '
                  'stopping.',
 'wavelet_denoising': 'Correct to recursive first-order baseline subtraction; no wavelet stage established.',
 'weighted_blend': 'Correct to equal probability mean and signal-label MCC threshold from in-training '
                   'scores; strict greater-than decisions broadcast over triples.'}
SCOPE='Preserve full independent source reconstruction: raw signal triples through baseline removal, custom peaks, descriptors and phase aggregates; all 125 repeated-fold tree models; equal query probability averaging; signal-label MCC threshold and strict decisions broadcast over triples. Return one probability per query measurement in input order, three ordered signal decisions per measurement, selected threshold, model count and declared evaluation limitations. The intake scalar shorthand is corrected to this batched numerical result. No file or submission artifact is required by the original root output. Numeric filtering, peak ranking and knee controls, descriptor and phase operations, generic-width fold fitting and threshold search can be selected separately. Consecutive three-signal grouping, nine-feature permutation, random five-way fold coverage and threshold routing remain explicit recipe assumptions. Retain independent mathematics and source attributions; no notebook code copying or new license-grant claim. Both heldouts influence stopping and threshold selection uses in-training scores; no unbiased validation or historical performance claim. Actual-length descriptors and FFT phase remain declared generalizations; preserve undefined-case rejection and missing aggregates.'


def build_graph():
    scope_path='docs/reviews/vsb_original_scope.json'
    raw=(ROOT/scope_path).read_bytes()
    if raw!=subprocess.check_output(['git','show','6e093e16:'+scope_path],cwd=ROOT):
        raise ValueError('Source branch qualification changed')
    scope=json.loads(raw)
    if not scope['passed'] or set(scope['original_stages'])!=set(CORRECTIONS):
        raise ValueError('Full seven-stage scope required')
    graph=build_decomposed_vsb_graph()
    retained=json.loads((ROOT/'docs/reviews/decomposed_vsb_graph_execution.json').read_text())
    if encode_execution_graph(graph)[0]!=retained['graph_sha256']:
        raise ValueError('Qualified base graph drift')
    contracts=provider_publication.contracts()
    for node in graph.nodes:
        ports={p['name']:p for p in contracts[node.matched_primitive]['ports'] if p['direction']=='input'}
        for port in node.inputs:
            port.required=ports[port.name]['required']
            port.default_value_repr=ports[port.name]['default_value_repr']
    output_sources={p.name:dict(node_id=n.node_id,output_name=p.name)
                    for n in graph.nodes if n.node_id == 'encode_result' for p in n.outputs}
    graph.metadata.update(original_source_version_id=ORIGINAL,original_source_content_hash=ORIGINAL_HASH,
        qualified_reference_version_id=PARENT,catalog_output_sources=output_sources,
        artifact_source='corrected_original_competition_execution',
        scope=SCOPE,source_scope_sha256=sha(ROOT/scope_path),
        source_corrections=CORRECTIONS,
        qualification_status='Stored execution and semantic publication checks remain required.')
    return graph


def boundaries(graph):
    consumed={(e.target_id,e.input_name) for e in graph.edges};inputs={}
    callable_only={n.node_id for n in graph.nodes if any(e.source_id==n.node_id for e in graph.edges)
                   and all(e.edge_kind==EdgeKind.CALLABLE_INJECTION for e in graph.edges if e.source_id==n.node_id)}
    for node in graph.nodes:
        if node.node_id in callable_only:continue
        for port in node.inputs:
            if (node.node_id,port.name) in consumed:continue
            value=port.model_dump()
            if port.name in inputs and inputs[port.name]!=value:raise ValueError('Repeated root input declarations differ')
            inputs[port.name]=value
    by_node={n.node_id:n for n in graph.nodes};outputs=[]
    for alias,source in graph.metadata['catalog_output_sources'].items():
        port=next(p for p in by_node[source['node_id']].outputs if p.name==source['output_name']).model_dump()
        port.update(name=alias,constraints='Graph output '+source['node_id']+'/'+source['output_name']+'.')
        outputs.append(port)
    if set(inputs)!={'payload'} or set(graph.metadata['catalog_output_sources'])!={'result'}:
        raise ValueError('Complete root boundary differs')
    return list(inputs.values()),outputs


def check_anchor(db,plan):
    parent=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest
        FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s''',(PARENT,)).fetchone()
    if parent!=dict(artifact_id='9ddba25b-899d-5c65-b35c-525e8730275c',fqdn=FQDN+'.execution',
        status='approved',is_publishable=True,is_latest=True,
        content_hash='33fa3095e0d940b9efc1b5ace745b635c43c2812ad166821490de32295d1050a'):
        raise ValueError('Approved reference derivative changed')
    anchor=db.execute('SELECT a.artifact_id::text,a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(ORIGINAL,)).fetchone()
    if anchor!=dict(artifact_id=IDENTITY,fqdn=FQDN,content_hash=ORIGINAL_HASH):raise ValueError('Original identity differs')
    evidence=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(ORIGINAL,)).fetchall()
    if len(evidence)!=1 or hashlib.sha256(_canonical(evidence[0]['details']['snapshot']).encode()).hexdigest()!=ORIGINAL_HASH:
        raise ValueError('Original snapshot differs')
    from sciona.competition_graph import load_competition_graph
    if {n.node_id for n in load_competition_graph(db,version_id=ORIGINAL).nodes}!=set(CORRECTIONS):
        raise ValueError('Original source stage coverage differs')
    for version,key in [(ORIGINAL,'original_history'),(PARENT,'parent_history')]:
        if history(db,version)!=plan[key]:raise ValueError('Preserved source history differs')


def make_plan():
    providers=provider_publication.qualify_plan()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        provider_publication.check(db,providers)
    verification=dict(passed=True,exact_providers=23,native_eligibility_checked=True,scope="Exact approved and served provider versions")
    graph=build_graph();digest,nodes,edges=encode_execution_graph(graph)
    inputs,outputs=boundaries(graph)
    version=str(uuid5(UUID(IDENTITY),'corrected-complete-execution:'+digest))
    by_runtime={r['runtime_fqdn']:r for r in providers['records']}
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
    result=dict(passed=True,approved=False,artifact_id=IDENTITY,fqdn=FQDN,version_id=version,graph_sha256=digest,
        original_version_id=ORIGINAL,original_content_hash=ORIGINAL_HASH,original_history=original_history,parent_history=parent_history,
        nodes=len(nodes),edges=len(edges),inputs=inputs,outputs=outputs,bindings=bindings,dependencies=dependencies,
        provider_publication_plan_sha256=sha(provider_publication.PLAN),provider_verification=verification,
        builder_sha256=sha(__file__),qualified_base_execution_sha256=sha(ROOT/'docs/reviews/decomposed_vsb_graph_execution.json'))
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        check_anchor(db,result)
    return result


def qualified_plan():
    plan=json.loads(PLAN.read_text())
    provider_plan=provider_publication.qualify_plan()
    if sha(provider_publication.PLAN)!=plan['provider_publication_plan_sha256'] or sha(__file__)!=plan['builder_sha256']:
        raise ValueError('Qualified plan or builder changed')
    if sha(ROOT/'docs/reviews/decomposed_vsb_graph_execution.json')!=plan['qualified_base_execution_sha256']:
        raise ValueError('Base execution evidence changed')
    graph=build_graph();digest,nodes,edges=encode_execution_graph(graph)
    if (digest,len(nodes),len(edges))!=(plan['graph_sha256'],26,60) or boundaries(graph)!=(plan['inputs'],plan['outputs']):
        raise ValueError('Complete corrected graph differs')
    return plan,provider_plan


def check_candidate(db,plan,provider_plan,approved=False):
    check_anchor(db,plan);provider_publication.check(db,provider_plan)
    state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(IDENTITY,)).fetchone()
    if state!=dict(status='approved' if approved else 'draft',is_publishable=approved):raise ValueError('Original publication state differs')
    version=db.execute('SELECT artifact_id::text,content_hash,is_latest,trust_tier,derives_from::text FROM artifact_versions WHERE version_id=%s',(plan['version_id'],)).fetchone()
    if version!=dict(artifact_id=IDENTITY,content_hash=plan['graph_sha256'],is_latest=approved,trust_tier=3,derives_from=ORIGINAL):raise ValueError('Corrected version differs')
    latest=db.execute('SELECT version_id::text FROM artifact_versions WHERE artifact_id=%s AND is_latest',(IDENTITY,)).fetchall()
    if latest!=[{'version_id':plan['version_id'] if approved else ORIGINAL}]:raise ValueError('Latest selection differs')
    bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s ORDER BY node_id',(plan['version_id'],)).fetchall()
    if sorted(bindings,key=lambda b:b['node_id'])!=sorted(plan['bindings'],key=lambda b:b['node_id']):raise ValueError('Exact graph bindings differ')
    dependencies=db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional,port_name,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s ORDER BY dependency_artifact_fqdn',(plan['version_id'],)).fetchall()
    if sorted(dependencies,key=lambda d:d['dependency_artifact_fqdn'])!=sorted(plan['dependencies'],key=lambda d:d['dependency_artifact_fqdn']):raise ValueError('Exact graph dependencies differ')
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
        gates=json.loads((ROOT/'docs/reviews/vsb_corrected_intake_transaction.json').read_text())
        if not gates['passed'] or gates['importer_sha256']!=sha(__file__) or gates['plan_sha256']!=sha(PLAN):raise ValueError('Matching staging gates required')
    identity,version=UUID(IDENTITY),UUID(plan['version_id']);writes=0
    digest,nodes,edges=encode_execution_graph(build_graph())
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        db.execute('SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE',(identity,)).fetchone()
        check_anchor(db,plan);provider_publication.check(db,provider_plan)
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
        nodes=len(nodes),edges=len(edges),reused_providers=23,new_providers=0,root_inputs=1,root_outputs=1,
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
    elif action=='validate':result=validate_transaction();path=ROOT/'docs/reviews/vsb_corrected_intake_transaction.json'
    else:
        result=stage(apply=True)
        if action=='repeat' and result['rows_created']:raise ValueError('Repeat changed rows')
        path=ROOT/('docs/reviews/vsb_corrected_intake_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(planned=True,nodes=result['nodes'],edges=result['edges']) if action=='plan' else result))
