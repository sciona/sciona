"""Stage both original OpenVaccine identities using shared approved providers.

Staging preserves source history and latest selection; it never approves a CDG.
Each instance owns immutable target configuration, so plans cannot cross targets.
"""
import argparse
import hashlib
import json
from uuid import UUID, uuid5
from psycopg.types.json import Jsonb
from sciona.decomposed_openvaccine_graph import build_decomposed_openvaccine_graph
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.architect.models import EdgeKind
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.competition_import import _canonical
from sciona.competition_graph import load_competition_graph
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.import_residual_execution_drafts import ensure_row
from scripts.stage_remaining_legacy_physics_revisions import history
from scripts import decomposed_openvaccine_approved_provider_state as provider_publication
from scripts.plan_flavours_provider_intake import sha

PARENT='ac5e5792-b3c6-59c6-861a-52433b976866'
PARENT_IDENTITY='19cf56e7-8fc1-5dad-ac8a-51d58c08f6e9'
PARENT_FQDN='cdg.competition.solution.kaggle.covid_vaccine_mrna_1st.execution'
PARENT_HASH='c8dead8658202fc579e7a9d1a9123f77cf722061f418b1c4cbcef1c729f6530e'
IO_FIELDS=('name','type_desc','constraints','required','default_value_repr','dim_signature')
TARGETS={
    'covid': dict(identity='d132e834-0634-59a6-adc5-2fe3b00082e3',
        fqdn='cdg.competition.solution.kaggle.covid_vaccine_mrna_1st',
        original='9510a335-f191-5642-9ae1-20459e7d4eba',
        original_hash='5a6c3af650d1ea240d4b211a57c69d1136fe68f1780250cbb47bab790060c121'),
    'degradation': dict(identity='26cf554f-1920-5652-853d-99b62dc3aade',
        fqdn='cdg.competition.solution.kaggle.openvaccine_mrna_degradation_1st',
        original='bc2e55f3-60b8-56a3-8519-87f31ed27246',
        original_hash='a5b83b0ed23f1c38cc1212632e1ac01906c4b2553a29c411ebdcd3d4325ab32c'),
}
CORRECTIONS={
    'covid': {
        'secondary_structure_parsing': 'features: pinned CONTRAfold-SE plus EternaFold coefficients and MEA decoding replace the ungrounded ViennaRNA/Arnie claim.',
        'node_feature_generation': 'features: complete 55 node and eight adjacency channels, with corrected per-record batching and strict balanced structures.',
        'sequence_to_sequence_model': 'fit_forecast: twenty full TensorFlow legacy-Keras members, five states each of GRU, LSTM, forward and wave heads.',
        'graph_neural_network': 'fit_forecast: shared graph base with autoencoder pretraining; no separate parallel GCN branch.',
        'feature_fusion': 'fit_forecast: shared graph-base features feed diversity heads; claimed parallel recurrent/GCN concatenation is excluded.',
        'output_head': 'fit_forecast, blend_inputs, blend, result: five per-position targets, reverse averaging, corrected all-member clipping, external generic ordered sum and arithmetic mean.',
    },
    'degradation': {
        'base_pair_extraction': 'features: pinned CONTRAfold-SE plus EternaFold coefficients replace the ungrounded ViennaRNA claim.',
        'tensor_formatting': 'features: validated dense float32 features with 55 node and eight adjacency channels; no PyTorch tensors.',
        'gcn_layer': 'fit_forecast: source TensorFlow shared graph base with autoencoder pretraining; no PyG GCN.',
        'rnn_layer': 'fit_forecast: all twenty full GRU, LSTM, forward and wave members, rather than a single bidirectional GRU.',
        'target_weighted_loss': 'fit_forecast: source weighted per-sample per-target RMSE with masked-position denominator and epsilon floor; caller-controlled uncertainty masking/perturbation and population weights, complete supervised and pseudo-label refinement.',
    },
}
SCOPE=(
    'Complete independent source-corrected twenty-member reconstruction shared by two overlapping original intakes. '
    'Raw RNA folding, complete 55/8 features, shared autoencoder pretraining, all configured initial supervised steps, '
    'all configured supervised/pseudo-label refinement rounds, checkpoint restoration, reverse-averaged predictions, '
    'all-member clipping and external generic ordered summation are retained. Recovered source is inference-only; '
    'training is reconstructed from the winner writeup with caller-controlled populations, splits, weights, steps, '
    'noise, reversal and rollback policy. Original winning clustered blend, historical schedule and accuracy are unproven. '
    'Both originals retain separate immutable provenance but do not represent two distinct algorithms. '
    'The ordered sum is domain independent; dense-feature training can accept compatible non-RNA features within fixed '
    '55/8/5 dimensions and pinned architectures. Folding and ensemble/training policy remain recipe-specific. '
    'Shared pretraining and teachers are transductive; no independent cross-validation accuracy is claimed. '
    'Automated Tier 3 candidate; Tier 1 human certification and Tier 2 qualification are not claimed.'
)


def check_sources(db):
    for key, target in TARGETS.items():
        anchor=db.execute('SELECT a.artifact_id::text,a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s', (target['original'],)).fetchone()
        if anchor != dict(artifact_id=target['identity'],fqdn=target['fqdn'],content_hash=target['original_hash']):
            raise ValueError('Overlapping original identity differs')
        evidence=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'", (target['original'],)).fetchall()
        if len(evidence)!=1 or hashlib.sha256(_canonical(evidence[0]['details']['snapshot']).encode()).hexdigest()!=target['original_hash']:
            raise ValueError('Overlapping original snapshot differs')
        graph=load_competition_graph(db,version_id=target['original'])
        if {n.node_id for n in graph.nodes}!=set(CORRECTIONS[key]):
            raise ValueError('Original source coverage differs')


class InjectedFailure(RuntimeError):
    pass


class Intake:
    def __init__(self,key):
        target=TARGETS[key]
        self.key=key
        self.IDENTITY=target['identity']
        self.FQDN=target['fqdn']
        self.ORIGINAL=target['original']
        self.ORIGINAL_HASH=target['original_hash']
        self.other=TARGETS[next(k for k in TARGETS if k!=key)]
        self.PLAN=self.report_path('plan')
        self.RUNNER='openvaccine-'+key+'-corrected-intake-draft.v1'

    def report_path(self,suffix):
        return ROOT/('docs/reviews/openvaccine_'+self.key+'_corrected_intake_'+suffix+'.json')

    def build_graph(self):
        graph=build_decomposed_openvaccine_graph()
        retained=json.loads((ROOT/'docs/reviews/decomposed_openvaccine_graph_execution.json').read_text())
        if encode_execution_graph(graph)[0]!=retained['serialized_graph_sha256']:
            raise ValueError('Qualified base graph drift')
        contracts=provider_publication.contracts()
        for node in graph.nodes:
            ports={p['name']:p for p in contracts[node.matched_primitive]['ports'] if p['direction']=='input'}
            for port in node.inputs:
                port.required=ports[port.name]['required']
                port.default_value_repr=ports[port.name]['default_value_repr']
        output_sources={p.name:dict(node_id=n.node_id,output_name=p.name)
                        for n in graph.nodes if n.node_id == 'result' for p in n.outputs}
        graph.metadata.update(original_source_version_id=self.ORIGINAL,original_source_content_hash=self.ORIGINAL_HASH,
            qualified_reference_version_id=PARENT,catalog_output_sources=output_sources,
            artifact_source='corrected_original_competition_execution',
            scope=SCOPE, source_corrections=CORRECTIONS[self.key],
            overlapping_source_versions=[t['original'] for t in TARGETS.values()],
            shared_solution_family=True,
            qualification_status='Stored execution and semantic publication checks remain required.')
        return graph


    def boundaries(self,graph):
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


    def check_anchor(self,db,plan):
        check_sources(db)
        parent=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,v.is_latest
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s''',(PARENT,)).fetchone()
        if parent!=dict(artifact_id=PARENT_IDENTITY,fqdn=PARENT_FQDN,
            status='approved',is_publishable=True,is_latest=True,
            content_hash=PARENT_HASH):
            raise ValueError('Approved reference derivative changed')
        anchor=db.execute('SELECT a.artifact_id::text,a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(self.ORIGINAL,)).fetchone()
        if anchor!=dict(artifact_id=self.IDENTITY,fqdn=self.FQDN,content_hash=self.ORIGINAL_HASH):raise ValueError('Original identity differs')
        evidence=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'",(self.ORIGINAL,)).fetchall()
        if len(evidence)!=1 or hashlib.sha256(_canonical(evidence[0]['details']['snapshot']).encode()).hexdigest()!=self.ORIGINAL_HASH:
            raise ValueError('Original snapshot differs')
        for version,key in [(self.ORIGINAL,'original_history'),(self.other['original'],'other_original_history'),(PARENT,'parent_history')]:
            if history(db,version)!=plan[key]:raise ValueError('Preserved source history differs')


    def make_plan(self):
        providers=provider_publication.qualify_plan()
        with connect() as db:
            db.execute('SET TRANSACTION READ ONLY')
            provider_publication.check(db,providers)
        verification=dict(passed=True,exact_providers=6,native_eligibility_checked=True,scope="Exact approved and served provider versions")
        graph=self.build_graph();digest,nodes,edges=encode_execution_graph(graph)
        inputs,outputs=self.boundaries(graph)
        version=str(uuid5(UUID(self.IDENTITY),'corrected-complete-execution:'+digest))
        by_runtime={r['runtime_fqdn']:r for r in providers['records']}
        bindings=[dict(node_id=n.node_id,bound_artifact_fqdn=by_runtime[n.matched_primitive]['catalog_fqdn'],
            bound_version_content_hash=by_runtime[n.matched_primitive]['content_hash'],status='active',
            evidence_summary=dict(runtime_fqdn=n.matched_primitive,provider_version_id=by_runtime[n.matched_primitive]['version_id'],
                output_aliases_by_ordinal=[p.name for p in n.outputs])) for n in graph.nodes]
        with connect() as db:
            db.execute('SET TRANSACTION READ ONLY')
            parent=db.execute('SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE version_id=%s',(PARENT,)).fetchone()
            original_history=history(db,self.ORIGINAL);parent_history=history(db,PARENT)
            other_original_history=history(db,self.other['original'])
            state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(self.IDENTITY,)).fetchone()
            if state!=dict(status='draft',is_publishable=False):raise ValueError('Original must be draft')
        dependencies=[dict(dependency_artifact_fqdn=self.FQDN,dependency_content_hash=self.ORIGINAL_HASH,dependency_role='cdg',
                          optional=False,port_name='',binding_metadata=dict(source_version_id=self.ORIGINAL,scope='Immutable original intake provenance.')),
                      dict(dependency_artifact_fqdn=parent['fqdn'],dependency_content_hash=parent['content_hash'],dependency_role='cdg',
                          optional=False,port_name='',binding_metadata=dict(source_version_id=PARENT,scope='Preserved executable reference provenance; computation uses explicit atom bindings.'))]
        dependencies.append(dict(dependency_artifact_fqdn=self.other['fqdn'],dependency_content_hash=self.other['original_hash'],
            dependency_role='cdg',optional=False,port_name='',binding_metadata=dict(source_version_id=self.other['original'],
            scope='Overlapping immutable original intake provenance; shared solution family, not an independent algorithm.')))
        for runtime in sorted({n.matched_primitive for n in graph.nodes}):
            record=by_runtime[runtime]
            dependencies.append(dict(dependency_artifact_fqdn=record['catalog_fqdn'],dependency_content_hash=record['content_hash'],
                dependency_role='logic_atom',optional=False,port_name='',binding_metadata=dict(runtime_fqdn=runtime,provider_version_id=record['version_id'])))
        result=dict(passed=True,approved=False,artifact_id=self.IDENTITY,fqdn=self.FQDN,version_id=version,graph_sha256=digest,
            original_version_id=self.ORIGINAL,original_content_hash=self.ORIGINAL_HASH,original_history=original_history,parent_history=parent_history,other_original_history=other_original_history,
            nodes=len(nodes),edges=len(edges),inputs=inputs,outputs=outputs,bindings=bindings,dependencies=dependencies,
            provider_publication_plan_sha256=sha(provider_publication.PLAN),provider_verification=verification,
            builder_sha256=sha(__file__),qualified_base_execution_sha256=sha(ROOT/'docs/reviews/decomposed_openvaccine_graph_execution.json'))
        with connect() as db:
            db.execute('SET TRANSACTION READ ONLY')
            self.check_anchor(db,result)
        return result


    def qualified_plan(self):
        plan=json.loads(self.PLAN.read_text())
        provider_plan=provider_publication.qualify_plan()
        if sha(provider_publication.PLAN)!=plan['provider_publication_plan_sha256'] or sha(__file__)!=plan['builder_sha256']:
            raise ValueError('Qualified plan or builder changed')
        if sha(ROOT/'docs/reviews/decomposed_openvaccine_graph_execution.json')!=plan['qualified_base_execution_sha256']:
            raise ValueError('Base execution evidence changed')
        graph=self.build_graph();digest,nodes,edges=encode_execution_graph(graph)
        if (digest,len(nodes),len(edges))!=(plan['graph_sha256'],6,7) or self.boundaries(graph)!=(plan['inputs'],plan['outputs']):
            raise ValueError('Complete corrected graph differs')
        expected_identity=(self.IDENTITY,self.FQDN,self.ORIGINAL,self.ORIGINAL_HASH,
            str(uuid5(UUID(self.IDENTITY),'corrected-complete-execution:'+digest)))
        if tuple(plan[k] for k in ('artifact_id','fqdn','original_version_id','original_content_hash','version_id'))!=expected_identity:
            raise ValueError('Plan belongs to a different original or candidate')
        by_runtime={r['runtime_fqdn']:r for r in provider_plan['records']}
        expected_bindings=[dict(node_id=n.node_id,bound_artifact_fqdn=by_runtime[n.matched_primitive]['catalog_fqdn'],
            bound_version_content_hash=by_runtime[n.matched_primitive]['content_hash'],status='active',
            evidence_summary=dict(runtime_fqdn=n.matched_primitive,provider_version_id=by_runtime[n.matched_primitive]['version_id'],
                output_aliases_by_ordinal=[p.name for p in n.outputs])) for n in graph.nodes]
        if plan['bindings']!=expected_bindings:
            raise ValueError('Plan provider bindings differ')
        expected_dependencies={(t['fqdn'],t['original_hash'],'cdg',t['original']) for t in TARGETS.values()}
        expected_dependencies.add((PARENT_FQDN,PARENT_HASH,'cdg',PARENT))
        expected_dependencies.update((r['catalog_fqdn'],r['content_hash'],'logic_atom',r['version_id']) for r in by_runtime.values())
        actual_dependencies={(d['dependency_artifact_fqdn'],d['dependency_content_hash'],d['dependency_role'],
            d['binding_metadata'].get('source_version_id',d['binding_metadata'].get('provider_version_id'))) for d in plan['dependencies']}
        if (actual_dependencies!=expected_dependencies or len(plan['dependencies'])!=9
                or any(d['optional'] is not False or d['port_name']!='' for d in plan['dependencies'])):
            raise ValueError('Plan must preserve both sources, reference and six exact providers')
        return plan,provider_plan


    def check_candidate(self,db,plan,provider_plan,approved=False):
        self.check_anchor(db,plan);provider_publication.check(db,provider_plan)
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(self.IDENTITY,)).fetchone()
        if state!=dict(status='approved' if approved else 'draft',is_publishable=approved):raise ValueError('Original publication state differs')
        version=db.execute('SELECT artifact_id::text,content_hash,is_latest,trust_tier,derives_from::text FROM artifact_versions WHERE version_id=%s',(plan['version_id'],)).fetchone()
        if version!=dict(artifact_id=self.IDENTITY,content_hash=plan['graph_sha256'],is_latest=approved,trust_tier=3,derives_from=self.ORIGINAL):raise ValueError('Corrected version differs')
        latest=db.execute('SELECT version_id::text FROM artifact_versions WHERE artifact_id=%s AND is_latest',(self.IDENTITY,)).fetchall()
        if latest!=[{'version_id':plan['version_id'] if approved else self.ORIGINAL}]:raise ValueError('Latest selection differs')
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
        evidence=db.execute('SELECT details FROM artifact_audit_evidence WHERE evidence_id=%s AND passed AND status=%s',(uuid5(UUID(plan['version_id']),self.RUNNER),'completed')).fetchone()
        if not evidence or evidence['details']!=dict(plan_sha256=sha(self.PLAN),scope='Complete decomposed candidate; not publication approval.'):
            raise ValueError('Staging evidence differs')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(self.FQDN,)).fetchone()['d']
        graph=_artifact_document_to_cdg(document,version_id=plan['version_id'],content_hash=plan['graph_sha256'],require_execution_envelope=True)
        if graph!=self.build_graph():raise ValueError('Stored graph differs')
        served=db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(self.IDENTITY,)).fetchone()
        if bool(served)!=approved:raise ValueError('Serving state differs')
        return graph


    def stage(self,apply=False,fail_after=None):
        plan,provider_plan=self.qualified_plan()
        if apply:
            gates=json.loads((self.report_path('transaction')).read_text())
            if not gates['passed'] or gates['importer_sha256']!=sha(__file__) or gates['plan_sha256']!=sha(self.PLAN):raise ValueError('Matching staging gates required')
        identity,version=UUID(self.IDENTITY),UUID(plan['version_id']);writes=0
        digest,nodes,edges=encode_execution_graph(self.build_graph())
        with connect() as db:
            db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(self.RUNNER,))
            db.execute('SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE',(identity,)).fetchone()
            self.check_anchor(db,plan);provider_publication.check(db,provider_plan)
            def ensure(table,keys,row):
                nonlocal writes
                writes+=int(ensure_row(db,table,keys,row))
                if writes==fail_after:raise InjectedFailure('Injected corrected intake failure')
            ensure('artifact_versions',{'version_id':version},dict(version_id=version,artifact_id=identity,content_hash=digest,
                semver='0.1.0+corrected.'+digest[:12],is_latest=False,trust_tier=3,derives_from=UUID(self.ORIGINAL)))
            for table,rows,keys in [('artifact_cdg_nodes',nodes,['node_id']),('artifact_cdg_edges',edges,['source_id','target_id','output_name','input_name'])]:
                for item in rows:
                    row=dict(version_id=version,**item);ensure(table,{k:row[k] for k in ['version_id',*keys]},row)
            for direction in ('input','output'):
                for ordinal,port in enumerate(plan[direction+'s']):
                    if any(port.get(k) for k in ('data_kind','time_basis','provenance')):raise ValueError('Unsupported populated root metadata')
                    row=dict(artifact_id=identity,version_id=version,direction=direction,ordinal=ordinal,**{k:port[k] for k in IO_FIELDS})
                    ensure('artifact_io_specs',{k:row[k] for k in ('artifact_id','version_id','direction','name')},row)
            for binding in plan['bindings']:
                row=dict(binding,version_id=version,binding_confidence=1.,binding_source=self.RUNNER,
                         alternatives=Jsonb([]),evidence_summary=Jsonb(binding['evidence_summary']))
                ensure('artifact_cdg_bindings',dict(version_id=version,node_id=binding['node_id']),row)
            for dependency in plan['dependencies']:
                row=dict(dependency,dependent_version_id=version,binding_metadata=Jsonb(dependency['binding_metadata']))
                ensure('artifact_dependencies',{k:row[k] for k in ('dependent_version_id','dependency_artifact_fqdn','dependency_content_hash','port_name')},row)
            evidence=uuid5(version,self.RUNNER)
            ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=identity,version_id=version,
                audit_type='asset_integrity_check',passed=True,status='completed',source_kind='automated',runner_version=self.RUNNER,
                details=Jsonb(dict(plan_sha256=sha(self.PLAN),scope='Complete decomposed candidate; not publication approval.'))))
            self.check_candidate(db,plan,provider_plan)
            if fail_after is not None:raise ValueError('Fault point not reached')
            if not apply:db.rollback()
        return dict(applied=apply,approved=False,rows_created=writes,version_id=str(version),graph_sha256=digest,
            nodes=len(nodes),edges=len(edges),reused_providers=6,new_providers=0,root_inputs=1,root_outputs=1,
            source_histories_preserved=True,stored_roundtrip_verified=True)


    def validate_transaction(self):
        plan,_=self.qualified_plan()
        def unchanged():
            with connect() as db:
                self.check_anchor(db,plan)
                if db.execute('SELECT 1 FROM artifact_versions WHERE version_id=%s',(plan['version_id'],)).fetchone():raise ValueError('Candidate survived rollback')
                state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(self.IDENTITY,)).fetchone()
                if state!=dict(status='draft',is_publishable=False):raise ValueError('Publication state changed')
        unchanged();dry=self.stage();unchanged();points=[3,dry['rows_created']//2,dry['rows_created']]
        for point in points:
            try:self.stage(fail_after=point)
            except InjectedFailure:pass
            else:raise ValueError('Expected injected rollback')
            unchanged()
        return dict(passed=True,catalog_mutations=0,dry_run=dry,injected_failure_points=points,
            all_rollbacks_complete=True,importer_sha256=sha(__file__),plan_sha256=sha(self.PLAN))



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target',choices=list(TARGETS))
    parser.add_argument('action',choices=['plan','validate','apply','repeat'])
    args=parser.parse_args();intake=Intake(args.target)
    if args.action=='plan':result=intake.make_plan();path=intake.PLAN
    elif args.action=='validate':result=intake.validate_transaction();path=intake.report_path('transaction')
    else:
        result=intake.stage(apply=True)
        if args.action=='repeat' and result['rows_created']:raise ValueError('Repeat changed rows')
        path=intake.report_path(args.action)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(target=args.target,planned=True,nodes=result['nodes'],edges=result['edges']) if args.action=='plan' else result))
