"""Review every graph call against exact staged native and catalog interfaces."""
import importlib
import inspect
import json

from scripts.complete_flavours_provider_contracts import expected_dependencies, PORT_FIELDS
from scripts.import_flavours_provider_drafts import connect, reviewed_plan, ROOT
from scripts.inventory_flavours_decomposed_providers import inventory
from scripts.plan_flavours_provider_intake import sha
from sciona.flavours_decomposed_evaluation import build_evaluated_graph


def normalized_type(value):
    # Shape/dtype validity comes from exact provider contracts and execution,
    # not this catalog spelling normalization.
    return 'numpy.ndarray' if value in ('NDArray[np.float32]','NDArray[np.float64]') else value


def check_call(node, function, catalog_ports):
    signature=inspect.signature(function)
    inputs={p['name']:p for p in catalog_ports if p['direction']=='input'}
    if set(inputs)!=set(signature.parameters):
        raise ValueError('Catalog signature differs')
    for name,parameter in signature.parameters.items():
        port=inputs[name]
        required=parameter.default is inspect.Parameter.empty
        default='' if required else repr(parameter.default)
        if port['required']!=required or port['default_value_repr']!=default:
            raise ValueError('Catalog default or required flag differs')
    bound={p.name:p for p in node.inputs}
    if len(bound)!=len(node.inputs) or not set(bound)<=set(inputs):
        raise ValueError('Unknown or duplicate graph input')
    if any(port['required'] and name not in bound for name,port in inputs.items()):
        raise ValueError('Graph omits required native input')
    for name,port in bound.items():
        if normalized_type(port.type_desc)!=normalized_type(inputs[name]['type_desc']):
            raise ValueError('Graph input type differs')
    outputs=sorted((p for p in catalog_ports if p['direction']=='output'),key=lambda p:p['ordinal'])
    if [p['ordinal'] for p in outputs]!=list(range(len(node.outputs))):
        raise ValueError('Output ordinals or arity differ')
    aliases=[]
    for graph_port,catalog_port in zip(node.outputs,outputs):
        if normalized_type(graph_port.type_desc)!=normalized_type(catalog_port['type_desc']):
            raise ValueError('Graph output type differs')
        if graph_port.name!=catalog_port['name']:
            if not (node.matched_primitive=='sciona.atoms.ml.xgboost.model_io.predict_binary_model'
                    and len(outputs)==1 and graph_port.name=='predictions' and catalog_port['name']=='probabilities'):
                raise ValueError('Unreviewed output alias')
            aliases.append(dict(graph_name=graph_port.name,catalog_name=catalog_port['name'],ordinal=0))
    defaults={name:p['default_value_repr'] for name,p in inputs.items() if name not in bound}
    return dict(omitted_native_defaults=defaults,output_aliases=aliases)


def review():
    reviewed_plan()
    current=inventory()
    evidence=ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json'
    execution=json.loads(evidence.read_text())
    if not (execution['passed'] and execution['all_nodes_uncached'] and execution['executed_nodes']==841
            and execution['graph_digest']==current['graph_digest']):
        raise ValueError('Complete exact graph execution required')
    graph=build_evaluated_graph()
    records={r['runtime_fqdn']:r for r in current['providers']}
    declarations={}
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for runtime,record in records.items():
            for catalog in ('legacy','canonical'):
                versions=record[catalog+'_latest']
                if len(versions)!=1 or versions[0]['version_id']!=record['version_id'] or versions[0]['content_hash']!=record['content_hash'] or versions[0]['trust_tier']!=3:
                    raise ValueError('Exact version is not staged in both catalogs')
            columns=','.join(PORT_FIELDS)
            canonical=db.execute('SELECT '+columns+' FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(record['version_id'],)).fetchall()
            legacy=db.execute('SELECT '+columns+' FROM atom_io_specs WHERE version_id=%s ORDER BY direction,ordinal',(record['version_id'],)).fetchall()
            if not canonical or canonical!=legacy:
                raise ValueError('Legacy/canonical interfaces differ')
            declarations[runtime]=canonical
        for expected in expected_dependencies(current['providers']):
            actual=db.execute('SELECT dependent_version_id::text,dependency_artifact_fqdn,dependency_content_hash,port_name,dependency_role,optional,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s AND dependency_artifact_fqdn=%s',
                              (expected['dependent_version_id'],expected['dependency_artifact_fqdn'])).fetchall()
            if actual!=[expected]:
                raise ValueError('Required registered dependency differs')
    calls={}
    for node in graph.nodes:
        runtime=node.matched_primitive
        module,name=runtime.rsplit('.',1)
        result=check_call(node,getattr(importlib.import_module(module),name),declarations[runtime])
        if runtime in calls and calls[runtime]!=result:
            raise ValueError('Repeated calls differ in default or alias semantics')
        calls[runtime]=result
    return dict(passed=True,approved=False,catalog_mutations=0,review_source='automated',proposed_tier=3,
        graph_digest=current['graph_digest'],graph_calls_checked=len(graph.nodes),direct_providers_checked=len(calls),
        closure_provider_contracts_checked=len(records),required_dependency_links_checked=current['registered_dependency_links'],
        call_semantics=calls,execution_evidence_sha256=sha(evidence),reviewer_sha256=sha(__file__),
        limitations=['Interface and exact registered dependency review only; final semantic/publication gates remain required.',
            'Array type spelling normalization does not broaden dtype, shape or numerical validity bounds.',
            'The binary predictor single return is assigned the graph output name by the qualified production runner.',
            'Weighted AUC retains native threshold and weight defaults when the graph omits those optional arguments.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/flavours_decomposed_binding_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('call_semantics','limitations')}))
