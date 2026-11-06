"""Plan exact DSB catalog ports without publishing or claiming semantic approval."""
import importlib
import inspect
import json
from typing import get_origin
from pathlib import Path
import numpy as np
from sciona.dsb_decomposed_evaluation_graph import build_dsb_evaluation_graph
from scripts.inventory_dsb_decomposed_providers import inventory, ROOT, sha, SPATIAL_RECIPES

# Indirect providers have no graph output names. These names specify actual
# return tuple order; values are runtime types, not claims of serialization.
INDIRECT_OUTPUTS = {
    'scale_coordinate_rows': [('rows','numpy.ndarray')],
    'centered_window': [('window','torch.Tensor')],
    'repeat_rows_by_threshold': [('rows','numpy.ndarray')],
    'sequential_softmax_indices': [('indices','numpy.ndarray')],
    'resample_regular_volume': [('volume','numpy.ndarray'),('effective_spacing','numpy.ndarray')],
    'noisy_or_with_baseline': [('probabilities','torch.Tensor')],
    'thresholded_log_penalty': [('penalty','torch.Tensor')],
    'highest_scoring_indices': [('indices','torch.Tensor')],
    'crop_detection': [('sample','numpy.ndarray'),('target','numpy.ndarray'),('boxes','numpy.ndarray'),('coordinates','numpy.ndarray')],
    'augment_detection': [('sample','numpy.ndarray'),('target','numpy.ndarray'),('boxes','numpy.ndarray'),('coordinates','numpy.ndarray')],
    'augment_classifier': [('sample','numpy.ndarray'),('coordinates','numpy.ndarray')],
    'crop_classifier_proposal': [('sample','numpy.ndarray'),('coordinates','numpy.ndarray')],
    'assign_anchor_labels': [('labels','numpy.ndarray')],
    'validate_detector': [('metrics','dict')],
    'validate_classifier': [('metrics','dict')],
    'export_model_checkpoint': [('checkpoint','bytes')],
}


def type_name(annotation):
    value=get_origin(annotation) or annotation
    if value is np.ndarray:return 'numpy.ndarray'
    if value in (object,dict,list,tuple,int,float,bool,str,bytes):return value.__name__
    raise ValueError('Unreviewed annotation: '+str(annotation))


def plan():
    current=inventory(offline=True)
    grouped={}
    for node in build_dsb_evaluation_graph().nodes:
        grouped.setdefault(node.matched_primitive,[]).append(node)
    records=[];findings=[]
    for record in current['providers']:
        runtime=record['runtime_fqdn'];module,name=runtime.rsplit('.',1)
        function=getattr(importlib.import_module(module),name)
        signature=inspect.signature(function);nodes=grouped.get(runtime,[])
        if any({p.name for p in node.inputs}!=set(signature.parameters) for node in nodes):
            raise ValueError('Graph input names do not match signature: '+runtime)
        ports=[]
        for ordinal,(parameter_name,parameter) in enumerate(signature.parameters.items()):
            if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,inspect.Parameter.KEYWORD_ONLY):
                raise ValueError('Variadic provider contract requires explicit review')
            declared=type_name(parameter.annotation)
            graph_types={p.type_desc for node in nodes for p in node.inputs if p.name==parameter_name}
            if len(graph_types)>1:raise ValueError('Repeated input type differs')
            graph_type=next(iter(graph_types)) if graph_types else None
            # Graph object ports can carry concrete dictionaries, bytes, arrays
            # and model state. Preserve their graph type and record refinement.
            if graph_type and graph_type not in ('object',declared) and declared!='object':
                raise ValueError('Incompatible explicit graph annotation')
            semantic_required=runtime in SPATIAL_RECIPES and parameter_name=='rng'
            port=dict(direction='input',name=parameter_name,ordinal=ordinal,
                      type_desc=graph_type or declared,declared_type=declared,
                      parameter_kind=parameter.kind.name.lower(),
                      required=semantic_required or parameter.default is inspect.Parameter.empty,
                      default_value_repr='' if semantic_required or parameter.default is inspect.Parameter.empty else repr(parameter.default),
                      signature_required=parameter.default is inspect.Parameter.empty,
                      signature_default_repr='' if parameter.default is inspect.Parameter.empty else repr(parameter.default),
                      non_null_required=semantic_required)
            if semantic_required:
                if parameter.default is not None:
                    raise ValueError('RNG signature changed; review the catalog override')
                port['runtime_contract'] = (
                    'Caller-owned random.Random for anchor assignment; caller-owned '
                    'numpy.random.RandomState for crop and augmentation. Execution '
                    'advances the supplied stream; None and omitted values are rejected.'
                )
                findings.append(dict(provider=runtime,parameter=parameter_name,
                    resolution='Catalog input is required with no default and non-null; rejected Python default retained separately for audit.'))
            if parameter_name=='prepare_epoch':
                port['callback_contract']='Caller-owned per-epoch ordering callback. Only the qualified callback behavior is covered; arbitrary callbacks are not certified.'
            ports.append(port)
        if nodes:
            outputs=[(p.name,p.type_desc) for p in nodes[0].outputs]
            if any([(p.name,p.type_desc) for p in node.outputs]!=outputs for node in nodes):
                raise ValueError('Repeated output contracts differ')
            provenance='qualified_graph_port_order'
        else:
            if name not in INDIRECT_OUTPUTS:raise ValueError('Missing explicit indirect output contract: '+runtime)
            outputs=INDIRECT_OUTPUTS[name];provenance='reviewed_implementation_return_order'
        for ordinal,(output_name,output_type) in enumerate(outputs):
            ports.append(dict(direction='output',name=output_name,ordinal=ordinal,type_desc=output_type,
                              required=True,default_value_repr='',contract_source=provenance))
        records.append(dict(runtime_fqdn=runtime,version_id=record['version_id'],content_hash=record['content_hash'],
            provider_sha256=record['provider_sha256'],ports=ports,
            operation_contract=inspect.getdoc(function),graph_nodes=[node.node_id for node in nodes],
            state_contract='Runtime model, factory, RNG and tensor objects remain in-process unless an explicit bytes export is requested.'))
    return dict(approved=False,catalog_mutations=0,graph_digest=current['graph_digest'],
                providers=len(records),ports=sum(len(r['ports']) for r in records),
                findings=findings,provider_contracts=records,planner_sha256=sha(Path(__file__)),
                inventory_sha256=sha(ROOT/'docs/reviews/dsb_decomposed_provider_inventory.json'),
                limitations=['Draft interface plan; does not replace semantic or dependency review.',
                    'Generic object annotations retain runtime constraints from implementation contracts.',
                    'Catalog identities, existing versions and eligibility still require live database reconciliation.'])


if __name__=='__main__':
    report=plan()
    (ROOT/'docs/reviews/dsb_provider_interface_plan.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('providers','ports','findings')},indent=2))
