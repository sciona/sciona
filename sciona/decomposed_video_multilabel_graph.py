"""Draft sharded recipe with explicit reusable differentiable/threshold edges."""
import inspect
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, EdgeKind, IOSpec, NodeStatus
from sciona.atoms.ml.masked_attention import masked_attention_pool, sigmoid_context_gate
from sciona.atoms.ml.calibration.empirical_f1 import fit_empirical_f1_thresholds, apply_column_thresholds
from sciona.atoms.ml.video_multilabel_sharded_execution import execute
from sciona.atoms.ml.model_selection.materialized_prediction import average


def build_decomposed_video_multilabel_graph():
    def node(name, function, inputs, outputs):
        return AlgorithmicNode(node_id=name, name=name, description=inspect.getdoc(function),
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=function.__module__+'.'+function.__name__,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs])
    operations = [
        ('pool',masked_attention_pool,[('values','object'),('logits','object'),('mask','object')], [('pooled','object'),('weights','object')]),
        ('gate',sigmoid_context_gate,[('values','object'),('logits','object')],[('gated','object')]),
        ('mean',average,[('vectors','list')],[('mean','ndarray')]),
        ('threshold_fit',fit_empirical_f1_thresholds,[('scores','ndarray'),('labels','ndarray')],[('thresholds','ndarray')]),
        ('threshold_apply',apply_column_thresholds,[('scores','ndarray'),('thresholds','ndarray')],[('predictions','ndarray')]),
    ]
    nodes=[node(*operation) for operation in operations]
    nodes.append(node('execute',execute,
        [('training','object'),('calibration','object'),('query','object'),('controls','dict')]
        +[(name+'_function','object') for name,_,_,_ in operations], [('result','dict')]))
    edges=[DependencyEdge(source_id=name,target_id='execute',output_name=outputs[0][0],
        input_name=name+'_function',source_type='object',target_type='object',
        edge_kind=EdgeKind.CALLABLE_INJECTION) for name,_,_,outputs in operations]
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(
        artifact_source='video_multilabel_sharded_decomposition',publication_status='draft',
        num_nodes=len(nodes),num_edges=len(edges),
        scope='Complete configured sharded training with explicit pooling, gating and empirical threshold callables.',
        limits=['Sparse fitting and prediction still require reuse review and decomposition.',
                'Caller supplies private runtime shard stores; no data or paths are embedded.',
                'Draft execution evidence only; catalog qualification and original publication pending.']))
