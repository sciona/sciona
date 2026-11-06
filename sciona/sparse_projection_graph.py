"""Dataflow graph for fitting numeric sparse factors and projecting rows."""
import inspect
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus
from sciona.atoms.ml.sklearn.decomposition import sparse_projection as provider


def build_sparse_projection_graph():
    nodes=[]
    for name,function,inputs,outputs in [
        ('fit',provider.fit_sparse_projection,
         [('matrix','object'),('rank','int'),('iterations','int'),('oversamples','int'),('seed','int')],
         [('components','ndarray'),('singular_values','ndarray')]),
        ('project',provider.transform_sparse_projection,
         [('matrix','object'),('components','ndarray')],[('projected','ndarray')])]:
        contract=inspect.getdoc(function)
        nodes.append(AlgorithmicNode(node_id=name,name=name,description=contract,
            concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=function.__module__+'.'+function.__name__,
            inputs=[IOSpec(name=n,type_desc=t,required=True,default_value_repr='',constraints=contract) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t,required=True,default_value_repr='',constraints=contract) for n,t in outputs]))
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='fit',target_id='project',
        output_name='components',input_name='components',source_type='ndarray',target_type='ndarray')],
        metadata=dict(scope='Fit sparse numeric projection and transform the supplied rows with explicit controls.',
            limits=['No text, labels, split policy or catalog approval.',
                    'Sparse matrices are supplied as concrete runtime objects; no data embedded.']))
