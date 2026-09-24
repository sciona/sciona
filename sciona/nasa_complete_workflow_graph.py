"""Explicit training-to-prediction composition for a corrected original intake.

All computation remains in the approved population graph regions. The only new
edge carries complete training state into the inference partitioner.
"""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import DependencyEdge
from sciona.nasa_population_graphs import build_population_graph


def build_complete_workflow_graph(active_query_populations):
    if type(active_query_populations) is not int or not 1<=active_query_populations<=10:
        raise ValueError('One through ten active query populations required')
    nodes=[];edges=[]
    for kind,count in [('training',10),('inference',active_query_populations)]:
        prefix=kind+'_'
        graph=build_population_graph(kind,count)
        for original in graph.nodes:
            node=original.model_copy(deep=True)
            node.node_id=prefix+node.node_id
            nodes.append(node)
        for original in graph.edges:
            edge=original.model_copy(deep=True)
            edge.source_id=prefix+edge.source_id;edge.target_id=prefix+edge.target_id
            edges.append(edge)
    edges.append(DependencyEdge(source_id='training_exhausted',target_id='inference_partition',
        output_name='result',input_name='model_states',source_type='dict',target_type='dict'))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(num_nodes=len(nodes),num_edges=len(edges),
        scope='Corrected source-derived ten-population training and complete query prediction with explicit state handoff.',
        active_query_populations=active_query_populations,
        final_node_id='inference_population_output',
        exclusions=['Historical backend identity','Empirical competition performance','Whole-graph static table propagation',
            'Source filesystem/notebook orchestration']))


def select_complete_workflow_graph(payload):
    """Validate active count before model execution; no hidden numerical work."""
    from sciona.atoms.ml.domain_adapters.airport_population import training_jobs
    from sciona.nasa_population_dispatch import partition_records
    jobs=training_jobs(payload['training_records'],payload['airports'],payload['maximum_errors'])
    if len(jobs)!=10:raise ValueError('Complete source workflow requires ten training populations')
    queries,_=partition_records(payload['prediction_records'],payload['airports'])
    return build_complete_workflow_graph(len(queries))
