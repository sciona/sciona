"""Complete March reconstruction with externally bound calibration and reduction."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.march_graph import build_march_graph


def build_decomposed_march_graph():
    graph = build_march_graph()
    def node(name, description, inputs, outputs, fqdn=None):
        return AlgorithmicNode(node_id=name, name=name, description=description,
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=fqdn or 'sciona.atoms.ml.decomposed_march_execution.'+name,
            inputs=[IOSpec(name=n, type_desc=t, required=n!='sample_weight',
                           default_value_repr='None' if n=='sample_weight' else '') for n,t in inputs],
            outputs=[IOSpec(name=n, type_desc=t) for n,t in outputs])
    calibration = 'sciona.atoms.ml.sklearn.calibration.atoms.'
    graph.nodes = [graph.nodes[0],
        node('features','Aggregate complete season efficiencies, graph ranks and Elo ratings',
             [('prepared','object')],[('featured','dict')]),
        node('pairings','Build chronological mirrored fitting and calibration populations',
             [('featured','dict')],[('arrays','dict')]),
        node('fit_scores','Fit standardized binary logistic scores from numeric matrices',
             [('arrays','dict')],[('calibration_scores','ndarray'),('labels','ndarray'),('sample_weight','object'),
                                 ('forward_scores','ndarray'),('reverse_scores','ndarray'),('counts','dict')]),
        node('calibrate','Fit a shared held-out sigmoid without refitting the base classifier',
             [('predictions','ndarray'),('y','ndarray'),('sample_weight','object')],[('state','object')],
             calibration+'sigmoid_calibration_fit'),
        node('forward','Calibrate forward prediction scores',
             [('predictions','ndarray'),('state','object')],[('probabilities','ndarray')],calibration+'sigmoid_calibration_predict'),
        node('reverse','Calibrate reverse prediction scores with the same fitted sigmoid',
             [('predictions','ndarray'),('state','object')],[('probabilities','ndarray')],calibration+'sigmoid_calibration_predict'),
        node('projection_inputs','Prepare forward plus one minus reverse in fixed arithmetic order',
             [('forward','ndarray'),('reverse','ndarray')],[('values','ndarray'),('coefficients','ndarray')]),
        node('sum','Execute generic ordered numeric reduction',
             [('values','ndarray'),('coefficients','ndarray')],[('total','ndarray')],
             'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'),
        node('result','Scale the external sum and restore complete strict JSON result',
             [('total','ndarray'),('counts','dict')],[('result','dict')])]
    links = [
        ('prepare','features','prepared','prepared','object'),
        ('features','pairings','featured','featured','dict'),
        ('pairings','fit_scores','arrays','arrays','dict'),
        ('fit_scores','calibrate','calibration_scores','predictions','ndarray'),
        ('fit_scores','calibrate','labels','y','ndarray'),
        ('fit_scores','calibrate','sample_weight','sample_weight','object'),
        ('fit_scores','forward','forward_scores','predictions','ndarray'),
        ('fit_scores','reverse','reverse_scores','predictions','ndarray'),
        ('calibrate','forward','state','state','object'),
        ('calibrate','reverse','state','state','object'),
        ('forward','projection_inputs','probabilities','forward','ndarray'),
        ('reverse','projection_inputs','probabilities','reverse','ndarray'),
        ('projection_inputs','sum','values','values','ndarray'),
        ('projection_inputs','sum','coefficients','coefficients','ndarray'),
        ('sum','result','total','total','ndarray'),
        ('fit_scores','result','counts','counts','dict')]
    graph.edges = [DependencyEdge(source_id=s,target_id=t,output_name=o,input_name=i,source_type=k,target_type=k)
                   for s,t,o,i,k in links]
    graph.metadata.update(num_nodes=10,num_edges=16,
        catalog_output_sources={'result':dict(node_id='result',output_name='result')},
        reuse_boundary='Dense standardized score fitting accepts arbitrary positive feature widths. Sigmoid fit/predict and ordered reduction are separate generic providers. Season features, chronology, mirrored pairings and orientation projection remain explicit recipe policy.',
        qualification_status='Full registered graph execution and corrected original publication pending.')
    return graph
