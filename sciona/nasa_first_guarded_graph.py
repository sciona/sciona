"""Primary and baseline graph composition with bounded source fallback routing."""
import sciona.atoms.ml.tabular.available_features as features
from sciona.available_feature_contracts import contracts
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.nasa_first_primary_graph import build_nasa_first_primary_graph
from sciona.guarded_prediction_graphs import graph_descriptor, guarded_prediction


def build_nasa_first_baseline_graph():
    inputs, outputs = contracts('estimate_fallback', features.estimate_fallback)
    inputs, outputs = [IOSpec(**p) for p in inputs], [IOSpec(**p) for p in outputs]
    prepare = AlgorithmicNode(node_id='prepare',name='baseline_inputs',description='Expose observed-as-of baseline operands',
        concept_type='data_extraction',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.domain_adapters.first_place_fallback.baseline_inputs',
        inputs=[IOSpec(name='queries',type_desc='pd.DataFrame',constraints='Query identities and naive timestamps in original row order.'),
                IOSpec(name='raw_options',type_desc='dict',constraints='Runtime estimate history with per-record observation time.')],outputs=inputs)
    baseline = AlgorithmicNode(node_id='baseline',name='estimate_fallback',description='Observation-safe latest valid estimate fallback',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.tabular.available_features.estimate_fallback',inputs=inputs,outputs=outputs)
    edges = [DependencyEdge(source_id='prepare',target_id='baseline',output_name=p.name,input_name=p.name,
                            source_type=p.type_desc,target_type=p.type_desc) for p in inputs]
    return CDGExport(nodes=[prepare,baseline],edges=edges,metadata=dict(publication_status='draft',
        scope='Latest observed valid estimate minus source offset with fractional results, bounds and missing default.'))


def branch_descriptors():
    primary = build_nasa_first_primary_graph()
    primary.nodes.append(AlgorithmicNode(node_id='values',name='prediction_values',description='Primary vector for guarded output validation',
        concept_type='data_extraction',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.domain_adapters.first_place_fallback.prediction_values',
        inputs=[IOSpec(name='predictions',type_desc='pd.DataFrame',constraints='Primary query table with aligned predictions.')],
        outputs=[IOSpec(name='values',type_desc='NDArray[np.int64]',constraints='Owned positional primary prediction vector in minutes.')]))
    primary.edges.append(DependencyEdge(source_id='format',target_id='values',output_name='predictions',input_name='predictions',
                                       source_type='pd.DataFrame',target_type='pd.DataFrame'))
    return graph_descriptor(primary,'values','values'), graph_descriptor(build_nasa_first_baseline_graph(),'baseline','result')


async def execute_nasa_first_prediction_graph(queries, models, population, *, raw_options,
        feature_columns, numeric_fills, categorical_fills):
    primary, baseline = branch_descriptors()
    inputs = dict(queries=queries,models=models,population=population,raw_options=raw_options,
        feature_columns=feature_columns,numeric_fills=numeric_fills,categorical_fills=categorical_fills,integer_dtype='int16')
    values, route = await guarded_prediction(primary,baseline,inputs,len(queries),30.)
    result = queries.copy(deep=True)
    result['minutes_until_pushback'] = values
    return result, route


def build_nasa_first_guarded_graph():
    """Outer control graph; branch envelopes remain explicit in graph metadata."""
    def p(name, typ, contract):
        return IOSpec(name=name,type_desc=typ,constraints=contract)
    inputs = [p('queries','pd.DataFrame','Caller query identities in original order; empty and duplicate rows supported.'),
        p('models','dict','Local slots 0/1/2 and shared global model; invalid or missing models may select baseline.'),
        p('population','str','Explicit source population label.'), p('raw_options','dict','Explicit raw tables and availability/policy context.'),
        p('feature_columns','list','Ordered model features.'), p('numeric_fills','dict','Fixed numeric fills.'),
        p('categorical_fills','dict','Fixed category fills.')]
    control = [p('primary','dict','Hash-bound primary branch envelope.'),p('baseline','dict','Hash-bound baseline branch envelope.'),
        p('inputs','dict','Private caller-owned branch inputs.'),p('count','int','Nonnegative query row count.'),
        p('constant','float','Source constant fallback in minutes.')]
    values = p('values','np.ndarray','Finite aligned values in minutes; int64 primary, fractional float64 fallback, empty float64.')
    route = p('route','str','One of primary, baseline, constant, empty.')
    def n(identity, runtime, ins, outs):
        return AlgorithmicNode(node_id=identity,name=identity,description='Lazy prediction branch orchestration',concept_type='conditional_routing',
            status=NodeStatus.ATOMIC,matched_primitive=runtime,inputs=ins,outputs=outs)
    nodes = [n('inputs','sciona.atoms.ml.domain_adapters.first_place_fallback.execution_inputs',inputs,control),
        n('predict','sciona.atoms.ml.model_selection.graph_fallbacks.predict',control,[values,route]),
        n('format','sciona.atoms.ml.domain_adapters.first_place_prediction.format_predictions',[inputs[0],values],
            [p('predictions','pd.DataFrame','Owned input query table with positional prediction values.')])]
    edges = [DependencyEdge(source_id='inputs',target_id='predict',output_name=port.name,input_name=port.name,
                            source_type=port.type_desc,target_type=port.type_desc) for port in control]
    edges.append(DependencyEdge(source_id='predict',target_id='format',output_name='values',input_name='values',
                                source_type=values.type_desc,target_type=values.type_desc))
    primary, baseline = branch_descriptors()
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',primary_branch=primary,baseline_branch=baseline,
        scope='Complete single-population inference: lazy primary/baseline/constant and empty query handling.',
        exclusions=['Training graph', 'Native model serialization', 'Static symbolic propagation']))
