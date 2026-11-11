"""Validate explicit callable edges through the native runner against retained source."""
import contextlib
import hashlib
import io
import importlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts import audit_decomposed_trackml_readiness as readiness
from scripts import validate_trackml_lifecycle as comparison
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph


def validate():
    root=readiness.ROOT
    state=readiness.audit()
    if state['implementation_drift']:raise ValueError('Retained implementation changed')
    previous=json.loads((root/'docs/reviews/competition_trackml_lifecycle.json').read_text())
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    vendor=root.parent/'sciona-atoms-physics/src/sciona/atoms/particle_tracking/source_tracking'
    paths=['scripts/validate_decomposed_trackml_tangent_graph.py','scripts/audit_decomposed_trackml_readiness.py',
           'sciona/visualizer/runner.py','sciona/services/catalog_artifact_retrieval.py',
           'sciona/services/execution_graph_codec.py']
    before={p:readiness.sha(root/p) for p in paths}
    from sciona.decomposed_trackml_tangent_graph import build_decomposed_trackml_tangent_graph
    graph=build_decomposed_trackml_tangent_graph()
    from sciona.atoms.physics import decomposed_trackml_tangent_execution as provider
    import inspect
    provider_hash=readiness.sha(inspect.getfile(provider))
    paths += ['sciona/decomposed_trackml_graph.py','sciona/decomposed_trackml_lifecycle.py','sciona/decomposed_trackml_geometry.py']
    paths += ['sciona/decomposed_trackml_projection_graph.py','sciona/decomposed_trackml_projection_lifecycle.py',
              'sciona/decomposed_trackml_nearest.py','sciona/decomposed_trackml_correction_graph.py',
              'sciona/decomposed_trackml_correction_lifecycle.py','sciona/decomposed_trackml_intersections.py',
              'sciona/decomposed_trackml_coordinate_graph.py','sciona/decomposed_trackml_coordinate_lifecycle.py',
              'sciona/decomposed_trackml_intersection_graph.py','sciona/decomposed_trackml_corrections.py',
              'sciona/decomposed_trackml_neighbor_graph.py','sciona/decomposed_trackml_neighbor_lifecycle.py',
              'sciona/decomposed_trackml_bayesian_graph.py','sciona/decomposed_trackml_bayesian_lifecycle.py',
              'sciona/decomposed_trackml_commit_graph.py','sciona/decomposed_trackml_commit_lifecycle.py',
              'sciona/decomposed_trackml_detector_graph.py','sciona/decomposed_trackml_detector_lifecycle.py',
              'sciona/decomposed_trackml_residual_graph.py','sciona/decomposed_trackml_residual_lifecycle.py',
              'sciona/decomposed_trackml_grid_graph.py','sciona/decomposed_trackml_grid_lifecycle.py',
              'sciona/decomposed_trackml_direction_graph.py','sciona/decomposed_trackml_direction_lifecycle.py',
              'sciona/decomposed_trackml_tangent_graph.py','sciona/decomposed_trackml_tangent_lifecycle.py']
    before={p:readiness.sha(root/p) for p in paths}
    runner=comparison.runner
    original_execute=runner.CDGExecutionSession.execute
    traces=[];bindings=[]
    from sciona import decomposed_trackml_tangent_lifecycle as lifecycle
    from sciona.atoms.particle_tracking.helix_geometry import atoms
    from sciona.atoms.physics.coordinate_projection import cylindrical_projection, plane_projection
    from sciona.atoms.physics.helix_cylinder import helix_cylinder_intersection
    from sciona.atoms.particle_tracking.track_matching.atoms import helix_cap_intersection, bayesian_neighbor_evaluation, greedy_track_commit
    from sciona.atoms.particle_tracking.detector_corrections.atoms import perturbative_cap_correction, perturbative_cylinder_correction
    from sciona.atoms.ml.sklearn.neighbors.fitted_search import fit_euclidean_search, query_euclidean_k, query_euclidean_radius
    from sciona.atoms.ml.sklearn.cluster.lloyd_fit import fit_lloyd_clusters
    from sciona.atoms.ml.sklearn.cluster.atoms import mean_shift
    from sciona.atoms.physics.tangent_residuals import hemisphere_tangent_displacement, combine_scaled_uncertainty
    from sciona.atoms.physics.regular_grid import fit_linear_grid, evaluate_linear_grid
    from sciona.atoms.physics.helix_tangent import construct_helix_from_tangent, oriented_helix_tangent
    expected_functions=dict(circle_function=atoms.circle_from_three_points,
        pitch_two_function=atoms.helix_pitch_from_two_points,pitch_three_function=atoms.helix_pitch_least_squares,
        nearest_function=atoms.helix_nearest_point_distance,
        cylinder_function=cylindrical_projection,plane_function=plane_projection,
        cylinder_intersection_function=helix_cylinder_intersection,plane_intersection_function=helix_cap_intersection,
        cap_correction_function=perturbative_cap_correction,cylinder_correction_function=perturbative_cylinder_correction,
        search_fit_function=fit_euclidean_search,search_k_function=query_euclidean_k,search_radius_function=query_euclidean_radius,
        decision_function=bayesian_neighbor_evaluation,commit_function=greedy_track_commit,
        kmeans_function=fit_lloyd_clusters,mean_shift_function=mean_shift,displacement_function=hemisphere_tangent_displacement,
        uncertainty_function=combine_scaled_uncertainty,grid_fit_function=fit_linear_grid,
        grid_evaluate_function=evaluate_linear_grid, direction_function=atoms.helix_direction_from_two_points,
        construction_function=construct_helix_from_tangent,tangent_function=oriented_helix_tangent)
    run=lifecycle.execute
    def exact_binding(*args,**kwargs):
        if {key:kwargs.get(key) for key in expected_functions}!=expected_functions:
            raise ValueError('Runner must pass exact registered geometry callables')
        bindings.append(True)
        return run(*args,**kwargs)
    provider_files={}
    for node in graph.nodes:
        module,symbol=node.matched_primitive.rsplit('.',1)
        provider_files[node.matched_primitive]=inspect.getfile(inspect.unwrap(getattr(importlib.import_module(module),symbol)))
    provider_hashes={name:readiness.sha(path) for name,path in provider_files.items()}
    async def checked(session,*args,**kwargs):
        result=await original_execute(session,*args,**kwargs)
        trace={t['node_id']:t for t in result['trace']}
        if result['status']!='completed' or set(trace)!={n.node_id for n in graph.nodes}:
            raise ValueError('Complete draft graph must execute')
        if any(t['cached'] for t in trace.values()):raise ValueError('Uncached execution required')
        if not all(trace[n].get('callable_reference') is True for n in ('circle','pitch_two','pitch_three','nearest','cylinder','plane','cylinder_intersection','plane_intersection','cap_correction','cylinder_correction','search_fit','search_k','search_radius','decision','commit','kmeans','mean_shift','displacement','uncertainty','grid_fit','grid_evaluate','direction','construction','tangent')):
            raise ValueError('Explicit numeric callable references required')
        traces.append(True)
        return result
    with tempfile.TemporaryDirectory(prefix='trackml-public-source-') as directory:
        source=Path(directory)
        for name in ('geometry.py','neighbors.py','corrections.py','candidates.py','cells.py','algorithm.py'):
            relative='trackml_solution/'+name
            data=(vendor/('solution_'+relative.replace('/','_')+'.source')).read_bytes()
            if hashlib.sha256(data).hexdigest()!=pins['files'][relative]:raise ValueError('Retained source pin differs')
            target=source/relative;target.parent.mkdir(exist_ok=True);target.write_bytes(data)
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()), \
             patch.object(comparison,'build_trackml_graph',return_value=graph), \
             patch.object(runner.CDGExecutionSession,'execute',checked), \
             patch.object(lifecycle,'execute',exact_binding):
            result=comparison.validate(root,source)
    for key in ('checks','implementation_sha256','limitations','provider_sha256'):
        if result[key]!=previous[key]:raise ValueError('Full retained lifecycle evidence differs: '+key)
    if len(bindings)!=4 or len(traces)!=4 or (result['graph_nodes'],result['graph_edges'])!=(27,26):
        raise ValueError('Four complete twenty-seven-node executions required')
    if readiness.sha(inspect.getfile(provider))!=provider_hash or provider_hashes!={name:readiness.sha(path) for name,path in provider_files.items()}:raise ValueError('Provider changed')
    if readiness.audit()!=state or before!={p:readiness.sha(root/p) for p in paths}:
        raise ValueError('Baseline changed during execution')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        draft_graph_executed=True,exact_registered_bindings=True,all_provider_sha256=provider_hashes,all_nodes_uncached=True,explicit_callable_edges=24,provider_sha256=provider_hash,checks=result['checks'],graph_sha256=result['serialized_graph_sha256'],
        implementation_sha256=before,retained_report_sha256=readiness.sha(root/'docs/reviews/competition_trackml_lifecycle.json'),
        limits=['Complete tracking behavior with explicit geometry injection; remaining decomposition and canonical provider qualification precede original publication.',
                'Synthetic analytic maps exercise correction application; learned calibration and historical accuracy remain unqualified.'])


if __name__=='__main__':
    result=validate()
    (readiness.ROOT/'docs/reviews/decomposed_trackml_tangent_graph_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
