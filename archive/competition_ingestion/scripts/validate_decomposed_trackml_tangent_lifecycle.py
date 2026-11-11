"""Check twenty-four numeric providers including helix direction across the retained lifecycle."""
import contextlib
import hashlib
import io
import inspect
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
    previous=json.loads((root/'docs/reviews/competition_trackml_lifecycle.json').read_text())
    pins=json.loads((root/'docs/reviews/competition_trackml_source_pins.json').read_text())
    vendor=root.parent/'sciona-atoms-physics/src/sciona/atoms/particle_tracking/source_tracking'
    paths=['scripts/validate_decomposed_trackml_tangent_lifecycle.py','scripts/audit_decomposed_trackml_readiness.py',
           'sciona/visualizer/runner.py','sciona/services/catalog_artifact_retrieval.py',
           'sciona/services/execution_graph_codec.py']
    paths += ['sciona/'+name+'.py' for name in ('decomposed_trackml_geometry','decomposed_trackml_lifecycle',
                                               'decomposed_trackml_nearest','decomposed_trackml_projection_lifecycle','decomposed_trackml_coordinate_lifecycle','decomposed_trackml_intersections','decomposed_trackml_correction_lifecycle','decomposed_trackml_corrections','decomposed_trackml_neighbor_lifecycle','decomposed_trackml_bayesian_lifecycle','decomposed_trackml_commit_lifecycle','decomposed_trackml_detector_lifecycle','decomposed_trackml_residual_lifecycle','decomposed_trackml_grid_lifecycle','decomposed_trackml_direction_lifecycle','decomposed_trackml_tangent_lifecycle')]
    for name,digest in previous['implementation_sha256'].items():
        if readiness.sha(root/name)!=digest:
            raise ValueError('Unrelated qualified implementation changed: '+name)
    paths=list(dict.fromkeys(paths+list(previous['implementation_sha256'])))
    before={p:readiness.sha(root/p) for p in paths}
    with readiness.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(state['derivative']['fqdn'],)).fetchone()['d']
        graph=_artifact_document_to_cdg(document,version_id=readiness.DERIVATIVE,
            content_hash=state['derivative']['content_hash'],require_execution_envelope=True)
    if encode_execution_graph(graph)[0]!=previous['serialized_graph_sha256']:
        raise ValueError('Served graph changed')
    with tempfile.TemporaryDirectory(prefix='trackml-public-source-') as directory:
        source=Path(directory)
        for name in ('geometry.py','neighbors.py','corrections.py','candidates.py','cells.py','algorithm.py'):
            relative='trackml_solution/'+name
            data=(vendor/('solution_'+relative.replace('/','_')+'.source')).read_bytes()
            if hashlib.sha256(data).hexdigest()!=pins['files'][relative]:raise ValueError('Retained source pin differs')
            target=source/relative;target.parent.mkdir(exist_ok=True);target.write_bytes(data)
        from sciona import trackml_payload as payload
        from sciona.atoms.particle_tracking.helix_geometry import atoms
        provider_path=inspect.getfile(inspect.unwrap(atoms.circle_from_three_points))
        provider_hash=readiness.sha(provider_path)
        counts=dict(circle=0,two=0,three=0,nearest=0,cylinder=0,plane=0,cylinder_intersection=0,plane_intersection=0,cap_correction=0,cylinder_correction=0,search_fit=0,search_k=0,search_radius=0,decision=0,commit=0,kmeans=0,mean_shift=0,displacement=0,uncertainty=0,grid_fit=0,grid_evaluate=0,direction=0,construction=0,tangent=0)
        from sciona.atoms.physics import coordinate_projection as projection
        projection_path=inspect.getfile(projection)
        projection_hash=readiness.sha(projection_path)
        from sciona.atoms.physics.helix_cylinder import helix_cylinder_intersection
        from sciona.atoms.particle_tracking.track_matching.atoms import helix_cap_intersection, bayesian_neighbor_evaluation, greedy_track_commit
        intersection_paths = {name: inspect.getfile(inspect.unwrap(fn)) for name, fn in
            [('cylinder', helix_cylinder_intersection), ('plane', helix_cap_intersection)]}
        intersection_hashes = {name: readiness.sha(path) for name, path in intersection_paths.items()}
        from sciona.atoms.particle_tracking.detector_corrections.atoms import perturbative_cap_correction, perturbative_cylinder_correction
        correction_path = inspect.getfile(inspect.unwrap(perturbative_cap_correction))
        correction_hash = readiness.sha(correction_path)
        from sciona.atoms.ml.sklearn.neighbors.fitted_search import fit_euclidean_search, query_euclidean_k, query_euclidean_radius
        search_path = inspect.getfile(inspect.unwrap(fit_euclidean_search))
        search_hash = readiness.sha(search_path)
        def observed(name,fn):
            def call(*args,**kwargs):
                counts[name]+=1
                return fn(*args,**kwargs)
            return call
        from sciona.decomposed_trackml_tangent_lifecycle import execute
        from sciona.atoms.physics.helix_tangent import construct_helix_from_tangent, oriented_helix_tangent
        tangent_path = inspect.getfile(inspect.unwrap(construct_helix_from_tangent))
        tangent_hash = readiness.sha(tangent_path)
        from sciona.atoms.physics.regular_grid import fit_linear_grid, evaluate_linear_grid
        grid_path = inspect.getfile(inspect.unwrap(fit_linear_grid))
        grid_hash = readiness.sha(grid_path)
        from sciona.atoms.physics.tangent_residuals import hemisphere_tangent_displacement, combine_scaled_uncertainty
        residual_path = inspect.getfile(inspect.unwrap(hemisphere_tangent_displacement))
        residual_hash = readiness.sha(residual_path)
        from sciona.atoms.ml.sklearn.cluster.lloyd_fit import fit_lloyd_clusters
        from sciona.atoms.ml.sklearn.cluster.atoms import mean_shift
        clustering_paths = {name: inspect.getfile(inspect.unwrap(fn)) for name, fn in
            [("kmeans", fit_lloyd_clusters), ("mean_shift", mean_shift)]}
        clustering_hashes = {name: readiness.sha(path) for name, path in clustering_paths.items()}
        providers=dict(circle_function=observed('circle',atoms.circle_from_three_points),
                       pitch_two_function=observed('two',atoms.helix_pitch_from_two_points),
                       pitch_three_function=observed('three',atoms.helix_pitch_least_squares),
                       nearest_function=observed('nearest',atoms.helix_nearest_point_distance),
                       cylinder_function=observed('cylinder',projection.cylindrical_projection),
                       plane_function=observed('plane',projection.plane_projection),
                       cylinder_intersection_function=observed('cylinder_intersection',helix_cylinder_intersection),
                       plane_intersection_function=observed('plane_intersection',helix_cap_intersection),
                       cap_correction_function=observed('cap_correction',perturbative_cap_correction),
                       cylinder_correction_function=observed('cylinder_correction',perturbative_cylinder_correction),
                       search_fit_function=observed('search_fit',fit_euclidean_search),
                       search_k_function=observed('search_k',query_euclidean_k),
                       search_radius_function=observed('search_radius',query_euclidean_radius),
                       decision_function=observed('decision',bayesian_neighbor_evaluation),
                       commit_function=observed('commit',greedy_track_commit),
                       kmeans_function=observed('kmeans',fit_lloyd_clusters),
                       mean_shift_function=observed('mean_shift',mean_shift),
                       displacement_function=observed('displacement',hemisphere_tangent_displacement),
                       uncertainty_function=observed('uncertainty',combine_scaled_uncertainty),
                       grid_fit_function=observed('grid_fit',fit_linear_grid),
                       grid_evaluate_function=observed('grid_evaluate',evaluate_linear_grid),
                       direction_function=observed('direction',atoms.helix_direction_from_two_points),
                       construction_function=observed('construction',construct_helix_from_tangent),
                       tangent_function=observed('tangent',oriented_helix_tangent))
        def injected(*args,**kwargs):return execute(*args,**kwargs,**providers)
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()), \
             patch.object(comparison,'build_trackml_graph',return_value=graph), \
             patch.object(comparison,'execute_tracking',injected), \
             patch.object(payload,'execute_tracking',injected):
            result=comparison.validate(root,source)
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()), \
             patch.object(comparison,'build_trackml_graph',return_value=graph):
            default=comparison.validate(root,source)
        if default!=result:raise ValueError('Injected and default lifecycle results differ')
        if clustering_hashes != {name: readiness.sha(path) for name, path in clustering_paths.items()}:raise ValueError('Clustering provider changed')
        if readiness.sha(tangent_path)!=tangent_hash:raise ValueError('Tangent provider changed')
        if readiness.sha(grid_path)!=grid_hash:raise ValueError('Grid provider changed')
        if readiness.sha(residual_path)!=residual_hash:raise ValueError('Residual provider changed')
        if readiness.sha(search_path)!=search_hash:raise ValueError('Search provider changed')
        if readiness.sha(correction_path)!=correction_hash:raise ValueError('Correction provider changed')
        if intersection_hashes != {name: readiness.sha(path) for name, path in intersection_paths.items()}:raise ValueError('Intersection provider changed')
        if readiness.sha(provider_path)!=provider_hash or readiness.sha(projection_path)!=projection_hash:raise ValueError('Reused provider changed')
    expected=dict(previous);actual=dict(result)
    expected.pop('implementation_sha256');actual.pop('implementation_sha256')
    if actual!=expected:raise ValueError('Full retained lifecycle evidence differs')
    if not all(counts.values()):raise ValueError('All registered geometry providers must execute')
    if readiness.audit()!=state or before!={p:readiness.sha(root/p) for p in paths}:
        raise ValueError('Baseline changed during execution')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        stored_graph_executed=True,default_path_equal=True,provider_sha256=provider_hash,projection_provider_sha256=projection_hash,intersection_provider_sha256=intersection_hashes,correction_provider_sha256=correction_hash,search_provider_sha256=search_hash,clustering_provider_sha256=clustering_hashes,residual_provider_sha256=residual_hash,grid_provider_sha256=grid_hash,tangent_provider_sha256=tangent_hash,registered_provider_calls=counts,checks=result['checks'],graph_sha256=result['serialized_graph_sha256'],
        implementation_sha256=before,retained_report_sha256=readiness.sha(root/'docs/reviews/competition_trackml_lifecycle.json'),
        limits=['Exercises the existing stored graph with invocation-local numeric injection and separately checks the unchanged default path; new serialized callable bindings and original eleven-stage publication remain pending.',
                'Synthetic analytic maps exercise correction application; learned calibration and historical accuracy remain unqualified.'])


if __name__=='__main__':
    result=validate()
    (readiness.ROOT/'docs/reviews/decomposed_trackml_tangent_lifecycle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
