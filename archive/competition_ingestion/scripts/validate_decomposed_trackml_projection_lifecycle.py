"""Check explicit circle, pitch and nearest-point providers across retained lifecycle evidence."""
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
    paths=['scripts/validate_decomposed_trackml_projection_lifecycle.py','scripts/audit_decomposed_trackml_readiness.py',
           'sciona/visualizer/runner.py','sciona/services/catalog_artifact_retrieval.py',
           'sciona/services/execution_graph_codec.py']
    paths += ['sciona/'+name+'.py' for name in ('decomposed_trackml_geometry','decomposed_trackml_lifecycle',
                                               'decomposed_trackml_nearest','decomposed_trackml_projection_lifecycle')]
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
        counts=dict(circle=0,two=0,three=0,nearest=0)
        def observed(name,fn):
            def call(*args,**kwargs):
                counts[name]+=1
                return fn(*args,**kwargs)
            return call
        from sciona.decomposed_trackml_projection_lifecycle import execute
        providers=dict(circle_function=observed('circle',atoms.circle_from_three_points),
                       pitch_two_function=observed('two',atoms.helix_pitch_from_two_points),
                       pitch_three_function=observed('three',atoms.helix_pitch_least_squares),
                       nearest_function=observed('nearest',atoms.helix_nearest_point_distance))
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
        if readiness.sha(provider_path)!=provider_hash:raise ValueError('Reused provider changed')
    expected=dict(previous);actual=dict(result)
    expected.pop('implementation_sha256');actual.pop('implementation_sha256')
    if actual!=expected:raise ValueError('Full retained lifecycle evidence differs')
    if not all(counts.values()):raise ValueError('All registered geometry providers must execute')
    if readiness.audit()!=state or before!={p:readiness.sha(root/p) for p in paths}:
        raise ValueError('Baseline changed during execution')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        stored_graph_executed=True,default_path_equal=True,provider_sha256=provider_hash,registered_provider_calls=counts,checks=result['checks'],graph_sha256=result['serialized_graph_sha256'],
        implementation_sha256=before,retained_report_sha256=readiness.sha(root/'docs/reviews/competition_trackml_lifecycle.json'),
        limits=['Exercises the existing stored graph with invocation-local numeric injection and separately checks the unchanged default path; new serialized callable bindings and original eleven-stage publication remain pending.',
                'Synthetic analytic maps exercise correction application; learned calibration and historical accuracy remain unqualified.'])


if __name__=='__main__':
    result=validate()
    (readiness.ROOT/'docs/reviews/decomposed_trackml_projection_lifecycle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
