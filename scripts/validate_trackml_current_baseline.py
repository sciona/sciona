"""Recheck the served tracking lifecycle against retained pinned source software."""
import contextlib
import hashlib
import io
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
    paths=['scripts/validate_trackml_current_baseline.py','scripts/audit_decomposed_trackml_readiness.py',
           'sciona/visualizer/runner.py','sciona/services/catalog_artifact_retrieval.py',
           'sciona/services/execution_graph_codec.py']
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
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()), \
             patch.object(comparison,'build_trackml_graph',return_value=graph):
            result=comparison.validate(root,source)
    if result!=previous:raise ValueError('Full retained lifecycle evidence differs')
    if readiness.audit()!=state or before!={p:readiness.sha(root/p) for p in paths}:
        raise ValueError('Baseline changed during execution')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        stored_graph_executed=True,checks=result['checks'],graph_sha256=result['serialized_graph_sha256'],
        implementation_sha256=before,retained_report_sha256=readiness.sha(root/'docs/reviews/competition_trackml_lifecycle.json'),
        limits=['Revalidates the existing three-node tracking derivative, not original eleven-stage publication or reusable decomposition.',
                'Synthetic analytic maps exercise correction application; learned calibration and historical accuracy remain unqualified.'])


if __name__=='__main__':
    result=validate()
    (readiness.ROOT/'docs/reviews/trackml_current_baseline.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
