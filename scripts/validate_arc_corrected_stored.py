"""Execute the stored corrected ARC candidate through its exact catalog graph."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from scripts.import_flavours_provider_drafts import ROOT
from scripts.promote_arc_execution import sha
from scripts.review_arc_original_scope import review
from scripts import stage_arc_corrected_intake as intake
from sciona.catalog_arc_graph import build_catalog_arc_graph
from sciona.services.execution_graph_codec import encode_execution_graph


def validate():
    plan=intake.qualified_plan()
    with intake.connect() as db:
        db.execute("SET TRANSACTION READ ONLY")
        stored=intake.check(db,plan)
    scope=review()
    if scope!=json.loads((ROOT/'docs/reviews/arc_original_scope.json').read_text()):
        raise ValueError('Current original scope review required')
    previous=json.loads((ROOT/'docs/reviews/competition_arc_graph_execution.json').read_text())
    paths=list(previous['code_sha256'])+[
        '../sciona-atoms-ml/src/sciona/atoms/ml/arc_execution.py',
        'sciona/catalog_arc_graph.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py',
        'scripts/review_arc_original_scope.py','scripts/validate_catalog_arc_graph.py','scripts/validate_arc_corrected_stored.py',
        'scripts/stage_arc_corrected_intake.py','docs/reviews/arc_corrected_intake_plan.json',
        'tests/test_catalog_arc_graph.py','tests/test_arc_corrected_stored.py','tests/test_text_pair_provider_graphs.py',
        'docs/reviews/arc_original_scope.json','docs/reviews/arc_provider_contracts.json']
    before={p:sha(ROOT/p) for p in paths}
    digest,nodes,edges=encode_execution_graph(stored)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONPATH=str(ROOT),
        MPLCONFIGDIR='/private/tmp/sciona-mpl-cache')
    # A caller may supply a different location containing the exact reviewed bundle.
    env.setdefault('SCIONA_ARC_BUILD_DIR','/private/tmp/sciona_arc_reviewed_build_v1')
    with tempfile.TemporaryDirectory(prefix='synthetic-arc-catalog-') as temp:
        xml=Path(temp)/'results.xml'
        result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q','tests/test_arc_corrected_stored.py',
                               '--junitxml='+str(xml)],cwd=ROOT,env=env,capture_output=True,text=True)
        if result.returncode:
            raise ValueError('Synthetic graph execution failed: '+result.stdout[-5000:]+result.stderr[-1000:])
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=2,failures=0,errors=0,skipped=0):
        raise ValueError('Both complete solver cases required')
    if before!={p:sha(ROOT/p) for p in paths} or review()!=scope:
        raise ValueError('Qualification changed during execution')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if intake.check(db,plan)!=stored:
            raise ValueError('Stored graph changed during execution')
    return dict(format='arc-corrected-stored-execution.v1',stored_catalog_graph_executed=True,version_id=plan['version_id'],passed=True,approved=False,synthetic_only=True,
        catalog_mutations=0,graph_sha256=digest,nodes=len(nodes),edges=len(edges),tests=counts,
        full_solver_searches=12,all_nodes_uncached=True,exact_state_edge_identity=True,
        codec_roundtrip=True,software_sha256=before,
        resource_policy=dict(serial_cases=True,max_solver_children=2,native_threads=1),
        limits=['Two synthetic identity-transformation cases with one and two test inputs and all four adaptive phases.',
                'No independent accuracy, universal synthesis or cross-domain correctness claim.',
                'Corrected original is stored but not yet approved or selected from the served catalog.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/arc_corrected_stored_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=report['tests'],full_solver_searches=12,approved=False)))
