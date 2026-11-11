"""Qualify full synthetic lifecycles using the stored original candidate."""
import json
import os
import subprocess
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts import stage_resource_route_corrected_intake as intake


def validate():
    plan, providers = intake.qualified_plan()
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        graph = intake.check_candidate(db, plan, providers)
    baseline = json.loads((intake.ROOT/'docs/reviews/resource_route_current_runtime.json').read_text())
    paths = list(baseline['software_sha256']) + [
        'scripts/stage_resource_route_corrected_intake.py', 'scripts/resource_route_approved_provider_state.py',
        'scripts/validate_resource_route_corrected_execution.py', 'tests/test_resource_route_corrected_stored_execution.py',
        'sciona/catalog_resource_route_graph.py', 'docs/reviews/resource_route_corrected_intake_plan.json', 'docs/reviews/resource_route_approved_provider_state.json']
    before = {p:intake.sha(intake.ROOT/p) for p in paths}
    if any(before[p] != digest for p,digest in baseline['software_sha256'].items()):
        raise ValueError('Previously executed software changed')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/sciona-mpl-cache', PYTHONPATH=str(intake.ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-stored-resource_route-') as temporary:
        xml = Path(temporary)/'tests.xml'
        run = subprocess.run([str(intake.ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_resource_route_corrected_stored_execution.py', '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)],
            cwd=intake.ROOT, env=env, capture_output=True, text=True)
        if run.returncode:raise ValueError('Stored lifecycle validation failed: '+run.stdout[-5000:]+run.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts != dict(tests=4,failures=0,errors=0,skipped=0):raise ValueError('Four complete lifecycle cases required')
    if before != {p:intake.sha(intake.ROOT/p) for p in paths}:raise ValueError('Executed software changed')
    if intake.qualified_plan() != (plan, providers):raise ValueError('Qualification changed')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if intake.check_candidate(db, plan, providers) != graph:raise ValueError('Stored graph changed')
    return dict(format='resource_route-corrected-stored-execution.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, stored_catalog_graph_executed=True,
        artifact_id=plan['artifact_id'], version_id=plan['version_id'], graph_sha256=plan['graph_sha256'],
        tests=counts, software_sha256=before, graph_nodes=5, state_edges=4,
        exact_registered_bindings=True, all_nodes_uncached=True, full_configured_lifecycle_parity=True,
        resource_policy=dict(serial_tests=True,native_threads=1),
        limits=['Four synthetic route lifecycles: optimal, infeasible, search limit and identity route; no historical accuracy claim.',
                'Original CDG publication and served selection remain pending.'])


if __name__ == '__main__':
    result = validate()
    (intake.ROOT/'docs/reviews/resource_route_corrected_intake_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=result['tests'],stored_graph=True,approved=False)))
