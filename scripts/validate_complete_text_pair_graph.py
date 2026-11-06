"""Qualify the draft lifecycle and inventory its exact approved numeric dependencies."""
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.text_pair_complete_graph import build_complete_text_pair_graph, STAGES, adapters
from scripts.validate_complete_text_pair_providers import ROOT, ML, sha
from scripts.review_video_numeric_runtime import review as runtime_review
from scripts.import_flavours_provider_drafts import connect
from scripts.promote_text_pair_providers import qualify_plan, check as check_text_providers


DEPENDENCIES = {
    'sciona.atoms.ml.text_nlp.atoms.clean_text',
    'sciona.atoms.ml.text_nlp.atoms.levenshtein',
    'sciona.atoms.ml.text_nlp.bm25.score_bm25',
    'sciona.atoms.ml.sklearn.isotonic.atoms.isotonic_regression',
    'sciona.atoms.ml.sklearn.isotonic.atoms.isotonic_regression_fit',
    'sciona.atoms.ml.sklearn.isotonic.atoms.isotonic_regression_transform',
    'sciona.atoms.ml.sklearn.decomposition.sparse_projection.fit_sparse_projection',
    'sciona.atoms.ml.sklearn.decomposition.sparse_projection.transform_sparse_projection',
    'sciona.atoms.ml.model_selection.materialized_prediction.average',
    'sciona.atoms.ml.calibration.empirical_f1.fit_empirical_f1_thresholds',
    'sciona.atoms.ml.calibration.empirical_f1.apply_column_thresholds',
}


def provider_inventory():
    functions = {getattr(adapters, name).__module__ + '.' + name: getattr(adapters, name) for name in STAGES}
    specs = {}
    for spec in _parse_registered_atoms(repo=ProviderRepo(ML.name, ML), artifact_root=ML/'src/sciona/atoms'):
        name = spec.import_module + '.' + spec.source_symbol
        if name in DEPENDENCIES | set(functions):
            if name in specs:
                raise ValueError('Ambiguous provider source identity')
            specs[name] = spec
    if set(specs) != DEPENDENCIES | set(functions):
        raise ValueError('Missing adapter or numerical provider source')
    adapter_records = [dict(runtime_fqdn=name, catalog_fqdn=specs[name].fqdn,
        version_id=str(specs[name].version_id), content_hash=specs[name].content_hash,
        signature=str(inspect.signature(fn))) for name, fn in sorted(functions.items())]
    dependencies = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        check_text_providers(db, qualify_plan(), published=True)
        for name in sorted(DEPENDENCIES):
            spec = specs[name]
            pair = []
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                rows = db.execute(f'''SELECT a.{key}::text AS artifact_id,a.fqdn,v.version_id::text,
                    v.content_hash,v.trust_tier FROM {prefix}s a JOIN {prefix}_versions v USING({key})
                    WHERE a.fqdn=%s AND v.is_latest AND a.status='approved' AND a.is_publishable
                    AND {prefix}_is_publishable(a.{key})''', (spec.fqdn,)).fetchall()
                if len(rows) != 1:
                    raise ValueError('Exact approved native provider required: '+name)
                row = rows[0]
                if row['version_id'] != str(spec.version_id) or row['content_hash'] != spec.content_hash or row['trust_tier'] != 3:
                    raise ValueError('Approved provider source/version differs: '+name)
                if not db.execute(f'SELECT 1 FROM catalog_{prefix}s_served WHERE {key}=%s', (row['artifact_id'],)).fetchone():
                    raise ValueError('Dependency is not served: '+name)
                pair.append(row)
            if pair[0] != pair[1]:
                raise ValueError('Legacy and canonical dependency identities differ')
            if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (pair[0]['version_id'],)).fetchone():
                raise ValueError('Failed dependency evidence')
            dependencies.append(dict(runtime_fqdn=name, **pair[0]))
    return adapter_records, dependencies


def validate():
    prior_path = 'docs/reviews/complete_text_pair_provider_qualification.json'
    raw = (ROOT/prior_path).read_bytes()
    if raw != subprocess.check_output(['git', 'show', '4f1ef5e5:'+prior_path], cwd=ROOT):
        raise ValueError('Immutable recipe qualification changed')
    prior = json.loads(raw)
    for path, digest in prior['software_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Previously qualified recipe source changed: '+path)
    paths = sorted(set(prior['software_sha256']) | {prior_path,
        'sciona/text_pair_complete_stages.py', 'sciona/text_pair_complete_graph.py',
        'tests/test_text_pair_complete_graph.py', 'tests/test_text_pair_provider_graphs.py',
        'scripts/validate_complete_text_pair_graph.py', 'sciona/services/execution_graph_codec.py',
        'sciona/visualizer/runner.py', '../sciona-atoms-ml/src/sciona/atoms/ml/text_pair_complete_execution.py'})
    before = {p: sha(ROOT/p) for p in paths}
    runtime = runtime_review()
    if runtime != prior['runtime']:
        raise ValueError('Qualified numeric runtime changed')
    adapter_records, dependencies = provider_inventory()
    graph_hash = encode_execution_graph(build_complete_text_pair_graph())[0]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-complete-pair-') as temp:
        xml = Path(temp)/'tests.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q',
            'tests/test_text_pair_complete_graph.py', '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)],
            cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('Native lifecycle failed: '+result.stdout[-6000:]+result.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=5, failures=0, errors=0, skipped=0):
        raise ValueError('All five lifecycle cases required')
    if before != {p: sha(ROOT/p) for p in paths} or runtime != runtime_review():
        raise ValueError('Source/runtime changed during qualification')
    if (adapter_records, dependencies) != provider_inventory():
        raise ValueError('Provider catalog changed during qualification')
    return dict(format='complete-text-pair-graph-qualification.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, tests=counts, native_executions=3,
        adapters=adapter_records, approved_dependencies=dependencies, graph_sha256=graph_hash,
        software_sha256=before, runtime=runtime, resource_policy=dict(serial=True, native_threads=1),
        limits=['Draft recipe adapters; no original CDG or adapter catalog approval.',
            'Three serialized native executions match the independently invoked complete recipe exactly.',
            'Concrete execution-local states only; arbitrary forged or concurrently mutated intermediates are unqualified.',
            'Eleven registered numerical/text dependencies are approved, source-matched and served in both catalogs.',
            'Local set arithmetic, vector distances, feature assembly and sklearn model/vectorizer policy remain recipe implementation.',
            'The prior 115-test qualification is source-checked, not rerun.',
            'No independent accuracy, abstract-witness equivalence or named competition submission claim.'])


if __name__ == '__main__':
    result = validate()
    (ROOT/'docs/reviews/text_pair_complete_graph_qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], native_executions=result['native_executions'],
        adapters=len(result['adapters']), approved_dependencies=len(result['approved_dependencies']), approved=False)))
