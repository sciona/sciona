"""Source-bound runtime evidence and read-only provider inventory for tabular promotion."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.competition_import import _canonical
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.tabular_complete_graph import build_complete_tabular_graph, STAGES, adapters
from scripts.import_flavours_provider_drafts import connect
from scripts.review_video_numeric_runtime import review as runtime_review

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT.parent/'sciona-atoms-ml'
DEPENDENCIES = {
    'sciona.atoms.ml.tabular.gradient_boosting.atoms.'+name
    for name in ('frequency_encode_fit', 'frequency_encode', 'target_encode', 'group_aggregate')
} | {
    'sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping.atoms.rfe_active_feature_indices',
    'sciona.atoms.ml.sklearn.decomposition.lda_shell.atoms.lda_normalize_document_topics',
}
PARTS = ('category_features', 'group_features', 'feature_selection', 'text_topics', 'entity_embeddings',
    'complete_features', 'complete_training', 'complete_contract', 'complete_graph',
    'ensemble_features', 'ensemble_training', 'ensemble_calibration', 'ensemble_contract')
TESTS = ['tests/test_tabular_'+name+'.py' for name in PARTS]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    names = DEPENDENCIES | {adapters.__name__+'.'+name for name in STAGES}
    specs = {}
    for spec in _parse_registered_atoms(repo=ProviderRepo(ML.name, ML), artifact_root=ML/'src/sciona/atoms'):
        name = spec.import_module+'.'+spec.source_symbol
        if name in names:
            if name in specs: raise ValueError('Ambiguous provider identity')
            specs[name] = spec
    if set(specs) != names: raise ValueError('Missing registered provider source')
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        source = db.execute('''SELECT v.content_hash,e.details FROM artifact_versions v
            JOIN artifact_audit_evidence e USING(version_id) WHERE v.version_id=%s
            AND e.runner_version='competition-intake.v1' ''', ('9f3c8c4f-ce85-5e9d-b169-8b055562ce65',)).fetchall()
        expected = '3cc5a5fdfab302b5c1a37abe86916ac996f577d11b733cba0bbf7b7d4438b2cd'
        if len(source) != 1 or source[0]['content_hash'] != expected or hashlib.sha256(
                _canonical(source[0]['details']['snapshot']).encode()).hexdigest() != expected:
            raise ValueError('Original immutable source changed')
        for name, spec in sorted(specs.items()):
            record = dict(runtime_fqdn=name, catalog_fqdn=spec.fqdn, version_id=str(spec.version_id),
                content_hash=spec.content_hash, source_file=os.path.relpath(spec.file_path, ROOT),
                provider_sha256=sha(spec.file_path), role='dependency' if name in DEPENDENCIES else 'adapter')
            if record['role'] == 'adapter':
                record['signature'] = str(inspect.signature(getattr(adapters, spec.source_symbol)))
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                rows = db.execute(f'''SELECT a.{key}::text AS artifact_id,a.status,a.is_publishable,
                    v.version_id::text,v.content_hash,v.trust_tier,{prefix}_is_publishable(a.{key}) AS native_eligible,
                    EXISTS(SELECT 1 FROM catalog_{prefix}s_served s WHERE s.{key}=a.{key}) AS served
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.fqdn=%s AND v.is_latest''',
                    (spec.fqdn,)).fetchall()
                record[prefix] = rows
                record[prefix+'_source_matches'] = len(rows) == 1 and rows[0]['version_id'] == str(spec.version_id) and rows[0]['content_hash'] == spec.content_hash
            records.append(record)
    return records


def validate():
    providers = inventory()
    paths = sorted(set(TESTS + ['sciona/tabular_'+name+'.py' for name in PARTS] + [
        'sciona/tabular_complete_stages.py', 'scripts/tabular_ensemble_synthetic.py',
        'tests/test_text_pair_provider_graphs.py', 'scripts/validate_complete_tabular_graph.py',
        'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py'] + [r['source_file'] for r in providers]))
    before = {p: sha(ROOT/p) for p in paths}
    runtime = runtime_review()
    graph_hash = encode_execution_graph(build_complete_tabular_graph())[0]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-tabular-qualification-') as temp:
        xml = Path(temp)/'tests.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
            '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)], cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode: raise ValueError('Runtime qualification failed: '+result.stdout[-4000:]+result.stderr[-1000:])
        counts = {k: sum(int(s.get(k, '0')) for s in ET.parse(xml).getroot().iter('testsuite'))
            for k in ('tests', 'failures', 'errors', 'skipped')}
    if not counts['tests'] or any(counts[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Complete successful test execution required')
    if before != {p: sha(ROOT/p) for p in paths} or runtime != runtime_review() or providers != inventory():
        raise ValueError('Software, runtime or catalog changed during qualification')
    return dict(format='complete-tabular-runtime-qualification.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, tests=counts, native_executions=3,
        graph_sha256=graph_hash, software_sha256=before, runtime=runtime, providers=providers,
        resource_policy=dict(serial=True, native_threads=1),
        next_actions=['Qualify and reconcile exact reusable dependencies in both catalogs.',
            'Stage and semantically review four concrete recipe adapters with exact dependency bindings.',
            'Preserve original and derivative history; stage, execute and publish the corrected original with Tier 3 evidence.'],
        limits=['Runtime qualification is not catalog approval or a claim of source-complete publication.',
            'Six directly invoked registered dependencies inventoried; sklearn models and preprocessing remain recipe policy.',
            'Synthetic behavior, split isolation and native graph parity only; no empirical accuracy or named submission claim.',
            'Execution-local state only; forged or concurrently mutated intermediates unqualified.',
            'Software/runtime fingerprints are not a full installed-package or clean-install integrity audit.'])


if __name__ == '__main__':
    result = validate()
    (ROOT/'docs/reviews/complete_tabular_runtime_qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, tests=result['tests'], providers=len(result['providers']), approved=False,
        canonical_present=sum(bool(r['artifact']) for r in result['providers']))))
