"""Audit original tabular scope and recheck the retained synthetic recipe."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from sciona.competition_import import _canonical
from scripts.import_flavours_provider_drafts import ROOT, connect

SOURCE = '9f3c8c4f-ce85-5e9d-b169-8b055562ce65'
SOURCE_HASH = '3cc5a5fdfab302b5c1a37abe86916ac996f577d11b733cba0bbf7b7d4438b2cd'
TESTS = ['tests/test_tabular_ensemble_'+part+'.py' for part in ('features', 'training', 'calibration', 'contract')]
FILES = ['sciona/tabular_ensemble_'+part+'.py' for part in ('features', 'training', 'calibration', 'contract', 'graph')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    old_execution = json.loads((ROOT/'docs/reviews/competition_tabular_ensemble_graph_execution.json').read_text())
    old_review = json.loads((ROOT/'docs/reviews/competition_tabular_ensemble_semantic_review.json').read_text())
    paths = FILES+TESTS+['scripts/audit_classical_tabular_readiness.py']
    before = {p: sha(ROOT/p) for p in paths}
    versions = {name: metadata.version(name) for name in old_review['dependencies']}
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,e.details
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e USING(version_id)
            WHERE v.version_id=%s AND e.runner_version='competition-intake.v1' ''', (SOURCE,)).fetchall()
        if len(rows) != 1 or rows[0]['content_hash'] != SOURCE_HASH:
            raise ValueError('Exact original source required')
        original = dict(rows[0]); snapshot = original.pop('details')['snapshot']
        if hashlib.sha256(_canonical(snapshot).encode()).hexdigest() != SOURCE_HASH:
            raise ValueError('Source snapshot changed')
        reference = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier,
            EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND v.is_latest''',
            (original['fqdn']+'.execution',)).fetchall()
        if len(reference) != 1 or reference[0]['content_hash'] != old_execution['serialized_graph_sha256']:
            raise ValueError('Retained derivative changed')
        candidates = {}
        patterns = ['%frequency_encode_fit', '%frequency_encode', '%target_encode', '%logit', '%expit',
            '%sigmoid_calibration%', '%variance_threshold%', '%one_hot_encoder_fit', '%simple_imputer_fit']
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            candidates[prefix] = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier,
                {prefix}_is_publishable(a.{key}) AS native_eligible FROM {prefix}s a JOIN {prefix}_versions v USING({key})
                WHERE v.is_latest AND a.fqdn ILIKE ANY(%s) ORDER BY a.fqdn''', (patterns,)).fetchall()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-tabular-readiness-') as temp:
        xml = Path(temp)/'tests.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
            '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)], cwd=ROOT, env=environment, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('Retained recipe runtime failed: '+result.stdout[-5000:]+result.stderr[-1000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k: sum(int(s.get(k, '0')) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if not counts['tests'] or any(counts[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Complete retained test suite required')
    if before != {p: sha(ROOT/p) for p in paths} or versions != {n: metadata.version(n) for n in versions}:
        raise ValueError('Sources or dependency versions changed during audit')
    return dict(format='classical-tabular-readiness.v1', read_only=True, catalog_mutations=0, approved=False,
        original=original, source_version_id=SOURCE, reference=reference[0],
        source_stages=[dict(stage_id=s['stage_id'], description=s['description'], guarantees=s['guarantees'])
            for s in snapshot['template']['stages']],
        tests=counts, synthetic_only=True, software_sha256=before, current_dependency_versions=versions,
        historical_dependency_drift={n: dict(qualified=v, current=versions[n]) for n, v in old_review['dependencies'].items() if versions[n] != v},
        reuse_candidates=candidates, resource_policy=dict(serial=True, native_threads=1),
        findings=[
            'Cleaning removes exact constant and duplicate numeric columns after imputation/clipping, but no explicit quasi-constant policy exists. Categorical columns are not deduplicated or screened.',
            'Features implement one-hot and frequency encoding only. Original target/count encodings, interactions, entity embeddings, text topics, group aggregates, log transforms and feature selection need explicit source-scope resolution and implementation where required; derivative approval does not waive them.',
            'The model-zoo source explicitly permits alternatives. Logistic regression and ExtraTrees form a valid declared branch; every base model has fold-local features and one out-of-fold score per training row.',
            'The OOF logistic stacker and full-training base refit are implemented. Caller group identifiers enforce fold and population isolation; duplicate/entity discovery and chronological split policy are not inferred.',
            'The binary held-out sigmoid procedure and fixed cutoff require an explicit realization of the original metric/output alternatives. Generic probability arrays are not a named competition submission format.',
            'Numerical clipping/logit conversion, frequency fitting/application and feature screening are reuse candidates. Inspect source contracts and both legacy/canonical identities before adding providers or replacing arithmetic.',
            'Current-runtime tests establish the retained derivative behavior only. Complete original scope, exact provider bindings and stored original graph publication remain unqualified.'],
        next_actions=['Resolve complete feature-stage scope and explicit applicability policy without silently dropping named requirements.',
            'Qualify reusable operations and keep table schema, folds, model selection and output policy in the recipe.',
            'Implement and test the missing original capabilities, then qualify the complete stored original graph.'],
        limits=['Generic algorithm-template and software metadata only; all executed inputs are synthetic.',
            'Dependency versions are recorded, not a full installed-package integrity or clean-install qualification.',
            'Name-pattern inventory is a starting point, not evidence that equivalent reusable providers are absent.'])


if __name__ == '__main__':
    result = audit()
    (ROOT/'docs/reviews/classical_tabular_readiness.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(original=result['original']['status'], reference_served=result['reference']['served'],
        tests=result['tests'], historical_dependency_drift=result['historical_dependency_drift'],
        reuse_candidates={k: len(v) for k, v in result['reuse_candidates'].items()}, approved=False)))
