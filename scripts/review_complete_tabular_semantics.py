"""Automated semantic review for staged tabular providers; no approval writes."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from scripts.stage_complete_tabular_providers import ROOT, PLAN, reviewed, stage, sha, require

CONTRACTS = {
    'frequency_encode_fit': ('Compute relative occurrence counts for categorical keys.',
        'Nonempty one-dimensional mutually sortable, hashable category keys. Returns count divided by total count for each observed key; no labels, split policy or domain semantics. Qualified string keys; mixed incomparable objects excluded.', []),
    'frequency_encode': ('Look up a supplied nonnegative value for each category.',
        'One-dimensional hashable category keys, finite nonnegative mapping values and finite nonnegative unknown_value. Preserves query order and duplicates; never updates the mapping. Mapping and fallback values need not sum to one or be bounded by one.', []),
    'target_encode': ('Compute category means shrunk toward an explicit or observed prior.',
        'Nonempty aligned category keys and finite real targets; mutually sortable hashable keys, finite nonnegative smoothing, optional finite prior. Output is (group sum + smoothing * prior)/(group count + smoothing). Default prior is the full supplied target mean. No fold exclusion is intrinsic; caller must select fitting rows. Requires representable intermediate arithmetic.', []),
    'group_aggregate': ('Reduce numeric values separately for each key.',
        'Nonempty aligned finite float64 values and mutually sortable keys. Output follows sorted unique-key order. Supports count, sum, mean, population var/std, min, max, first and last. First/last preserve input encounter order within key, not time order. No missing-value handling or fold isolation is intrinsic. Requires finite intermediate reductions.', []),
    'lda_normalize_document_topics': ('Scale each row of nonnegative weights to sum to one.',
        'Finite nonnegative two-dimensional float64 matrix with positive finite row sums; output preserves shape and divides each row by its sum. Generic mass normalization: no LDA model fitting or topic meaning required. Overflowing row sums and zero-mass rows excluded.', []),
    'rfe_active_feature_indices': ('Return the positions selected by a boolean mask.',
        'Nonempty one-dimensional boolean mask with at least one true entry. Returns ascending original positions as int64; no fitting, importance scoring or domain interpretation occurs.', []),
    'prepare': ('Validate explicit feature policies and separated populations.',
        'Finite JSON version 2 with exact controls and populations. Aligned mixed numeric/string rows; binary labels with both classes in training, calibration and each outer fitting split. Group IDs enforce train/calibration/query separation and contiguous group-consistent folds. Immutable serialized configuration. Fold-specific widths, empty vocabularies and convergence remain runtime conditions.', []),
    'fit_stack': ('Fit fold-local features and models, then combine out-of-fold predictions.',
        'Validated PreparedComplete from the qualified prepare stage. Training-only median/clipping/screening, one-hot/count/frequency and group-excluded target means; explicit interactions, signed log copies, aggregates, LDA text topics, supervised categorical embeddings and optional RFE. Each outer fold independently fits features and logistic/ExtraTrees models. Logistic stacker uses exactly one OOF score pair per row; final base models refit all training rows. Group identity and feature applicability remain caller policy. Serial native pools and n_jobs=1; dense large-table resource use unqualified.',
        ['frequency_encode_fit', 'frequency_encode', 'target_encode', 'group_aggregate', 'lda_normalize_document_topics', 'rfe_active_feature_indices']),
    'calibrate': ('Fit a sigmoid score map using separated calibration labels.',
        'Unmodified FittedComplete from qualified fitting stage. Frozen base and stacker score calibration rows; regularized logistic sigmoid fits clipped score logits using calibration labels only. Does not refit features or base/stacker models. No empirical probability-calibration guarantee. Concrete execution-local state; fabricated intermediates excluded.', ['frequency_encode', 'lda_normalize_document_topics']),
    'predict': ('Return ordered query probabilities and binary decisions.',
        'Unmodified CalibratedComplete from qualified calibration stage. Frozen feature maps, models and sigmoid; no query fitting. Preserve query order and repeated rows, with inclusive probability >= 0.5 decisions and aggregate execution counts. Generic output, not a named competition submission. Concrete execution-local state; forged or concurrently mutated intermediates excluded.', ['frequency_encode', 'lda_normalize_document_topics']),
}


def review():
    raw = PLAN.read_bytes()
    require(raw == subprocess.check_output(['git', 'show', '4d6f8673:'+str(PLAN.relative_to(ROOT))], cwd=ROOT), 'Immutable intake plan changed')
    frozen = reviewed()
    require(stage()['rows_created'] == 0, 'Exact staged records required')
    records = {r['seed_fields']['source_symbol']: r for r in frozen['providers']}
    require(set(records) == set(CONTRACTS), 'Every provider requires semantic coverage')
    paths = ['tests/test_tabular_provider_semantics.py', 'scripts/review_complete_tabular_semantics.py']
    before = {p:sha(ROOT/p) for p in paths}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-tabular-semantics-') as temp:
        xml = Path(temp)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', paths[0],
            '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)], cwd=ROOT, env=env, capture_output=True, text=True)
        require(run.returncode == 0, 'Semantic tests failed: '+run.stdout[-3000:])
        counts = {k:sum(int(s.get(k,'0')) for s in ET.parse(xml).getroot().iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    require(counts == dict(tests=13, failures=0, errors=0, skipped=0), 'Complete semantic checks required')
    require(before == {p:sha(ROOT/p) for p in paths} and reviewed() == frozen, 'Review inputs changed')
    providers = []
    for name, r in records.items():
        plain, bounds, deps = CONTRACTS[name]
        providers.append(dict(artifact_id=r['atom_id'], fqdn=r['catalog_fqdn'], version_id=r['version_id'],
            content_hash=r['content_hash'], plain_description=plain, validity_bounds=bounds,
            interfaces=[dict(p, constraints=bounds) for p in r['interfaces']],
            dependencies=[dict(artifact_id=records[d]['atom_id'], version_id=records[d]['version_id'],
                content_hash=records[d]['content_hash']) for d in deps]))
    return dict(format='complete-tabular-provider-semantics.v1', passed=True, approved=False,
        proposed_tier=3, catalog_mutations=0, synthetic_only=True, providers=providers, tests=counts,
        software_sha256=before, intake_plan_sha256=sha(PLAN), runtime_evidence_sha256=frozen['review_sha256'],
        limits=['Automated Tier 3 candidate review only; no human Tier 1 or Tier 2 designation.',
            'Six numeric operations retain cross-domain semantics; four adapters carry explicit binary recipe policy.',
            'Thirteen semantic checks supplement immutable 144-test recipe evidence; direct numeric native graph checks remain pending.',
            'Approval requires source-bound metadata, dependency and rollback gates; this script does not publish.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/complete_tabular_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=result['tests'], approved=False)))
