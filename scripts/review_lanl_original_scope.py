"""Bind a source-scope correction review to the unchanged original intake.

This is a read-only planning gate. It does not qualify implementations, license
the complete winning submission, or approve a smaller demonstration as the CDG.
"""
import hashlib
import json
from pathlib import Path

from sciona.competition_import import _canonical, _id
from scripts.import_flavours_provider_drafts import ROOT, connect

FQDN = 'cdg.competition.solution.kaggle.lanl_earthquake_1st'
VERSION = 'bfba9df2-591c-538b-9810-7c8511921ea4'
DIGEST = 'f99854e020974328e9c5f25ad66beb3e210a83a67c6ae64f36aec52ddab95840'
WRITEUP = 'https://www.kaggle.com/c/LANL-earthquake-Prediction/discussion/94390'

# Paraphrased source findings; implementation details still require pinned code.
STAGES = {
    'signal_chunking': 'Keep fixed windows; add Gaussian noise before subtracting each window median. The intake omits this preprocessing.',
    'time_domain_features': 'Resolve peak-count support and rolling-standard-deviation quantiles from code; generic means and quantiles do not establish coverage.',
    'frequency_domain_features': 'The winner identifies MFCC means. Welch band extraction in the intake is not established by this writeup.',
    'adversarial_validation': 'The writeup uses KS feature screening. Do not equate the reported drift diagnosis with a mandatory classifier-filter implementation.',
    'model_training': 'Cover LightGBM, SVR and a multitask neural model, including binary near-failure and time-since-failure losses. Kernel Ridge lacks source support here.',
    'stacking': 'Recover the hill-climbing blend. Neither plain averaging nor the intake floor/ceiling is established; a smaller public demonstration cannot stand for all submitted models.',
}


def review():
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        rows = db.execute("""SELECT e.details FROM artifact_audit_evidence e
            JOIN artifacts a USING(artifact_id) WHERE a.fqdn=%s AND e.version_id=%s
            AND e.runner_version='competition-intake.v1'""", (FQDN, VERSION)).fetchall()
        if len(rows) != 1:
            raise ValueError('Unique original intake snapshot required')
        snapshot = rows[0]['details']['snapshot']
        digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
        if digest != DIGEST or _id(FQDN+'/'+digest) != VERSION:
            raise ValueError('Original intake identity changed')
        stages = snapshot['template']['stages']
        if len(stages) != 6 or {s['stage_id'] for s in stages} != set(STAGES):
            raise ValueError('Original six-stage scope changed')
    return dict(format='lanl-original-source-scope.v1', passed=True, approved=False,
        catalog_mutations=0, review_source='automated primary-source inspection',
        source_version_id=VERSION, source_content_hash=DIGEST, fqdn=FQDN,
        primary_writeup=WRITEUP,
        source_stage_review=[dict(stage_id=s['stage_id'], correction=STAGES[s['stage_id']]) for s in stages],
        code_leads=[dict(url='https://www.kaggle.com/code/ilu000/1-private-lb-kernel-lanl-lgbm',
            license_declaration='Apache-2.0', source_bytes_pinned=False,
            scope='Public LightGBM demonstration; insufficient for full original-CDG coverage.')],
        reuse_requirements=[
            'Separate windowing, explicit noise injection, median centering, feature extraction, distribution comparison, model fitting and blending.',
            'Expose physical units, sample rate, window geometry, randomness and feature coordinate ordering as contracts.',
            'Treat target-population distribution screening as an explicit transductive policy; it is unsuitable when future query populations are unavailable.',
            'Keep event definitions and auxiliary response construction in application adapters; never infer universal failure thresholds.',
            'Compare each operation with pinned source on independent synthetic signals, then execute the full graph.'],
        remaining=[
            'Obtain and pin executable sources and separate permissions for all three model families and the final blend.',
            'Recover exact feature settings, training populations, validation folds, schedules and ensemble selection.',
            'Implement and qualify reusable providers without omitting source stages or substituting public demonstration scope.',
            'Execute the corrected stored original and pass metadata, dependency, transaction and served-selection gates.'],
        limitations=['Source-scope corrections only; no numerical execution or historical-performance claim.',
            'Notebook page license declaration does not establish permissions for every external dependency or the complete submission.',
            'No real runtime signals, labels, dataset metadata or model artifacts were accessed.'],
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/lanl_original_source_scope.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, original_stages=len(result['source_stage_review']), approved=False)))
