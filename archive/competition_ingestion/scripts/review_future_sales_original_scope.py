"""Resolve selectable feature branches against the exact immutable intake."""
import hashlib
import json
import subprocess

from sciona.competition_import import _canonical
from scripts.audit_decomposed_future_sales_readiness import ROOT, ORIGINAL, SOURCE_HASH, STAGES, sha
from scripts.import_flavours_provider_drafts import connect

COMMIT = '356193bc371a7f9adc7b210abde04b9b370f55b6'
PRIOR = 'eb2aabea654e25815e99cac996ce244a66dde164'
ASSET = 'data/solution_cdgs/predict_future_sales_benchmark.json'


def review():
    repo = ROOT.parent/'sciona-atoms'
    raw = subprocess.check_output(['git','show',COMMIT+':'+ASSET],cwd=repo)
    prior = json.loads(subprocess.check_output(['git','show',PRIOR+':'+ASSET],cwd=repo))
    template = json.loads(raw)
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows = db.execute('''SELECT v.content_hash,e.details FROM artifact_versions v
            JOIN artifact_audit_evidence e USING(version_id) WHERE v.version_id=%s
            AND e.runner_version='competition-intake.v1' AND e.audit_type='asset_integrity_check' ''', (ORIGINAL,)).fetchall()
    if len(rows)!=1 or rows[0]['content_hash']!=SOURCE_HASH:
        raise ValueError('Unique immutable source required')
    snapshot=rows[0]['details']['snapshot']
    if hashlib.sha256(_canonical(snapshot).encode()).hexdigest()!=SOURCE_HASH or snapshot['template']!=template:
        raise ValueError('Committed template and canonical intake differ')
    stages={s['stage_id']:s for s in template['stages']}
    if set(stages)!=set(STAGES):raise ValueError('Complete five-stage scope required')
    choices=template['applicability']['swappable_stages']
    encoding='Can use store, shop, item, category, day-of-week, month, holiday, promotion, target encoding, or learned entity embeddings.'
    if choices['mean_encoding']!=encoding or 'embeddings or encodings' not in template['summary']:
        raise ValueError('Explicit encoding alternatives no longer established')
    if 'as long as future target leakage is excluded' not in choices['lag_feature_generation']:
        raise ValueError('Temporal feature safety requirement changed')
    old={s['stage_id']:s for s in prior['stages']}
    if old['mean_encoding']['description']!='calculate the expanding mean of sales per shop and per item over time to generate robust continuous features':
        raise ValueError('Earlier source history changed')
    return dict(format='future-sales-original-scope.v1',passed=True,approved=False,catalog_mutations=0,
        source_version_id=ORIGINAL,source_content_hash=SOURCE_HASH,
        template_commit=COMMIT,prior_template_commit=PRIOR,template_sha256=hashlib.sha256(raw).hexdigest(),
        exact_committed_template_equals_intake=True,original_stages=STAGES,
        decision='The full intake explicitly makes mean-encoding representations swappable. Select expanding entity/store/item/category/global means and cyclical period features; learned embeddings, holiday and promotion features are alternative branches, not mandatory additional computations.',
        evidence=dict(explicit_alternatives=encoding,summary_uses_embeddings_or_encodings=True,
            earlier_mean_encoding=old['mean_encoding']['description'],
            scope_expansion_commit=COMMIT),
        selected_computations={
            'time_aggregation':'Caller-defined periods and fixed store/item/category universe; compensated net transaction summation, absence means zero.',
            'target_clipping':'Clip net observed period/entity quantities to [0,20] only after summation. Preserve unknown forecast targets.',
            'lag_feature_generation':'All caller-configured lags and complete-width shifted rolling means, without current or future target access.',
            'mean_encoding':'Prior-only expanding means for entity/store/item/category/global populations and explicit cyclic period features, using the source-authorized encoding branch.',
            'gbdt_modeling':'LightGBM squared error with clipped-prediction RMSE validation, every configured TPE trial including model-based sampling, early stopping and final selected-round refit.'},
        limits=[
            'The template is a generic benchmark topology with a competition landing-page citation, not a historical winning implementation.',
            'Stage descriptions must be interpreted together with the explicit applicability/swappable_stages contract; isolated descriptions overstated the mandatory feature set in the previous readiness audit.',
            'Do not claim implementation of omitted holiday, promotion or embedding branches. Cross-domain reuse requires the selected feature and observation-availability assumptions.',
            'Clipping does not by itself replicate a competition evaluation metric. The realization declares its squared-error objective and clipped-prediction RMSE; no historical metric or accuracy equivalence is certified.',
            'All configured trials and feature operations remain mandatory. This resolution does not approve providers, stored execution or the original CDG.'],reviewer_sha256=sha(__file__))


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/future_sales_original_scope.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,exact_template_match=True,explicit_encoding_alternatives=True,approved=False)))
