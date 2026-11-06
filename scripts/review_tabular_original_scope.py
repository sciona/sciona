"""Map every immutable classical-tabular source stage to the qualified recipe."""
import hashlib
import json

from sciona.competition_import import _canonical
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.plan_flavours_provider_intake import sha
from scripts.review_text_pair_original_scope import anchor
from scripts import tabular_approved_provider_state as providers

ORIGINAL = '9f3c8c4f-ce85-5e9d-b169-8b055562ce65'
ORIGINAL_HASH = '3cc5a5fdfab302b5c1a37abe86916ac996f577d11b733cba0bbf7b7d4438b2cd'
STAGES = {
    'table_audit_cleaning': dict(
        anchors=[('tabular_ensemble_features', 'validate_table'), ('tabular_ensemble_features', 'TabularFeatures.fit'),
            ('tabular_complete_features', 'fit_binary_tabular_features'), ('tabular_complete_contract', 'prepare'),
            ('tabular_complete_training', '_fit_complete_stacker')],
        realization='Select explicit numeric/string column positions; validate finite values or nulls, fit numeric medians and clipping quantiles per outer training fold, screen constant/duplicate and modal-share quasi-constant columns, and validate caller-provided group-consistent folds.',
        limits='Schema meaning and group discovery remain caller policy. No automatic chronology, duplicate-entity detection or general leakage guarantee. All-null numeric columns fill zero and are screened. Dense tables and wide-category resource use are not qualified.'),
    'split_aware_feature_engineering': dict(
        anchors=[('tabular_category_features', 'fit_category_features'), ('tabular_group_features', 'fit_group_features'),
            ('tabular_text_topics', 'fit_text_topics'), ('tabular_entity_embeddings', 'fit_entity_embeddings'),
            ('tabular_feature_selection', 'select_training_features'), ('tabular_complete_features', 'fit_binary_tabular_features'),
            ('tabular_complete_training', '_fit_base')],
        realization='Implement count/frequency and group-excluded target means, collision-free category-pair interactions, supervised categorical embedding vectors, training-only count-vocabulary LDA topics, explicit-key numeric aggregates, signed log1p copies, quantile clipping and RFE-selected dense features. Fit each feature family only within its outer training fold; preserve row order.',
        limits='Empty specification lists explicitly disable inapplicable families; qualification exercises every family together. Text fields use explicit whitespace/case policy. Group aggregate keys are feature choices distinct from validation groups. Entity vectors use a binary MLP objective, full fitting-fold labels and zero vectors for unseen categories. RFE is the selected alternative to permutation selection. Only target means exclude each training row group; supervised embeddings and RFE use all fitting-fold labels. Fold coordinate systems need not align because only prediction columns are stacked.'),
    'tabular_model_zoo': dict(
        anchors=[('tabular_complete_training', '_fit_base'), ('tabular_complete_training', '_fit_complete_stacker')],
        realization='Select standardized logistic regression and ExtraTrees as two source-listed model-family branches for binary classification. Fit each independently per outer fold and produce one held-out probability from each model for every training row.',
        limits='Other listed model families are alternatives and are not claimed. No SMOTE, class weighting, early stopping or hyperparameter search; caller chooses applicability and controls. Convergence warnings fail execution. ExtraTrees n_jobs and native pools are one.'),
    'ensemble_stacking_blending': dict(
        anchors=[('tabular_complete_training', '_fit_complete_stacker'), ('tabular_ensemble_training', 'StackedModel.predict')],
        realization='Select a second-level logistic stacker trained on the two OOF prediction columns. Refit base models and their feature maps on all training rows for later calibration and prediction.',
        limits='No rank-average, pseudo-labeling or stability-based model selection is claimed. OOF coverage and group/fold separation are checked; OOF scores are training inputs to the stacker, not an independent evaluation of the final stacked model.'),
    'metric_calibration': dict(
        anchors=[('tabular_ensemble_calibration', 'calibrate_stack'), ('tabular_ensemble_calibration', 'predict_calibrated'),
            ('tabular_complete_contract', 'execute'), ('tabular_complete_stages', 'calibrate'), ('tabular_complete_stages', 'predict')],
        realization='Realize probability calibration for the explicit two-class specialization: fit a regularized sigmoid to clipped class-1 score logits on separated calibration rows, then return finite ordered query probabilities and inclusive 0.5 decisions.',
        limits='This specializes the source probability-calibration family to binary classes; it does not implement arbitrary multiclass calibration, QWK search, regression transforms or a named benchmark file format. Generic result arrays preserve query order and duplicates. A fitted calibration procedure is not a guarantee of empirical calibration or accuracy. Native stored-original execution and publication remain pending.'),
}


def review():
    qualified = providers.qualify_plan()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        providers.check(db, qualified)
        rows = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,v.content_hash,e.details
            FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e USING(version_id)
            WHERE v.version_id=%s AND e.runner_version='competition-intake.v1' ''', (ORIGINAL,)).fetchall()
        if len(rows) != 1 or rows[0]['content_hash'] != ORIGINAL_HASH:
            raise ValueError('Original source identity changed')
        original = dict(rows[0]); snapshot = original.pop('details')['snapshot']
        if hashlib.sha256(_canonical(snapshot).encode()).hexdigest() != ORIGINAL_HASH:
            raise ValueError('Immutable original snapshot changed')
        source_stages = snapshot['template']['stages']
    if set(STAGES) != {s['stage_id'] for s in source_stages}:
        raise ValueError('Every original stage must be covered')
    return dict(format='tabular-original-scope.v1', read_only=True, catalog_mutations=0, approved=False,
        original=original, source_version_id=ORIGINAL, source_content_hash=ORIGINAL_HASH,
        source_stage_review=[dict(source_stage=dict(stage_id=s['stage_id'], description=s['description'], guarantees=s['guarantees']),
            realization=STAGES[s['stage_id']]['realization'], limits=STAGES[s['stage_id']]['limits'],
            source_anchors=[anchor(*a) for a in STAGES[s['stage_id']]['anchors']]) for s in source_stages],
        reviewer_sha256=sha(__file__), approved_provider_state_sha256=sha(providers.PLAN),
        remaining_requirements=['Stage corrected original graph with exact approved bindings and immutable lineage.',
            'Execute the stored original graph, verify source-scope fidelity, then run publication corruption and rollback checks.',
            'Verify approved latest-version retrieval and served selection without changing the retained derivative.'],
        limits=['Explicit binary realization of the generic source topology; not every listed model or metric alternative.',
            'All named feature families implemented with explicit applicability policies; original source does not prescribe a universal schema.',
            'Automated synthetic qualification only; no historical winning-solution reproduction or Tier 1 human review.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/tabular_original_scope.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(stages=len(result['source_stage_review']), approved_providers=10, approved=False)))
