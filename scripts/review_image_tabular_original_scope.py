"""Map the exact original intake to the qualified group-separated image/tabular realization."""
import hashlib
import json
import subprocess

from sciona.competition_import import FORMAT, _canonical, _id
from scripts.review_text_pair_original_scope import anchor
from scripts.review_image_tabular_provider_semantics import review as provider_review
from scripts.validate_image_tabular_complete_runtime import ROOT, sha

ORIGINAL = '4d7b7da1-6568-56b0-aa5e-7e2e7a275556'
ORIGINAL_HASH = '41907e416953f825b93f7b409d3fb36f16fce58303147e7540566f90bdcc622f'
IDENTITY = 'ea5352da-5061-5537-a723-9c27095e43ca'
STAGES = {
    'image_feature_extraction': dict(
        anchors=[('image_tabular_features','image_features')],
        realization='Select the explicitly allowed handcrafted image descriptor branch: 31 RGB moments, four-bin histograms, horizontal/vertical absolute differences and aspect ratio, preserving input order.',
        limits='Caller supplies normalized RGB arrays with consistent channel meaning. No learned CNN/ViT, image decoding, resizing or color conversion. Handcrafted descriptors instantiate the source alternative; they do not reproduce a historical neural image backbone.'),
    'tabular_feature_preparation': dict(
        anchors=[('image_tabular_features','metadata'), ('image_tabular_features','FusedFeatures.fit')],
        realization='Validate row-aligned numeric and categorical metadata; each fitting fold learns numeric median imputation and categorical vocabulary from its own training rows.',
        limits='All-missing numeric columns become zero; unseen categories become zero indicator vectors. Caller guarantees semantic alignment and consistent units. No automatic record join, PCA or missingness feature branch.'),
    'feature_fusion': dict(
        anchors=[('image_tabular_features','FusedFeatures._joined'), ('image_tabular_features','FusedFeatures.transform')],
        realization='Concatenate fixed visual descriptors, imputed numeric values and categorical indicators, then standardize the fused vector using the fitting fold only.',
        limits='Positional alignment is validated by equal row counts; caller owns correct image/metadata pairing. No attention fusion or independently weighted branch blending.'),
    'hybrid_head_training': dict(
        anchors=[('image_tabular_network','HybridHead'), ('image_tabular_network','smoothed_loss'), ('image_tabular_training','fit')],
        realization='Train the selected fused MLP branch with explicit hidden width, seeded dropout, binary label smoothing, CPU float64 Adam and every configured full-batch epoch.',
        limits='Binary classification only, fixed epochs and no early stopping, boosting, pretrained model or checkpoint search. Feature materialization and full-batch optimization have no large-data resource qualification.'),
    'fold_ensemble_calibration': dict(
        anchors=[('image_tabular_training','validate_split'), ('image_tabular_training','HybridEnsemble.predict'), ('image_tabular_complete_stages','calibrate'), ('image_tabular_complete_stages','predict')],
        realization='Validate caller-supplied group-consistent folds with both fitting classes; retain every fold model and complete OOF coverage. Average fold query/calibration probabilities equally, then select a held-out empirical-F1 threshold and apply inclusive decisions using the existing reusable numeric providers.',
        limits='Groups are disjoint across training/calibration/query. Fold generation and meaningful entity identity remain caller responsibilities. No full-data refit, OOF stacking or probability-calibration guarantee. Output is ordered generic binary scores/classes and counts; a named competition submission requires an explicit external formatting adapter.'),
}

def review():
    semantics_path = 'docs/reviews/image_tabular_provider_semantics.json'
    raw = (ROOT/semantics_path).read_bytes()
    semantics = json.loads(raw)
    if semantics != provider_review():
        raise ValueError('Qualified provider semantics changed')
    directory = ROOT.parent/'sciona-atoms/data/solution_cdgs'
    name = 'image_tabular_hybrid_classification_topology'
    snapshot = dict(format=FORMAT, template=json.loads((directory/(name+'.json')).read_text()),
        bindings=json.loads((directory/(name+'_bindings.json')).read_text()))
    digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
    fqdn = 'cdg.competition.'+snapshot['template']['asset_id']
    if (digest, _id(fqdn+'/'+digest), _id(fqdn)) != (ORIGINAL_HASH, ORIGINAL, IDENTITY):
        raise ValueError('Local source does not reconstruct the exact original intake')
    stages = snapshot['template']['stages']
    if len(stages) != len(STAGES) or {s['stage_id'] for s in stages} != set(STAGES):
        raise ValueError('Every original stage must be covered exactly once')
    return dict(format='image_tabular-original-scope.v1', passed=True, approved=False,
        catalog_mutations=0, catalog_state_checked=False, source_snapshot_reconstructed=True,
        source_version_id=ORIGINAL, source_content_hash=ORIGINAL_HASH, artifact_id=IDENTITY, fqdn=fqdn,
        provider_semantics_sha256=sha(ROOT/semantics_path), reviewer_sha256=sha(__file__),
        source_stage_review=[dict(source_stage={k:s[k] for k in ('stage_id','description','guarantees')},
            realization=STAGES[s['stage_id']]['realization'], limits=STAGES[s['stage_id']]['limits'],
            source_anchors=[anchor(*a) for a in STAGES[s['stage_id']]['anchors']]) for s in stages],
        reuse_decision='Reuse two existing empirical-F1 providers without changing their catalog identities or contracts. Four recipe adapters expose preparation, fitting, calibration and query prediction. Image meaning, units, group separation, descriptors, metadata transforms, neural models and fold averaging remain explicit recipe policy.',
        remaining_requirements=['Recheck original snapshot and preserved derivative against live catalog state.',
            'Reconcile and approve exact provider versions with metadata, dependency and rollback checks.',
            'Bind, store and execute the corrected original, then verify Tier 3 publication and served retrieval.'],
        limits=['Source reconstruction proves exact intake content identity, not current database status or approval.',
            'All five source stages map to the permitted handcrafted-image and fused binary MLP realization. Decision thresholds do not establish calibrated probabilities, cross-domain predictive utility or independent accuracy.',
            'Automated synthetic Tier 3 candidate evidence only; no human Tier 1 designation or empirical deployment guarantee.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/image_tabular_original_scope.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, stages=len(result['source_stage_review']), exact_source=True, approved=False)))
