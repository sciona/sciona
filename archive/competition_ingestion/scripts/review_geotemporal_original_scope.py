"""Map the exact original intake to the qualified causal planar regression realization."""
import hashlib
import json
import subprocess

from sciona.competition_import import FORMAT, _canonical, _id
from scripts.review_text_pair_original_scope import anchor
from scripts.review_geotemporal_provider_semantics import review as provider_review
from scripts.validate_geotemporal_complete_runtime import ROOT, sha

ORIGINAL = '9bb6428c-0529-56c8-b51d-96e1b4d6990c'
ORIGINAL_HASH = 'b22878a5b3fa77fcf721b4343515a4284b9dd7777758bd9e86b1ed56e5c6790e'
IDENTITY = '8fd737be-1d14-55d1-a896-d946f86545fd'
STAGES = {
    'geo_temporal_join': dict(
        anchors=[('geotemporal_contract','prepare'),('geotemporal_features','feature_matrix'),('geotemporal_stages','prepare')],
        realization='Select explicit shared planar metric coordinates and seconds clock. Validate finite aligned records and unique site/time identities. Join latest context per site available at or before the feature time, within lookback and radius. Keep query row order.',
        limits='Caller supplies a common coordinate system and clock. Context time is interpreted as availability time; late publication or revised historical values require caller preprocessing. No automatic projection, raster resampling, track alignment, road or external-source ingestion.'),
    'spatial_temporal_features': dict(
        anchors=[('geotemporal_features','feature_matrix'),('geotemporal_stages','features')],
        realization='Select coordinates, seasonal sine/cosine, nearest-context mean/distance/age, strict-past target mean/count and availability indicators. Preserve row identities and create separate training/validation feature matrices for each forward fold, plus full-training and query matrices.',
        limits='Nearest-site ties use coordinate order. Missing aggregates use zero plus explicit indicators; zero is not an observed value claim. No raster, tile embedding, road network or arbitrary rolling-statistic branch. Dense in-memory joins have no large-population qualification.'),
    'leakage_safe_geo_validation': dict(
        anchors=[('geotemporal_validation','forward_splits'),('geotemporal_stages','features'),('geotemporal_stages','fit')],
        realization='Select strictly ordered expanding time blocks with an explicit exclusion gap. Each fold builds target-history features from its training rows only. Earlier warmup rows fit models but receive no out-of-fold predictions. Score only held-out rows; smoothing sees only co-timed held-out predictions, never their labels.',
        limits='This establishes the selected forward-time boundary, not spatial holdout independence. Locations may repeat across folds. Context availability is caller asserted. Hyperparameter tuning on these metrics would require additional evaluation; no unseen-location accuracy or independent final accuracy claim.'),
    'fusion_model': dict(
        anchors=[('geotemporal_training','model'),('geotemporal_stages','fit'),('geotemporal_stages','predict')],
        realization='Select ExtraTrees regression on eleven joined spatial/temporal features, with explicit seeded tree controls and one worker. Fit separate forward-fold models and a final model on all observations. Predict query features without refitting, preserving requested row order.',
        limits='Tree regression is one source-permitted family. No neural, image, sequence or multimodal hybrid branch is implemented. Trusted in-process model and feature state; no portable persistence, arbitrary schema or historical winning-model reproduction.'),
    'geographic_postprocessing': dict(
        anchors=[('geotemporal_stages','postprocess'),('neighborhood_ops','blend_neighbor_means'),('geotemporal_stages','finish')],
        realization='Select nonnegative clipping then convex neighbor-mean blending. Recipe constructs same-time, within-radius, non-self adjacency; the registered generic numeric operation accepts that mask and an explicit strength. Return ordered predictions and forward-validation/population/model summaries.',
        limits='Generic numeric blending permits signed values and arbitrary directed masks; geography and clipping remain recipe policy. Empty neighborhoods retain the clipped value. Query population affects smoothing; no general sum conservation, map matching, zero-sum correction or named submission-format claim.'),
}

def review():
    semantics_path = 'docs/reviews/geotemporal_provider_semantics.json'
    raw = (ROOT/semantics_path).read_bytes()
    semantics = json.loads(raw)
    if semantics != provider_review():
        raise ValueError('Qualified provider semantics changed')
    directory = ROOT.parent/'sciona-atoms/data/solution_cdgs'
    name = 'geospatial_spatiotemporal_fusion_topology'
    snapshot = dict(format=FORMAT, template=json.loads((directory/(name+'.json')).read_text()),
        bindings=json.loads((directory/(name+'_bindings.json')).read_text()))
    digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
    fqdn = 'cdg.competition.'+snapshot['template']['asset_id']
    if (digest, _id(fqdn+'/'+digest), _id(fqdn)) != (ORIGINAL_HASH, ORIGINAL, IDENTITY):
        raise ValueError('Local source does not reconstruct the exact original intake')
    stages = snapshot['template']['stages']
    if len(stages) != len(STAGES) or {s['stage_id'] for s in stages} != set(STAGES):
        raise ValueError('Every original stage must be covered exactly once')
    return dict(format='geotemporal-original-scope.v1', passed=True, approved=False,
        catalog_mutations=0, catalog_state_checked=False, source_snapshot_reconstructed=True,
        source_version_id=ORIGINAL, source_content_hash=ORIGINAL_HASH, artifact_id=IDENTITY, fqdn=fqdn,
        provider_semantics_sha256=sha(ROOT/semantics_path), reviewer_sha256=sha(__file__),
        source_stage_review=[dict(source_stage={k:s[k] for k in ('stage_id','description','guarantees')},
            realization=STAGES[s['stage_id']]['realization'], limits=STAGES[s['stage_id']]['limits'],
            source_anchors=[anchor(*a) for a in STAGES[s['stage_id']]['anchors']]) for s in stages],
        reuse_decision='One independently reusable numeric neighborhood operation with five recipe adapters. Coordinate/clock units, adjacency, forward validation, model family and clipping remain explicit recipe policy.',
        remaining_requirements=['Recheck original snapshot and preserved derivative against live catalog state.',
            'Reconcile and approve exact provider versions with metadata, dependency and rollback checks.',
            'Bind, store and execute the corrected original, then verify Tier 3 publication and served retrieval.'],
        limits=['Source reconstruction proves exact intake content identity, not current database status or approval.',
            'All five source stages covered for the explicit causal planar regression realization; forward-time validation does not imply spatial holdout independence.',
            'Automated synthetic Tier 3 candidate evidence only; no human Tier 1 designation or empirical deployment guarantee.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/geotemporal_original_scope.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, stages=len(result['source_stage_review']), exact_source=True, approved=False)))
