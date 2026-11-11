"""Review DFDC reuse assumptions against implementation without granting approval."""
import json
from pathlib import Path
from scripts.audit_dfdc_original_readiness import ROOT, sha
from scripts.review_text_pair_original_scope import anchor

BOUNDARIES = [
    dict(name='media_and_targets',
        contract='Decoded uint8 RGB clips, explicit original/altered pair relationships and positional part groups. Labels derive from whether a clip references itself as its original. Both classes must occur in each selected training and validation population.',
        limitation='This is a face-focused alteration recipe, not a generic video classifier. Caller must justify RGB encoding, frame order, paired target meaning and valid face geometry; matching array shapes is insufficient.',
        anchors=[('dfdc_runtime','prepare'),('dfdc_runtime','_frames')]),
    dict(name='population_independence',
        contract='Fold assignment partitions fifty positional groups and keeps each original and its alterations in the same fold. Originals must reference themselves directly.',
        limitation='Positional group numbers and pair-fold agreement do not prove independence of people, recording sessions or acquisition sources. Caller must supply a grouping justified for the new task.',
        anchors=[('dfdc_folds','assign_folds'),('dfdc_runtime','prepare')]),
    dict(name='detectors_and_landmarks',
        contract='Neural face detection and five-point preprocessing landmarks are separate from a native predictor with the source sixty-eight-point convention used for polygon occlusion. The predictor is caller-supplied at runtime; no automatic download.',
        limitation='A sixty-eight-point count does not establish the semantic ordering or quality of landmarks. Native predictor deserialization is a trusted-file boundary. The operation named convex hull uses an ordered landmark polygon, and source failures silently leave the image unchanged.',
        anchors=[('dfdc_hull_backend','load_hull_backend'),('dfdc_convex_hull','blackout_convex_hull'),('dfdc_landmark_removal','remove_landmark')]),
    dict(name='training_and_checkpoints',
        contract='Full EfficientNet-B7 classifier with explicit random, encoder or complete state initialization. Runs specify seed, fold and epoch count; requests specify zero-based checkpoint epochs. Every requested run must be configured.',
        limitation='Source default forty-epoch runs cannot supply a request for zero-based epoch forty. An explicit executable adaptation is required; do not silently extend runs or relabel checkpoints. Tensor shape checks and seed labels do not authenticate historical weights or predictive parity.',
        anchors=[('dfdc_classifier','build_classifier'),('dfdc_selection','selection_plan'),('dfdc_selection','select_snapshots')]),
    dict(name='geometry_and_aggregation',
        contract='Source aspect-preserving resize and central padding feed the classifier at size380. Per-model confident face aggregation precedes ensemble averaging. Strict confidence thresholds and count conditions are retained.',
        limitation='Face detection, augmentation invariance and confidence thresholds are source assumptions requiring justification in another application. Scores are not calibrated cross-domain probabilities. Generic resize or aggregation would need independently qualified provider contracts before separate selection.',
        anchors=[('dfdc_transforms','create_train_transforms'),('dfdc_inference_primitives','confident_strategy'),('dfdc_prediction','predict_video')]),
    dict(name='fallback_and_prepared_state',
        contract='Prediction returns a score plus explicit predicted/source_fallback status and reason. Empty input, missing faces and caught execution failures can return0.5. Execution accepts only trusted prepared state and returns summaries and predictions, not reusable trained models.',
        limitation='A fallback0.5 is not a successful neutral prediction; downstream consumers must inspect status. Prepared mappings are mutable and execute_prepared does not repeat envelope validation. One-face scalar behavior can cause the source execution-error fallback.',
        anchors=[('dfdc_prediction','predict_video'),('dfdc_runtime','execute_prepared')]),
]


def review():
    return dict(format='dfdc-reuse-boundaries.v1',passed=True,approved=False,catalog_mutations=0,
        review_source='automated',scope='Static implementation-bound reuse review; full execution and publication remain separate.',
        boundaries=[{k:v for k,v in item.items() if k!='anchors'} |
                    dict(source_anchors=[anchor(*a) for a in item['anchors']]) for item in BOUNDARIES],
        reuse_decision='Other paired face-video alteration tasks may reuse this recipe only with justified target meanings, population grouping, detector and landmark conventions, augmentation invariances and checkpoint plans. Preparation and execution remain recipe adapters. General image geometry, sampling and aggregation require independent contracts before standalone selection.',
        remaining=['Qualify full detector, augmentation and native-predictor dependencies and execution.',
            'Review all original source stages and exact provider interfaces.',
            'Execute complete current and corrected stored ensembles with explicit bounded resources before publication.'],
        reviewer_sha256=sha(Path(__file__)))


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/dfdc_reuse_boundaries.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,boundaries=len(report['boundaries']),approved=False)))
