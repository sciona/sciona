"""Review source-specific reuse assumptions without approving execution coverage."""
import json
from pathlib import Path
from scripts.audit_contrails_original_readiness import ROOT, sha
from scripts.review_text_pair_original_scope import anchor

BOUNDARIES = [
    dict(name='thermal_encoding',
         contract='Finite float32-compatible arrays shaped (4,3,256,256), with caller-defined ordered times and the source thermal-channel physical scale. False color is 1-((channel2-channel1+4)/6, (channel1-channel0+4)/9, (channel1-243)/60), without clipping or per-image normalization.',
         limitation='Shape and finiteness checks do not establish physical units, channel identity or temporal spacing. Arbitrary RGB or unrelated numeric channels do not satisfy the semantic contract merely by matching shape.',
         anchors=[('contrails_runtime','prepare'), ('contrails_preprocessing','ash_color'), ('contrails_prediction','prepare_inference')]),
    dict(name='temporal_and_spatial_geometry',
         contract='Single-frame branch uses frame3; temporal branch arranges frames3,0 / 1,2 in fixed quadrants. Spatial rotations/flips preserve temporal quadrant assignment. Training has shifted targets and source interpolation/resize policy.',
         limitation='Caller must justify spatial transformation invariance and target alignment in any new application. This is a fixed two-branch recipe; generic temporal averaging is not implemented.',
         anchors=[('contrails_temporal','create_four_panels'), ('contrails_temporal','create_four_panels_tta'), ('contrails_preparation','prepare_training_example')]),
    dict(name='populations_and_targets',
         contract='At least ten ordered training examples and nonempty scoring/prediction populations. Training/scoring supply label and annotation_mean arrays shaped (1,256,256) in [0,1]. Globally distinct nonempty keys; source shuffled ten-fold selection and explicit v43/v47 branch-fold subsets.',
         limitation='Distinct keys do not prove independent subjects, locations, times or arrays. Scoring participates in diagnostics and is not an independent final evaluation. Annotation means must retain their intended meaning.',
         anchors=[('contrails_runtime','prepare'), ('contrails_population','select_folds'), ('contrails_fold','train_fold')]),
    dict(name='prepared_state',
         contract='Preparation returns a trusted Python mapping of decoded populations, initialization states and configuration. Existing float32 input arrays may be aliased; execution shallow-copies mappings and does not repeat the original envelope validation.',
         limitation='Prepared state is neither deeply immutable nor a safe untrusted deserialization boundary. Callers must avoid mutation between preparation and execution; detached copying is not promised.',
         anchors=[('contrails_runtime','_array'), ('contrails_runtime','prepare'), ('contrails_runtime','execute_prepared')]),
    dict(name='training_and_resources',
         contract='Explicit seed and initialization policy: random, supplied encoder states or supplied full-model states per selected fold. Complete MaxViT/U-Net models, source loss/schedule, terminal checkpoints and all selected folds. Optional epoch_limit explicitly shortens the schedule.',
         limitation='Weight shape checks do not certify provenance. No pretrained downloads or historical predictive parity. Callers must set num_workers=0 and bound Torch, native pools and OpenCV for the shared resource budget; source defaults do not enforce that budget.',
         anchors=[('contrails_runtime','_state'), ('contrails_lifecycle','run_lifecycle'), ('contrails_fold','train_fold')]),
    dict(name='output_and_decisions',
         contract='Ordered per-example (1,256,256) unnormalized weighted fold/branch sums, plus column-major one-based run-length masks using strict threshold0.5 and the empty marker "-". Return variant, seed, initialization, epoch limit and per-fold optimizer-update summaries.',
         limitation='The field named probabilities is a source-weighted sum that can exceed one; it is not a calibrated posterior. Diagnostic threshold sweeps do not replace the fixed final threshold. No trained model is returned for later reuse.',
         anchors=[('contrails_prediction','ensemble_probabilities'), ('contrails_prediction','encode_mask'), ('contrails_runtime','execute_prepared')]),
]


def review():
    records = []
    for item in BOUNDARIES:
        records.append({k:v for k,v in item.items() if k != 'anchors'} |
                       dict(source_anchors=[anchor(*a) for a in item['anchors']]))
    return dict(format='contrails-reuse-boundaries.v1', passed=True, approved=False,
        catalog_mutations=0, review_source='automated',
        scope='Static implementation-bound semantic review; does not qualify full runtime execution or authorize publication.',
        boundaries=records,
        reuse_decision='Other aligned thermal-image segmentation tasks may reuse the recipe only when the source physical encoding, temporal order, target meaning, transformation invariances and fold assumptions are justified. Domain-neutral keys and arrays do not remove those assumptions. Preparation and execution remain recipe adapters; generic spatial augmentation or mask encoding would need independent provider contracts and qualification before separate selection.',
        remaining=['Complete both full-model variants under the bounded resource policy.',
                   'Bind final provider contracts to immutable full-execution evidence and repair catalog metadata transactionally.',
                   'Review all original intake stages, execute the corrected stored original and qualify publication/selection.'],
        reviewer_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/contrails_reuse_boundaries.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True, boundaries=len(report['boundaries']), approved=False)))
