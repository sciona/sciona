"""Reproducible automated Tier3 semantic/dependency evidence audit for DFDC."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
EVIDENCE=['competition_dfdc_checkpoint_selection_review.json', 'competition_dfdc_checkpoints.json', 'competition_dfdc_classifier.json', 'competition_dfdc_codec.json', 'competition_dfdc_codec_b7.json', 'competition_dfdc_convex_hull.json', 'competition_dfdc_crop_annotations.json', 'competition_dfdc_dataset.json', 'competition_dfdc_detector.json', 'competition_dfdc_dlib_dependency_pin.json', 'competition_dfdc_encoder_compatibility.json', 'competition_dfdc_ensemble_b7.json', 'competition_dfdc_ensemble_prediction.json', 'competition_dfdc_ensemble_training.json', 'competition_dfdc_epoch.json', 'competition_dfdc_evaluation.json', 'competition_dfdc_facenet_dependency_review.json', 'competition_dfdc_facenet_source_pins.json', 'competition_dfdc_faces.json', 'competition_dfdc_folds.json', 'competition_dfdc_frame_prediction_edges.json', 'competition_dfdc_graph.json', 'competition_dfdc_hull_backend.json', 'competition_dfdc_inference_primitives.json', 'competition_dfdc_landmark_removal.json', 'competition_dfdc_population.json', 'competition_dfdc_prediction.json', 'competition_dfdc_preparation.json', 'competition_dfdc_records.json', 'competition_dfdc_runtime.json', 'competition_dfdc_runtime_full.json', 'competition_dfdc_selection.json', 'competition_dfdc_source_pins.json', 'competition_dfdc_source_review.json', 'competition_dfdc_stochastic_depth.json', 'competition_dfdc_timm_dependency_pin.json', 'competition_dfdc_training.json', 'competition_dfdc_training_b7.json', 'competition_dfdc_training_crops.json', 'competition_dfdc_training_masks.json', 'competition_dfdc_training_math.json', 'competition_dfdc_transforms.json']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reviews=ROOT/'docs/reviews';documents={}
    for name in EVIDENCE:
        document=json.loads((reviews/name).read_text());documents[name]=document
        for path,digest in document.get('sha256',{}).items():
            target=(ROOT/path).resolve()
            assert target.is_relative_to(ROOT) and target.is_file() and sha(target)==digest,path
    def evidence(suffix):return documents['competition_dfdc_'+suffix+'.json']
    for suffix in ('graph','runtime_full','stochastic_depth','selection','codec_b7','ensemble_b7'):
        assert evidence(suffix)['result']=='passed'
    full=evidence('graph')
    assert full['checks']=={'provider_witness_contracts':2,'serialized_full_runtime_graphs':1,
                           'actual_B7_training_runs':5,'training_updates':10,'ensemble_states':7,'runner_rejections':1}
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('dfdc_*.py')} <= set(full['sha256'])
    assert evidence('stochastic_depth')['checks']['parameter_gradient_mismatches']==0
    assert evidence('stochastic_depth')['checks']['rng_continuation_exact']
    assert evidence('selection')['checks']['inconsistent_source_default_rejected']
    assert evidence('selection')['checks']['explicit_extension_flagged']
    assert evidence('runtime_full')['checks']['nonfallback_prediction']
    pins=evidence('source_pins')
    assert pins['commit']=='89c6290490bac96b29193a4061b3db9dd3933e36'
    assert pins['intake_version']=='95e3414b-c0ed-5816-b944-4466d697d9fc'
    assert pins['intake_content_hash']=='66f965f75d83ba4ec2daf15afcfa7a657589938dc98c00722b9ccea7eec92679'
    assert sha(ROOT/'docs/licenses/DFDC-MIT.txt')==next(f['sha256'] for f in pins['files'] if f['path']=='LICENSE')
    assert sha(ROOT/'docs/licenses/FacenetPytorch-MIT.txt')==next(f['sha256'] for f in evidence('facenet_source_pins')['files'] if f['path']=='LICENSE.md')
    for item in evidence('dlib_dependency_pin')['installed_license_notices']:
        assert sha(ROOT/item['path'])==item['sha256']
    provider=ROOT.parent/'sciona-atoms-dl/src/sciona/atoms/dl/dfdc_execution.py'
    assert sha(provider)==full['provider_sha256']
    from sciona.dfdc_graph import build_dfdc_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    assert encode_execution_graph(build_dfdc_graph())[0]==full['serialized_graph_sha256']
    dependencies={}
    for line in (ROOT/'requirements/dfdc-execution.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            name,version=line.split('==');assert importlib.metadata.version(name)==version
            dependencies[name]=version
    dependency_review={'format':'dfdc-execution-dependencies.v1','result':'verified_installed_environment',
        'python':platform.python_version(),'dependencies':dependencies,
        'provider_package_sha256':sha(ROOT.parent/'sciona-atoms-dl/pyproject.toml'),
        'sha256':{'requirements/dfdc-execution.txt':sha(ROOT/'requirements/dfdc-execution.txt')},
        'limits':'Exact versions verified in matcher venv; no isolated clean-install test. Native dlib built locally and contains its bundled HOG detector. No external landmark predictor or pretrained neural states bundled. Facenet source-derived modules and historical timm stochastic-depth adaptation are attributed separately.'}
    (reviews/'competition_dfdc_execution_dependencies.json').write_text(json.dumps(dependency_review,indent=2)+'\n')
    limitations=[
        'Tier3 automated Community reconstruction only; no Tier1 human certification or Tier2 usage qualification.',
        'Original intake stays draft. Source correction selects MTCNN and B7, no RetinaFace/Se-ResNeXt or inference landmark alignment.',
        'Complete decoded arrays replace media codecs; caller supplies positional parts/pairs, explicit states and native68-point predictor. No historical membership lists or weights bundled.',
        'Source frame duplicate behavior, one-face/error fallback,127-face default cap and per-checkpoint confidence heuristic are preserved.',
        'Full paired crop/landmark/difference-mask preparation, augmentation, population balancing, smoothed class-balanced BCE, SGD, mutating scheduler and numbered snapshots are included.',
        'Source submission requests7 states from5 runs, including suffix40; source default40-epoch run produces0–39. Default combination is rejected. Caller must explicitly supply an executable plan; no automatic extension/rename or historical checkpoint authentication.',
        'Graph evidence runs actual MTCNN, native synthetic dlib and5 full B7 models with2epochs/1batch1 update each and7-state prediction. Shortened plan is explicit, not full default-scale training or historical accuracy.',
        'Historical stochastic-depth function restored with exact synthetic CPU encoder outputs/gradients/RNG; historical Torch/CUDA/Apex/distributed runtime equivalence is not established.',
        'Current facenet/dlib/albumentations/skimage adaptations remain explicit; numeric conversion fixes current MTCNN object-landmark rounding. No positive pretrained detector-quality claim.',
        'DFDC execution restores caller RNG after provider discovery. First-time discovery of unrelated packages changes NumPy state; startup isolation is not claimed.',
        'Dependencies pinned to tested installed versions; clean independent installation not tested. Explicit initialization supplies pretrained states without automatic downloads.',
        'Runtime input arrays, states and native predictor paths remain private; validation uses synthetic fixtures and publishes aggregate evidence only.'
    ]
    stage_map=[
        {'intake':'frame_extraction','realization':'Decoded RGB boundary and source frame-index/fallback semantics','evidence':['frame_prediction_edges','codec','runtime_full']},
        {'intake':'face_detection','realization':'Actual three-network MTCNN with stage-specific thresholds and half-size inference','evidence':['detector','training_crops','runtime_full']},
        {'intake':'face_alignment_cropping','realization':'Source margin crop without inference alignment;5point and68point training augmentation','evidence':['faces','landmark_removal','hull_backend','crop_annotations']},
        {'intake':'deep_fake_classification','realization':'Full B7 paired-record training,5run orchestration and explicit7state selection','evidence':['records','training','ensemble_b7','stochastic_depth','graph']},
        {'intake':'temporal_aggregation','realization':'Per-checkpoint confidence heuristic then checkpoint mean','evidence':['inference_primitives','prediction','ensemble_prediction','runtime_full']}
    ]
    auxiliary=['requirements/dfdc-execution.txt','scripts/review_dfdc_execution.py',
               'docs/licenses/DFDC-MIT.txt','docs/licenses/FacenetPytorch-MIT.txt',
               'docs/licenses/Dlib-Boost.txt','docs/licenses/Timm-Apache-2.0.txt']
    semantic={'format':'dfdc-semantic-review.v1','review_source':'automated','proposed_tier':3,
              'verdict':'acceptable_with_limits','source_commit':pins['commit'],
              'source_version_id':pins['intake_version'],'source_hash':pins['intake_content_hash'],
              'serialized_graph_sha256':full['serialized_graph_sha256'],'provider_sha256':sha(provider),
              'limitations':limitations,'intake_stage_mapping':stage_map,
              'finding':'Complete corrected computation and actual serialized graph evidence support automated Tier3 selection with explicit initialization and executable checkpoint plan. This supersedes the initial implementation-in-progress source review, without changing its historical record. Catalog mutation and served retrieval gates remain separate.',
              'evidence_sha256':{name:sha(reviews/name) for name in EVIDENCE+['competition_dfdc_execution_dependencies.json']},
              'auxiliary_sha256':{path:sha(ROOT/path) for path in auxiliary},'catalog_mutations':0}
    (reviews/'competition_dfdc_semantic_review.json').write_text(json.dumps(semantic,indent=2)+'\n')
    print(json.dumps({'evidence_documents':len(semantic['evidence_sha256']),'intake_stages_mapped':len(stage_map),
                      'proposed_tier':3,'verdict':semantic['verdict'],'catalog_mutations':0}))


if __name__=='__main__':main()
