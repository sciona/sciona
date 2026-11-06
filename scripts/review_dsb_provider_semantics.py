"""Bound automated Tier 3 review of the exact staged DSB provider versions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from scripts.inventory_dsb_decomposed_providers import inventory, ROOT, sha
from scripts.plan_dsb_provider_interfaces import plan as interface_plan
from scripts.audit_dsb_static_dependencies import audit
from scripts.qualify_dsb_runtime_environment import environment
from scripts.complete_dsb_draft_metadata import OPERATIONS

REPORT=ROOT/'docs/reviews/dsb_provider_semantic_review.json'
TESTS=['tests/'+name+'.py' for name in (
    'test_threshold_sampling','test_volume_tiles','test_regular_volume_resampling','test_coordinate_scaling',
    'test_centered_window','test_ranked_selection','test_thresholded_log_penalty',
    'test_dsb_decomposed_sampling','test_dsb_decomposed_pooling','test_dsb_decomposed_loss',
    'test_dsb_decomposed_runtime','test_dsb_decomposed_preparation','test_dsb_decomposed_annotations',
    'test_dsb_decomposed_evaluation','test_dsb_spatial_recipes','test_dsb_lifecycle_contracts',
    'test_dsb_ranked_loss','test_dsb_checkpoint_adaptation','test_dsb_inference_steps',
    'test_dsb_augmentation','test_dsb_schedule')]
SPECIAL={
 'augment_classifier':'Source rotation changes the image but not the coordinate grid; disabled by default. Preserve configured flips and axis swaps; no claim of geometrically corrected augmentation.',
 'augment_detection':'Source flip positions use size-coordinate. Finite positive-size targets and boxes; explicit caller RandomState advances in source order.',
 'assign_anchor_labels':'Selected-target anchor assignment, not generic assignment of every surrounding box. Caller-owned Python random.Random; None target means a negative crop. Historical Python 2 random.sample equivalence is not claimed.',
 'crop_classifier_proposal':'Retain clamped crop starts and padded-frame coordinates, Python 2 integer crop halves, source jitter and scale rules. Returned images are not yet normalized. Explicit caller RandomState required.',
 'crop_detection':'Retain source random-crop and scale policies with explicit caller RandomState. Missing target requires unscaled random crop; crop size must be divisible by stride.',
 'restore_shared_training_runtime':'CPU source network classes, shared detector identity, fresh SGD optimizers with source momentum and weight decay. Model warm restart only: optimizer and RNG state are not restored.',
 'restore_inference_models':'Separate source detector and classifier networks on CPU, restored from caller checkpoint bytes and placed in eval mode. Checkpoint provenance is caller responsibility.',
 'dsb_adapt_detector':'Transfer caller detector weights into the source classifier and initialize a new case head using caller Torch RNG. Epoch-zero model-only checkpoint; not a trained classifier or prediction-quality claim.',
 'train_shared_runtime':'Qualified source networks and persistent factory classes, shared detector identity, matching optimizer ownership, explicit orders and random streams. Variant 3 epochs 49–51 and variant 4 epochs 159–161 are source-qualified. Arbitrary callbacks, replacement models/factories, other schedules and full competition training are not certified.',
 'export_model_checkpoint':'In-memory bytes containing only epoch and CPU-copied state_dict; no optimizer or RNG resume state. The caller controls storage. Concrete model required.',
 'prepare_raw_collections':'Explicit source training/validation population roles and voxel/world coordinate conventions; orientation and sentinel policies are not interchangeable. No file loading or inferred provenance.',
 'preprocess_volume':'Source segmentation and cleanup with requested-spacing crop conventions. Effective resizing spacing may differ after shape rounding. This is a source preprocessing recipe, not a universal image-cleaning method.',
 'validation_batches':'Caller-owned source arrays, orders and persistent random streams. Keep the three source validation population roles distinct; factory iteration remains in process.',
 'dsb_decomposed_training_batches':'Caller-owned source arrays, orders and persistent random streams. Preserve threshold multiplicities, sampled proposal order, partial batches, and source spatial adapters across passes.',
 'validate_detector':'Source detector model and nonempty batches. Sets eval mode, disables gradients, averages each batch equally and sums counts. training=False excludes ranked hard-negative selection.',
 'validate_classifier':'Source classifier model and nonempty batches. Sets eval mode; weights losses by case count. Uses strict score > .5; model and batch objects must satisfy source contracts.',
 'predict_detector_tiles':'Source eval-mode detector and aligned image/coordinate tiles; serial ordered CPU batches. No substitute model architecture or non-eval state is qualified.',
 'predict_classifier_crops':'Source eval-mode classifier and crops in the expected normalized coordinate convention. Model scores are not a calibration or predictive-quality guarantee.',
 'prepare_classifier_crops':'Source score order, top-k proposal selection, empty-slot retention and padded coordinate frame. Crops must match the source classifier architecture.',
 'prepare_inference_geometry':'Source stride four, max stride sixteen and global padded-volume coordinates. Tile geometry must satisfy the validated divisibility and margin checks.',
 'decode_detector_proposals':'Source fixed anchor sizes, stride four, score threshold strictly greater than -3 and cubic proposal encoding.',
 'filter_classifier_proposals':'Source score threshold strictly greater than -1. Scores are not interpreted as probabilities.',
 'suppress_classifier_proposals':'Source cubic-box overlap convention and threshold .05, with retained score ordering.',
 'normalize_and_align_tiles':'Matching paired tile grids and source (float32(image)-128)/128 intensity transform; caller-supplied image semantics.',
}


def review(run_tests=False):
    current=inventory(offline=True);contracts=interface_plan();static=audit()
    path=ROOT/'docs/reviews/dsb_corrected_intake_execution.json'
    execution=json.loads(path.read_text())
    if not execution['passed'] or not execution['stored_candidate'] or not execution['synthetic_only']:
        raise ValueError('Stored full-graph qualification required')
    if environment()!=execution['environment']:raise ValueError('Qualified environment changed')
    if [(c['variant'],c['executed_nodes'],c['exact_checkpoint_rng_validation'],c['exact_final_inference']) for c in execution['cases']]!=[(3,19,True,True),(4,19,True,True)]:
        raise ValueError('Complete qualification cases required')
    if contracts!=json.loads((ROOT/'docs/reviews/dsb_provider_interface_plan.json').read_text()):
        raise ValueError('Interface evidence drift')
    if static!=json.loads((ROOT/'docs/reviews/dsb_static_dependency_audit.json').read_text()):
        raise ValueError('Static source evidence drift')
    by_runtime={r['runtime_fqdn']:r for r in contracts['provider_contracts']};providers=[]
    for record in current['providers']:
        runtime=record['runtime_fqdn'];name=runtime.rsplit('.',1)[1];contract=by_runtime[runtime]
        generic=record['classification']=='generic_operation'
        if not generic and name not in SPECIAL:raise ValueError('Missing recipe semantic review')
        bounds=contract['operation_contract'] if generic else SPECIAL[name]
        providers.append(dict(runtime_fqdn=runtime,catalog_fqdn=record['catalog_fqdn'],version_id=record['version_id'],
            content_hash=record['content_hash'],provider_sha256=record['provider_sha256'],
            reuse_class='generic' if generic else 'recipe_adapter',
            operation=OPERATIONS.get(name,'Initialize a source classifier from explicit detector weights.'),
            validity_bounds=bounds,interfaces=contract['ports']))
    if len(providers)!=34 or sum(p['reuse_class']=='generic' for p in providers)!=10:
        raise ValueError('Semantic review coverage differs')
    if run_tests:
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        env['PYTHONPATH']=str(ROOT);env['MPLCONFIGDIR']='/private/tmp/sciona-mpl'
        with tempfile.TemporaryDirectory(prefix='dsb-semantic-') as temp:
            junit=Path(temp)/'results.xml'
            command='import torch,pytest,sys; torch.set_num_threads(1); torch.set_num_interop_threads(1); sys.exit(pytest.main(sys.argv[1:]))'
            result=subprocess.run([sys.executable,'-c',command,'-q',*TESTS,'--junitxml='+str(junit)],cwd=ROOT,env=env,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stdout+result.stderr)
            suites=list(ET.parse(junit).getroot().iter('testsuite'))
            counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
            if counts['tests']<100 or any(counts[k] for k in ('failures','errors','skipped')):
                raise ValueError('Complete passing semantic tests required')
    else:counts=json.loads(REPORT.read_text())['test_results']
    result=dict(passed=True,approved=False,catalog_mutations=0,proposed_tier=3,review_source='automated',
        providers=providers,provider_count=34,execution_evidence_sha256=sha(path),
        static_audit_sha256=sha(ROOT/'docs/reviews/dsb_static_dependency_audit.json'),
        environment=environment(),test_results=counts,test_source_sha256={p:sha(ROOT/p) for p in TESTS},
        reviewer_sha256=sha(Path(__file__)),limitations=[
            'Automated Tier 3 review only; no Tier 1 human review or Tier 2 community-usage evidence claimed.',
            'Ten generic operations support cross-domain reuse within explicit numerical contracts; source recipes retain their conventions.',
            'Exact CPU library versions and provisioned in-process runtime only; no clean-install or historical-library reproduction claim.',
            'Synthetic computational parity does not establish predictive quality, calibration, or domain validity.',
            'Model/factory/callback objects require the qualified bindings; arbitrary replacements are not certified.',
            'Publication rollback/corruption gates and production selection remain required.'])
    if not run_tests and result!=json.loads(REPORT.read_text()):raise ValueError('Semantic review evidence changed')
    return result


if __name__=='__main__':
    report=review(run_tests=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','provider_count','test_results','proposed_tier')}))
