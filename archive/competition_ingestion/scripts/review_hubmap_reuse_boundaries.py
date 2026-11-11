"""Implementation-bound HuBMAP reuse review; grants no catalog approval."""
import json
from pathlib import Path
from scripts.audit_hubmap_original_readiness import ROOT, sha
from scripts.review_text_pair_original_scope import anchor

BOUNDARIES = [
    dict(name='raster_targets_and_preparation',
        contract='Decoded RGB uint8 rasters and aligned binary masks feed unshifted and half-shifted tiles, mask-area ordering, JPEG roundtrip, standard-deviation filtering and rounded-area presence bins.',
        limitation='Binary foreground semantics, physical scale, color statistics and lossy preprocessing must be justified for each application. A nonempty mask can enter the background sampling population. TIFF decoding and stain normalization are not implemented by this graph.',
        anchors=[('hubmap_raw_population','prepare_population'),('hubmap_preparation','prepare_tile'),('hubmap_preparation','select_training_tiles')]),
    dict(name='population_independence',
        contract='Caller-supplied group labels partition whole rasters before tile preparation. Both training and validation must retain selected tiles. Pseudo populations append to training only.',
        limitation='Group labels do not establish independence of subjects, acquisition sessions or sources. The caller must exclude validation overlap from pseudo sources; the implementation does not detect that overlap.',
        anchors=[('hubmap_raw_training','prepare_fold'),('hubmap_retraining','prepare_retraining')]),
    dict(name='fresh_training_and_checkpoint_selection',
        contract='Each FoldStage constructs a full SE-ResNeXt101 segmentation model from explicit encoder state with fresh decoder, Adam optimizer and cosine scheduler. Ordered selectors identify produced fold/checkpoint roles; retraining starts fresh and consumes generated pseudo masks.',
        limitation='Encoder-state shape compatibility and deterministic seeds do not authenticate pretrained weights or prove accuracy. Explicit stage schedules and checkpoint roles must be executable. The graph stage functions use _train_stages directly and do not inherit all run_lifecycle selector validation.',
        anchors=[('hubmap_lifecycle','train_fold'),('hubmap_lifecycle','_train_stages'),('hubmap_graph_runtime','initial_training'),('hubmap_graph_runtime','retraining')]),
    dict(name='pseudo_label_policy',
        contract='Initial models generate masks with four flip views and threshold0.5. Ordered pseudo populations retain only rounded-bin-positive tiles for retraining.',
        limitation='Pseudo masks are model outputs, not independent ground truth. Positive-only filtering, area bins and augmentation invariance are recipe assumptions; application transfer requires justification and independent evaluation.',
        anchors=[('hubmap_graph_runtime','pseudo_labels'),('hubmap_retraining','append_pseudo'),('hubmap_augmentation','augment')]),
    dict(name='inference_geometry_and_batch_dependence',
        contract='Final inference uses three flip views, ordered model averaging, resize before strict thresholding and source grid stitching. The classifier shortcut skips the segmentation path when all examples in a batch fall below its classifier threshold.',
        limitation='Batch composition, tile scale, padding, normalization and thresholds affect meaning and output. An empty final mask can result from the classifier shortcut; it is not evidence of absent foreground. Binary outputs are not calibrated uncertainty estimates.',
        anchors=[('hubmap_graph_runtime','final_prediction'),('hubmap_raw_inference','predict_slide'),('hubmap_network','UNET_SERESNEXT101.forward'),('hubmap_inference','predict_batch')]),
    dict(name='runtime_objects_and_reusable_units',
        contract='Graph boundaries pass runtime arrays, FoldStage objects, explicit RNG streams, checkpoint bytes and evaluation models. Model restoration requires a nonempty ordered checkpoint ensemble; inference uses zero data-loader workers.',
        limitation='These provider interfaces are recipe stages and trusted runtime-object boundaries, not a generic JSON service. Image tiling, balanced sampling and aggregation need independently qualified contracts before standalone cross-domain selection.',
        anchors=[('hubmap_graph_runtime','restore_models'),('hubmap_lifecycle','FoldStage'),('hubmap_raw_inference','predict_slide')]),
]


def review():
    return dict(format='hubmap-reuse-boundaries.v1',passed=True,approved=False,catalog_mutations=0,
        review_source='automated',scope='Static implementation-bound review; full execution and publication remain separate.',
        boundaries=[{k:v for k,v in item.items() if k!='anchors'} |
                    dict(source_anchors=[anchor(*a) for a in item['anchors']]) for item in BOUNDARIES],
        reuse_decision='The recipe may transfer to other binary raster-segmentation tasks with justified foreground semantics, physical scale, color and augmentation assumptions, independent grouped populations and pseudo-source isolation. Keep explicit stage and inference policies. Generic image or sampling operations require independent qualification before standalone selection.',
        remaining=['Review original source stages and exact provider interfaces.',
                   'Qualify current full networks, source-sized geometry and complete fold routing with bounded resources.',
                   'Execute the corrected stored graph and verify transactional publication and selection.'],
        reviewer_sha256=sha(Path(__file__)))


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/hubmap_reuse_boundaries.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,boundaries=len(report['boundaries']),approved=False)))
