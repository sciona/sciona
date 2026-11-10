"""Qualify HuBMAP recipe interfaces after committed complete lifecycle execution."""
import argparse
import importlib
import inspect
import json
import subprocess
from pathlib import Path
from scripts.audit_hubmap_original_readiness import ROOT, sha
from scripts.review_hubmap_reuse_boundaries import review as reuse_review
from scripts.hubmap_execution_evidence import review as execution_review
from scripts.review_text_pair_original_scope import anchor

READINESS='docs/reviews/hubmap_original_readiness.json'
REUSE='docs/reviews/hubmap_reuse_boundaries.json'
CONTRACTS={
    'slides':'Ordered decoded RGB uint8 raster and aligned binary uint8 mask pairs. Caller defines foreground meaning and physical scale. Source preparation uses unshifted and half-shifted grids, JPEG roundtrip, std>10 filtering and rounded-area bins. Both foreground-bin and background-bin training populations must remain.',
    'groups':'One string or integer group label per labeled raster. Validation groups hold out whole rasters before tiling; both partitions must retain tiles. Caller must establish subject/acquisition independence and exclude validation overlap from pseudo sources.',
    'initial_stages':'Nonempty ordered FoldStage objects with explicit full encoder tensor state, integer model seed, validation group labels and training options. Options include positive ordered start/end epochs, maximum_bin, batch sizes, input_side, explicit Python Random, NumPy RandomState and CPU Torch Generator. Training batch size is at least2 and model input dimensions are multiples of32. Every fold constructs a fresh decoder, Adam optimizer and scheduler.',
    'retraining_stages':'Nonempty ordered FoldStage objects with the same explicit encoder, grouping, schedule and RNG contract as initial stages. Retraining constructs fresh model/optimizer state and appends positive pseudo tiles; it does not continue the initially trained segmentation model.',
    'initial_selectors':'Nonempty ordered pairs of configured fold index and actually produced checkpoint role. Roles are best_loss, best_score or snapshot_<epoch>. These stage adapters do not perform the separate run_lifecycle selector precheck. Invalid selectors must not be supplied; tensor-compatible checkpoints do not establish pretrained provenance.',
    'final_selectors':'Nonempty ordered pairs of retraining fold index and actually produced checkpoint role, preserving ensemble order. Callers must select checkpoints produced by their explicit schedule. Historical notebook checkpoint selection is not authenticated by shape compatibility.',
    'tile_size':'Integer tile side at least2; source default1024. Both source grid offsets must produce a nonempty grid for each raster. Tile scale changes target meaning and sampling, so application transfer requires justification.',
    'checkpoints':'Nonempty ordered tensor-only model checkpoint byte payloads. Restoration uses weights_only=True on CPU and requires exact state keys, shapes, dtypes, strided layout and finite tensors. No optimizer or training RNG state is restored.',
    'input_resolution':'Positive integer model input side compatible with the network, normally320 and a multiple of32. It must agree with inference_geometry.input_resolution. This interface does not check that agreement before model execution.',
    'models':'Nonempty ordered evaluation-only full inference model ensemble with source classifier threshold0.5. Batch-wide classifier shortcut may produce zero logits and hence empty binary predictions. Model ordering and batch composition can affect results.',
    'pseudo_sources':'Ordered collections of decoded RGB uint8 rasters from caller-approved pseudo populations. Caller must prevent validation overlap. Four-view predictions become binary masks; only rounded-bin-positive pseudo tiles are later appended to training.',
    'pseudo_rng':'Mapping with python_rng as Python Random and torch_generator as CPU Torch Generator. Streams are consumed by inference; callers control and preserve their lifecycle. No hidden global seed or device scheduling equivalence is promised.',
    'final_rng':'Mapping with python_rng as Python Random and torch_generator as CPU Torch Generator, consumed during final inference. These are mutable trusted runtime objects, not JSON values.',
    'inference_geometry':'None or a mapping containing only resolution, input_resolution, pad_size and batch_size. Defaults are1024,320,256,12. Positive integer sizes and batch size are required; padding must satisfy0<=pad_size<resolution/2. Model and inference input resolution must agree; whole decoded input and output rasters remain resident.',
    'pseudo_labeled_sources':'Ordered collections of decoded RGB uint8 rasters paired with generated binary masks. Pseudo masks are not ground truth. Retraining prepares source grids and appends only rounded-bin-positive tiles; labeled validation receives no pseudo tiles.',
    'final_images':'Ordered decoded nonempty RGB uint8 rasters for final prediction. Foreground semantics, color/normalization, physical scale and augmentation invariance must be justified for the new application.',
    'initial_checkpoints':'Ordered selected model-state byte payloads from the requested initial fold/checkpoint roles. These contain model tensors only and do not resume optimizer or RNG state.',
    'final_checkpoints':'Ordered selected model-state byte payloads from fresh retraining folds. Their shapes do not establish historical weight provenance or predictive quality.',
    'initial_metrics':'Ordered per-fold epoch records with training and validation aggregates plus checkpoint/early-stop decisions. These are training diagnostics, not independent predictive-quality evidence.',
    'retraining_metrics':'Ordered per-fold epoch records after fresh training with positive pseudo tiles. Validation metrics do not prove independence of caller-supplied populations.',
    'predictions':'Ordered full-raster uint8 binary masks using three flip views, model averaging, resize, strict threshold0.5 and interior stitching. Empty masks may reflect the batch classifier shortcut. Outputs are not calibrated uncertainty estimates.',
}
SPECS={
    'hubmap_initial_training':('Prepare independent labeled groups and train full models using explicit model states and schedules.','initial_training'),
    'hubmap_restore_models':('Restore an ordered group of inference models from compatible model-only checkpoints.','restore_models'),
    'hubmap_pseudo_labels':('Generate binary pseudo masks from an ordered model ensemble using four image views.','pseudo_labels'),
    'hubmap_retraining':('Append positive pseudo tiles and train fresh models with explicit independent validation groups.','retraining'),
    'hubmap_final_prediction':('Combine three image views and ordered model predictions into binary raster masks.','final_prediction'),
}


def committed(path):
    raw=subprocess.check_output(['git','show','993e042d:'+path],cwd=ROOT)
    if raw!=(ROOT/path).read_bytes():raise ValueError('Immutable readiness or reuse qualification changed')
    return json.loads(raw)


def contracts():
    baseline=committed(READINESS);reuse=committed(REUSE)
    if reuse!=reuse_review():raise ValueError('Implementation-bound reuse review changed')
    from sciona.hubmap_execution import build_hubmap_lifecycle_graph
    from sciona.visualizer.runner import _ensure_atoms_imported
    _ensure_atoms_imported()
    graph=build_hubmap_lifecycle_graph()
    nodes={n.matched_primitive:n for n in graph.nodes}
    limits=[b['limitation'] for b in reuse['boundaries']]+[
        'Automated Tier 3 qualification only; no human Tier 1 designation.',
        'Current qualification uses synthetic random encoder state and the source epoch schedule with resource-adapted batch size. No historical final checkpoint policy/provenance, predictive accuracy or clean-install claim.',
        'Caller must bound CPU threads and memory. Model ensembles, checkpoint bytes, full decoded inputs and output masks remain resident; zero loader workers is not a memory limit.']
    providers=[];derivative=None
    for record in baseline['records']:
        identity=record['identity']
        if identity['artifact_kind']!='atom':
            if identity['fqdn'].endswith('.execution'):
                derivative=dict(plain_description='Train grouped raster models, generate pseudo masks, retrain fresh models and combine predictions with explicit geometry and target assumptions.',
                    ports=[dict(p,constraints=CONTRACTS[p['name']]) for p in record['catalogs']['artifact']['ports']])
            continue
        name=identity['fqdn'];symbol=name.rsplit('.',1)[1]
        if name not in nodes or symbol not in SPECS or identity['status']!='approved' or identity['trust_tier']!=3:
            raise ValueError('Exact retained Tier 3 provider required')
        fn=getattr(importlib.import_module(name.rsplit('.',1)[0]),symbol)
        signature=inspect.signature(fn)
        node=nodes[name]
        if list(signature.parameters)!=[p.name for p in node.inputs]:raise ValueError('Callable input order changed')
        ports=record['catalogs']['artifact']['ports']
        if record['catalogs']['atom']['ports']!=ports:raise ValueError('Provider catalog interfaces differ')
        for direction,items in [('input',node.inputs),('output',node.outputs)]:
            registered=sorted((p for p in ports if p['direction']==direction),key=lambda p:p['ordinal'])
            if [(p['name'],p['type_desc']) for p in registered]!=[(p.name,p.type_desc) for p in items]:
                raise ValueError('Graph interface changed')
            if direction=='input':
                for p in registered:
                    param=signature.parameters[p['name']]
                    required=param.default is inspect.Parameter.empty
                    if p['required']!=required or p['default_value_repr']!=('' if required else repr(param.default)):
                        raise ValueError('Callable defaults differ')
        providers.append(dict(artifact_id=identity['artifact_id'],runtime_fqdn=name,version_id=identity['version_id'],
            content_hash=identity['content_hash'],plain_description=SPECS[symbol][0],
            ports=[dict(p,constraints=CONTRACTS[p['name']]) for p in ports],
            source_anchors=[anchor('hubmap_graph_runtime',SPECS[symbol][1])],role='recipe_adapter',
            reuse=reuse['reuse_decision'],validity_bounds=limits))
    if len(providers)!=5 or derivative is None:raise ValueError('Five providers and retained derivative required')
    return dict(providers=providers,derivative=derivative,reuse_decision=reuse['reuse_decision'],limitations=limits)


def review(commit):
    execution=execution_review(commit)
    values=contracts()
    values['limitations']=list(dict.fromkeys(values['limitations']+execution['limits']))
    for provider in values['providers']:provider['validity_bounds']=values['limitations']
    return dict(format='hubmap-provider-contract-review.v1',passed=True,approved=False,review_source='automated',
        proposed_tier=3,verdict='acceptable_with_limits',catalog_mutations=0,**values,evidence_commit=commit,execution=execution,
        readiness_sha256=sha(ROOT/READINESS),reuse_sha256=sha(ROOT/REUSE),reviewer_sha256=sha(Path(__file__)),
        remaining=['Transactional metadata repair and both native eligibility predicates.',
                   'Resolve historical final-selection coverage, execute corrected stored original and verify publication.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence-commit',required=True)
    args=parser.parse_args();report=review(args.evidence_commit)
    (ROOT/'docs/reviews/hubmap_provider_contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(report['providers']),approved=False,catalog_mutations=0)))
