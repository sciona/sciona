"""Qualify retained DFDC interfaces only after complete graph execution pass."""
import argparse
import json
import subprocess
from pathlib import Path
from scripts.audit_dfdc_original_readiness import ROOT, sha
from scripts.review_text_pair_original_scope import anchor
from scripts.review_dfdc_reuse_boundaries import review as reuse_review
from scripts.dfdc_execution_evidence import review as execution_review

READINESS = 'docs/reviews/dfdc_original_readiness.json'
REUSE = 'docs/reviews/dfdc_reuse_boundaries.json'


def committed(path, commit):
    raw = subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
    if raw != (ROOT/path).read_bytes():
        raise ValueError('Immutable contract prerequisite changed')
    return json.loads(raw)


def port(direction, name, type_desc, constraints):
    return dict(direction=direction,name=name,type_desc=type_desc,ordinal=0,
                required=True,default_value_repr='',constraints=constraints)


def contracts():
    baseline = committed(READINESS, '5adca315')
    reuse = committed(REUSE, '32d07a28')
    if reuse != reuse_review():
        raise ValueError('Implementation-bound reuse review changed')
    bounds = {r['name']:r for r in reuse['boundaries']}
    input_contract = ('Exact version-1 envelope with training, prediction, detector, classifier, hull, plan and config. '
        'Training entries carry numerical-transport frames, positional part in [0,49] and original index; prediction carries numerical-transport clips. '
        'Frames decode to uint8 RGB with four axes, final axis3 and spatial dimensions at least2; training clips are nonempty. '
        'Config requires record_order_seed in [0,2**32); optional n_splits in [1,50], positive batch_size, batches_per_epoch, test_every and frames_per_video, and precision float16/float32. '
        'Detector policy is random or supplied state with explicit seed. Classifier policy is random, encoder or complete per-run states. '
        'Hull requires a caller-supplied native predictor path, kept out of report fields. Runs and requests are explicit integer triples. '
        + ' '.join(bounds[name]['contract'] for name in ('media_and_targets','population_independence','training_and_checkpoints')))
    prepared = ('Trusted mutable mapping returned by preparation, containing decoded clips, states, configuration, plan and private native predictor reference. '
        'execute_prepared does not revalidate the public envelope; callers must not mutate prepared state before execution.')
    output = ('Version1 result mapping with explicit executable plan, training record count, per-run training/validation summaries and ordered predictions. '
        'Each prediction includes score, status, reason, selected frame count, classified face count, model count and precision. '
        'Inspect status: source_fallback score0.5 can indicate empty input, no faces or caught execution failure. '
        'No trained model or native predictor path is returned. Scores are not calibrated cross-domain probabilities.')
    limits = [r['limitation'] for r in reuse['boundaries']] + [
        'Automated Tier 3 synthetic execution only; no human Tier 1 or community Tier 2 qualification.',
        'Qualification uses five full B7 runs with random initialization, two epochs and one batch1 update per epoch, seven selected checkpoints and CPU float32 inference; full default-scale schedules and external pretrained states are not empirically qualified.',
        'The neural detector uses constructed synthetic state and the native predictor is trained solely on synthetic landmarks. This does not establish pretrained detector quality or positive HOG face quality.',
        'Current-platform serial execution only; caller must bound Torch, native pools and OpenCV. Source default workload sizes are not shared-budget controls.',
        'Prepared values and native files remain trusted caller inputs. No clean-install, GPU, large-input or predictive-accuracy qualification.']
    specifications = {
        'dfdc_prepare': dict(plain='Check video populations, original and altered pairs, model states, fold settings and the requested training checkpoints.',
            ports=[port('input','payload','dict',input_contract),port('output','prepared','object',prepared)],
            anchors=[('dfdc_runtime','prepare'),('dfdc_codec','decode_array'),('dfdc_selection','selection_plan')]),
        'dfdc_execute': dict(plain='Prepare face crops, train all requested models, then combine the selected checkpoints into video scores with explicit fallback status.',
            ports=[port('input','prepared','object',prepared),port('output','result','dict',output)],
            anchors=[('dfdc_runtime','execute_prepared'),('dfdc_ensemble_training','train_ensemble'),('dfdc_prediction','predict_video')]),
    }
    providers = []
    for record in baseline['records']:
        identity = record['identity']
        if identity['artifact_kind'] != 'atom':
            continue
        spec = specifications[identity['fqdn'].rsplit('.',1)[1]]
        if identity['status'] != 'approved' or identity['served'] is not True or identity['trust_tier'] != 3:
            raise ValueError('Retained approved Tier 3 provider required')
        for catalog in record['catalogs'].values():
            for before,after in zip(catalog['ports'],spec['ports'],strict=True):
                if {k:v for k,v in before.items() if k!='constraints'} != {k:v for k,v in after.items() if k!='constraints'}:
                    raise ValueError('Contract must preserve registered interface')
        providers.append(dict(artifact_id=identity['artifact_id'],runtime_fqdn=identity['fqdn'],
            version_id=identity['version_id'],content_hash=identity['content_hash'],
            plain_description=spec['plain'],ports=spec['ports'],
            source_anchors=[anchor(*a) for a in spec['anchors']],role='recipe_adapter',
            reuse=reuse['reuse_decision'],validity_bounds=limits))
    if len(providers)!=2:
        raise ValueError('Exact retained provider pair required')
    return dict(providers=providers,reuse_decision=reuse['reuse_decision'],limitations=limits,
        derivative=dict(plain_description='Check paired video populations, prepare face crops, train all requested models and combine selected checkpoints into scores with explicit fallback status.',
            ports=[port('input','payload','dict',input_contract),port('output','result','dict',output)]))


def review(evidence_commit):
    execution = execution_review(evidence_commit)
    values = contracts()
    return dict(format='dfdc-provider-contract-review.v1',passed=True,approved=False,
        review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',catalog_mutations=0,
        **values,evidence_commit=evidence_commit,execution=execution,
        readiness_sha256=sha(ROOT/READINESS),reuse_sha256=sha(ROOT/REUSE),reviewer_sha256=sha(Path(__file__)),
        remaining=['Transactional metadata repair and both native eligibility predicates.',
                   'Corrected stored original execution, transactional publication and production selection.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-commit',required=True)
    args=parser.parse_args()
    report=review(args.evidence_commit)
    (ROOT/'docs/reviews/dfdc_provider_contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=2,approved=False,catalog_mutations=0)))
