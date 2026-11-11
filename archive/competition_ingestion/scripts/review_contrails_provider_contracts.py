"""Qualify retained Contrails interfaces only after both full variants pass."""
import argparse
import json
import subprocess
from pathlib import Path
from scripts.audit_contrails_original_readiness import ROOT, sha
from scripts.review_text_pair_original_scope import anchor
from scripts.review_contrails_reuse_boundaries import review as reuse_review
from scripts.contrails_execution_evidence import review as execution_review

READINESS = 'docs/reviews/contrails_original_readiness.json'
REUSE = 'docs/reviews/contrails_reuse_boundaries.json'


def committed(path, commit):
    raw = subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
    if raw != (ROOT/path).read_bytes():
        raise ValueError('Immutable contract prerequisite changed')
    return json.loads(raw)


def port(direction, name, type_desc, constraints):
    return dict(direction=direction,name=name,type_desc=type_desc,ordinal=0,
                required=True,default_value_repr='',constraints=constraints)


def contracts():
    baseline = committed(READINESS, 'b07e4b40')
    reuse = committed(REUSE, 'd3fc51c7')
    if reuse != reuse_review():
        raise ValueError('Implementation-bound reuse review changed')
    bounds = {r['name']:r for r in reuse['boundaries']}
    input_contract = ('Exact version-1 envelope with training, scoring, prediction, initialization and config. '
        'Config requires integer seed in [0,2**32); optional variant v43/v47, positive integer batch_size, nonnegative integer num_workers and integer epoch_limit in [1,35]. '
        'Qualification uses explicit batch_size=1, num_workers=0 and epoch_limit=1. '
        + ' '.join(bounds[name]['contract'] for name in ('thermal_encoding','populations_and_targets','training_and_resources')))
    prepared = bounds['prepared_state']['contract']+' '+bounds['prepared_state']['limitation']
    output = bounds['output_and_decisions']['contract']+' '+bounds['output_and_decisions']['limitation']
    limits = [r['limitation'] for r in reuse['boundaries']] + [
        'Automated Tier 3 synthetic execution only; no human Tier 1, community Tier 2 or historical accuracy claim.',
        'Both full variants are required, each with one explicit epoch per fold and random initialization; default 35/40-epoch schedules and external pretrained states are not empirically qualified.',
        'Fixed spatial interpolation, shifted targets, loss masking, source fold selection, terminal-checkpoint policy and unnormalized weights remain recipe choices.',
        'Validation covers current-platform serial CPU execution; no GPU, large-input memory or clean-install qualification.']
    specifications = {
        'contrails_prepare': dict(plain='Check ordered thermal image populations, targets, training settings and initial model state, then prepare them for execution.',
            ports=[port('input','payload','dict',input_contract),port('output','prepared','object',prepared)],
            anchors=[('contrails_runtime','prepare'),('contrails_runtime','_array'),('contrails_runtime','_state')]),
        'contrails_execute': dict(plain='Train all selected models, reload their final states, and combine spatial predictions into weighted image scores and encoded masks.',
            ports=[port('input','prepared','object',prepared),port('output','result','dict',output)],
            anchors=[('contrails_runtime','execute_prepared'),('contrails_lifecycle','run_lifecycle'),('contrails_prediction','ensemble_probabilities')]),
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
        derivative=dict(plain_description='Check thermal image populations, train all selected models, and combine their spatial predictions into weighted scores and masks.',
            ports=[port('input','payload','dict',input_contract),port('output','result','dict',output)]))


def review(evidence_commit):
    execution = execution_review(evidence_commit)
    values = contracts()
    return dict(format='contrails-provider-contract-review.v1',passed=True,approved=False,
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
    (ROOT/'docs/reviews/contrails_provider_contracts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=2,approved=False,catalog_mutations=0)))
