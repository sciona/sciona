"""Source-bound contracts for reuse of the retained audio-event ensemble."""
import json
import subprocess
from pathlib import Path

from scripts.import_flavours_provider_drafts import ROOT
from scripts.audit_cornell_original_readiness import sha
from scripts.validate_cornell_current_graph import prerequisites
from scripts.review_text_pair_original_scope import anchor

COMMIT = 'c1d23e39'
READINESS = 'docs/reviews/cornell_original_readiness.json'
EXECUTION = 'docs/reviews/cornell_current_graph.json'
INPUT = ('Exact version-1 dict containing populations, background and short-noise banks, inference, epochs, batch_size, seed and initialization. '
         'Exactly four fourfold and five fivefold training/validation splits; unique nonempty record keys within each population and disjoint keys within each split. '
         'Finite nonempty mono arrays explicitly sampled at 32000 Hz; primary and distinct secondary class indices in 0..263. Training requires a primary label; empty-label validation records are permitted. '
         'Nonempty noise banks, each waveform at least 3200 samples with finite positive float32 RMS. Inference contains only sample_rate and waveform. '
         'At least two epochs, positive even batch size at least two with a complete training batch per fold, bounded integer seed, and explicit synthetic or provisioned_backbone initialization. '
         'Caller owns consistent units, label meaning, fold membership and population independence; no audio decoding, resampling or automatic duplicate detection.')
PREPARED = ('Trusted sciona.cornell_contract.Prepared containing a detached deep copy of the request. The frozen dataclass contains mutable arrays and dictionaries; it is not deeply immutable. '
            'Execution revalidates the payload, but concurrent mutation and untrusted object deserialization are not qualified. '
            'Pinned source and historical augmentation software must be provisioned via deployment configuration; provisioned initialization additionally requires a complete DenseNet121 backbone state.')
OUTPUT = ('Finite Python dict containing models_completed=13, seventeen training-phase summaries including four continuations, integer window_votes and boolean window_decisions for each five-second interval and 264 output slots, '
          'plus 264 whole_record_decisions. Clip and frame threshold is fixed at 0.3; four of thirteen model votes select a window. Whole-record decisions require four already-selected windows. '
          'Final partial windows are allowed; padding and endpoints are excluded by half-open intervals. Singleton and empty events are supported. '
          'Training summaries report selected epoch and validation score; they are not unbiased performance estimates or calibrated probabilities. No named submission formatting or persisted reusable model is returned.')
LIMITS = [
    'Automated Tier 3 synthetic execution only; no human Tier 1 review, historical training parity or independent predictive accuracy.',
    'Fixed 32000 Hz mono waveform representation, 264 output slots, full DenseNet121 sound-event heads, fourfold/fivefold ensemble and fixed threshold/voting policy remain explicit recipe choices.',
    'All thirteen models and seventeen phases execute. Four mixup members use model-only continuation with fresh optimizer state. Epoch and batch controls are caller supplied; qualification uses two epochs per phase and explicit random initialization.',
    'Pinned historical noise augmentation, class-balanced sampling with replacement, training drop-last, two validation clips, two-epoch validation cadence and five score-ranked checkpoints are retained. Best score/latest retained tie replaces hand-picked historical checkpoints.',
    'Ten evaluation copies are averaged despite evaluation disabling spectrogram masking. Primary-label dropout loss remains stochastic during validation for the configured members. No probability calibration is claimed.',
    'External backbone training provenance is not certified by shape checks. Noise and audio inputs stay runtime values; linked notebook code is comparison evidence only and is not copied into runtime.',
    'Fold separation checks record keys within each supplied split; they do not establish subject separation, correct cross-fold membership or inference independence. Caller must enforce those properties for the intended application.',
    'Serial current-platform execution used one numerical thread. Runtime callers must separately bound thread pools and reserve the shared worker budget. No large-input memory, GPU or clean-install qualification.',
]
REUSE = ('Reusable for other weakly supervised audio-event applications when callers supply faithful mono 32000 Hz waveforms, consistent class-index meaning across 264 output slots, meaningful fold populations and appropriate noise banks. '
         'No bird taxonomy or domain meaning is inferred. Preparation can be selected independently; execution is the complete source-specific ensemble recipe. '
         'STFT, neural training, augmentation and voting helpers are not newly registered independent primitives here; their interfaces and fixed source assumptions would require separate qualification before independent selection. '
         'Cross-domain applicability is a contract possibility, not demonstrated predictive effectiveness.')


def port(direction, name, type_desc, constraints):
    return dict(direction=direction, name=name, type_desc=type_desc, ordinal=0,
                required=True, default_value_repr='', constraints=constraints)


def review():
    baseline = {}
    for path in (READINESS, EXECUTION):
        raw = subprocess.check_output(['git', 'show', COMMIT + ':' + path], cwd=ROOT)
        if raw != (ROOT/path).read_bytes():
            raise ValueError('Committed Cornell revalidation evidence changed')
        baseline[path] = json.loads(raw)
    prerequisites()
    execution = baseline[EXECUTION]
    if (execution['passed'] is not True
            or (execution['models_completed'], execution['training_phases'], execution['continuations']) != (13, 17, 4)
            or execution['actual_runner_nodes'] != 2
            or not all(execution[key] is True for key in (
                'all_nodes_uncached', 'exact_state_routing', 'full_thirty_second_training',
                'ten_copy_inference', 'strict_json_output', 'synthetic_only'))):
        raise ValueError('Full current source execution evidence required')
    for path,digest in execution['software_sha256'].items():
        if sha(ROOT/path) != digest: raise ValueError('Executed software changed')
    records = baseline[READINESS]['records']
    providers = []
    specifications = {
        'cornell_prepare': dict(plain='Check audio arrays, labels, fold populations, noise banks and training settings, then copy the request.',
            ports=[port('input','payload','dict',INPUT),port('output','prepared','object',PREPARED)],
            anchors=[('cornell_contract','prepare'),('cornell_population','prepare_records'),('cornell_population','check_split')]),
        'cornell_execute': dict(plain='Train the full audio-event ensemble, reload selected models, and combine their votes for short intervals and the whole recording.',
            ports=[port('input','prepared','object',PREPARED),port('output','result','dict',OUTPUT)],
            anchors=[('cornell_contract','execute'),('cornell_lifecycle','execute'),('cornell_training','train_phase'),('cornell_lifecycle','predict'),('cornell_voting','vote')]),
    }
    for record in records:
        identity = record['identity']
        if identity['artifact_kind'] != 'atom':
            continue
        name = identity['fqdn'].rsplit('.', 1)[1]
        specification = specifications[name]
        if identity['status'] != 'approved' or not identity['served'] or identity['trust_tier'] != 3:
            raise ValueError('Expected retained Tier 3 provider')
        for catalog in record['catalogs'].values():
            for before, after in zip(catalog['ports'], specification['ports'], strict=True):
                if {k:v for k,v in before.items() if k != 'constraints'} != {k:v for k,v in after.items() if k != 'constraints'}:
                    raise ValueError('Review must preserve callable interfaces')
        providers.append(dict(artifact_id=identity['artifact_id'], runtime_fqdn=identity['fqdn'],
            version_id=identity['version_id'], content_hash=identity['content_hash'],
            plain_description=specification['plain'], ports=specification['ports'],
            source_anchors=[anchor(*a) for a in specification['anchors']],
            role='recipe_adapter', reuse=REUSE, validity_bounds=LIMITS))
    if len(providers) != 2:
        raise ValueError('Two existing providers required')
    return dict(format='cornell-provider-contract-review.v1', passed=True, approved=False,
        review_source='automated', proposed_tier=3, verdict='acceptable_with_limits', catalog_mutations=0,
        providers=providers, reuse_decision=REUSE, limitations=LIMITS,
        derivative=dict(plain_description='Check audio and fold populations, train the full ensemble, and combine model votes into interval and whole-record decisions.',
            ports=[port('input','payload','dict',INPUT), port('output','result','dict',OUTPUT)]),
        evidence_commit=COMMIT, readiness_sha256=sha(ROOT/READINESS), execution_sha256=sha(ROOT/EXECUTION),
        reviewer_sha256=sha(Path(__file__)),
        remaining=['Transactional metadata reconciliation and both native eligibility checks.',
                   'Original source-scope review, corrected stored execution, publication and served selection.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/cornell_provider_contracts.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(report['providers']), approved=False, catalog_mutations=0)))
