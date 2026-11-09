"""Source-bound contracts for reuse of the retained grid transformation solver."""
import json
import subprocess
from pathlib import Path

from scripts.import_flavours_provider_drafts import ROOT
from scripts.promote_arc_execution import review as existing_review, sha
from scripts.review_text_pair_original_scope import anchor

COMMIT = '1953ff38'
READINESS = 'docs/reviews/arc_original_readiness.json'
EXECUTION = 'docs/reviews/arc_retained_execution_revalidation.json'
INPUT = ('Exact version-1 dict with task and budget only. Task has train and test lists, '
         '1..20 input/output training pairs and 1..20 unlabeled test inputs. Grids are rectangular '
         '1..30 by 1..30 lists of non-boolean integers 0..9. Budget has seconds and memory_mib, '
         'finite positive non-boolean numbers at most 30780 seconds and 15564.8 MiB. '
         'No caller paths, test answers, arbitrary executable or alternate search plan.')
PREPARED = ('Trusted immutable sciona.arc_runtime.Prepared returned by arc_prepare, containing '
            'tuple-backed validated task and explicit budgets. In-process use only; execute checks '
            'the exact Prepared type and budgets but does not revalidate a manually forged Task. '
            'Do not construct or deserialize arbitrary state objects.')
OUTPUT = ('Version-1 dict with ordered per-test ranked candidate grids and scores, four search arguments '
          '3/23/33/4, explicit time/memory budget, per-attempt status/time/peak sampled RSS, missing test '
          'indices, completion flag and reviewed build identity. Up to three unique grids per test, '
          'descending score then serialized-grid order. Partial training agreement may be accepted. '
          'Scores are heuristic and are not probabilities; predictions are not saved executable programs. '
          'Limited runs may return incomplete predictions; missing runs never imply successful zero-grid output.')
LIMITS = [
    'Automated Tier 3 synthetic evidence only; no human Tier 1 review or historical competition accuracy.',
    'Caller supplies a faithful categorical-grid encoding. No numeric units, spatial resolution, physical laws or domain meaning are inferred.',
    'Search is restricted to the pinned image DSL, cost rules, composition and candidate scoring. No universal program synthesis or cross-domain accuracy guarantee.',
    'The exact reviewed macOS arm64 bundle must be provisioned through deployment configuration. No runtime download or automatic build.',
    'Adaptive phases use up to four solver children, then two in the last phase. Callers must reserve the total allocation before invoking; two-input validation uses at most two children.',
    'Process RSS is sampled and excludes parent memory. Global monotonic deadline and explicit kill/reap differ from the original monitor; budgets are not instantaneous operating-system limits.',
    'Timing-dependent baseline estimates and resource contention can affect the search. A successful invocation may still report incomplete runs.',
]
REUSE = ('Use for discrete grid transformation tasks in other disciplines only when callers explicitly '
         'encode their examples as 0..9 categorical grids and the pinned DSL expresses the desired rules. '
         'Preparation is separately selectable. The execution provider is a source-specific search recipe, '
         'not a generic numerical primitive. C++ image operations, search state and scoring are coupled '
         'inside the reviewed binary; exposing them as independent Python atoms would require new interfaces '
         'and independent qualification. Do not claim that existing opaque wrappers provide that decomposition.')


def port(direction, name, type_desc, constraints):
    return dict(direction=direction, name=name, type_desc=type_desc, ordinal=0,
                required=True, default_value_repr='', constraints=constraints)


def review():
    baseline = {}
    for path in (READINESS, EXECUTION):
        raw = subprocess.check_output(['git', 'show', COMMIT + ':' + path], cwd=ROOT)
        if raw != (ROOT/path).read_bytes():
            raise ValueError('Committed ARC revalidation evidence changed')
        baseline[path] = json.loads(raw)
    existing_review(ROOT)
    execution = baseline[EXECUTION]
    if execution['catalog_graph_runtime_comparison'] != 'passed' or execution['full_source_search_runs'] != 8:
        raise ValueError('Full retained source execution evidence required')
    records = baseline[READINESS]['records']
    providers = []
    specifications = {
        'arc_prepare': dict(plain='Check example grids and search limits, then keep an immutable copy for the solver.',
            ports=[port('input', 'payload', 'dict', INPUT), port('output', 'prepared', 'object', PREPARED)],
            anchors=[('arc_contract', '_grid'), ('arc_contract', 'validate_task'), ('arc_runtime', 'prepare')]),
        'arc_execute': dict(plain='Search for grid transformations that explain the examples. Return ranked guesses and report unfinished searches.',
            ports=[port('input', 'prepared', 'object', PREPARED), port('output', 'result', 'dict', OUTPUT)],
            anchors=[('arc_runtime', 'execute'), ('arc_runtime', 'reviewed_binary'), ('arc_schedule', 'run_schedule'),
                     ('arc_schedule', 'commands_for_phase'), ('arc_process', 'run_once'), ('arc_contract', 'merge_answers')]),
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
    return dict(format='arc-provider-contract-review.v1', passed=True, approved=False,
        review_source='automated', proposed_tier=3, verdict='acceptable_with_limits', catalog_mutations=0,
        providers=providers, reuse_decision=REUSE, limitations=LIMITS,
        derivative=dict(plain_description='Use example grid pairs to search for transformations, then return ranked guesses for new grids with explicit search limits and completion status.',
            ports=[port('input','payload','dict',INPUT), port('output','result','dict',OUTPUT)]),
        evidence_commit=COMMIT, readiness_sha256=sha(ROOT/READINESS), execution_sha256=sha(ROOT/EXECUTION),
        reviewer_sha256=sha(Path(__file__)),
        remaining=['Transactional metadata reconciliation and both native eligibility checks.',
                   'Original source-scope review, corrected stored execution, publication and served selection.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/arc_provider_contracts.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(report['providers']), approved=False, catalog_mutations=0)))
