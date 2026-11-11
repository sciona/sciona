"""Review scoped Tier 3 eligibility for separate learned-state lifecycle graphs."""
import json
from pathlib import Path

from scripts.plan_nasa_lifecycle_drafts import ROOT,audit,plan,sha
from scripts.review_nasa_domain_publication import review as domain_review
from scripts.validate_nasa_lifecycle_database_gates import fault_cases

LIMITATIONS=[
    'Automated Tier 3 corrected single-airport training and inference graphs in the provisioned in-process runtime; no Tier 1 human certification or Tier 2 empirical-performance claim.',
    'Training accepts no future query records. Its metadata check uses the observed training matrix as a shape probe; unseen queries are validated when inference runs.',
    'Whole-graph static table propagation remains unsupported. These graphs are not qualified for workflows requiring a successful pre-execution static table simulation.',
    'Complete state carries native regressor/classifier models, ordered feature names, prediction feature name, signed calibration offsets, probability threshold and airport vocabulary/unit metadata.',
    'State is tied to the exact installed backend version. Digests detect corruption but do not establish authenticity; native model contents are also checked during prediction.',
    'Learned models, feature names, vocabulary and derived tables from non-public inputs are private runtime material. This catalog approval publishes no model files, records or data-derived metadata.',
    'Inference recomputes corrected domain features using the saved vocabulary and performs no fitting. Missing features retain NaN backend semantics; output rows retain query identity order.',
    'The source correction policy and explicit minute/int32/float32 boundaries remain those of the approved domain graph; original source bugs are not reproduced.',
    'Both parent and fresh inference children were qualified with optional server packages blocked. HTTP transport, clean installation and binary/package redistribution remain unqualified.',
    'Synthetic comparisons establish execution behavior, not empirical domain effectiveness. Multi-airport graph composition and original-intake completion remain separate work.',
]


def review():
    semantic,proposed,parent=audit(),plan(),domain_review()
    directory=ROOT/'docs/reviews'
    names=['nasa_lifecycle_runtime_profile.json','nasa_lifecycle_catalog_execution.json','nasa_lifecycle_database_gates.json']
    runtime,execution,gates=[json.loads((directory/name).read_text()) for name in names]
    if not runtime['passed'] or not execution['passed'] or runtime['execution']!=execution:
        raise ValueError('Matching lifecycle runtime/catalog evidence required')
    expected={kind:graph['graph_sha256'] for kind,graph in proposed['graphs'].items()}
    if execution['graph_sha256']!=expected or gates['graph_sha256']!=expected:
        raise ValueError('Lifecycle graph identity differs')
    for report,filename in [(runtime,'validate_nasa_lifecycle_runtime_profile.py'),
        (execution,'validate_nasa_lifecycle_catalog.py'),(gates,'validate_nasa_lifecycle_database_gates.py')]:
        if report['validator_sha256']!=sha(ROOT/'scripts'/filename):raise ValueError('Lifecycle qualification code drift')
    if not gates['passed'] or not gates['rollback_verified'] or gates['rejected_faults']!=[case[0] for case in fault_cases(proposed)]:
        raise ValueError('Lifecycle database fault evidence missing')
    base=json.loads((directory/'nasa_domain_runtime_profile.json').read_text())
    for key in ['manifest_sha256','psycopg_implementation','libpq_version','excluded_optional_packages']:
        if runtime[key]!=base[key]:raise ValueError('Qualified runtime changed: '+key)
    if not runtime['dependency_closure']['compatible'] or runtime['dependency_closure']['versions']!=parent['dependency_versions']:
        raise ValueError('Lifecycle dependency closure differs')
    if len(runtime['child_profiles'])!=10 or execution['catalog_inference_children']!=10:
        raise ValueError('Fresh child coverage missing')
    for child in runtime['child_profiles']:
        if not child['passed'] or child['excluded_optional_packages']!=runtime['excluded_optional_packages'] or child['psycopg_implementation']!='python':
            raise ValueError('Fresh child runtime differs')
    names+=['competition_nasa_lifecycle_graphs.json','nasa_domain_publication_review.json']
    return dict(format='nasa-lifecycle-publication-review.v1',review_source='automated',proposed_tier=3,
        eligible_for_approval_transaction=True,approved=False,catalog_mutations=0,graph_sha256=expected,
        scope=semantic['scope'],limitations=LIMITATIONS,new_state_atoms=len(proposed['atoms']),
        reused_approved_atoms=len(proposed['reused_atoms']),dependency_versions=parent['dependency_versions'],
        notice_review=parent['notice_review'],source_sha256=semantic['lifecycle_source_sha256'],
        shared_source_sha256=semantic['shared_source_sha256'],evidence_sha256={name:sha(directory/name) for name in names},
        reviewer_sha256=sha(Path(__file__)))


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/nasa_lifecycle_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,new_state_atoms=report['new_state_atoms'],graphs=len(report['graph_sha256']),approved=False)))
