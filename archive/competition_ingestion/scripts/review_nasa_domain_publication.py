"""Review scoped automated Tier 3 eligibility for the materialized domain graph."""
import json
from pathlib import Path

from scripts.plan_nasa_domain_draft import ROOT, audit, plan, sha
from scripts.review_residual_classifier_publication import review as numerical_review, expected_faults

LIMITATIONS = [
    'Automated Tier 3 corrected single-airport domain workflow in the provisioned in-process runner only; no Tier 1 human certification or Tier 2 empirical usage claim.',
    'Whole-graph static table propagation is unsupported. Do not select these adapters for workflows requiring a successful pre-execution whole-graph Ghost simulation.',
    'After the feature joins materialize, a mandatory checkpoint validates observed array shapes, schema and time dimension through all 25 numerical witnesses before model fitting.',
    'Caller supplies explicit runtime tables in one timezone-naive clock, one airport per invocation, and the documented public software identity convention. No implicit dataset discovery or loading.',
    'Corrected behavior includes final/singleton entities, fixed fallback categories, deterministic vocabulary order and observation-availability-safe history snapshots; original bugs are not reproduced.',
    'Training uses source inner/left joins and complete-row filtering; inference retains missing features as NaN and restores original query identities.',
    'Inputs, derived tables, learned category names and native model state remain private runtime material when inputs are non-public. This transaction publishes catalog metadata, not records or model files.',
    'HTTP transport, clean installation and binary/package redistribution are not qualified. Existing dependency-notice scope and local package dispositions remain unchanged.',
    'Synthetic source comparisons establish implemented behavior, not empirical competition performance. Multi-airport graph composition and persistent training lifecycle remain separate work.',
    'The numerical atoms remain cross-domain reusable; these domain adapters deliberately expose airport-specific field and clock semantics.',
]


def review():
    semantic, proposed, core = audit(), plan(), numerical_review()
    directory=ROOT/'docs/reviews'
    names=['nasa_domain_runtime_profile.json','nasa_domain_catalog_execution.json','nasa_domain_database_gates.json']
    runtime,execution,gates=[json.loads((directory/name).read_text()) for name in names]
    if not runtime['passed'] or not execution['passed'] or runtime['execution']!=execution:
        raise ValueError('Matching fresh runtime and catalog execution required')
    if execution['graph_sha256']!=semantic['graph_sha256'] or execution['exact_bindings_verified']!=40:
        raise ValueError('Domain execution identity differs')
    for report,filename in [(runtime,'validate_nasa_domain_runtime_profile.py'),
                            (execution,'validate_nasa_domain_catalog.py'),(gates,'validate_nasa_domain_database_gates.py')]:
        if report['validator_sha256']!=sha(ROOT/'scripts'/filename):raise ValueError('Qualification code drift')
    if not execution['execution']['materialized_numerical_preflight_before_fitting'] or execution['execution']['whole_graph_symbolic_qualified']:
        raise ValueError('Expected staged metadata qualification scope differs')
    expected=expected_faults(proposed)+['removed_handoff_edge','numerical_provenance_metadata']
    for index,atom in enumerate(proposed['reused_atoms']):expected.extend([f'reused_hash_{index}',f'reused_approval_{index}'])
    if not gates['passed'] or not gates['rollback_verified'] or gates['rejected_faults']!=expected or gates['graph_sha256']!=semantic['graph_sha256']:
        raise ValueError('Domain negative database gates missing')
    base=json.loads((directory/'residual_classifier_runtime_profile.json').read_text())
    for key in ['manifest_sha256','psycopg_implementation','libpq_version','excluded_optional_packages']:
        if runtime[key]!=base[key]:raise ValueError('Qualified runtime scope drift: '+key)
    if not runtime['dependency_closure']['compatible'] or runtime['dependency_closure']['versions']!=core['dependency_versions']:
        raise ValueError('Domain dependency closure differs')
    names+=['competition_nasa_domain_graph.json','residual_classifier_publication_review.json',
            'residual_classifier_dependency_notices.json','residual_classifier_tokenizers_notice.json']
    return dict(format='nasa-domain-publication-review.v1',review_source='automated',proposed_tier=3,
        eligible_for_approval_transaction=True,approved=False,catalog_mutations=0,
        graph_sha256=semantic['graph_sha256'],scope=semantic['scope'],limitations=LIMITATIONS,
        execution_source_sha256=semantic['source_sha256'],dependency_versions=core['dependency_versions'],
        notice_review=core['notice_review'],reused_approved_atoms=len(proposed['reused_atoms']),
        new_domain_atoms=len(proposed['atoms']),evidence_sha256={name:sha(directory/name) for name in names},
        reviewer_sha256=sha(Path(__file__)))


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/nasa_domain_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=report['eligible_for_approval_transaction'],proposed_tier=3,approved=False)))
