"""Review scoped automated Tier 3 eligibility for the population graph family."""
import json
from pathlib import Path

from scripts.plan_nasa_population_drafts import ROOT,audit,plan,sha
from scripts.review_nasa_lifecycle_publication import review as lifecycle_review,LIMITATIONS as LIFECYCLE_LIMITATIONS
from scripts.validate_nasa_population_database_gates import fault_cases

LIMITATIONS=[
    'Automated Tier 3 explicit ten-population training and inference variants for one through ten active populations in the provisioned in-process runtime; no Tier 1 certification or Tier 2 empirical-effectiveness claim.',
    'Select the fixed graph variant from validated active population count before execution. Every lifecycle branch remains visible; routing atoms do not execute nested graphs or models. Exhaustion checks reject count mismatches.',
    'Training requires labeled rows and an explicit finite nonnegative residual cutoff for every declared slot. Inference requires a nonempty query batch and complete matching saved states; unqueried states may be present but their prediction branches do not execute.',
    'Four keyed routing atoms operate on domain-independent mappings without mutating caller-owned values. Five airport adapters enforce domain partitioning, saved-state identity and int32 minute output alignment; all thirty reused atoms keep their prior contracts.',
    'The graph documents retain their draft-generation metadata. Current catalog approval, trust tier and this review govern selection; original-intake completion is not established by publishing these realizations.',
    *LIFECYCLE_LIMITATIONS[1:9],
    'Synthetic evidence covers source calibration, all ten source configurations, shuffled query identities, every active-count inference variant and normal cold provider discovery. It does not establish empirical competition effectiveness.',
]


def review():
    semantic,proposed,parent=audit(),plan(),lifecycle_review()
    directory=ROOT/'docs/reviews'
    names=['nasa_population_runtime_profile.json','nasa_population_catalog_execution.json','nasa_population_database_gates.json']
    runtime,execution,gates=[json.loads((directory/name).read_text()) for name in names]
    if not runtime['passed'] or not execution['passed'] or runtime['execution']!=execution:
        raise ValueError('Matching population runtime/catalog evidence required')
    expected={key:graph['graph_sha256'] for key,graph in proposed['graphs'].items()}
    if execution['graph_sha256']!=expected or gates['graph_sha256']!=expected:
        raise ValueError('Population graph identities differ')
    for report,filename in [(runtime,'validate_nasa_population_runtime_profile.py'),
        (execution,'validate_nasa_population_catalog.py'),(gates,'validate_nasa_population_database_gates.py')]:
        if report['validator_sha256']!=sha(ROOT/'scripts'/filename):raise ValueError('Population qualification code drift')
    checker=sha(ROOT/'scripts/check_nasa_population_catalog.py')
    if execution['checker_sha256']!=checker or gates['checker_sha256']!=checker:
        raise ValueError('Population integrity checker drift')
    if not gates['passed'] or not gates['rollback_verified'] or gates['committed_catalog_mutations']!=0:
        raise ValueError('Population database rollback evidence missing')
    if gates['rejected_faults']!=[case[0] for case in fault_cases(proposed)]:
        raise ValueError('Population fault coverage differs')
    base=json.loads((directory/'nasa_lifecycle_runtime_profile.json').read_text())
    for key in ['manifest_sha256','psycopg_implementation','libpq_version','excluded_optional_packages']:
        if runtime[key]!=base[key]:raise ValueError('Qualified runtime changed: '+key)
    if not runtime['dependency_closure']['compatible'] or runtime['dependency_closure']['versions']!=parent['dependency_versions']:
        raise ValueError('Population dependency closure differs')
    if len(runtime['child_profiles'])!=10 or execution['catalog_inference_children']!=10:
        raise ValueError('All guarded inference variants required')
    for child in runtime['child_profiles']:
        if not child['passed'] or child['excluded_optional_packages']!=runtime['excluded_optional_packages'] or child['psycopg_implementation']!='python':
            raise ValueError('Child runtime differs')
    names+=['competition_nasa_population_graphs.json','nasa_population_registration.json','nasa_population_draft_transaction_gates.json']
    return dict(format='nasa-population-publication-review.v1',review_source='automated',proposed_tier=3,
        eligible_for_approval_transaction=True,approved=False,catalog_mutations=0,graph_sha256=expected,
        scope=semantic['scope'],limitations=LIMITATIONS,new_atoms=len(proposed['atoms']),reused_approved_atoms=len(proposed['reused_atoms']),
        dependency_versions=parent['dependency_versions'],notice_review=parent['notice_review'],
        source_sha256=semantic['population_source_sha256'],shared_source_sha256=semantic['shared_source_sha256'],
        registration_source_sha256=semantic['registration_source_sha256'],binding_fault_scope=gates['binding_fault_scope'],
        evidence_sha256={name:sha(directory/name) for name in names},reviewer_sha256=sha(Path(__file__)))


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/nasa_population_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,new_atoms=report['new_atoms'],graphs=len(report['graph_sha256']),approved=False)))
