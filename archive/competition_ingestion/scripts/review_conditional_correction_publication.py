"""Qualify the provisioned in-process reusable graph for automated Tier 3."""
import json
import tomllib

import psycopg

from scripts.review_conditional_correction import ROOT, audit, require, sha
from scripts.review_conditional_correction_environment import dependency_closure

LIMITATIONS = [
    'Automated Tier 3 reusable correction subgraph only; complete NASA workflow, Tier 2 usage qualification and Tier 1 human certification are not claimed.',
    'Provisioned in-process CDGExecutionSession with Python psycopg/system libpq; HTTP deployment and clean installation are not qualified.',
    'Installed provider metadata is stale; all declared requirements in the current source manifest were checked against the provisioned environment.',
    'Finite float64 numerical contract; historical float32 estimator arithmetic and submission rounding remain separate.',
    'Caller establishes population alignment, target-unit consistency, calibration validity and underestimation-probability semantics.',
    'Synthetic cross-domain execution demonstrates composability, not empirical effectiveness or guaranteed accuracy improvement.',
    'Threshold ties remain unchanged; both calibration groups must be nonempty and signed offsets can reverse the predicted direction.',
]
FAULTS = ['provider_hash', 'provider_tier', 'graph_latest', 'graph_contract', 'legacy_contract',
          'binding_hash', 'runtime_binding', 'optional_provenance', 'failed_audit']


def review():
    semantic = audit()
    directory = ROOT / 'docs/reviews'
    runtime = json.loads((directory / 'conditional_correction_runtime_profile.json').read_text())
    execution = json.loads((directory / 'conditional_correction_catalog_execution.json').read_text())
    gates = json.loads((directory / 'conditional_correction_database_gates.json').read_text())
    require(runtime['passed'] is True and execution['passed'] is True, 'Runtime qualification missing')
    require(runtime['execution'] == execution, 'Runtime execution evidence differs')
    require(execution['graph_sha256'] == semantic['serialized_graph_sha256'], 'Runtime graph differs')
    for report, name in [(runtime, 'validate_conditional_correction_runtime_profile.py'),
                         (execution, 'validate_conditional_correction_catalog.py'),
                         (gates, 'validate_conditional_correction_database_gates.py')]:
        require(report['validator_sha256'] == sha(ROOT / 'scripts' / name), 'Qualification code drift: ' + name)
    files = {'sciona': ROOT / 'pyproject.toml', 'sciona-atoms-ml': ROOT.parent / 'sciona-atoms-ml/pyproject.toml'}
    require(runtime['manifest_sha256'] == {name: sha(path) for name, path in files.items()}, 'Manifest evidence drift')
    projects = {name: tomllib.loads(path.read_text())['project'] for name, path in files.items()}
    closure = dependency_closure(['sciona-atoms-ml', 'sciona', 'psycopg', 'python-dotenv'], projects)
    require(closure['compatible'] and closure['versions'] == runtime['dependency_closure']['versions']
            and runtime['dependency_closure']['compatible'], 'Runtime dependency drift')
    require(runtime['psycopg_implementation'] == psycopg.pq.__impl__ == 'python'
            and runtime['libpq_version'] == psycopg.pq.version(), 'Database client profile differs')
    require(runtime['excluded_optional_packages'] == ['httptools', 'psycopg_binary', 'uvloop', 'watchfiles'],
            'Optional-package exclusion evidence differs')
    require(gates['passed'] is True and gates['rollback_verified'] is True and gates['rejected_faults'] == FAULTS
            and gates['graph_sha256'] == semantic['serialized_graph_sha256'], 'Database negative gates missing')
    names = ['conditional_correction_runtime_profile.json', 'conditional_correction_catalog_execution.json',
             'conditional_correction_database_gates.json', 'conditional_correction_reuse.json',
             'conditional_correction_environment.json']
    return dict(format='conditional-correction-publication-review.v1', review_source='automated', proposed_tier=3,
        eligible_for_approval_transaction=True, source_version_id=semantic['source_version_id'], source_hash=semantic['source_hash'],
        graph_sha256=semantic['serialized_graph_sha256'], provider_sha256=semantic['provider_sha256'],
        scope=semantic['scope'], applicability=semantic['applicability'], limitations=LIMITATIONS,
        dependency_versions=closure['versions'], evidence_sha256={name: sha(directory / name) for name in names},
        reviewer_sha256=sha(ROOT / 'scripts/review_conditional_correction_publication.py'),
        prior_environment_blocker_resolution='Fresh catalog execution passed with optional server packages blocked; qualify the actual in-process runtime. Deployment exclusions remain explicit.',
        approved=False, catalog_mutations=0)


if __name__ == '__main__':
    result = review()
    (ROOT / 'docs/reviews/conditional_correction_publication_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(eligible=result['eligible_for_approval_transaction'], proposed_tier=3, approved=False)))
