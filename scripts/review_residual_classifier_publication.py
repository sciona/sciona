"""Qualify the provisioned in-process reusable graph for automated Tier 3."""
import json
import tomllib

import psycopg

from scripts.review_conditional_correction import ROOT, SOURCE_VERSION, SOURCE_HASH, require, sha
from scripts.plan_residual_classifier_draft import audit, plan
from scripts.review_conditional_correction_environment import dependency_closure

LIMITATIONS = [
    'Automated Tier 3 numerical workflow only; original NASA domain workflows, Tier 2 empirical usage and Tier 1 human certification remain unqualified.',
    'Provisioned in-process execution with Python psycopg/system libpq. HTTP deployment, clean installation, binary/package redistribution and full native dependency licensing are outside this approval.',
    'Current source manifests override stale installed provider dependency metadata.',
    'Caller supplies aligned numeric matrices, independent query rows, explicit group identities, ordered feature schema and common target units.',
    'Nearest-even int32 quantization is explicit and required; this workflow is unsuitable when continuous unquantized outputs are required.',
    'Group holdout and residual filtering must leave sufficient fit rows, both classifier classes and both strict calibration groups; unsupported populations fail.',
    'Training uses rounded predictions in the classifier feature; inference uses unrounded predictions and explicit float32 signed-offset arithmetic.',
    'Native model state is an in-memory private runtime value for non-public inputs. No model or training data is published by this catalog transaction.',
    'Synthetic cross-domain execution establishes computational reuse, not empirical predictive quality or historical XGBoost engine identity.',
    'No repository-wide license is assigned to the local provider. Source attribution and dependency notices are retained; package redistribution requires a separate review.',
]


def expected_faults(proposed):
    names = ['graph_latest', 'graph_contract', 'runtime_binding', 'optional_provenance', 'failed_audit']
    for index, atom in enumerate(proposed['atoms']):
        names.extend(f'{kind}_{index}' for kind in ['provider_hash', 'provider_tier', 'legacy_hash', 'legacy_contract'])
    for binding in proposed['bindings']:
        names.extend(kind+binding['node_id'] for kind in ['missing_binding_', 'binding_hash_'])
    return names


def review_notices(directory):
    inventory = json.loads((directory/'residual_classifier_dependency_notices.json').read_text())
    upstream = json.loads((directory/'residual_classifier_tokenizers_notice.json').read_text())
    require(inventory['inventory_completed'] and inventory['packages_without_installed_notices'] == ['sciona','sciona-atoms-ml','tokenizers'], 'Notice inventory gaps changed')
    for package in inventory['packages'].values():
        for notice in package['retained_notices']:
            require(sha(ROOT/notice['retained_notice']) == notice['sha256'], 'Retained notice drift')
    require(upstream['version'] == inventory['packages']['tokenizers']['version'] == '0.22.2' and
            upstream['source_commit'] == 'f383101a26663708484cac0727792aad74f78234' and
            sha(ROOT/upstream['retained_notice']) == upstream['sha256'] == 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4', 'Upstream notice differs')
    require(sha(ROOT/'docs/licenses/conditional-correction/DrivenData-MIT.txt') == '0d6822721946585bb39e5ced6cd594f658a9321c36a78ccd41d20001fae39e1e', 'Original source notice drift')
    return dict(scope='Existing provisioned runtime and catalog metadata selection; no dependency binaries or package distributions are published.',
        notices_verified=73, upstream_tokenizers_notice_verified=True,
        local_package_dispositions={
            'sciona':'Existing source manifest declares MIT; installed notice file absent. No packaging license assignment made.',
            'sciona-atoms-ml':'Local user-authorized implementation work. No repository-wide license declaration; no new redistribution license inferred.'},
        redistribution_qualified=False)


def review():
    semantic = audit()
    directory = ROOT / 'docs/reviews'
    runtime = json.loads((directory / 'residual_classifier_runtime_profile.json').read_text())
    execution = json.loads((directory / 'residual_classifier_catalog_execution.json').read_text())
    gates = json.loads((directory / 'residual_classifier_database_gates.json').read_text())
    require(runtime['passed'] is True and execution['passed'] is True, 'Runtime qualification missing')
    require(runtime['execution'] == execution, 'Runtime execution evidence differs')
    require(execution['graph_sha256'] == semantic['graph_sha256'], 'Runtime graph differs')
    for report, name in [(runtime, 'validate_residual_classifier_runtime_profile.py'),
                         (execution, 'validate_residual_classifier_catalog.py'),
                         (gates, 'validate_residual_classifier_database_gates.py')]:
        require(report['validator_sha256'] == sha(ROOT / 'scripts' / name), 'Qualification code drift: ' + name)
    files = {'sciona': ROOT / 'pyproject.toml', 'sciona-atoms-ml': ROOT.parent / 'sciona-atoms-ml/pyproject.toml'}
    require(runtime['manifest_sha256'] == {name: sha(path) for name, path in files.items()}, 'Manifest evidence drift')
    projects = {name: tomllib.loads(path.read_text())['project'] for name, path in files.items()}
    closure = dependency_closure(['sciona-atoms-ml[xgboost]', 'sciona', 'psycopg', 'python-dotenv'], projects)
    require(closure['compatible'] and closure['versions'] == runtime['dependency_closure']['versions']
            and runtime['dependency_closure']['compatible'], 'Runtime dependency drift')
    require(runtime['psycopg_implementation'] == psycopg.pq.__impl__ == 'python'
            and runtime['libpq_version'] == psycopg.pq.version(), 'Database client profile differs')
    require(runtime['excluded_optional_packages'] == ['httptools', 'psycopg_binary', 'uvloop', 'watchfiles'],
            'Optional-package exclusion evidence differs')
    require(gates['passed'] is True and gates['rollback_verified'] is True and gates['rejected_faults'] == expected_faults(plan())
            and gates['graph_sha256'] == semantic['graph_sha256'], 'Database negative gates missing')
    names = ['residual_classifier_runtime_profile.json', 'residual_classifier_catalog_execution.json',
             'residual_classifier_database_gates.json', 'residual_classifier_bound_execution.json',
             'residual_classifier_reuse.json', 'residual_classifier_dependency_notices.json',
             'residual_classifier_tokenizers_notice.json']
    return dict(format='residual-classifier-publication-review.v1', review_source='automated', proposed_tier=3,
        eligible_for_approval_transaction=True, source_version_id=SOURCE_VERSION, source_hash=SOURCE_HASH,
        graph_sha256=semantic['graph_sha256'], provider_source_sha256={b['runtime']:b['source_sha256'] for b in semantic['bindings']},
        notice_review=review_notices(directory),
        scope=semantic['scope'], limitations=LIMITATIONS,
        dependency_versions=closure['versions'], evidence_sha256={name: sha(directory / name) for name in names},
        reviewer_sha256=sha(ROOT / 'scripts/review_residual_classifier_publication.py'),
        prior_environment_blocker_resolution='Fresh catalog execution passed with optional server packages blocked; qualify the actual in-process runtime. Deployment exclusions remain explicit.',
        approved=False, catalog_mutations=0)


if __name__ == '__main__':
    result = review()
    (ROOT / 'docs/reviews/residual_classifier_publication_review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(eligible=result['eligible_for_approval_transaction'], proposed_tier=3, approved=False)))
