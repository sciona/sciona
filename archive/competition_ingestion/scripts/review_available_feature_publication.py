"""Bind automated Tier 3 review to the thirteen materialized provider versions."""
import json
from pathlib import Path
import tomllib

from scripts.plan_available_feature_providers import ROOT, plan, sha
from scripts.review_conditional_correction_environment import dependency_closure
from scripts.review_residual_classifier_publication import review_notices


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review():
    proposed = plan()
    directory = ROOT/'docs/reviews'
    names = ['available_feature_provider_runtime.json', 'available_feature_provider_graph_execution.json',
             'available_feature_provider_database_gates.json', 'available_feature_provider_staging_gates.json']
    runtime, execution, gates, staging = [json.loads((directory/name).read_text()) for name in names]
    require(all(report['passed'] for report in [runtime, execution, gates, staging]), 'Qualification failed')
    require(runtime['execution'] == execution, 'Scoped runtime execution differs')
    expected = {(atom['runtime_fqdn'], atom['version_id'], atom['content_hash'], len(atom['outputs']))
                for atom in proposed['atoms']}
    actual = {(case['runtime_fqdn'], case['version_id'], case['provider_content_hash'], case['outputs_checked'])
              for case in execution['cases'] if case['passed']}
    require(actual == expected and len(execution['cases']) == 13, 'Exact provider execution coverage missing')
    require(execution['stored_ports_used'] and execution['codec_roundtrips'] and execution['production_executor'],
            'Stored contract execution missing')
    for report, filename in [(runtime, 'validate_available_feature_runtime.py'),
                             (gates, 'validate_available_feature_database_gates.py')]:
        require(report['validator_sha256'] == sha(ROOT/'scripts'/filename), 'Validator drift')
    for report in [execution, staging]:
        for name, digest in report['implementation_sha256'].items():
            require(sha(ROOT/name) == digest, 'Implementation drift: '+name)
    faults = []
    for index in range(13):
        for table in ['artifact_versions', 'atom_versions']:
            faults.extend([f'{table}_hash_{index}', f'{table}_tier_{index}'])
        faults.extend(f'{table}_contract_{index}' for table in ['artifact_io_specs', 'atom_io_specs'])
    faults += ['missing_staging_evidence', 'evidence_payload', 'failed_audit', 'runtime_pointer',
               'canonical_latest', 'legacy_latest', 'canonical_publishable', 'legacy_publishable', 'source_hash']
    require(gates['rejected_faults'] == faults and gates['rollback_verified'], 'Catalog rejection coverage differs')
    require(gates['plan_sha256'] == sha(directory/'available_feature_provider_plan.json'), 'Plan evidence drift')
    require(staging['original_and_target_rows_unchanged'] and staging['dry_run_rows'] == 265
            and staging['failure_after_write_checks'] == [1, 132, 265], 'Staging rollback evidence differs')
    manifests = {'sciona': ROOT/'pyproject.toml', 'sciona-atoms-ml': ROOT.parent/'sciona-atoms-ml/pyproject.toml'}
    require(runtime['manifest_sha256'] == {name: sha(path) for name, path in manifests.items()}, 'Manifest drift')
    projects = {name: tomllib.loads(path.read_text())['project'] for name, path in manifests.items()}
    closure = dependency_closure(['sciona-atoms-ml', 'sciona', 'psycopg', 'python-dotenv'], projects)
    require(closure == runtime['dependency_closure'] and closure['compatible'], 'Dependency drift')
    require(runtime['excluded_optional_packages'] == ['httptools', 'psycopg_binary', 'uvloop', 'watchfiles']
            and runtime['psycopg_implementation'] == 'python', 'Runtime scope differs')
    for name, digest in runtime['evidence_sha256'].items():
        require(sha(directory/name) == digest, 'Notice evidence drift')
    notices = review_notices(directory)
    require(notices == runtime['notice_review'], 'Notice review differs')
    return dict(format='available-feature-publication-review.v1', review_source='automated', proposed_tier=3,
        eligible_for_approval_transaction=True, approved=False, catalog_mutations=0,
        provider_versions=[{key: atom[key] for key in ['artifact_id', 'version_id', 'content_hash', 'runtime_fqdn']}
                           for atom in proposed['atoms']],
        qualification=proposed['evidence'], source_provenance=proposed['source_provenance'],
        scope=proposed['scope'], dependency_versions=closure['versions'], notice_review=notices,
        evidence_sha256={name: sha(directory/name) for name in names}, reviewer_sha256=sha(Path(__file__)),
        limitations=['Automated Tier 3 materialized feature operations only; no human Tier 1 or empirical Tier 2 claim.',
            'Cross-domain use requires the explicit units, availability, shape and caller policy contracts on each port.',
            'Static symbolic propagation is unsupported; select only for materialized execution.',
            'The original competition workflow remains unapproved by this review; these are independently reusable operations.',
            'Synthetic tests establish computational behavior, not empirical prediction performance.',
            'Provisioned in-process runtime only; HTTP deployment, clean installation and package redistribution are unqualified.',
            'Caller records, policies and derived values remain private runtime material when non-public.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/available_feature_provider_publication_review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(eligible=report['eligible_for_approval_transaction'], providers=len(report['provider_versions']),
                         proposed_tier=3, approved=False)))
