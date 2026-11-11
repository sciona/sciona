"""Qualify materialized feature providers in the provisioned in-process runtime."""
import json
from pathlib import Path
import sys
import tomllib

from scripts.validate_nasa_domain_runtime_profile import EXCLUDED, RejectOptionalServerImports


def validate():
    if any(name.split('.')[0] in EXCLUDED for name in sys.modules):
        raise ValueError('Fresh process without optional server implementations required')
    guard = RejectOptionalServerImports()
    sys.meta_path.insert(0, guard)
    try:
        import psycopg
        from scripts.plan_available_feature_providers import ROOT, sha
        from scripts.validate_available_feature_graph_execution import validate as execute
        from scripts.review_conditional_correction_environment import dependency_closure
        from scripts.review_residual_classifier_publication import review_notices
        manifests = {'sciona': ROOT/'pyproject.toml',
                     'sciona-atoms-ml': ROOT.parent/'sciona-atoms-ml/pyproject.toml'}
        projects = {name: tomllib.loads(path.read_text())['project'] for name, path in manifests.items()}
        closure = dependency_closure(['sciona-atoms-ml', 'sciona', 'psycopg', 'python-dotenv'], projects)
        if not closure['compatible']:
            raise ValueError('Dependency closure incompatible: '+json.dumps(closure['failures']))
        directory = ROOT/'docs/reviews'
        notices = review_notices(directory)
        inventory = json.loads((directory/'residual_classifier_dependency_notices.json').read_text())
        for package, version in closure['versions'].items():
            if inventory['packages'].get(package, {}).get('version') != version:
                raise ValueError('Retained notice version differs: '+package)
        execution = execute()
        if not execution['passed'] or psycopg.pq.__impl__ != 'python':
            raise ValueError('Expected execution/client profile differs')
    finally:
        sys.meta_path.remove(guard)
    return dict(passed=True, approved=False, catalog_mutations=0,
        dependency_closure=closure, notice_review=notices, execution=execution,
        manifest_sha256={name: sha(path) for name, path in manifests.items()},
        psycopg_implementation=psycopg.pq.__impl__, libpq_version=psycopg.pq.version(),
        excluded_optional_packages=sorted(EXCLUDED), unavailable_import_attempts=guard.attempted,
        validator_sha256=sha(Path(__file__)),
        evidence_sha256={name: sha(directory/name) for name in [
            'residual_classifier_dependency_notices.json', 'residual_classifier_tokenizers_notice.json']},
        limitations=['Provisioned in-process materialized execution only; clean installation and HTTP deployment are unqualified.',
            'Source manifests determine dependencies because installed local provider metadata is stale.',
            'No native binaries, model files or dependency packages are redistributed.',
            'Eager discovery may load unrelated providers; this does not establish minimal packaging isolation.'])


if __name__ == '__main__':
    report = validate()
    root = Path(__file__).resolve().parents[1]
    (root/'docs/reviews/available_feature_provider_runtime.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, packages=len(report['dependency_closure']['versions']),
        provider_graphs=report['execution']['provider_graphs'], approved=False)))
