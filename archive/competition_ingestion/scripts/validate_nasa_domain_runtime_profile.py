"""Qualify the exercised in-process catalog runner without optional server accelerators."""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'uvloop', 'watchfiles', 'httptools', 'psycopg_binary'}


class RejectOptionalServerImports(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in EXCLUDED:
            self.attempted.append(fullname)
            raise ModuleNotFoundError('Optional server implementation unavailable: ' + fullname, name=fullname)
        return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path,required=True)
    args=parser.parse_args()
    if any(name.split('.')[0] in EXCLUDED for name in sys.modules):
        raise ValueError('Fresh process without excluded packages required')
    guard = RejectOptionalServerImports()
    sys.meta_path.insert(0, guard)
    try:
        import psycopg
        from scripts.validate_nasa_domain_catalog import validate
        from scripts.review_conditional_correction_environment import dependency_closure
        files = {'sciona': ROOT / 'pyproject.toml', 'sciona-atoms-ml': ROOT.parent / 'sciona-atoms-ml/pyproject.toml'}
        projects = {name: tomllib.loads(path.read_text())['project'] for name, path in files.items()}
        closure = dependency_closure(['sciona-atoms-ml[xgboost]', 'sciona', 'psycopg', 'python-dotenv'], projects)
        if not closure['compatible']:
            raise ValueError('In-process base dependency closure incompatible')
        execution = validate(args.source_directory)
        if not execution['passed'] or psycopg.pq.__impl__ != 'python':
            raise ValueError('Expected execution/client profile differs')
    finally:
        sys.meta_path.remove(guard)
    closure['scope'] = 'Source-declared provider/base Sciona dependencies and local catalog client; optional HTTP server extras excluded.'
    report = dict(format='nasa-domain-runtime-profile.v1', passed=True, approved=False,
        catalog_mutations=0, execution=execution, dependency_closure=closure,
        psycopg_implementation=psycopg.pq.__impl__, libpq_version=psycopg.pq.version(),
        excluded_optional_packages=sorted(EXCLUDED), unavailable_import_attempts=guard.attempted,
        manifest_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Provisioned in-process CDGExecutionSession and local catalog client only; HTTP server deployment is separately unqualified.',
                    'Installed provider metadata is stale; dependency review uses the current source manifest. Clean installation remains unqualified.',
                    'Eager atom discovery can import unrelated providers; this records successful execution, not minimal packaging isolation.',
                    'This supplementary evidence does not itself change publication gates or approve catalog artifacts.'])
    (ROOT / 'docs/reviews/nasa_domain_runtime_profile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, packages=len(closure['versions']), psycopg_implementation=psycopg.pq.__impl__,
                         unavailable_import_attempts=guard.attempted, catalog_mutations=0)))


if __name__ == '__main__':
    main()
