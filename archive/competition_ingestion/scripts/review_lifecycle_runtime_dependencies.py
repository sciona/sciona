"""Verify provisioned lifecycle requirements and retain additional notices.

Only package license files are copied. No datasets, native libraries or model
payloads are exported. This does not qualify binary redistribution.
"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import tomllib
from urllib.request import urlopen

from scripts.review_conditional_correction_environment import dependency_closure
from scripts.review_residual_classifier_publication import review_notices

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084'
CATBOOST_NOTICE_SHA256 = 'a2574dd20dd0af8e0bd25b34b38c1bb11912ff319ee4f41436395df0b1d19309'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review():
    manifests = {'sciona':ROOT/'pyproject.toml','sciona-atoms-ml':ROOT.parent/'sciona-atoms-ml/pyproject.toml'}
    profile = ROOT/'requirements/lifecycle-cpu.txt'
    supplemental = [line.strip() for line in profile.read_text().splitlines() if line.strip() and not line.startswith('#')]
    projects = {name:tomllib.loads(path.read_text())['project'] for name,path in manifests.items()}
    closure = dependency_closure(['sciona-atoms-ml','sciona','psycopg','python-dotenv',*supplemental],projects)
    if not closure['compatible']:
        raise ValueError('Runtime requirements incompatible: '+json.dumps(closure['failures']))
    directory = ROOT/'docs/reviews'
    baseline = review_notices(directory)
    prior = json.loads((directory/'residual_classifier_dependency_notices.json').read_text())['packages']
    retained = {}
    destination = ROOT/'docs/licenses/lifecycle-cpu'
    for name,version in closure['versions'].items():
        if name in prior:
            if prior[name]['version'] != version:
                raise ValueError('Retained base notice version changed: '+name)
            continue
        distribution = metadata.distribution(name)
        notices = []
        for file in distribution.files or []:
            relative = Path(str(file))
            # Include package and dist-info text notices, never Python modules
            # that happen to be called copying.py or external share paths.
            if '..' in relative.parts or not relative.name.lower().startswith(('license','notice','copying')) or relative.suffix in {'.py','.pyc'}:
                continue
            source = Path(distribution.locate_file(file))
            payload = source.read_bytes()
            payload.decode('utf-8')
            target = destination/name/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=payload:
                raise ValueError('Retained notice changed: '+name)
            target.write_bytes(payload)
            notices.append(dict(retained_notice=str(target.relative_to(ROOT)),sha256=sha(target)))
        if name=='catboost':
            url=f'https://raw.githubusercontent.com/catboost/catboost/{COMMIT}/LICENSE'
            target=destination/'catboost/UPSTREAM-LICENSE.txt'
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                with urlopen(url,timeout=30) as response:payload=response.read()
                if b'Apache License' not in payload or b'YANDEX LLC' not in payload:
                    raise ValueError('Unexpected upstream CatBoost notice')
                target.write_bytes(payload)
            if sha(target)!=CATBOOST_NOTICE_SHA256:
                raise ValueError('Pinned upstream CatBoost notice changed')
            notices.append(dict(retained_notice=str(target.relative_to(ROOT)),sha256=sha(target),
                upstream_url=url,source_commit=COMMIT,version=version))
        if not notices:
            raise ValueError('Additional package notice missing: '+name)
        retained[name]=dict(version=version,retained_notices=notices)
    return dict(passed=True,approved=False,catalog_mutations=0,dependency_closure=closure,
        additional_package_notices=retained,base_notice_review=baseline,
        source_manifest_sha256={name:sha(path) for name,path in manifests.items()},
        profile_sha256=sha(profile),reviewer_sha256=sha(Path(__file__)),
        limitations=['Provisioned runtime and metadata selection only; clean installation and binary redistribution are not qualified.',
                     'The explicit supplemental runtime profile supplies CatBoost; base provider installation alone does not supply this backend.',
                     'Installed local package metadata is stale; current source manifests determine dependencies.',
                     'Upstream CatBoost notice does not inventory every statically linked native dependency.'])


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/competition_nasa_first_lifecycle_dependencies.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,packages=len(report['dependency_closure']['versions']),additional_notice_packages=len(report['additional_package_notices']))))
