"""Check installed Santa dependency closure and its own RECORD hashes."""
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def audit():
    executed = json.loads((ROOT/'docs/reviews/decomposed_santa_graph_execution.json').read_text())['runtime']
    pending = list(executed)
    packages = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in packages:
            continue
        distribution = metadata.distribution(name)
        version = distribution.version
        if name in executed and version != executed[name]:
            raise ValueError('Executed direct dependency changed: ' + name)
        dependencies = []
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            if requirement.extras or requirement.url:
                raise ValueError('Explicit review required for extra or direct URL dependency')
            installed = metadata.version(requirement.name)
            if not requirement.specifier.contains(installed, prereleases=True):
                raise ValueError('Unsatisfied installed dependency: ' + str(requirement))
            dependencies.append(str(requirement))
            pending.append(requirement.name)
        hashed = 0
        files = distribution.files
        if not files:
            raise ValueError('Installed package file inventory missing: ' + name)
        for relative in files:
            if relative.hash is None:
                continue
            path = distribution.locate_file(relative)
            if not path.is_file():
                raise ValueError('Installed hashed file missing in package: ' + name)
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, relative.hash.mode).digest()
            if base64.urlsafe_b64encode(digest).rstrip(b'=').decode() != relative.hash.value:
                raise ValueError('Installed file differs from own RECORD in package: ' + name)
            hashed += 1
        if hashed == 0:
            raise ValueError('No hashed installed files: ' + name)
        packages[name] = dict(version=version, hashed_files_verified=hashed, active_requirements=sorted(dependencies))
    return dict(passed=True, approved=False, catalog_mutations=0,
                direct_versions=executed, packages=dict(sorted(packages.items())),
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations=['Installed files are checked against their own RECORD, not independently authenticated publisher signatures.',
                             'Default active dependency closure only; optional extras and clean deployment are not qualified.'])


if __name__ == '__main__':
    result = audit()
    (ROOT/'docs/reviews/decomposed_santa_package_files.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], packages=len(result['packages']),
                         hashed_files_verified=sum(p['hashed_files_verified'] for p in result['packages'].values()))))
