"""Record provisioned dependency compatibility and retained software notices."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_closure(roots, source_projects):
    """Follow active requirements, including extras, using local source manifests."""
    queue = [(None, Requirement(text)) for text in roots]
    visited, versions, failures = set(), {}, []
    while queue:
        parent, requirement = queue.pop()
        name = canonicalize_name(requirement.name)
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            failures.append(dict(parent=parent, requirement=str(requirement), reason='missing'))
            continue
        versions[name] = version
        if not requirement.specifier.contains(version, prereleases=True):
            failures.append(dict(parent=parent, requirement=str(requirement), installed=version, reason='version_conflict'))
        key = (name, tuple(sorted(requirement.extras)))
        if key in visited:
            continue
        visited.add(key)
        if name in source_projects:
            project = source_projects[name]
            children = list(project.get('dependencies', []))
            for extra in sorted(requirement.extras):
                if extra not in project.get('optional-dependencies', {}):
                    failures.append(dict(parent=name, extra=extra, reason='undeclared_extra'))
                children.extend(project.get('optional-dependencies', {}).get(extra, []))
        else:
            children = metadata.requires(name) or []
        for text in children:
            child = Requirement(text)
            if child.marker and not any(child.marker.evaluate({'extra': extra}) for extra in ('', *requirement.extras)):
                continue
            queue.append((name, child))
    failures.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return dict(compatible=not failures, versions=dict(sorted(versions.items())), failures=failures,
                roots=roots, scope='Declared provider plus architect/visualizer requirements; active platform markers and requested extras.')


def audit():
    project = ROOT.parent / 'sciona-atoms-ml/pyproject.toml'
    declared = tomllib.loads(project.read_text())['project']['dependencies']
    installed = metadata.requires('sciona-atoms-ml') or []
    matcher_project = ROOT / 'pyproject.toml'
    closure = dependency_closure(['sciona-atoms-ml', 'sciona[architect,visualizer]'], {
        'sciona-atoms-ml': tomllib.loads(project.read_text())['project'],
        'sciona': tomllib.loads(matcher_project.read_text())['project'],
    })
    versions = {}
    for text in declared:
        requirement = Requirement(text)
        version = metadata.version(requirement.name)
        if not requirement.specifier.contains(version, prereleases=True):
            raise ValueError('Provider dependency incompatible: ' + requirement.name)
        versions[requirement.name] = version
    notices = ROOT / 'docs/licenses/conditional-correction'
    if sha(notices / 'DrivenData-MIT.txt') != '0d6822721946585bb39e5ced6cd594f658a9321c36a78ccd41d20001fae39e1e':
        raise ValueError('Pinned source notice differs')
    numpy = metadata.distribution('numpy')
    count = 0
    for file in numpy.files or []:
        if '.dist-info/licenses/' not in str(file):
            continue
        relative = str(file).split('.dist-info/licenses/', 1)[1]
        if sha(notices / 'numpy' / relative) != sha(Path(numpy.locate_file(file))):
            raise ValueError('Installed NumPy notice differs')
        count += 1
    if not count:
        raise ValueError('NumPy license inventory missing')
    return dict(
        format='conditional-correction-environment.v1',
        provisioned_direct_dependencies_compatible=True,
        dependency_versions=versions,
        transitive_dependency_closure=closure,
        matcher_package_sha256=sha(matcher_project),
        python_version=platform.python_version(), platform=platform.system(), architecture=platform.machine(),
        provider_package_sha256=sha(project),
        installed_provider_metadata_matches_source=set(installed) == set(declared),
        declared_provider_dependencies=declared, installed_provider_dependencies=installed,
        license_sha256={str(p.relative_to(notices)): sha(p) for p in sorted(notices.rglob('*')) if p.is_file()},
        reviewer_sha256=sha(Path(__file__)),
        limitations=[
            'Provisioned environment only; clean installation and other platforms are not qualified.',
            'Installed provider metadata differs from the source manifest; no dependency installation performed while training jobs are active.',
            'Transitive checks cover declared provider and architect/visualizer requirements; undeclared dynamic imports are covered only by recorded execution, not a complete packaging audit.',
            'Retained notices cover pinned DrivenData source and installed NumPy distribution; this does not assign a repository-wide provider license.',
        ], approved=False, catalog_mutations=0)


if __name__ == '__main__':
    report = audit()
    (ROOT / 'docs/reviews/conditional_correction_environment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(direct_dependencies=len(report['dependency_versions']),
                          retained_notices=len(report['license_sha256']),
                          metadata_matches=report['installed_provider_metadata_matches_source'])))
