"""Read-only active dependency closure for observed TGS imports and optimizer."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    import torch
    import sciona.tgs_workflow
    from sciona.tgs_training_state import optimizer_for
    model = torch.nn.Linear(2, 1)
    optimizer_for(model, 'keras', .0001)
    owners = metadata.packages_distributions()
    roots = {'numpy', 'scipy', 'torch', 'torchvision', 'opencv-python', 'h5py'}
    local = set()
    for module in list(sys.modules):
        top = module.split('.')[0]
        if top == 'cv2':
            continue  # Explicit qualified overlay owner; shared cv2 overlaps.
        for package in owners.get(top, []):
            name = canonicalize_name(package)
            if name.startswith(('sciona', 'ageo')):
                local.add(name)
            else:
                roots.add(name)
    pending, seen, packages, failures = [(name, frozenset()) for name in sorted(roots)], set(), {}, []
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            failures.append(dict(package=name, reason='missing'))
            continue
        item = packages.setdefault(name, dict(version=dist.version, active_requirements=[]))
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not any(requirement.marker.evaluate({'extra': extra}) for extra in {''} | set(extras)):
                continue
            if text not in item['active_requirements']:
                item['active_requirements'].append(text)
            try:
                version = metadata.version(requirement.name)
                if version not in requirement.specifier:
                    failures.append(dict(package=name, requirement=text, installed=version, reason='version mismatch'))
            except metadata.PackageNotFoundError:
                failures.append(dict(package=name, requirement=text, reason='missing'))
            pending.append((requirement.name, frozenset(requirement.extras)))
    report = dict(approved=False, catalog_mutations=0, dependency_constraints_satisfied=not failures,
                  roots=sorted(roots), local_source_packages=sorted(local), packages=dict(sorted(packages.items())),
                  python_version=sys.version.split()[0], failures=failures,
                  limits=['Observed workflow imports and optimizer initialization only; optional lazy branches may require additional dependencies.',
                          'Publisher authentication, installed file provenance, licenses and native-library qualification remain separate gates.'],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path('docs/reviews/competition_tgs_environment_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(packages=len(packages), failures=failures)))


if __name__ == '__main__':
    main()
