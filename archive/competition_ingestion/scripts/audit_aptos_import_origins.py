"""Observe shared runner/numerical imports and attribute their source files.

This is import/startup plus optimizer warmup coverage, not a claim that every
possible lazy module or optional execution branch has been imported.
"""
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig


def main():
    import scripts.aptos_graph_execution
    import sciona.aptos_pipeline
    import pretrainedmodels
    import torch
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    files = {}
    for name, module in list(sys.modules.items()):
        location = getattr(module, '__file__', None)
        if not location or not Path(location).is_file():
            continue
        path = Path(location).resolve()
        if path.suffix == '.pyc':
            path = Path(importlib.util.source_from_cache(str(path))).resolve()
        files.setdefault(path, []).append(name)
    owners = {}
    for dist in metadata.distributions():
        for item in dist.files or []:
            path = Path(dist.locate_file(item)).resolve()
            if path in files:
                owners.setdefault(path, []).append((dist.metadata['Name'], dist.version))
    packages, unmatched, first_party, standard = {}, [], 0, 0
    root = Path(__file__).resolve().parents[2]
    stdlib = Path(sysconfig.get_path('stdlib')).resolve()
    for path, modules in files.items():
        if path in owners:
            for name, version in owners[path]:
                row = packages.setdefault(name, {'version': version, 'module_files': 0})
                if row['version'] != version:
                    raise ValueError('Multiple observed versions of a distribution')
                row['module_files'] += 1
        elif path.is_relative_to(root) and '.venv' not in path.parts and 'site-packages' not in path.parts:
            first_party += 1
        elif path.is_relative_to(stdlib) and 'site-packages' not in path.parts:
            standard += 1
        else:
            unmatched.append({'modules': modules, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {'approved': False, 'catalog_mutations': 0,
        'scope': 'Shared runner/numerical startup and one synthetic optimizer update. Distribution-file attribution; not publisher authentication or exhaustive lazy-execution coverage.',
        'packages': dict(sorted(packages.items())), 'unmatched': unmatched,
        'first_party_module_files': first_party, 'stdlib_module_files': standard,
        'sha256': {'scripts/audit_aptos_import_origins.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    Path('docs/reviews/competition_aptos_import_origins.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'packages': len(packages), 'unmatched': len(unmatched), 'first_party': first_party}))


if __name__ == '__main__':
    main()
