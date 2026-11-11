"""Attribute framework imports to selected distribution RECORD entries."""
import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import sysconfig


def main():
    report_path = Path('docs/reviews/competition_cassava_environment_audit.json')
    audit = json.loads(report_path.read_text())
    selected = audit['packages']
    index = {}
    for name in selected:
        distribution = metadata.distribution(name)
        for entry in distribution.files or []:
            if entry.hash:
                path = Path(distribution.locate_file(entry)).resolve()
                index.setdefault(path, []).append((name, str(entry)))
    # Import the real validation entry point without executing its guarded main.
    importlib.import_module('scripts.validate_cassava_pipeline')
    for module in ['timm', 'torchvision', 'safetensors.torch', 'tf_keras']:
        importlib.import_module(module)
    standard = Path(sysconfig.get_path('stdlib')).resolve()
    repository = Path.cwd().resolve()
    attributed, local, unmatched = {}, [], []
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_file():
            continue
        if path in index:
            attributed[name] = [{'distribution': owner, 'record_path': record} for owner, record in index[path]]
        elif path.is_relative_to(repository) and not path.is_relative_to(repository / '.venv'):
            local.append(name)
        elif path.is_relative_to(standard) and 'site-packages' not in path.parts:
            continue
        else:
            unmatched.append(name)
    report = {'approved': False, 'selected_distribution_modules': len(attributed),
        'local_modules': local, 'unattributed_modules': unmatched, 'origins': attributed,
        'scope': 'Import-time origins for the real workflow entry point and major frameworks; training-time lazy imports and shared-library linkage are not covered. RECORD contents are checked by the separate file audit.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
            str(report_path), 'scripts/audit_cassava_import_origins.py', 'scripts/validate_cassava_pipeline.py']}}
    Path('docs/reviews/competition_cassava_import_origin_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'attributed_modules': len(attributed), 'unattributed_count': len(unmatched),
                      'unattributed_roots': sorted({name.split('.')[0] for name in unmatched})}))


if __name__ == '__main__':
    main()
