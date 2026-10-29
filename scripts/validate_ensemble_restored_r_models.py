"""Requalify unchanged full R model oracles against the restored binary set.

Only the runtime-audit input is replaced; source programs, model budgets,
synthetic populations, exact comparisons and invalid-window checks are retained.
Historical qualification files are never modified.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(reference_dir, library):
    original_path = ROOT/'docs/reviews/andriy_r_contract_audit.json'
    restored_path = ROOT/'docs/reviews/ensemble_restored_r_contract_audit.json'
    historical_paths = [original_path] + [ROOT/f'docs/reviews/andriy_r_{b}_parity.json' for b in ['xgb', 'svm', 'glm']]
    historical_hashes = {p.name: sha(p) for p in historical_paths}
    original = json.loads(original_path.read_text())
    restored = json.loads(restored_path.read_text())
    if {k:v for k,v in original.items() if k != 'compiled_library_sha256'} != {k:v for k,v in restored.items() if k != 'compiled_library_sha256'}:
        raise ValueError('Restored API audit differs beyond the compiled binary identity')
    for name, digest in restored['compiled_library_sha256'].items():
        if sha(library/name) != digest:
            raise ValueError('Restored library changed')
    for name, digest in restored['validation_sha256'].items():
        if sha(ROOT/'scripts'/name) != digest:
            raise ValueError('API audit implementation changed')
    read_text = Path.read_text
    def selected_audit(path, *args, **kwargs):
        if path.resolve() == original_path.resolve():
            return json.dumps(restored)
        return read_text(path, *args, **kwargs)
    reports = {}
    for branch in ['xgb', 'svm', 'glm']:
        module = importlib.import_module('scripts.validate_andriy_r_'+branch)
        historical = json.loads((ROOT/f'docs/reviews/andriy_r_{branch}_parity.json').read_text())
        if sha(module.__file__) != historical['validation_sha256']:
            raise ValueError('Full model oracle changed')
        output = ROOT/f'docs/reviews/ensemble_restored_r_{branch}_parity.json'
        argv = [module.__file__, '--reference-dir', str(reference_dir), '--library', str(library), '--output', str(output)]
        print('Executing full restored '+branch+' source and provider comparisons', flush=True)
        with patch.object(Path, 'read_text', selected_audit), patch.object(sys, 'argv', argv):
            module.main()
        result = json.loads(output.read_text())
        if result['compiled_library_sha256'] != restored['compiled_library_sha256']:
            raise ValueError('Model evidence binary identity differs')
        if {k:v for k,v in historical.items() if k != 'compiled_library_sha256'} != {k:v for k,v in result.items() if k != 'compiled_library_sha256'}:
            raise ValueError('Full model results differ from retained qualification')
        reports[branch] = dict(report_sha256=sha(output), unchanged_oracle_sha256=sha(module.__file__),
                               exact_historical_results_except_binary_hashes=True)
    if historical_hashes != {p.name: sha(p) for p in historical_paths}:
        raise ValueError('Historical evidence changed')
    report = dict(passed=True, approved=False, catalog_mutations=0, native_threads=1,
                  restored_api_audit_sha256=sha(restored_path), models=reports,
                  historical_evidence_sha256=historical_hashes,
                  validator_sha256=sha(__file__),
                  limitations=['Full original synthetic model comparisons under rebuilt binary identities; not historical-engine equivalence.',
                               'Raw feature extraction, complete stored ensemble execution and original-intake promotion remain separate.'])
    (ROOT/'docs/reviews/ensemble_restored_r_models.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, approved=False, models=len(reports), catalog_mutations=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    args = parser.parse_args()
    validate(args.reference_dir.resolve(), args.library.resolve())
