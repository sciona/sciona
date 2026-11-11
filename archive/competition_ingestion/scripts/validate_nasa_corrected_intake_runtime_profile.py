"""Qualify the corrected original intake in the provisioned base runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib

from scripts.validate_residual_classifier_runtime_profile import EXCLUDED, RejectOptionalServerImports, ROOT


def validate(source):
    if any(name.split('.')[0] in EXCLUDED for name in sys.modules):
        raise ValueError('Fresh process without excluded packages required')
    baseline_path=ROOT/'docs/reviews/nasa_corrected_intake_catalog_execution.json'
    baseline_bytes=baseline_path.read_bytes()
    baseline=json.loads(baseline_bytes)
    guard=RejectOptionalServerImports()
    sys.meta_path.insert(0,guard)
    try:
        import psycopg
        from scripts.validate_nasa_corrected_intake_catalog import validate as execute
        from scripts.review_conditional_correction_environment import dependency_closure
        from scripts.review_conditional_correction import require
        files={'sciona':ROOT/'pyproject.toml','sciona-atoms-ml':ROOT.parent/'sciona-atoms-ml/pyproject.toml'}
        manifests={name:path.read_bytes() for name,path in files.items()}
        projects={name:tomllib.loads(contents.decode())['project'] for name,contents in manifests.items()}
        closure=dependency_closure(['sciona-atoms-ml[xgboost]','sciona','psycopg','python-dotenv'],projects)
        require(closure['compatible'],'Base dependency closure incompatible')
        execution=execute(source)
        require(execution['passed'] and psycopg.pq.__impl__=='python','Expected runtime profile differs')
        require(execution==baseline,'Guarded execution differs from catalog qualification')
        require(baseline_path.read_bytes()==baseline_bytes,'Catalog evidence changed during validation')
        require(all(path.read_bytes()==manifests[name] for name,path in files.items()),'Source manifests changed')
        require(not any(name.split('.')[0] in EXCLUDED for name in sys.modules),'Excluded package loaded')
    finally:
        sys.meta_path.remove(guard)
    return dict(format='nasa-corrected-intake-runtime-profile.v1',passed=True,approved=False,catalog_mutations=0,
        execution=execution,dependency_closure=closure,psycopg_implementation=psycopg.pq.__impl__,
        libpq_version=psycopg.pq.version(),excluded_optional_packages=sorted(EXCLUDED),
        unavailable_import_attempts=guard.attempted,
        catalog_evidence_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
        manifest_sha256={name:hashlib.sha256(contents).hexdigest() for name,contents in manifests.items()},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        guard_source_sha256=hashlib.sha256((ROOT/'scripts/validate_residual_classifier_runtime_profile.py').read_bytes()).hexdigest(),
        limitations=[
            'Provisioned in-process runner and catalog client only; HTTP deployment and clean installation remain unqualified.',
            'Current source manifests define dependencies because installed provider metadata is stale.',
            'Eager discovery may import unrelated providers; this is successful guarded execution, not minimal packaging isolation.',
            '640 predictions execute the stored corrected version; 128 additional sparse predictions are supplementary candidate regression evidence.',
            'No new subprocess qualification: prior population lifecycle evidence covers saved-state inference in fresh processes.',
            'This report does not approve catalog artifacts or authorize package redistribution.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/nasa_corrected_intake_runtime_profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,packages=len(report['dependency_closure']['versions']),
        stored_predictions=report['execution']['stored_predictions_compared'],catalog_mutations=0)))
