"""Qualify restored native lifecycle execution in the explicit CPU profile."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from scripts.validate_nasa_domain_runtime_profile import EXCLUDED, RejectOptionalServerImports

ROOT = Path(__file__).resolve().parents[1]


def validate(source, checkpoint):
    if any(name.split('.')[0] in EXCLUDED for name in sys.modules):
        raise ValueError('Fresh process required for optional import exclusion')
    guard = RejectOptionalServerImports()
    sys.meta_path.insert(0,guard)
    try:
        from scripts.review_lifecycle_runtime_dependencies import review
        from scripts.validate_nasa_first_policy_state_graphs import validate as execute
        import psycopg
        dependencies=review()
        execution=execute(source,checkpoint)
        if not execution['passed'] or psycopg.pq.__impl__!='python':
            raise ValueError('Qualified execution or client profile differs')
    finally:
        sys.meta_path.remove(guard)
    return dict(passed=True,approved=False,catalog_mutations=0,native_threads=1,
        dependencies=dependencies,execution=execution,
        excluded_optional_packages=sorted(EXCLUDED),unavailable_import_attempts=guard.attempted,
        psycopg_implementation=psycopg.pq.__impl__,libpq_version=psycopg.pq.version(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Provisioned in-process execution only; clean installation and HTTP deployment remain unqualified.',
                     'Training qualification is compositional; this run encodes and restores the previously qualified graph-trained models.',
                     'This report does not publish artifacts or redistribute model payloads or dependency binaries.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--checkpoint-directory',type=Path,required=True)
    args=parser.parse_args()
    report=validate(args.source_directory,args.checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_lifecycle_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,packages=len(report['dependencies']['dependency_closure']['versions']),
        native_comparisons=report['execution']['exact_native_query_comparisons'],catalog_mutations=0)))
