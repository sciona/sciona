"""Qualify optional-server exclusion in the parent and every inference child."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from unittest.mock import patch

from scripts.validate_residual_classifier_runtime_profile import RejectOptionalServerImports,EXCLUDED

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path)
    parser.add_argument('--child-directory',type=Path)
    args=parser.parse_args()
    if not args.source_directory and not args.child_directory:parser.error('A source or child directory is required')
    if any(name.split('.')[0] in EXCLUDED for name in sys.modules):raise ValueError('Fresh runtime process required')
    guard=RejectOptionalServerImports()
    sys.meta_path.insert(0,guard)
    try:
        import psycopg
        import scripts.validate_nasa_lifecycle_catalog as catalog
        if args.child_directory:
            snapshot=json.loads((args.child_directory/'catalog_inference.json').read_text())
            with patch.object(catalog.execution,'build_nasa_inference_graph',return_value=catalog.materialize(snapshot)):
                catalog.execution.child(args.child_directory)
            if any(name.split('.')[0] in EXCLUDED for name in sys.modules):raise ValueError('Excluded optional package imported')
            profile=dict(passed=True,excluded_optional_packages=sorted(EXCLUDED),
                unavailable_import_attempts=guard.attempted,psycopg_implementation=psycopg.pq.__impl__)
            (args.child_directory/'optional_imports.json').write_text(json.dumps(profile))
            return
        from scripts.review_conditional_correction_environment import dependency_closure
        files={'sciona':ROOT/'pyproject.toml','sciona-atoms-ml':ROOT.parent/'sciona-atoms-ml/pyproject.toml'}
        projects={name:tomllib.loads(path.read_text())['project'] for name,path in files.items()}
        closure=dependency_closure(['sciona-atoms-ml[xgboost]','sciona','psycopg','python-dotenv'],projects)
        if not closure['compatible']:raise ValueError('In-process dependencies differ')
        original_run=subprocess.run
        children=[]
        def run(command,*positional,**keywords):
            selected=isinstance(command,list) and len(command)==4 and command[2]=='--child-directory' and Path(command[1]).name=='validate_nasa_lifecycle_catalog.py'
            if selected:command=[command[0],str(Path(__file__).resolve()),*command[2:]]
            result=original_run(command,*positional,**keywords)
            if selected and result.returncode==0:
                profile=json.loads((Path(command[3])/'optional_imports.json').read_text())
                if not profile['passed'] or profile['excluded_optional_packages']!=sorted(EXCLUDED) or profile['psycopg_implementation']!='python':
                    raise ValueError('Fresh child profile differs')
                children.append(profile)
            return result
        with patch.object(subprocess,'run',side_effect=run):
            execution=catalog.validate(args.source_directory)
        if not execution['passed'] or len(children)!=10 or psycopg.pq.__impl__!='python':
            raise ValueError('Complete parent/child profile qualification required')
        if any(name.split('.')[0] in EXCLUDED for name in sys.modules):raise ValueError('Excluded optional package imported')
        report=dict(passed=True,approved=False,catalog_mutations=0,execution=execution,dependency_closure=closure,
            psycopg_implementation=psycopg.pq.__impl__,libpq_version=psycopg.pq.version(),
            excluded_optional_packages=sorted(EXCLUDED),unavailable_import_attempts=guard.attempted,child_profiles=children,
            manifest_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()},
            validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limitations=['Provisioned in-process parent and fresh inference children only; no HTTP deployment or clean-install claim.',
                        'Every inference child executes the catalog snapshot with model fitting disabled.',
                        'Whole-graph static table propagation remains outside qualification.'])
        (ROOT/'docs/reviews/nasa_lifecycle_runtime_profile.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(passed=True,packages=len(closure['versions']),guarded_children=len(children),catalog_mutations=0)))
    finally:
        sys.meta_path.remove(guard)


if __name__=='__main__':main()
