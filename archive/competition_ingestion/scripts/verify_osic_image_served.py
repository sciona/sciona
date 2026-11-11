"""Resolve served image stages inside the qualified isolated execution worker."""
import argparse
from contextlib import ExitStack
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from scripts import promote_osic_image_providers as publisher
from scripts import validate_osic_image_population as population


def resolve_bindings(plan):
    bindings=[]
    with publisher.connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        publisher.check(db,plan,published=True)
        for record in plan['records']:
            module,symbol=record['fqdn'].rsplit('.',1)
            expected=dict(fqdn=record['fqdn'],import_module=module,source_symbol=symbol,
                version_id=record['version_id'],content_hash=record['content_hash'],trust_tier=3)
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                rows=db.execute(f'''SELECT s.fqdn,p.import_module,p.source_symbol,v.version_id::text,v.content_hash,v.trust_tier
                    FROM catalog_{prefix}s_served s JOIN {prefix}_versions v USING({key})
                    JOIN atoms p ON p.atom_id=s.{key}
                    WHERE s.{key}=%s AND v.is_latest''',(record['artifact_id'],)).fetchall()
                if not rows or any(row!=expected for row in rows):
                    raise ValueError('Exact served binding required in both catalogs')
            bindings.append(expected)
        original=db.execute('SELECT status,is_publishable FROM artifacts WHERE fqdn=%s',
            ('cdg.competition.solution.kaggle.osic_pulmonary_1st',)).fetchone()
        if original!=dict(status='draft',is_publishable=False):
            raise ValueError('Original workflow must remain pending')
    return bindings


def worker(args):
    plan=json.loads(publisher.PLAN.read_text())
    bindings=resolve_bindings(plan)
    calls={binding['source_symbol']:0 for binding in bindings}
    with ExitStack() as stack:
        for binding in bindings:
            module=importlib.import_module(binding['import_module'])
            if publisher.sha(inspect.getfile(module))!=plan['qualification']['software_sha256']['provider']:
                raise ValueError('Served provider source changed')
            symbol=binding['source_symbol']
            fn=getattr(module,symbol)
            def counted(*a,_fn=fn,_symbol=symbol,**kw):
                calls[_symbol]+=1
                return _fn(*a,**kw)
            stack.enter_context(patch.object(module,symbol,counted))
        execution=population.worker(args.source_directory,args.dependencies,args.state_directory,compose=True)
    if not all(count>0 for count in calls.values()):
        raise ValueError('Every resolved served interface must execute')
    if resolve_bindings(plan)!=bindings:
        raise ValueError('Served bindings changed during execution')
    execution['served_verification']=dict(bindings=bindings,calls=calls,catalog_mutations=0)
    return execution


def verify():
    plan=publisher.qualify_plan()
    bindings=resolve_bindings(plan)
    original_run=subprocess.run
    redirects=0
    def run_bound(command,*args,**kwargs):
        nonlocal redirects
        # Retain the frozen validator's archive checks, isolated OpenCV copy,
        # worker environment and thread limits. Only its worker entrypoint changes.
        command=list(command)
        target=str(Path(population.__file__).resolve())
        if target in command and '--worker' in command:
            command[command.index(target)]=str(Path(__file__).resolve())
            redirects+=1
        return original_run(command,*args,**kwargs)
    with patch.object(population.subprocess,'run',run_bound):
        report=population.validate(Path(os.environ['SCIONA_OSIC_DEPENDENCIES']),
            Path(os.environ['SCIONA_OSIC_SOURCE']),Path(os.environ['SCIONA_OSIC_IMAGE_STATE']),compose=True)
    if redirects!=1 or report['execution']['served_verification']['bindings']!=bindings:
        raise ValueError('Exactly one fresh catalog-bound worker required')
    expected=plan['qualification']['connected_execution']
    execution=dict(report['execution'])
    served=execution.pop('served_verification')
    report['execution']=execution
    if report!=expected:
        raise ValueError('Fresh served execution or implementation differs from qualified composition')
    if resolve_bindings(plan)!=bindings:
        raise ValueError('Serving changed during verification')
    return dict(passed=True,approved=True,trust_tier=3,catalog_mutations=0,served_catalogs=2,
        bindings=bindings,fresh_worker_calls=served['calls'],fresh_execution=execution,
        plan_sha256=publisher.sha(publisher.PLAN),verifier_sha256=publisher.sha(__file__),
        original_workflow_approved=False,
        limitations=['Four source-specific preparation/reduction bindings in the provisioned isolated worker; no HTTP transport or ranking qualification.',
            'Synthetic intermediate image checkpoint only; complete image training and the original CDG remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--dependencies')
    parser.add_argument('--source-directory')
    parser.add_argument('--state-directory',type=Path)
    parser.add_argument('--compose',action='store_true')
    args=parser.parse_args()
    if args.worker:
        if not args.compose or not all((args.dependencies,args.source_directory,args.state_directory)):
            parser.error('Complete isolated composition arguments required')
        print(json.dumps(worker(args)))
    else:
        result=verify()
        (publisher.ROOT/'docs/reviews/osic_image_served_verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,providers=len(result['bindings']),calls=result['fresh_worker_calls'])))
