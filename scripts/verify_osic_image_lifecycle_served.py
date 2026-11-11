"""Resolve and execute served lifecycle bindings in export and fresh replay workers."""
import argparse
from contextlib import ExitStack
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from scripts import promote_osic_image_lifecycle_providers as publisher
from scripts import validate_osic_image_lifecycle_providers as lifecycle


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


def bound_runner(observations):
    # Preserve the frozen validator's overlay, thread limits and subprocess
    # isolation. Remove only our catalog evidence before its exact comparison.
    original_run=lifecycle.RUN
    def run(command,*args,**kwargs):
        command=list(command)
        target=str(Path(lifecycle.__file__).resolve())
        bound=target in command and any(mode in command for mode in ('--worker','--replay'))
        if bound:
            command[command.index(target)]=str(Path(__file__).resolve())
        result=original_run(command,*args,**kwargs)
        if bound and result.returncode==0:
            payload=json.loads(result.stdout)
            observations.append(payload.pop('served_verification'))
            result=subprocess.CompletedProcess(result.args,result.returncode,
                stdout=json.dumps(payload)+'\n',stderr=result.stderr)
        return result
    return run


def worker(directory,replay=False):
    plan=json.loads(publisher.PLAN.read_text())
    bindings=resolve_bindings(plan)
    calls={binding['source_symbol']:0 for binding in bindings}
    children=[]
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
        stack.enter_context(patch.object(lifecycle,'RUN',bound_runner(children)))
        result=lifecycle.execute(directory,replay=replay)
    if calls!=result['lifecycle_provider_calls'] or resolve_bindings(plan)!=bindings:
        raise ValueError('Fresh served execution coverage or binding changed')
    if replay and children:
        raise ValueError('Replay must not create further execution workers')
    if not replay and (len(children)!=1 or children[0]['bindings']!=bindings or children[0]['mode']!='replay'):
        raise ValueError('Fresh catalog-bound replay worker required')
    result['served_verification']=dict(bindings=bindings,calls=calls,children=children,
        mode='replay' if replay else 'export',catalog_mutations=0)
    return result


def verify():
    plan=publisher.qualify_plan()
    bindings=resolve_bindings(plan)
    observations=[]
    with patch.object(lifecycle,'RUN',bound_runner(observations)):
        report=lifecycle.validate(Path(os.environ['SCIONA_OSIC_DEPENDENCIES']),
            Path(os.environ['SCIONA_OSIC_IMAGE_STATE']))
    if (report!=plan['qualification']['connected_execution'] or len(observations)!=1
            or observations[0]['bindings']!=bindings or observations[0]['mode']!='export'):
        raise ValueError('Fresh served lifecycle differs from frozen qualification')
    if resolve_bindings(plan)!=bindings:
        raise ValueError('Serving changed during verification')
    return dict(passed=True,approved=True,trust_tier=3,catalog_mutations=0,served_catalogs=2,
        bindings=bindings,execution=observations[0],
        exact_qualification_reproduced=True,private_source_unchanged=report['private_source_unchanged'],
        plan_sha256=publisher.sha(publisher.PLAN),verifier_sha256=publisher.sha(__file__),
        original_workflow_approved=False,
        limitations=['Fixed model and inference-state bindings in isolated export and replay workers; no HTTP transport or ranking qualification.',
            'Intermediate synthetic checkpoint only; full training and the original stored CDG remain pending.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--worker',action='store_true')
    modes.add_argument('--replay',action='store_true')
    parser.add_argument('--runtime-directory',type=Path)
    args=parser.parse_args()
    if args.worker or args.replay:
        if not args.runtime_directory:parser.error('--runtime-directory required')
        print(json.dumps(worker(args.runtime_directory,replay=args.replay)))
    else:
        result=verify()
        (publisher.ROOT/'docs/reviews/osic_image_lifecycle_served_verification.json').write_text(
            json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,providers=len(result['bindings']),
            export_calls=result['execution']['calls'],replay_calls=result['execution']['children'][0]['calls'])))
