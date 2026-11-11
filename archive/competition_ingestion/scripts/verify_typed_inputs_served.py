"""Fresh catalog-bound corrected generic conversions, without catalog mutation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import importlib
import inspect
import json
from scripts import promote_typed_inputs_providers as publisher


def verify():
    plan=publisher.qualify_plan()
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
            raise ValueError('Original workflow state differs from pending scope')
    from unittest.mock import patch
    from contextlib import ExitStack
    from scripts import validate_typed_inputs_unicode_correction as validator
    from scripts import review_typed_inputs_semantics as semantic
    counts={b['source_symbol']:0 for b in bindings}
    def counted(symbol,fn):
        def call(*args,**kwargs):
            counts[symbol]+=1
            return fn(*args,**kwargs)
        return call
    with ExitStack() as stack:
        for binding in bindings:
            module=importlib.import_module(binding['import_module'])
            if publisher.sha(inspect.getfile(module))!=plan['qualification']['software_sha256']['provider']:
                raise ValueError('Served implementation differs')
            symbol=binding['source_symbol']
            stack.enter_context(patch.object(module,symbol,counted(symbol,getattr(module,symbol))))
        execution=validator.validate()
        precision=semantic.precision_checks()
    if (execution!=plan['qualification']['connected_execution'] or precision!=plan['qualification']['precision_checks']
            or counts!=dict(cast_finite_array=20,encode_feature_descriptors=13)):
        raise ValueError('Exact served conversion qualification differs')
    checks=[dict(provider=symbol,resolved_calls=count,exact_qualification_reproduced=True) for symbol,count in counts.items()]
    return dict(passed=True,read_only=True,catalog_mutations=0,approved=True,trust_tier=3,
        served_catalogs=2,bindings=bindings,fresh_resolved_execution=checks,
        original_workflow_approved=False,history_and_publication_evidence_preserved=True,
        verifier_sha256=publisher.sha(__file__),
        limitations=['Direct served dtype conversion and corrected descriptor encoding only; no HTTP transport, semantic ranking or whole-CDG execution qualification.',
                    'Synthetic numerical checks only. The complete original OSIC workflow remains draft.'])


if __name__=='__main__':
    result=verify()
    (publisher.ROOT/'docs/reviews/typed_inputs_served_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['bindings']),execution_checks=len(result['fresh_resolved_execution']))))
