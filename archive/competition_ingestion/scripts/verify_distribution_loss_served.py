"""Fresh catalog-bound loss values and gradients, without catalog mutation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import importlib
import inspect
import json
import numpy as np
from scripts import promote_distribution_loss_providers as publisher


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
    checks=[]
    for binding in bindings:
        module=importlib.import_module(binding['import_module'])
        if publisher.sha(inspect.getfile(module))!=plan['qualification']['software_sha256']['provider']:
            raise ValueError('Served implementation differs')
        fn=getattr(module,binding['source_symbol'])
        for label,scale in [('synthetic_duration',1.),('synthetic_energy',1000.)]:
            targets=tf.ones((2,1),tf.float32)*scale
            if binding['source_symbol']=='mean_pinball':
                variable=tf.Variable(np.zeros((2,3),np.float32))
                with tf.GradientTape() as tape:
                    loss=fn(targets,variable,[.25,.5,.75])
                expected=.5*scale
                gradient=np.tile(-np.array([.25,.5,.75])/6,(2,1))
            else:
                variable=tf.Variable(np.tile(np.array([0.,0.,4.],np.float32)*scale,(2,1)))
                with tf.GradientTape() as tape:
                    loss=fn(targets,variable,2.*scale,5.*scale)
                expected=np.sqrt(2)/4+np.log(4*scale*np.sqrt(2))
                spread=(1/4-np.sqrt(2)/16)/(2*scale)
                gradient=np.tile([-spread,-np.sqrt(2)/(8*scale),spread],(2,1))
            assert loss.dtype==tf.float32 and loss.shape==()
            np.testing.assert_allclose(loss.numpy(),expected,rtol=3e-7)
            np.testing.assert_allclose(tape.gradient(loss,variable).numpy(),gradient,
                                       rtol=3e-7,atol=np.finfo(np.float32).eps/scale)
            checks.append(dict(provider=binding['source_symbol'],application=label,
                               analytic_value_and_gradient_passed=True))
    return dict(passed=True,read_only=True,catalog_mutations=0,approved=True,trust_tier=3,
        served_catalogs=2,bindings=bindings,fresh_resolved_execution=checks,
        original_workflow_approved=False,history_and_publication_evidence_preserved=True,
        verifier_sha256=publisher.sha(__file__),
        limitations=['Direct served Python bindings with in-process TensorFlow gradients only; no HTTP transport, semantic ranking or whole-CDG execution qualification.',
                    'Synthetic numerical checks only. The complete original OSIC workflow remains draft.'])


if __name__=='__main__':
    result=verify()
    (publisher.ROOT/'docs/reviews/distribution_loss_served_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['bindings']),execution_checks=len(result['fresh_resolved_execution']))))
