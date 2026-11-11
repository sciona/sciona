"""Fresh read-only catalog resolution and concrete affine provider execution."""
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
from scripts import promote_affine_providers as publisher


def verify():
    plan=publisher.qualify_plan()
    record=plan['records'][0]
    with publisher.connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        publisher.check(db,plan,published=True)
        resolved=[]
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            rows=db.execute(f'''SELECT s.fqdn,p.import_module,p.source_symbol,v.version_id::text,v.content_hash,v.trust_tier
                FROM catalog_{prefix}s_served s JOIN {prefix}_versions v USING({key})
                JOIN atoms p ON p.atom_id=s.{key}
                WHERE s.{key}=%s AND v.is_latest''',(record['artifact_id'],)).fetchall()
            expected=dict(fqdn=record['fqdn'],import_module='sciona.atoms.numerical.statistics.affine_fit',
                source_symbol='affine_least_squares',version_id=record['version_id'],content_hash=record['content_hash'],trust_tier=3)
            if not rows or any(row!=expected for row in rows):
                raise ValueError('Exact served runtime binding required in both catalogs')
            resolved.append(rows[0])
        original=db.execute("SELECT status,is_publishable FROM artifacts WHERE fqdn=%s",
            ('cdg.competition.solution.kaggle.osic_pulmonary_1st',)).fetchone()
        if original!=dict(status='draft',is_publishable=False):
            raise ValueError('Original workflow state differs from pending scope')
    binding=resolved[0]
    module=importlib.import_module(binding['import_module'])
    fn=getattr(module,binding['source_symbol'])
    if publisher.sha(inspect.getfile(module))!=plan['qualification']['software_sha256'][binding['import_module']]:
        raise ValueError('Served provider implementation changed')
    examples=[]
    for name,x,slope,intercept in [
        ('synthetic_motion',np.array([-3.,0.,2.,8.]),2.5,-4.),
        ('synthetic_sensor_calibration',np.array([1.,2.,5.,9.]),-.75,12.)]:
        result=fn(x,slope*x+intercept,-1)
        assert result['rank']==2
        np.testing.assert_allclose(result['coefficients'],[slope,intercept],rtol=1e-13,atol=1e-13)
        examples.append(dict(application=name,analytic_coefficients_passed=True))
    return dict(passed=True,read_only=True,catalog_mutations=0,approved=True,trust_tier=3,
        served_catalogs=2,artifact_id=record['artifact_id'],version_id=record['version_id'],
        content_hash=record['content_hash'],fresh_resolved_execution=examples,original_workflow_approved=False,
        history_and_publication_evidence_preserved=True,verifier_sha256=publisher.sha(__file__),
        limitations=['Direct execution of the exact served Python binding; HTTP transport, semantic ranking and whole-CDG execution are not qualified.',
            'Original OSIC workflow remains draft; this publication covers only reusable affine fitting.'])


if __name__=='__main__':
    result=verify()
    (publisher.ROOT/'docs/reviews/affine_served_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,served_catalogs=2,trust_tier=3,original_workflow_approved=False)))
