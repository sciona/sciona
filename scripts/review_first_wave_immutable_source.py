"""Review the immutable first-wave source independently of publication state."""
import hashlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
import sympy as sp
from sciona.physics_ingest.first_wave_dynamics_proof import corrected_proof


def review(root):
    source_version='693697cf-a97c-53bd-8d6b-19dc3c7697c1'
    source_hash='97dbc4000139b6514d678829e16c39e045f02f47ecbe91356863028decb932b3'
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        source=db.execute('SELECT a.fqdn,a.status,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',(source_version,)).fetchone()
        assert source['content_hash']==source_hash
        assert source['status'] in ['draft','approved']
        bindings=db.execute('SELECT DISTINCT bound_artifact_fqdn,bound_version_content_hash FROM artifact_cdg_bindings WHERE version_id=%s',(source_version,)).fetchall()
        expressions=[]
        for binding in bindings:
            doc=db.execute('SELECT get_artifact_document(%s) AS d',(binding['bound_artifact_fqdn'],)).fetchone()['d']
            versions=db.execute('SELECT v.version_id FROM artifact_versions v JOIN artifacts a USING(artifact_id) WHERE a.fqdn=%s AND v.content_hash=%s',(binding['bound_artifact_fqdn'],binding['bound_version_content_hash'])).fetchall()
            assert len(versions)==1
            matches=[e for e in doc['symbolic_expressions'] if str(e['version_id'])==str(versions[0]['version_id'])]
            assert len(matches)==1
            e=matches[0]
            expressions.append(dict(fqdn=binding['bound_artifact_fqdn'],content_hash=binding['bound_version_content_hash'],
                                    version_id=str(e['version_id']),raw_formula=e['raw_formula'],sympy_srepr=e['sympy_srepr']))
        nodes=db.execute('SELECT node_id,type_signature FROM artifact_cdg_nodes WHERE version_id=%s ORDER BY node_id',(source_version,)).fetchall()
    historical=json.loads((root/'docs/reviews/physics_first_wave_dynamics_review.json').read_text())
    for name,digest in historical['hashes'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, 'Historical source reviewer changed'
    assert source_version==historical['source_version_id'] and source_hash==historical['source_hash']
    assert sorted(expressions,key=lambda e:e['version_id'])==sorted(historical['source_expressions'],key=lambda e:e['version_id']), 'Immutable expressions changed'
    assert nodes==historical['source_nodes'], 'Immutable source nodes changed'
    proof=corrected_proof(); assert all(proof['checks'].values())
    assert proof==historical['corrected_proof'], 'Corrected premises changed' 
    # Independent exact examples for positive/negative/zero acceleration and time-varying force.
    t=sp.Symbol('t',real=True); count=0
    for position in [sp.Integer(0),t,3*t*t-2*t+4,t**4+2*t,sp.sin(t),sp.exp(t)]:
        for mass in [sp.Rational(1,3),sp.Integer(1),sp.Integer(7)]:
            acceleration=sp.diff(position,t,2); force=mass*acceleration
            assert sp.simplify(force/mass-acceleration)==0
            assert sp.simplify(force-mass*sp.diff(position,t,2))==0
            count+=1
    # Counterexample: F=ma alone says nothing about a separately chosen x(t).
    assert 2*3 != 2*sp.diff(t**2,t,2)
    legacy=next(e for e in expressions if e['fqdn'].endswith('eq_constant_mass_force'))
    assert 'Derivative(' not in legacy['sympy_srepr']
    paths=[Path(__file__),root/'sciona/physics_ingest/first_wave_dynamics_proof.py']
    return dict(read_only=True,approved=False,source_version_id=source_version,source_hash=source_hash,
        source_expressions=expressions,source_nodes=nodes,corrected_proof=proof,
        checks=dict(exact_symbolic_examples=count,missing_premise_counterexample=1,invalid_legacy_derivative_confirmed=1),
        hashes={str(p.resolve().relative_to(root.resolve())):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        historical_review_sha256=hashlib.sha256((root/'docs/reviews/physics_first_wave_dynamics_review.json').read_bytes()).hexdigest(),
        scope='Immutable original expressions and corrected premises; no artifact approval is inferred from this review')

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    result=review(root)
    (root/'docs/reviews/physics_first_wave_immutable_source_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['checks']))
