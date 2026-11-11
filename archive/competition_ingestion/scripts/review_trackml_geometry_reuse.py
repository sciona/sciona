"""Compare existing registered geometry with retained source semantics."""
import contextlib
import inspect
import io
import json
import numpy as np
from sciona.atoms.particle_tracking.helix_geometry import atoms
from sciona import trackml_geometry as circle
from sciona import trackml_helix as helix
from scripts.audit_decomposed_trackml_readiness import ROOT,sha,connect


def review():
    names=['circle_from_three_points','helix_pitch_from_two_points','helix_pitch_least_squares']
    functions=[getattr(atoms,n) for n in names]
    paths=['scripts/review_trackml_geometry_reuse.py','sciona/trackml_geometry.py','sciona/trackml_helix.py']
    before={p:sha(ROOT/p) for p in paths}
    provider_path=inspect.getfile(inspect.unwrap(functions[0]));provider_hash=sha(provider_path)
    counts=dict.fromkeys(names,0)
    def compare(index,reference,values,**options):
        saved=[v.copy() for v in values]
        actual=functions[index](*values,**options);expected=reference(*values,**options)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        for a,b in zip(values,saved):np.testing.assert_array_equal(a,b)
        counts[names[index]]+=1
    for seed in range(12):
        rng=np.random.RandomState(seed);n=64
        xm=rng.normal(size=n);ym=rng.normal(size=n);radius=rng.uniform(.5,10,n)
        phi=rng.uniform(-np.pi,np.pi,n);step=rng.uniform(.1,.9,n)*rng.choice([-1.,1.],n)
        pitch=rng.uniform(.3,12,n)*rng.choice([-1.,1.],n);z=rng.normal(size=n)
        points=[(xm+radius*np.cos(phi+j*step),ym+radius*np.sin(phi+j*step),z+j*step*pitch/(2*np.pi)) for j in range(3)]
        compare(0,circle.circle_from_three_points,[points[j][k] for j in range(3) for k in (0,1)])
        for replacement in (.001,-.02,0.):
            compare(1,helix.pitch_from_two_points,[*points[0],*points[1],xm,ym],zero_pitch=replacement)
            compare(2,helix.pitch_least_squares,[*points[0],*points[1],*points[2],xm,ym],zero_pitch=replacement)
    differences=[]
    for name,fn,reference,arity in [(names[0],functions[0],circle.circle_from_three_points,6),
                                  (names[2],functions[2],helix.pitch_least_squares,11)]:
        values=[np.zeros(1,dtype=np.float64) for _ in range(arity)]
        with contextlib.redirect_stderr(io.StringIO()):
            try:reference(*values)
            except ValueError:pass
            else:raise AssertionError('Expected undefined source geometry rejection')
        result=fn(*values)
        if not all(np.isfinite(v).all() for v in result):raise AssertionError('Expected finite substituted result')
        differences.append(dict(provider=name,case='coincident origin' if arity==6 else 'zero phase variance',
            source='rejects undefined nonfinite result',existing_provider='returns finite substituted geometry'))
    empty_rejections=[]
    for name,fn,reference,arity in [(names[0],functions[0],circle.circle_from_three_points,6),
                                  (names[1],functions[1],helix.pitch_from_two_points,8),
                                  (names[2],functions[2],helix.pitch_least_squares,11)]:
        values=[np.empty(0,dtype=np.float64) for _ in range(arity)]
        if not all(a.size==0 for a in reference(*values)):raise AssertionError('Expected source empty-vector support')
        import icontract
        try:fn(*values)
        except icontract.ViolationError:empty_rejections.append(name)
        else:raise AssertionError('Expected registered nonempty restriction')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows=db.execute('''SELECT a.fqdn,a.status,a.is_publishable,a.import_module,a.source_symbol,
            v.version_id::text,v.content_hash,v.trust_tier,
            EXISTS(SELECT 1 FROM catalog_atoms_served s WHERE s.atom_id=a.atom_id) AS served
            FROM atoms a JOIN atom_versions v USING(atom_id)
            WHERE v.is_latest AND a.import_module=%s AND a.source_symbol=ANY(%s) ORDER BY a.source_symbol''',
            (atoms.__name__,names)).fetchall()
    if before!={p:sha(ROOT/p) for p in paths} or sha(provider_path)!=provider_hash:
        raise ValueError('Compared implementation changed')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        exact_regular_cases=counts,singular_differences=differences,empty_input_restrictions=empty_rejections,
        catalog_candidates=rows,provider_sha256=provider_hash,implementation_sha256=before,
        decision='Reuse is conditional: preserve empty internal batches and reject source-undefined singular cases before invoking existing finite-domain providers. Full lifecycle parity and exact catalog qualification remain required.',
        limits=['Nondegenerate synthetic geometry only establishes the compared subset, not arbitrary input equivalence.',
                'No provider or CDG approval and no runtime observations persisted.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/trackml_geometry_reuse.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
