"""Assess reuse of the existing direction provider without changing catalog approval."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from scripts.audit_decomposed_trackml_readiness import ROOT, sha
from scripts.import_flavours_provider_drafts import connect


def review():
    paths = [ROOT/'tests/test_helix_direction_reuse.py', Path(__file__), ROOT/'sciona/trackml_nearest.py']
    directory = ROOT.parent/'sciona-atoms-physics/src/sciona/atoms/particle_tracking/helix_geometry'
    paths += [directory/name for name in ('atoms.py','expressions.py','witnesses.py')]
    before = {str(p.relative_to(ROOT.parent)): sha(p) for p in paths}
    env = dict(os.environ, OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='direction-reuse-synthetic-') as directory_name:
        xml = Path(directory_name)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q',
            'tests/test_helix_direction_reuse.py','--junitxml='+str(xml)],cwd=ROOT,env=env,capture_output=True,text=True)
        if run.returncode: raise ValueError('Synthetic direction checks failed: '+run.stdout[-2000:])
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        counts = {k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts != dict(tests=37,failures=0,errors=0,skipped=0): raise ValueError('Complete synthetic coverage required')
    from sciona.atoms.particle_tracking.helix_geometry.expressions import HELIX_DIRECTION_EXPR
    # A planar quarter-turn with positive pitch: runtime returns pitch, while
    # the declared main-branch expression yields zero. This is a scope limit.
    values = dict(hel_xm=0,hel_ym=0,hel_pitch=2,x1=1,y1=0,z1=0,x2=0,y2=1,z2=0)
    symbolic = HELIX_DIRECTION_EXPR.rhs.subs({s:values[str(s)] for s in HELIX_DIRECTION_EXPR.rhs.free_symbols})
    import numpy as np
    from sciona.atoms.particle_tracking.helix_geometry.atoms import helix_direction_from_two_points
    concrete = helix_direction_from_two_points(*[np.array([float(values[k])]) for k in
        ('hel_xm','hel_ym','hel_pitch','x1','y1','z1','x2','y2','z2')])[0]
    if float(symbolic)!=0. or concrete!=2.: raise ValueError('Reassess symbolic scope after implementation change')
    catalog = {}
    fqdn='sciona.atoms.particle_tracking.helix_geometry.helix_direction_from_two_points'
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            catalog[prefix] = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                FROM {prefix}s a LEFT JOIN {prefix}_versions v ON v.{key}=a.{key} AND v.is_latest WHERE a.fqdn=%s''',(fqdn,)).fetchall()
    if before != {str(p.relative_to(ROOT.parent)):sha(p) for p in paths}: raise ValueError('Reviewed source changed')
    return dict(format='trackml-direction-reuse.v1',read_only=True,catalog_mutations=0,approved=False,
        tests=counts,synthetic_only=True,source_parity_cases=36,software_sha256=before,catalog=catalog,
        findings=[
            'Reuse the existing concrete helix-direction implementation rather than duplicate it.',
            'The provider returns signed axial difference; equal-height points use pitch times transverse rotation sense. Coincident points can return zero.',
            'Synthetic cases cover both pitch signs, axial/planar/coincident points, point-order reversal, translation, positive unit scaling and input preservation.',
            'The symbolic expression is explicitly only the main branch and is not a proof of the equal-height fallback.',
            'Legacy approval does not supply a canonical artifact identity or exact versioned interface for graph selection.'],
        next_actions=[
            'Stage exact canonical source and interfaces, retaining legacy history.',
            'Qualify aligned finite nonempty float64 inputs and reject source-undefined or nonfinite arithmetic at the recipe boundary.',
            'Preserve explicit symbolic branch limits and validate source-parity injection through the complete tracking recipe.',
            'Review separate tangent construction, unit tangent and cell arithmetic for reusable numerical boundaries.'],
        limitations=['Synthetic numerical reuse review only; no publication approval or original-CDG completion.',
            'Inputs in the same-height fallback need a meaningful transverse rotation sense; zero output is possible.',
            'No general broadcasting, degenerate-geometry, floating-point error-bound or historical accuracy guarantee.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/trackml_direction_reuse.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(tests=result['tests'],canonical_records=len(result['catalog']['artifact']),approved=False)))
