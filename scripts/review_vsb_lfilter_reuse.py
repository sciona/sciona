"""Check existing registered lfilter reuse against the retained numeric baseline."""
import inspect
import json
from pathlib import Path
import numpy as np
from sciona.atoms.scipy.signal import lfilter
from sciona.vsb_baseline import subtract_baseline
from scripts.audit_decomposed_vsb_readiness import ROOT,sha
from scripts.import_flavours_provider_drafts import connect

FQDN='sciona.atoms.scipy.signal.lfilter'


def review():
    source=Path(inspect.getfile(inspect.unwrap(lfilter)))
    before=sha(source);cases=[]
    for dtype in (np.float32,np.float64,np.int64):
        for size in (1,7,129):
            for retention in (0.,.5,.99,1.):
                x=(np.arange(size)%11-5).astype(dtype);saved=x.copy()
                reference=subtract_baseline(x,retention=retention)
                numeric=x.astype(np.float64,copy=True)
                actual=lfilter([retention,-retention],[1.,-retention],numeric,
                    axis=-1,zi=np.array([-retention*numeric[0]]))
                actual[0]=0.
                if not np.array_equal(actual,reference) or not np.array_equal(saved,x):
                    raise ValueError('Registered filter baseline parity differs')
                cases.append(dict(dtype=np.dtype(dtype).name,length=size,retention=retention,exact_equal=True,input_unchanged=True))
    if sha(source)!=before:raise ValueError('Filter implementation changed')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows=db.execute('''SELECT a.atom_id::text,a.fqdn,a.status,a.is_publishable,a.import_module,a.source_symbol,
            v.version_id::text,v.content_hash,v.trust_tier,v.fingerprint
            FROM atoms a JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.is_latest''',(FQDN,)).fetchall()
        if len(rows)!=1:raise ValueError('Unique existing latest filter required')
        existing=rows[0]
        ports=db.execute('SELECT version_id::text,direction,name,ordinal,type_desc,required,default_value_repr,constraints FROM atom_io_specs WHERE atom_id=%s ORDER BY version_id NULLS FIRST,direction,ordinal',(existing['atom_id'],)).fetchall()
        mirrors=db.execute('SELECT artifact_id::text,status,is_publishable FROM artifacts WHERE fqdn=%s',(FQDN,)).fetchall()
        served=db.execute('SELECT count(*) AS n FROM catalog_atoms_served WHERE atom_id=%s',(existing['atom_id'],)).fetchone()['n']
    return dict(format='vsb-lfilter-reuse.v1',passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        existing_provider=existing,legacy_ports=ports,canonical_mirrors=mirrors,legacy_served_rows=served,
        provider_sha256=before,baseline_sha256=sha(ROOT/'sciona/vsb_baseline.py'),reviewer_sha256=sha(__file__),
        synthetic_cases=cases,
        decision='Reuse existing registered lfilter for the numeric filter operation after canonical/version/interface qualification; do not add another implementation of lfilter.',
        adapter_contract='Validate finite nonempty real vector and retention in [0,1], convert to float64, bind numerator [a,-a], denominator [1,-a], axis -1 and initial delay [-a*x[0]], then set first residual to zero and reject nonfinite output. The original zero-retention shortcut may remain explicit.',
        limits=['This bounded parity review does not qualify arbitrary filters, complex values, multidimensional axes or streaming final-state recovery.',
                'Existing provider returns filtered values only when initial delay is supplied; final delay state is discarded, which is sufficient for this full-vector operation.',
                'Canonical catalog publication, exact source fingerprint review and full registered interface validation remain required before graph binding.',
                'Generic filtering alone does not reproduce directional peak selection, descriptor construction, phase grouping or the full training lifecycle.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/vsb_lfilter_reuse.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(result['synthetic_cases']),canonical_mirrors=len(result['canonical_mirrors']),existing_status=result['existing_provider']['status'])))
