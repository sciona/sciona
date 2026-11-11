"""Run every learned branch with actual upstream predictions on synthetic sequences."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.amex_pipeline import fit


def population(rng,labels):
    return dict(numerics=[np.floor(rng.normal(loc=2*y,scale=.3,size=(13,2))*100) for y in labels],
        categories=[rng.integers(0,2,(13,1)).astype(float)*100 for _ in labels],
        times=[list(range(13)) for _ in labels],months=[list(range(13)) for _ in labels])


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    paths=sorted((ROOT/'sciona').glob('amex_*.py'))+[Path(__file__).resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    rng=np.random.default_rng(118);labels=[0]*800+[1]*800;query_labels=[0,1]*4
    training=population(rng,labels);query=population(rng,query_labels)
    result=fit(training,query,labels,zero_fill_columns=[0],progress=lambda s:print(s,flush=True))
    assert result['model_counts']==dict(row=5,downstream_tree=10,neural=10)
    b=result['branch_probabilities'];assert b.shape==(4,8) and np.isfinite(b).all()
    expected=np.array([sum(float(b[j,i])*w for j,w in enumerate((.3,.35,.15,.1))) for i in range(8)])
    np.testing.assert_allclose(result['scores'],expected,atol=1e-15,rtol=1e-15)
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,models=25,model_counts=result['model_counts'],manual_width=result['manual_width'],sequence_width=result['sequence_width'],score_kind=result['score_kind'],actual_upstream_predictions=True,sha256=before,
        limits=['Denoised/mapped synthetic inputs with explicit roles; raw schema mapping and private graph boundary remain pending.',
            'Full configured tree rounds and neural epochs invoked on independent CPU runtime; no historical volume, accuracy or GPU parity qualification.'])
    (ROOT/'docs/reviews/competition_amex_pipeline.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('sha256','limits')}),flush=True)


if __name__=='__main__':main()
