"""Configured CONTRAfold-SE/EternaFold coefficients through full features."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import warnings
import numpy as np
from sciona.openvaccine_features import folded_features

ROOT=Path(__file__).resolve().parents[1]


def read_posterior(path,length):
    matrix=np.zeros((length,length))
    for line in path.read_text().splitlines():
        parts=line.split()
        if len(parts)<2:continue
        i=int(parts[0])-1
        for field in parts[2:]:
            index,value=field.split(':');j=int(index)-1
            matrix[i,j]=matrix[j,i]=float(value)
    return matrix


def main():
    engine=Path('/private/tmp/sciona_openvaccine_contrafold/src/contrafold')
    binary_review=json.loads((ROOT/'docs/reviews/competition_openvaccine_folding_engine.json').read_text())
    assert hashlib.sha256(engine.read_bytes()).hexdigest()==binary_review['binary_sha256']
    params=Path('/private/tmp/sciona_openvaccine_eternafold_parameters/EternaFoldParams.v1')
    review=json.loads((ROOT/'docs/reviews/competition_openvaccine_eternafold_parameters.json').read_text())
    assert review['historical_bytes_match'] and hashlib.sha256(params.read_bytes()).hexdigest()==review['sha256']
    rng=np.random.default_rng(916)
    sequences=['A'*12,'G'*6+'A'*6+'C'*6]+[''.join(rng.choice(list('ACGU'),size=24)) for _ in range(4)]
    different=0
    with tempfile.TemporaryDirectory(prefix='sciona_synthetic_eternafold_') as temp:
        for k,sequence in enumerate(sequences):
            path=Path(temp)/'input';path.write_text(''.join(f'{i+1} {base} -1\n' for i,base in enumerate(sequence)))
            command=[str(engine),'predict',str(path)]
            configured=command+['--params',str(params)]
            structure=subprocess.run(configured,capture_output=True,text=True,check=True,timeout=30).stdout.strip().splitlines()[-1]
            posterior=Path(temp)/'posterior'
            subprocess.run(configured+['--posteriors','0.0000000001',str(posterior)],capture_output=True,check=True,timeout=30)
            matrix=read_posterior(posterior,len(sequence))
            subprocess.run(command+['--posteriors','0.0000000001',str(posterior)],capture_output=True,check=True,timeout=30)
            baseline=read_posterior(posterior,len(sequence))
            different+=int(not np.array_equal(matrix,baseline))
            if k==0:assert structure=='.'*len(sequence) and (matrix==0).all()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',FutureWarning)
                n,a=folded_features('/private/tmp/sciona_openvaccine_source',[sequence],[structure],[matrix])
            assert n.shape==(1,len(sequence)+2,55) and a.shape==(1,len(sequence)+2,len(sequence)+2,8)
            assert np.isfinite(n).all() and np.isfinite(a).all()
    assert different>0
    result=dict(status='passed',synthetic_only=True,configured_engine_cases=6,complete_feature_cases=6,
                nondefault_posterior_cases=different,engine_commit=binary_review['engine_commit'],
                binary_sha256=binary_review['binary_sha256'],parameters_sha256=review['sha256'],
                decoder='default MEA',posterior_cutoff=1e-10,
                scope='Pinned CONTRAfold-SE with historical-byte-matched EternaFold coefficients and full features on synthetic inputs; no historical competition-output or independent numeric folding parity claim.',
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_eternafold_engine.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
