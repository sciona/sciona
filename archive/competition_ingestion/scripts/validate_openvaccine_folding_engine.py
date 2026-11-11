"""Actual pinned CONTRAfold-SE execution using bundled default parameters only."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from sciona.openvaccine_structure import loop_labels

ROOT=Path(__file__).resolve().parents[1]


def main():
    cache=Path('/private/tmp/sciona_openvaccine_contrafold')
    manifest=json.loads((cache/'manifest.json').read_text())
    for p in manifest['pins']:assert hashlib.sha256((cache/p['software_path']).read_bytes()).hexdigest()==p['sha256']
    binary=cache/'src/contrafold'
    rng=np.random.default_rng(135)
    sequences=['A'*12,'G'*6+'A'*6+'C'*6]+[''.join(rng.choice(list('ACGU'),size=24)) for _ in range(4)]
    nonzero=0
    with tempfile.TemporaryDirectory(prefix='sciona_synthetic_contrafold_') as temp:
        for k,sequence in enumerate(sequences):
            path=Path(temp)/'input'
            path.write_text(''.join(f'{i+1} {base} -1\n' for i,base in enumerate(sequence)))
            command=[str(binary),'predict',str(path)]
            for mode in [[],['--viterbi']]:
                r=subprocess.run(command+mode,capture_output=True,text=True,check=True,timeout=30)
                structure=r.stdout.strip().splitlines()[-1]
                assert len(structure)==len(sequence)
                loop_labels(structure)
                if k==0:assert structure=='.'*len(sequence)
            posterior=Path(temp)/'posterior'
            subprocess.run(command+['--posteriors','0.0000000001',str(posterior)],capture_output=True,check=True,timeout=30)
            matrix=np.zeros((len(sequence),len(sequence)))
            for line in posterior.read_text().splitlines():
                parts=line.split()
                if len(parts)<2:continue
                i=int(parts[0])-1
                for field in parts[2:]:
                    index,value=field.split(':');j=int(index)-1
                    matrix[i,j]=matrix[j,i]=float(value)
            assert np.isfinite(matrix).all() and (matrix>=0).all() and (matrix<=1).all()
            assert (np.diag(matrix)==0).all() and (matrix.sum(axis=1)<=1+1e-5).all()
            if k==0:assert (matrix==0).all()
            nonzero+=int((matrix>0).any())
    assert nonzero>0
    report=dict(status='passed',synthetic_only=True,engine_commit=manifest['commit'],source_pins=manifest['pins'],
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_modified=False,
                compiler=subprocess.run(['clang++','--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0],
                build_flags=['-std=gnu++11','-O2','-include','unistd.h','-DEVIDENCE_SR','-DEVIDENCE_PARS','-DNDEBUG','-Wno-deprecated-declarations','-Wno-deprecated-register'],
                synthetic_cases=len(sequences),structure_modes=2,posterior_cases=len(sequences),nonzero_posterior_cases=nonzero,
                parameter_scope='Bundled CONTRAfold defaults; EternaFold parameter file not used',
                scope='Actual unmodified pinned engine basic execution/invariants, not OpenVaccine folding equivalence or an independent numeric folding oracle.',
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_folding_engine.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_pins'}))


if __name__=='__main__':main()
