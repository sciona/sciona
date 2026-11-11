"""Actual private adapter versus direct configured executable, synthetic only."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import warnings
from unittest.mock import patch
import numpy as np
from sciona.openvaccine_folding import fold_sequence,fold_features
from validate_openvaccine_eternafold_engine import read_posterior

ROOT=Path(__file__).resolve().parents[1]


def main():
    binary='/private/tmp/sciona_openvaccine_contrafold/src/contrafold'
    parameters='/private/tmp/sciona_openvaccine_eternafold_parameters/EternaFoldParams.v1'
    rng=np.random.default_rng(551);sequences=['A'*18,'G'*6+'A'*6+'C'*6]+[''.join(rng.choice(list('ACGU'),size=18)) for _ in range(3)]
    with tempfile.TemporaryDirectory(prefix='sciona_fold_reference_') as temp:
        for sequence in sequences:
            structure,matrix=fold_sequence(sequence,binary=binary,parameters=parameters)
            path=Path(temp)/'input';path.write_text(''.join(f'{i+1} {base} -1\n' for i,base in enumerate(sequence)))
            command=[binary,'predict',str(path),'--params',parameters]
            direct=subprocess.run(command,capture_output=True,text=True,check=True).stdout.strip().splitlines()[-1]
            assert direct==structure
            posterior=Path(temp)/'posterior'
            subprocess.run(command+['--posteriors','0.0000000001',str(posterior)],capture_output=True,check=True)
            np.testing.assert_array_equal(matrix,read_posterior(posterior,len(sequence)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',FutureWarning)
            n,a=fold_features('/private/tmp/sciona_openvaccine_source',sequences,binary=binary,parameters=parameters)
        assert n.shape==(5,20,55) and a.shape==(5,20,20,8)
        bad=Path(temp)/'bad';bad.write_bytes(b'synthetic invalid software')
        for kw in [dict(binary=bad,parameters=parameters),dict(binary=binary,parameters=bad)]:
            try:fold_sequence('A'*8,**kw)
            except ValueError:pass
            else:raise AssertionError('Unreviewed folding dependency accepted')
    for failure in [subprocess.TimeoutExpired('synthetic',1),subprocess.CompletedProcess([],1,stdout='',stderr='synthetic-private-marker')]:
        with patch('sciona.openvaccine_folding.subprocess.run',side_effect=failure if isinstance(failure,Exception) else None,return_value=failure):
            try:fold_sequence('A'*8,binary=binary,parameters=parameters)
            except RuntimeError as error:assert 'synthetic-private-marker' not in str(error)
            else:raise AssertionError('Subprocess failure not handled')
    result=dict(status='passed',synthetic_only=True,direct_engine_comparison_cases=5,complete_feature_batch=5,
                changed_dependencies_rejected=2,subprocess_failure_cases=2,
                scope='Reusable pinned private folding and feature adapter; exact direct-executable comparison, not independent solver or historical competition-output parity.',
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_folding.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_folding_runtime.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
