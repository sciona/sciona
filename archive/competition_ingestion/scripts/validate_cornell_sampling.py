"""Compare waveform sampling with original dataset methods using synthetic inputs."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.cornell_sampling import sample_waveform,PERIOD_SAMPLES


def main():
    cache=Path('/private/tmp/sciona_cornell_source')
    manifest=json.loads((cache/'manifest.json').read_text())
    name='src/dataloaders/sed_dataset.py';raw=(cache/name).read_bytes()
    expected=next(p['sha256'] for p in manifest['pins'] if p['software_path']==name)
    assert hashlib.sha256(raw).hexdigest()==expected
    ns=dict(np=np,torch=torch,Dataset=torch.utils.data.Dataset)
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='SedDataset')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<pinned-dataset-methods>','exec'),ns)
    checks=[]
    global_state=np.random.get_state()
    for training in (False,True):
        for length in (PERIOD_SAMPLES-1,PERIOD_SAMPLES,PERIOD_SAMPLES+1,2*PERIOD_SAMPLES+17):
            waveform=np.linspace(-.5,.5,length,dtype=np.float64)
            before=waveform.copy()
            dataset=ns['SedDataset'].__new__(ns['SedDataset'])
            dataset.transform=None;dataset.period=30;dataset.isTraining=training
            dataset.num_test_samples=1 if training else 2
            dataset.bird_code={'synthetic':0}
            dataset.get_audio=lambda _:dict(y=waveform.copy(),sr=32000,duration=length,
                all_labels=['synthetic'],primary_labels=['synthetic'],secondary_labels=[],filename='synthetic')
            source_rng=np.random.RandomState(1729)
            with patch.object(np.random,'randint',source_rng.randint):
                reference=dataset[0]['waveforms']
            runtime_rng=np.random.RandomState(1729)
            actual=sample_waveform(waveform,training=training,rng=runtime_rng)
            np.testing.assert_array_equal(actual,reference)
            np.testing.assert_array_equal(before,waveform)
            assert source_rng.randint(2**30)==runtime_rng.randint(2**30)
            checks.append(dict(training=training,synthetic_samples=length,exact_source_match=True,rng_advance_exact=True))
    after=np.random.get_state()
    assert global_state[0]==after[0] and global_state[2:]==after[2:]
    np.testing.assert_array_equal(global_state[1],after[1])
    failures=0
    for values in ([],[float('nan')],[[1,2]],[1e100]):
        try:sample_waveform(values,training=True,rng=np.random.RandomState(2))
        except ValueError:failures+=1
        else:raise AssertionError('Invalid waveform accepted')
    report=dict(status='passed',source_commit=manifest['commit'],checks=checks,
        invalid_inputs_rejected=failures,caller_waveform_and_global_rng_preserved=True,
        runtime_sha256=hashlib.sha256((ROOT/'sciona/cornell_sampling.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Original post-augmentation waveform sampling only. In-memory synthetic data bypasses source file loading and eval; complete augmentation/training remains separate.')
    (ROOT/'docs/reviews/competition_cornell_sampling_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(checks),invalid_inputs_rejected=failures,status='passed')))


if __name__=='__main__':main()
