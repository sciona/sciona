"""Compare reusable augmentation against the pinned source probe and RNG replay."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.cornell_augmentation import Augmentation
from validate_cornell_augmentation import namespace


def main():
    source=Path('/private/tmp/sciona_cornell_source')
    deps=Path('/private/tmp/sciona_cornell_dependencies/audiomentations')
    waveform=(np.sin(np.arange(960000)*.021)*.2).astype(np.float32)
    noise=(.15*np.sin(np.arange(32000)*.13)+.04).astype(np.float32)
    results=[]
    for variant in ('default','background'):
        for seed in range(12):
            ns,_=namespace(seed)
            tree=ast.parse((source/('src/augmentations/sed_'+variant+'_augment.py')).read_bytes())
            exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'<source-recipe>','exec'),ns)
            kwargs=dict(bckgrd_aug_dir='synthetic')
            if variant=='background':kwargs['secondary_bckgrd_aug_dir']='synthetic'
            reference=ns['get_transforms'](**kwargs)['train'](waveform.copy(),32000)
            runtime=Augmentation(source,deps,variant=variant,background=[noise],short_noises=[noise],seed=seed)
            actual=runtime(waveform)
            np.testing.assert_array_equal(reference,actual)
            np.testing.assert_array_equal(runtime(waveform,training=False),waveform)
            py_state=runtime.python_rng.getstate();np_state=runtime.numpy_rng.get_state()
            resumed=Augmentation(source,deps,variant=variant,background=[noise],short_noises=[noise],seed=999)
            resumed.python_rng.setstate(py_state);resumed.numpy_rng.set_state(np_state)
            np.testing.assert_array_equal(runtime(waveform),resumed(waveform))
        results.append(dict(variant=variant,seeds=12,exact_source_match=True,exact_rng_resume=True))
    rejected=0
    for bad in ([],[np.zeros(3200)],[np.ones(1)],[np.full(3200,np.nan)],[np.full(3200,1e30)]):
        try:Augmentation(source,deps,variant='default',background=bad,seed=0)
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid noise bank accepted')
    report=dict(status='passed',cases=results,invalid_noise_banks_rejected=rejected,
        runtime_sha256=hashlib.sha256((ROOT/'sciona/cornell_augmentation.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_validator_sha256=hashlib.sha256((ROOT/'scripts/validate_cornell_augmentation.py').read_bytes()).hexdigest(),
        scope='Synthetic in-memory banks at32kHz; exact source recipes and RNG continuation. Runtime guards invalid banks; no resampling or complete model-training claim.')
    (ROOT/'docs/reviews/competition_cornell_augmentation_runtime_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
