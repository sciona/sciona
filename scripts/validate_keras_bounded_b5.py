"""Exercise the generic provider in the frozen B5 batch-parity qualification."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
from scripts import validate_osic_image_predictor as original
from sciona.atoms.ml import keras_bounded_prediction as provider


def worker():
    from sciona import osic_image_predictor
    calls=0
    def factory(model):
        def predict(inputs):
            nonlocal calls
            calls+=1
            return provider.predict_bounded(model,inputs,32)
        return predict
    with patch.object(osic_image_predictor,'make_slice_predictor',factory):
        result=original.worker()
    if calls!=4:raise ValueError('Generic provider must execute the valid and three invalid calls')
    result['generic_provider_calls']=calls
    return result


def validate(dependencies):
    original_run=subprocess.run
    redirects=0
    def redirect(command,*args,**kwargs):
        nonlocal redirects
        command=list(command)
        target=str(Path(original.__file__).resolve())
        if target in command and '--worker' in command:
            command[command.index(target)]=str(Path(__file__).resolve())
            redirects+=1
        return original_run(command,*args,**kwargs)
    with patch.object(original.subprocess,'run',redirect):
        report=original.validate(dependencies)
    calls=report['execution'].pop('generic_provider_calls')
    baseline=json.loads((original.ROOT/'docs/reviews/osic_image_predictor.json').read_text())
    if redirects!=1 or report!=baseline:
        raise ValueError('Generic provider differs from frozen batch-parity qualification')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        generic_provider_calls=calls,source_batch_qualification=report,
        provider_sha256=original.sha(inspect.getfile(provider)),validator_sha256=original.sha(__file__),
        limitations=['Random-initialized full-size B5 qualification, not completion of image training or empirical effectiveness.',
            'The generic provider retains the native batch-size-32 and partial-batch numerical behavior; source image geometry remains outside it.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--dependencies',type=Path)
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(worker()))
    else:
        if not args.dependencies:parser.error('--dependencies required')
        result=validate(args.dependencies)
        (original.ROOT/'docs/reviews/keras_bounded_b5.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,queries=result['source_batch_qualification']['execution']['queries'],
            generic_provider_calls=result['generic_provider_calls'])))
