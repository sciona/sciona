"""Read-only preflight for the published EfficientDet Apex O1 execution path.

Readiness is not numerical qualification and cannot approve an artifact.
This command never installs packages or substitutes another precision mode.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import platform

import torch


def assess_environment(observed):
    missing = []
    if not observed.get('cuda_available'):
        missing.append('CUDA device unavailable')
    if not observed.get('cuda_build'):
        missing.append('Torch was built without CUDA')
    for name in ('amp_initialize', 'amp_scale_loss'):
        if observed.get(name) is not True:
            missing.append('Apex API unavailable: '+name)
    historical = (observed.get('python') == '3.7.6'
                  and str(observed.get('torch', '')).split('+')[0] == '1.4.0'
                  and observed.get('cuda_build') == '10.1'
                  and observed.get('cudnn') == 7501)
    return dict(runtime_ready=not missing, missing_requirements=missing,
                historical_core_versions_match=historical,
                historical_apex_revision_qualified=False,
                numerical_precision_qualified=False, approved=False)


def main(output):
    observed = dict(python=platform.python_version(), torch=torch.__version__,
                    cuda_available=torch.cuda.is_available(), cuda_build=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(), amp_initialize=False, amp_scale_loss=False)
    try:
        available = importlib.util.find_spec('apex') is not None
        if available:
            amp = importlib.import_module('apex.amp')
            observed['amp_initialize'] = callable(getattr(amp, 'initialize', None))
            observed['amp_scale_loss'] = callable(getattr(amp, 'scale_loss', None))
            # Record a content identity without exposing environment source paths.
            if getattr(amp, '__file__', None):
                observed['amp_init_sha256'] = hashlib.sha256(Path(amp.__file__).read_bytes()).hexdigest()
    except (ImportError, OSError, RuntimeError) as error:
        observed['apex_import_error_type'] = type(error).__name__
    report = dict(**assess_environment(observed), observed=observed, read_only=True,
                  catalog_mutations=0, requested_precision='Apex O1',
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limits=['Capability and core version inspection only; no detector or AMP step executed.',
                          'Apex APIs do not prove an exact historical revision or precision semantics.',
                          'Core versions omit OS, hardware, remaining dependencies and numerical qualification.'])
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    return 0 if report['runtime_ready'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().output))
