"""Refresh unchanged synthetic Web Traffic graph evidence on the current runtime."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import io
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import torch
from threadpoolctl import threadpool_info
from scripts.audit_decomposed_webtraffic_readiness import ROOT,sha
from scripts.validate_webtraffic_graph import validate


def runtime():
    return dict(python=platform.python_version(),packages={name:metadata.version(name)
        for name in ('numpy','pandas','torch','threadpoolctl','pydantic','icontract')})


def review(source):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pins=json.loads((ROOT/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    for name,digest in pins['files'].items():
        if sha(source/name)!=digest:raise ValueError('Pinned public source changed')
    historical_path='docs/reviews/competition_webtraffic_graph.json'
    raw=(ROOT/historical_path).read_bytes()
    if raw!=subprocess.check_output(['git','show','b6787ebf:'+historical_path],cwd=ROOT):
        raise ValueError('Historical evidence changed')
    historical=json.loads(raw)
    paths=set(historical['implementation_sha256'])|{
        'scripts/validate_webtraffic_current_baseline.py','sciona/visualizer/runner.py',
        'sciona/services/execution_graph_codec.py'}
    before={p:sha(ROOT/p) for p in sorted(paths)};versions=runtime()
    for p,digest in historical['implementation_sha256'].items():
        if before[p]!=digest:raise ValueError('Historical implementation changed')
    with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
        result=validate(ROOT,source)
    if result!=historical:raise ValueError('Complete retained graph checks differ')
    pools=threadpool_info()
    if not pools or any(p['num_threads']!=1 for p in pools):raise ValueError('One-thread native pools required')
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:raise ValueError('Torch thread budget differs')
    if before!={p:sha(ROOT/p) for p in sorted(paths)} or versions!=runtime():raise ValueError('Runtime changed during validation')
    for name,digest in pins['files'].items():
        if sha(source/name)!=digest:raise ValueError('Source changed during validation')
    return dict(format='webtraffic-current-baseline.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,complete_unchanged_graph_checks=result['checks'],
        direct_and_graph_outputs_equal=True,serialized_graph_sha256=result['serialized_graph_sha256'],
        source_commit=pins['commit'],source_files_verified=len(pins['files']),
        historical_evidence_sha256=sha(ROOT/historical_path),provider_sha256=result['provider_sha256'],
        implementation_sha256=before,runtime=versions,
        resource_policy=dict(serial_cases=True,native_threads=1,torch_intraop_threads=1,torch_interop_threads=1),
        limits=['Unchanged three-stage derivative baseline only; original CDG, reusable decomposition and final-output scope remain unapproved.',
                'Short synthetic schedule executes the complete configured lifecycle, not the historical full-duration competition training.',
                'No comparison with historical output bytes, empirical accuracy, independent package authenticity or legacy TensorFlow seeded replay.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    result=review(parser.parse_args().source_root)
    (ROOT/'docs/reviews/webtraffic_current_baseline.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=result['complete_unchanged_graph_checks'],runtime=result['runtime'],approved=False)))
