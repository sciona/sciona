"""Qualify the current CPU library versions against the unchanged DSB oracle."""
import argparse
import asyncio
import importlib.metadata
import json
import platform
from pathlib import Path
import torch

from scripts import validate_dsb_lifecycle_contract_graph as qualification
from scripts.audit_dsb_static_dependencies import audit
from scripts.stage_dsb_provider_dependencies import stage
from scripts.inventory_dsb_decomposed_providers import ROOT, sha


def environment():
    return dict(python=platform.python_version(),operating_system=platform.system(),
                architecture=platform.machine(),device='cpu',
                libraries={name:importlib.metadata.version(name) for name in
                           ('numpy','scipy','scikit-image','torch')},
                torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


async def validate(source_root):
    before=environment()
    source_audit=audit()
    dependency_check=stage()
    if dependency_check['rows_created']!=0:
        raise ValueError('Exact dependency declarations must already be staged')
    result=await qualification.validate(source_root)
    if before!=environment():raise ValueError('Runtime environment changed during qualification')
    if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
        raise ValueError('Single-thread CPU qualification required')
    for path,digest in source_audit['source_sha256'].items():
        if sha(ROOT.parent/path)!=digest:raise ValueError('Reviewed source changed during qualification')
    if result['execution_graph_sha256']!=source_audit['graph_digest']:
        raise ValueError('Qualified graph differs from dependency review')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
                environment=before,qualification=result,dependency_check=dependency_check,
                static_audit_sha256=sha(ROOT/'docs/reviews/dsb_static_dependency_audit.json'),
                validator_sha256=sha(Path(__file__)),
                limitations=['Qualification applies to these exact installed CPU library versions and reviewed source hashes.',
                    'Two synthetic three-epoch windows with inference are covered, not full competition training.',
                    'Arbitrary caller callbacks or replacement model/factory classes are outside the qualified bindings.',
                    'This validates local execution; stored-CDG execution and publication checks remain required.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    result=asyncio.run(validate(parser.parse_args().source_root))
    (ROOT/'docs/reviews/dsb_runtime_environment_qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],environment=result['environment'],
                         cases=result['qualification']['cases'])))
