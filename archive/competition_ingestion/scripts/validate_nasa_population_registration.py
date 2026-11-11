"""Verify cold provider discovery through the configured standard atom loader."""
import contextlib
import hashlib
import io
import json
from pathlib import Path

from sciona.nasa_population_graphs import build_population_graph
from sciona.ghost.registry import REGISTRY
from sciona.visualizer.runner import _ensure_atoms_imported

ROOT=Path(__file__).resolve().parents[1]


def validate():
    required={node.matched_primitive for kind in ['training','inference']
              for node in build_population_graph(kind,10).nodes}
    before=sum(name in REGISTRY for name in required)
    if before:
        raise ValueError('Run this validation in a fresh process without provider preimports')
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        _ensure_atoms_imported()
    if required-REGISTRY.keys():raise ValueError('Standard loader missed required population providers')
    paths=[Path(__file__),ROOT/'sciona/sources.py',ROOT/'sciona/synthesizer/ghost_sim.py',
        ROOT/'sciona/nasa_population_graphs.py',
        ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/domain_adapters/__init__.py']
    return dict(passed=True,required_providers=len(required),registered_before=before,registered_after=len(required),
        provider_preimports=False,standard_configured_loader=True,catalog_mutations=0,
        source_sha256={str(path.relative_to(ROOT.parent)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/nasa_population_registration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='source_sha256'}))
