#!/usr/bin/env python3
"""Full CDG runner validation using generated helices and packaged source."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from sciona.visualizer import runner
runner._ensure_atoms_imported()
from sciona.atoms.particle_tracking.source_tracking import atoms as provider
from sciona.tracking_source_execution import build_tracking_source_execution, SOURCE_VERSION
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from tests.test_source_tracking import fixture, assert_recovered


async def validate(root):
    graph = build_tracking_source_execution()
    digest, nodes, edges = encode_execution_graph(graph)
    restored = _artifact_document_to_cdg({'cdg_nodes': [{**n, 'version_id': SOURCE_VERSION} for n in nodes], 'cdg_edges': edges},
        version_id=SOURCE_VERSION, content_hash=digest, require_execution_envelope=True)
    if graph != restored:
        raise ValueError('Graph serialization differs')
    modules, hits = fixture()
    with tempfile.TemporaryDirectory(prefix='sciona-tracking-runner-') as directory:
        prior = runner.RUNS_DIR
        runner.RUNS_DIR = Path(directory)
        try:
            for i, options in enumerate([dict(extension_steps=3, commitment_rounds=1, commitment_limit=1000),
                                         dict(extension_steps=3, commitment_rounds=3, commitment_limit=1), {}]):
                run = 'case-'+str(i)
                result = await runner.CDGExecutionSession(None, 'synthetic-tracking', run).execute(
                    dict(detector_modules=modules, observations=hits, **options), cdg=restored)
                if result['status'] != 'completed':
                    raise ValueError('Full tracking runner failed: '+str(result))
                labels = np.load(Path(directory)/run/'tracking/out_track_labels.npy')
                assert_recovered(labels)
        finally:
            runner.RUNS_DIR = prior
    directory = Path(provider.__file__).parent
    closure = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
    files = ['sciona/tracking_source_execution.py', 'scripts/validate_tracking_source_execution.py',
             'tests/test_source_tracking.py', 'sciona/competition_dataframe_compat.py',
             'scripts/validate_tracking_detector_geometry.py', 'sciona/services/execution_graph_codec.py', 'sciona/visualizer/runner.py']
    return dict(approved=False, synthetic_only=True, graph_digest=digest, full_runner_cases=3,
                recovered_helices_per_case=8, exact_membership=True, provider_closure_sha256=closure,
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
                limitations=['Analytical observation-only execution; optional cell/calibrated/scoring paths excluded.',
                             'Synthetic geometric correctness only; no actual-event accuracy claim.',
                             'Source-compatible seven-ring, three-gap cap geometry and occupied neighbor layers required.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(validate(Path(__file__).resolve().parents[1]))
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['full_runner_cases', 'recovered_helices_per_case', 'exact_membership']}))
