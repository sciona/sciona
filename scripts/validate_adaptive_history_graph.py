"""Execute serialized three-output history CDG and symbolic unit contracts."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from sciona.adaptive_history_graph import build_adaptive_history_graph
import sciona.atoms.ml.calibration.adaptive_history as provider
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.dimensions import DimensionalSignature
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner

ROOT = Path(__file__).resolve().parents[1]


def validate():
    digest, nodes, edges = encode_execution_graph(build_adaptive_history_graph())
    graph = decode_execution_graph(nodes, edges, digest)
    if encode_execution_graph(graph)[0] != digest:
        raise ValueError('Graph roundtrip differs')
    runner._ensure_atoms_imported()
    cases = []
    for domain, scale in [('manufacturing_measurements', 1.), ('energy_sensor_readings', 10.)]:
        for empty in [False, True]:
            times = np.array([] if empty else [-70,-130], dtype=np.int64)
            values = np.array([] if empty else [10.*scale,30.*scale], dtype=np.float64)
            payload = dict(event_times=times, values=values, query_time=0,
                           lookbacks=np.array([60,120,180],dtype=np.int64), minimum_initial_count=3)
            captured = {}
            def capture(directory, node, name, value):
                if node == 'statistics' and name.startswith('out_'):
                    captured[name[4:]] = value
            with tempfile.TemporaryDirectory(prefix='adaptive-history-') as temporary:
                with patch.object(runner,'RUNS_DIR',Path(temporary)), patch.object(runner,'save_intermediate_value',side_effect=capture):
                    result = asyncio.run(runner.CDGExecutionSession(None,'synthetic-history',domain+str(empty)).execute(payload,cdg=graph))
            if result['status'] != 'completed' or set(captured) != {'count','mean','standard_deviation'}:
                raise ValueError('Three-output execution failed')
            expected = [0,np.nan,np.nan] if empty else [2,20.*scale,10.*scale]
            np.testing.assert_array_equal([captured[n] for n in ['count','mean','standard_deviation']],expected)
            if type(captured['count']) is not int:
                raise ValueError('Count output lost integer type')
            cases.append(dict(domain=domain,empty_window=empty,completed=True,output_ports=3))
    time, length = DimensionalSignature(T=1), DimensionalSignature(L=1)
    symbolic = provider.witness_adaptive_history_statistics(AbstractArray(shape=(2,),dtype='int64',dim=time),
        AbstractArray(shape=(2,),dtype='float64',dim=length), AbstractScalar(dtype='int64',dim=time),
        AbstractArray(shape=(3,),dtype='int64',dim=time), AbstractScalar(dtype='int64'))
    if len(symbolic)!=3 or symbolic[0].dtype!='int64' or symbolic[1].dim!=length or symbolic[2].dim!=length:
        raise ValueError('Witness units or count dtype differ')
    files=['sciona/adaptive_history_graph.py','scripts/validate_adaptive_history_graph.py','tests/test_adaptive_history.py']
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,graph_sha256=digest,
        scenarios=cases,dimensional_witness_passed=True,
        implementation_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        provider_sha256=hashlib.sha256(Path(provider.__file__).read_bytes()).hexdigest(),
        limitations=['Explicit three-output in-process execution; strict JSON transport of undefined statistics is not qualified.',
                    'Consumers must inspect count before using mean or standard deviation; no imputation is implicit.',
                    'Catalog publication and complete source workflow remain pending.'])


if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/adaptive_history_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
