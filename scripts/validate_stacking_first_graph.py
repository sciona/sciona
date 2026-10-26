"""Full native first-level execution through explicit source-entry graph nodes."""
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import sciona.atoms.ml.stacking_first as provider
from sciona.otto_execution import prepare, _first_controls
from sciona.otto_tsne import fit_population
from sciona.stacking_first_graph import build_first_graph
from sciona.stacking_first_operations import neighbor_groups
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from scripts.otto_synthetic import payload
from scripts.review_otto_candidate_runtime import review as runtime_review

ROOT=Path(__file__).resolve().parents[1]
JAVA_FLAGS='-XX:ActiveProcessorCount=1 -XX:+UseSerialGC -XX:-TieredCompilation -XX:CICompilerCount=1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not __debug__:raise RuntimeError('Validation assertions required')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError('Explicit one-thread native pools required')
    if os.environ.get('JAVA_TOOL_OPTIONS')!=JAVA_FLAGS:
        raise ValueError('Bounded Java processor, GC and compiler configuration required')
    runtime=runtime_review()
    paths=list((ROOT/'sciona').glob('otto_*.py'))+list((ROOT/'sciona').glob('otto_*.R'))+[
        ROOT/'sciona/stacking_first_operations.py',ROOT/'sciona/stacking_first_graph.py',
        ROOT/'scripts/otto_synthetic.py',Path(__file__).resolve()]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    provider_path=Path(inspect.getfile(provider));provider_hash=sha(provider_path)
    data=json.loads(prepare(payload(r_library=str(ROOT/'.venv/otto-r-library'),
                                   libfm=str(ROOT/'.venv/otto-libfm/bin/libFM'))).configuration)
    t,q,c=data['training'],data['query'],data['controls']
    controls=_first_controls(c['first_level'])
    print('Fitting fixed synthetic population embedding',flush=True)
    embedding=fit_population(np.vstack((t['values'],q['values'])),t['identities']+q['identities'],
                             seed=c['seed'],**c['embedding'])
    digest,nodes,edges=encode_execution_graph(build_first_graph(neighbor_groups(controls)))
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    completed=set();blocks={}
    def capture(directory,node,port,value):
        if port.startswith('out_'):
            completed.add(node)
            if node=='collect':blocks.update(value)
            if node!='policy':print('Completed first-level node: '+node,flush=True)
    inputs=dict(training=t['values'],labels=t['labels'],folds=t['first_folds'],training_ids=t['identities'],
                query=q['values'],query_ids=q['identities'],embedding=embedding,seed=c['seed'],controls=controls)
    with tempfile.TemporaryDirectory(prefix='synthetic-stacking-first-') as temporary:
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-first','native').execute(inputs,cdg=graph))
    assert result['status']=='completed'
    assert completed=={n.node_id for n in graph.nodes} and set(blocks)==set(range(1,34))
    for entry,block in blocks.items():
        width=5 if entry==10 else 9
        assert block.training.shape==(1350,width) and block.query.shape==(9,width)
        assert np.isfinite(block.training).all() and np.isfinite(block.query).all()
        assert block.training_ids==tuple(t['identities']) and block.query_ids==tuple(q['identities'])
        if entry==10:assert block.folds is None
        else:
            np.testing.assert_array_equal(block.folds,t['first_folds'])
            assert (block.training>=0).all() and (block.training<=1).all()
            assert (block.query>=0).all() and (block.query<=1).all()
    assert sum(block.training.shape[1] for block in blocks.values())==293
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    assert sha(provider_path)==provider_hash and runtime_review()==runtime
    report=dict(status='passed',approved=False,catalog_mutations=0,
        serialized_graph_sha256=digest,actual_runner_nodes=len(graph.nodes),source_entries=33,
        first_level_width=293,shared_neighbor_banks=len(neighbor_groups(controls)),
        synthetic_training_rows=1350,synthetic_query_rows=9,java_resource_flags=JAVA_FLAGS,
        provider_sha256=provider_hash,implementation_sha256=before,
        candidate_runtime_evidence_sha256=runtime['evidence_sha256'],
        limits=['Native first-level source-adapter computation on synthetic inputs only.',
                'Supplemental features and meta stage are not executed by this validator.',
                'No original-intake completion, historical winning accuracy or catalog approval claim.',
                'No private input values, identifiers, feature matrices or model state retained in evidence.'])
    (ROOT/'docs/reviews/competition_otto_decomposed_first_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',actual_runner_nodes=len(graph.nodes),source_entries=33)),flush=True)


if __name__=='__main__':main()
