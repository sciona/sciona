"""Run the unchanged full synthetic oracle against the stored corrected graph."""
import json
import os
import runpy
import tempfile
import time
from pathlib import Path

import pytest
from threadpoolctl import threadpool_limits

from scripts.stage_flavours_corrected_intake import ROOT,PLAN,qualified_plan,connect,check_candidate,sha


def validate():
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('Explicit single-thread environment required')
    plan,providers=qualified_plan()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        graph=check_candidate(db,plan,providers)
    retained=json.loads((ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json').read_text())
    oracle_path=ROOT/'tests/test_flavours_decomposed_evaluation.py'
    oracle_sha=sha(oracle_path)
    if oracle_sha!=retained['source_sha256']['tests/test_flavours_decomposed_evaluation.py']:
        raise ValueError('Full evaluated numerical oracle changed')
    oracle=runpy.run_path(str(oracle_path))['test_complete_evaluated_graph_matches_reference']
    started=time.monotonic()
    print('Executing catalog-loaded 841-node candidate and full 115-fit reference; native threads=1',flush=True)
    with tempfile.TemporaryDirectory(prefix='sciona-flavours-stored-') as temporary:
        with pytest.MonkeyPatch.context() as patch,threadpool_limits(limits=1):
            # Only graph acquisition changes. The original fixture, complete
            # training budgets, uncached runner and numerical assertions remain.
            patch.setitem(oracle.__globals__,'build_evaluated_graph',lambda:graph.model_copy(deep=True))
            oracle(Path(temporary),patch)
    if sha(oracle_path)!=oracle_sha:raise ValueError('Oracle changed during execution')
    after,after_providers=qualified_plan()
    if after!=plan:raise ValueError('Qualification changed during execution')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if check_candidate(db,after,after_providers)!=graph:raise ValueError('Stored candidate changed during execution')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,stored_candidate=True,
        graph_sha256=plan['graph_sha256'],candidate_version_id=plan['version_id'],original_artifact_id=plan['artifact_id'],
        executed_nodes=841,all_nodes_uncached=True,model_fits_per_workflow=115,
        reference_and_candidate_both_executed=True,exact_query_scores=True,exact_all_population_blends=True,
        exact_metrics_and_threshold_flags=True,exact_population_boundaries=True,codec_roundtrip=True,
        native_threads=1,elapsed_seconds=round(time.monotonic()-started,3),
        oracle_sha256=oracle_sha,validator_sha256=sha(__file__),candidate_checker_sha256=sha(ROOT/'scripts/stage_flavours_corrected_intake.py'),
        plan_sha256=sha(PLAN),retained_base_evidence_sha256=sha(ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json'),
        limitations=['Full-budget synthetic stored-candidate execution; no predictive-quality or historical-engine equivalence claim.',
            'The original CDG remains unpublished until its publication checks and transaction pass.',
            'All validation-loss nodes execute, while exact individual fit histories are separately covered by retained branch qualifications.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/flavours_corrected_intake_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
