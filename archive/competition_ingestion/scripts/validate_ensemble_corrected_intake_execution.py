"""Execute the complete stored corrected intake using the unchanged raw oracle.

The sole oracle substitution is graph acquisition: load the candidate from the
catalog instead of constructing its parent. All synthetic inputs, eleven model
branches, uncached execution and exact source-blend comparisons remain intact.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from threadpoolctl import threadpool_limits

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts import validate_ensemble_execution as oracle
from scripts.check_ensemble_corrected_intake import check_candidate
from scripts.plan_ensemble_corrected_intake import ROOT, plan, sha, require
from scripts.import_riemannian_execution import validated_graph
from scripts.audit_ensemble_original_intake_scope import COMPONENTS


def stored_candidate(proposed):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],
            row_factory=dict_row, options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_candidate(db,proposed)
        document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
        return _artifact_document_to_cdg(document,version_id=proposed['version_id'],
            content_hash=proposed['graph_sha256'],require_execution_envelope=True)


async def validate(args):
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        require(os.environ.get(name)=='1','Explicit single-thread environment required: '+name)
    library=Path(os.environ['R_LIBS_USER']).resolve()
    require(library==Path(os.environ.get('SCIONA_ENSEMBLE_R_LIBRARY',str(ROOT/'.venv/ensemble-r-runtime/library'))).resolve(),
        'Executed R library differs from qualified runtime')
    proposed=plan()
    graph=stored_candidate(proposed)
    oracle_sha=sha(oracle.__file__)
    retained=json.loads((ROOT/'docs/reviews/ensemble_execution_parity.json').read_text())
    require(oracle_sha==retained['validator_sha256'],'Raw full-ensemble oracle changed')
    for component in COMPONENTS:
        validated_graph(ROOT,args.reference_dir,component)
    checker_sha=sha(ROOT/'scripts/check_ensemble_corrected_intake.py')
    validator_sha=sha(__file__)
    print('Executing stored candidate: 155 nodes, eleven model branches, serial native threads=1',flush=True)
    # The retained oracle owns a fresh run directory and refuses to reuse one.
    with threadpool_limits(limits=1), patch.object(oracle,'build_ensemble_execution_graph',return_value=graph):
        await oracle.validate(args)
    result=json.loads(args.output.read_text())
    require(result['graph_digest']==proposed['graph_sha256'] and result['executed_nodes']==155
        and result['exact_final_source_blend'] and result['synthetic_only'],'Full stored execution incomplete')
    require(plan()==proposed and stored_candidate(proposed)==graph,'Qualification changed during execution')
    require(sha(oracle.__file__)==oracle_sha and sha(__file__)==validator_sha
        and sha(ROOT/'scripts/check_ensemble_corrected_intake.py')==checker_sha,'Execution/checker changed during validation')
    result.update(passed=True,approved=False,catalog_mutations=0,stored_candidate=True,
        candidate_version_id=proposed['version_id'],original_artifact_id=proposed['artifact_id'],
        plan_sha256=sha(ROOT/'docs/reviews/ensemble_corrected_intake_plan.json'),
        wrapper_sha256=validator_sha,candidate_checker_sha256=checker_sha,native_threads=1,
        restored_models_sha256=sha(ROOT/'docs/reviews/ensemble_restored_r_models.json'))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,stored_candidate=True,approved=False,executed_nodes=155)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    asyncio.run(validate(parser.parse_args()))
