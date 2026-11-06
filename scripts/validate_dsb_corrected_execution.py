"""Execute the stored original-identity DSB candidate against the pinned oracle."""
import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from scripts import validate_dsb_lifecycle_contract_graph as qualification
from scripts.qualify_dsb_runtime_environment import environment
from scripts.stage_dsb_corrected_intake import ROOT,PLAN,qualified_plan,connect,check_candidate,sha


async def validate(source_root):
    plan,providers=qualified_plan()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        graph=check_candidate(db,plan,providers)
    retained_path=ROOT/'docs/reviews/dsb_runtime_environment_qualification.json'
    retained=json.loads(retained_path.read_text())
    if environment()!=retained['environment']:raise ValueError('Qualified library environment differs')
    with patch.object(qualification,'build_dsb_evaluation_graph',lambda:graph.model_copy(deep=True)):
        result=await qualification.validate(source_root)
    for field in ('source_commit','cases','executions','implementation_sha256','provider_sha256'):
        if result[field]!=retained['qualification'][field]:raise ValueError('Stored execution differs: '+field)
    if result['execution_graph_sha256']!=plan['graph_sha256']:
        raise ValueError('Executed graph hash differs from stored candidate')
    after,after_providers=qualified_plan()
    if after!=plan or environment()!=retained['environment']:raise ValueError('Qualification changed during execution')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if check_candidate(db,after,after_providers)!=graph:raise ValueError('Stored graph changed during execution')
    result.update(stored_candidate=True,candidate_version_id=plan['version_id'],
        original_artifact_id=plan['artifact_id'],plan_sha256=sha(PLAN),environment=environment(),
        stored_validator_sha256=sha(Path(__file__)),candidate_checker_sha256=sha(ROOT/'scripts/stage_dsb_corrected_intake.py'),
        provider_checker_sha256=sha(ROOT/'scripts/dsb_staged_provider_state.py'),
        retained_environment_evidence_sha256=sha(retained_path))
    result['limitations'] += ['Stored original-identity candidate remains unpublished; native eligibility and production selection still require publication gates.']
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    result=asyncio.run(validate(parser.parse_args().source_root))
    (ROOT/'docs/reviews/dsb_corrected_intake_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','stored_candidate','candidate_version_id','cases')}))
