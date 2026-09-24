"""Requalify projectile execution using immutable source-version validation."""
import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch
from scripts.audit_projectile_original_scope import ROOT,audit,sha,require
from scripts.validate_projectile_immutable_source import validate as validate_source
import scripts.validate_projectile_execution as oracle


async def validate(source):
    scope=audit(source)
    with patch.object(oracle,'validate_source',side_effect=validate_source):
        result=await oracle.validate(ROOT,source/'symbols.cypher',source/'infrules.cypher',source/'expr_and_feed.cypher')
    require(result['source_proof']==scope['source_proof'],'Immutable proof changed during execution')
    require(result['graph_digest']==scope['approved_execution_graph_sha256'] and result['full_runner_cases']==6
        and result['synthetic_points']==29 and result['maximum_ulp_error']==0,'Qualified execution differs')
    retained=json.loads((ROOT/'docs/reviews/projectile_execution.json').read_text())
    require({k:v for k,v in result.items() if k!='source_proof'}=={k:v for k,v in retained.items() if k!='source_proof'},'Runtime evidence differs from qualified parent')
    return dict(passed=True,approved_original=False,catalog_mutations=0,execution=result,scope_sha256=sha(ROOT/'docs/reviews/physics_projectile_original_scope.json'),validator_sha256=sha(__file__),
        source_validator_sha256=sha(ROOT/'scripts/validate_projectile_immutable_source.py'),
        scope='Unchanged Decimal/reference/runtime checks with explicit immutable-version source validator injection. No database state is mocked or rewritten.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=asyncio.run(validate(parser.parse_args().source_directory))
    (ROOT/'docs/reviews/projectile_immutable_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,runner_cases=6,synthetic_points=29,maximum_ulp_error=0,approved_original=False)))
