"""Run actual stored constant_acceleration revision with immutable source and Decimal oracle."""
import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_constant_acceleration_original_revision import ROOT,sha,require
from scripts.check_constant_acceleration_revision import check
from scripts.audit_constant_acceleration_original_scope import audit
from scripts.validate_constant_acceleration_immutable_source import validate as validate_source
import scripts.validate_constant_acceleration_execution as oracle


def selected(approved):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:return check(db,approved)


async def validate(source,approved=False):
    scope=audit(source)
    require(scope==json.loads((ROOT/'docs/reviews/physics_constant_acceleration_original_scope.json').read_text()),'Qualified scope changed')
    graph,imported=selected(approved)
    # Explicit graph construction/source-validation dependency injection. The
    # numerical oracle and real stored catalog selection remain unchanged.
    with patch.object(oracle,'build_constant_acceleration_execution',return_value=graph),patch.object(oracle,'validate_source',side_effect=validate_source):
        execution=await oracle.validate(ROOT,source/'symbols.cypher',source/'infrules.cypher',source/'expr_and_feed.cypher')
    require(execution['graph_digest']==imported['graph_sha256'] and execution['full_runner_cases']==6
        and execution['synthetic_states']==29 and execution['maximum_ulp_error']==0,'Stored execution coverage differs')
    require(execution['source_proof']==scope['source_proof'],'Immutable source proof differs')
    require(selected(approved)==(graph,imported),'Stored graph changed during execution')
    return dict(passed=True,approved=approved,catalog_mutations=0,version_id=imported['version_id'],graph_sha256=imported['graph_sha256'],
        original_history_preserved=True,stored_execution=execution,validator_sha256=sha(__file__),checker_sha256=sha(ROOT/'scripts/check_constant_acceleration_revision.py'),
        source_validator_sha256=sha(ROOT/'scripts/validate_constant_acceleration_immutable_source.py'),scope_audit_sha256=imported['scope_audit_sha256'],
        limitations=['Corrected signed one-dimensional constant-acceleration proof with explicit quotient domains and independently proved zero-time/zero-acceleration extensions; no literal source-rule parity or path-length claim.',
            'Provisioned in-process stored execution; HTTP transport and clean installation are not covered.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True);parser.add_argument('--approved',action='store_true')
    args=parser.parse_args();report=asyncio.run(validate(args.source_directory,args.approved))
    name='constant_acceleration_revision_served_execution.json' if args.approved else 'constant_acceleration_revision_catalog_execution.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=args.approved,stored_states=29,maximum_ulp_error=0)))
