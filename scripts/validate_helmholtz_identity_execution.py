"""Execute each stored helmholtz revision with unchanged independent symbolic derivative and wave-residual equivalence checks."""
import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.check_helmholtz_identity_revision import check
from scripts.stage_helmholtz_identity_revisions import ROOT,sha,require
from scripts.validate_helmholtz_immutable_source import validate as validate_source
from scripts import validate_helmholtz_execution as oracle


def selected(family,approved=False):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        return check(db,family,approved)


async def validate(source,approved=False):
    items=json.loads((ROOT/'docs/reviews/helmholtz_identity_revision_import.json').read_text())['graphs']
    require(len(items)==2,'Both helmholtz identities required');reports=[]
    for item in items:
        graph,record=selected(item['family'],approved)
        with patch.object(oracle,'build_helmholtz_execution',return_value=graph),patch.object(oracle,'validate_source',side_effect=validate_source):
            execution=await oracle.validate(ROOT,source/'symbols.cypher',source/'infrules.cypher',source/'expr_and_feed.cypher')
        require(execution['graph_digest']==item['graph_sha256'] and execution['full_runner_cases']==6
            and execution['symbolic_derivative_checks']==6 and execution['formal_derivatives_preserved'] and execution['nonzero_residual_preserved'] and execution['cache_ast_strings_preserved'],'Stored helmholtz oracle differs')
        require(selected(item['family'],approved)==(graph,record),'Stored helmholtz changed during execution')
        reports.append(dict(family=item['family'],version_id=item['version_id'],graph_sha256=item['graph_sha256'],execution=execution))
    return dict(passed=True,approved=approved,catalog_mutations=0,native_threads=1,graphs=reports,
        validator_sha256=sha(__file__),checker_sha256=sha(ROOT/'scripts/check_helmholtz_identity_revision.py'),
        oracle_sha256=sha(ROOT/'scripts/validate_helmholtz_execution.py'),source_validator_sha256=sha(ROOT/'scripts/validate_helmholtz_immutable_source.py'),
        limitations=['Actual stored graphs replace only the construction boundary; independent symbolic derivative and wave-residual equivalence checks remain unchanged.',
            'Time-independent C2 Cartesian amplitude, real constant frequency including zero/negative values, positive constant speed. Retain nonzero residuals; no Maxwell divergence certification or boundary solver.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-directory',type=Path,required=True);p.add_argument('--approved',action='store_true');args=p.parse_args()
    report=asyncio.run(validate(args.source_directory,args.approved))
    name='helmholtz_identity_served_execution.json' if args.approved else 'helmholtz_identity_catalog_execution.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=args.approved,graphs=2,catalog_mutations=0)))
