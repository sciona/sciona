"""Execute five stored legacy revisions with unchanged independent family oracles."""
import argparse
import asyncio
import contextlib
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.check_remaining_legacy_physics_revision import check
from scripts.stage_remaining_legacy_physics_revisions import ROOT,sha,require


def selected(family,approved=False):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        return check(db,family,approved)


async def validate(source,approved=False):
    items=json.loads((ROOT/'docs/reviews/remaining_legacy_physics_revision_import.json').read_text())['graphs']
    scopes=json.loads((ROOT/'docs/reviews/remaining_legacy_physics_execution_scope.json').read_text())['graphs']
    expected={'two_body':(5,'synthetic_numeric_cases',208,1),'momentum':(5,'synthetic_vectors',84,0),
        'projectile':(6,'synthetic_points',29,0),'parallel_resistance':(6,'synthetic_states',29,0),
        'constant_acceleration':(6,'synthetic_states',29,0)}
    reports=[]
    for item in items:
        family=item['family'];graph,record=selected(family,approved)
        oracle=importlib.import_module('scripts.validate_'+family+'_execution')
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(oracle,'build_'+family+'_execution',return_value=graph))
            source_validator=None
            if family!='two_body':
                source_validator=importlib.import_module('scripts.validate_'+family+'_immutable_source')
                stack.enter_context(patch.object(oracle,'validate_source',side_effect=source_validator.validate))
            args=[ROOT,source/'symbols.cypher',source/'infrules.cypher']
            if family not in ['two_body','momentum']:args.append(source/'expr_and_feed.cypher')
            execution=await oracle.validate(*args)
        count,key,points,tolerance=expected[family]
        require(execution['graph_digest']==item['graph_sha256'] and execution['full_runner_cases']==count
            and execution[key]==points and execution['maximum_ulp_error']<=tolerance,'Stored oracle coverage differs: '+family)
        require(selected(family,approved)==(graph,record),'Stored legacy graph changed')
        reports.append(dict(family=family,version_id=item['version_id'],graph_sha256=item['graph_sha256'],execution=execution,
            oracle_sha256=sha(oracle.__file__),source_validator_sha256=sha(source_validator.__file__) if source_validator else None,
            inherited_topology_limits=next(s for s in scopes if s['family']==family)['inherited_missing_expression_dependency_edges']))
    return dict(passed=True,approved=approved,catalog_mutations=0,native_threads=1,graphs=reports,
        validator_sha256=sha(__file__),checker_sha256=sha(ROOT/'scripts/check_remaining_legacy_physics_revision.py'),
        limitations=['Actual stored legacy graphs are supplied at the family graph-construction boundary; independent reference calculations and invalid-input checks are unchanged.',
                     'Corrected family scope and interpretation limits remain binding. Historical topology omissions are preserved, not certified.',
                     'Provisioned in-process execution only; HTTP, search ranking and package redistribution remain unqualified.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True);parser.add_argument('--approved',action='store_true')
    args=parser.parse_args();report=asyncio.run(validate(args.source_directory,args.approved))
    name='remaining_legacy_physics_served_execution.json' if args.approved else 'remaining_legacy_physics_catalog_execution.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=args.approved,graphs=len(report['graphs']),catalog_mutations=0)))
