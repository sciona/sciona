"""Execute each stored legacy graph with unchanged qualified family oracles.

Dependency injection selects actual catalog graphs; no database state is mocked.
"""
import argparse
import asyncio
import importlib
import json
from pathlib import Path
from unittest.mock import patch
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.check_legacy_physics_revision import check
from scripts.stage_legacy_physics_revisions import ROOT,sha,require


def selected(family,approved=False):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        return check(db,family,approved)


async def validate(source,approved=False):
    items=json.loads((ROOT/'docs/reviews/legacy_physics_revision_import.json').read_text())['graphs'];reports=[]
    for item in items:
        family=item['family'];graph,record=selected(family,approved)
        if family in ['series','period_frequency']:
            oracle=importlib.import_module('scripts.validate_'+family+'_revision_execution')
            def actual_check(db,_approved=False):return check(db,family,approved)
            # The existing oracle's selection boundary reads the real legacy rows.
            # All independent reference calculations and invalid-input tests remain.
            with patch.object(oracle,'check',side_effect=actual_check):
                execution=await oracle.validate(source,approved=approved)
            require(execution['graph_sha256']==item['graph_sha256'],'Oracle executed another graph')
        else:
            oracle=importlib.import_module('scripts.validate_'+family+'_execution')
            with patch.object(oracle,'build_'+family+'_execution',return_value=graph):
                execution=await oracle.validate(ROOT) if family=='quadratic' else await oracle.validate(ROOT,source/'symbols.cypher',source/'infrules.cypher')
            require(execution['graph_digest']==item['graph_sha256'],'Oracle executed another graph')
        reuse=None
        if family=='quadratic':
            reuser=importlib.import_module('scripts.validate_quadratic_revision_reuse')
            with patch.object(reuser,'selected',side_effect=lambda state:selected('quadratic',state)):
                reuse=await reuser.validate(approved=approved)
            require(reuse['graph_sha256']==item['graph_sha256'],'Reuse executed another graph')
        require(selected(family,approved)==(graph,record),'Legacy graph changed during execution')
        reports.append(dict(family=family,version_id=item['version_id'],graph_sha256=item['graph_sha256'],execution=execution,reuse=reuse,
            oracle_sha256=sha(oracle.__file__),selection_boundary='Actual stored legacy graph and legacy integrity checker; family numerical/symbolic assertions unchanged. Nested historical checker/source labels describe the reused oracle; this wrapper binds actual legacy selection.'))
    return dict(passed=True,approved=approved,catalog_mutations=0,graphs=reports,
        validator_sha256=sha(__file__),checker_sha256=sha(ROOT/'scripts/check_legacy_physics_revision.py'),
        limits=['Provisioned in-process stored-graph execution; no HTTP/search-ranking, clean-install or redistribution claim.',
            'Family correction, interpretation and domain limits remain binding. Fixed Euler certificate does not claim general cross-domain computation.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True);parser.add_argument('--approved',action='store_true')
    args=parser.parse_args();report=asyncio.run(validate(args.source_directory,args.approved))
    name='legacy_physics_served_execution.json' if args.approved else 'legacy_physics_catalog_execution.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=args.approved,graphs=len(report['graphs']),catalog_mutations=0)))
