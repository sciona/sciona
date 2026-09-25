"""Version-bound Tier 3 review of both corrected integration_parts source identities."""
import json
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_integration_parts_identity_revisions import ROOT,sha,require
from scripts.audit_integration_parts_identity_scope import audit
from scripts.check_integration_parts_identity_revision import check
from scripts.validate_integration_parts_identity_database_gates import cases
from scripts.promote_integration_parts_execution import LIMITATIONS,REFERENCE

DESCRIPTION='Rewrite an indefinite integral by parts for scalar commutative C1 functions of a real parameter, retaining an independent integration constant and a symbolic residual integral. The caller establishes a common connected interval and excludes poles and branch discontinuities. Corrected parametrized realization; original incomplete differential AST remains provenance.'
PLAIN_DESCRIPTION='Transform an integral using integration by parts and retain the remaining integral symbolically. Supply two differentiable functions, a real variable and an independent constant. Establish a common valid interval before applying the result.'



def review(source):
    directory=ROOT/'docs/reviews';scope=audit(source)
    require(scope==json.loads((directory/'integration_parts_identity_execution_scope.json').read_text()),'Fresh integration_parts scope changed')
    reports={name:json.loads((directory/('integration_parts_identity_'+name+'.json')).read_text()) for name in
        ['revision_import','revision_repeat','revision_transaction','catalog_execution','database_gates']}
    imported=reports['revision_import'];repeat=reports['revision_repeat'];transaction=reports['revision_transaction'];execution=reports['catalog_execution'];gates=reports['database_gates']
    families={'integration_parts_legacy','integration_parts_projected'}
    for report in [scope,imported,repeat,execution,gates]:
        require(len(report['graphs'])==2 and {r['family'] for r in report['graphs']}==families,'Both integration_parts identities required')
    require(repeat['applied'] and repeat['rows_created']==0,'Repeat staging must make no changes')
    require(transaction['passed'] and transaction['rollback_verified'] and transaction['injected_failure_rollback_verified']
        and transaction['stager_sha256']==sha(ROOT/'scripts/stage_integration_parts_identity_revisions.py')
        and transaction['validator_sha256']==sha(ROOT/'scripts/validate_integration_parts_identity_transaction.py'),'Staging rollback changed')
    require(execution['passed'] and not execution['approved'] and execution['catalog_mutations']==0
        and execution['validator_sha256']==sha(ROOT/'scripts/validate_integration_parts_identity_execution.py')
        and execution['checker_sha256']==sha(ROOT/'scripts/check_integration_parts_identity_revision.py')
        and execution['oracle_sha256']==sha(ROOT/'scripts/validate_integration_parts_execution.py')
        and execution['source_validator_sha256']==sha(ROOT/'scripts/validate_integration_parts_immutable_source.py'),'Exact execution code required')
    require(gates['passed'] and gates['rollback_verified'] and gates['committed_catalog_mutations']==0
        and gates['validator_sha256']==sha(ROOT/'scripts/validate_integration_parts_identity_database_gates.py')
        and gates['checker_sha256']==execution['checker_sha256'],'Database fault qualification changed')
    reviewed=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        for item in imported['graphs']:
            family=item['family'];scoped=next(g for g in scope['graphs'] if g['family']==family)
            numerical=next(g for g in execution['graphs'] if g['family']==family)
            negative=next(g for g in gates['graphs'] if g['family']==family)
            for report in [numerical,negative]:require(report['version_id']==item['version_id'] and report['graph_sha256']==item['graph_sha256'],'Evidence version differs')
            require(negative['rejected_faults']==[c[0] for c in cases(item)],'Seventeen faults required per revision')
            result=numerical['execution']
            require(result['graph_digest']==item['graph_sha256'] and result['full_runner_cases']==6 and result['symbolic_derivative_checks']==6
                and result['residual_integrals_preserved'] and result['cache_ast_strings_preserved'] and result['source_proof']==scope['fresh_execution']['source_proof'],'Symbolic derivative/source coverage differs')
            for name,value in result['implementation_sha256'].items():require(sha(ROOT/name)==value,'Numerical dependency changed')
            state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(item['artifact_id'],)).fetchone()
            check(db,family,state==dict(status='approved',is_publishable=True))
            parent=scoped['qualification']
            refs=db.execute('SELECT ref_id,ref_key,title,url,source,verified,confidence,relevance_note FROM artifact_references WHERE artifact_id=%s AND ref_key=%s AND verified',(parent['artifact_id'],REFERENCE)).fetchall()
            bounds=db.execute("SELECT validity_statement,evidence_ref_key FROM artifact_validity_bounds WHERE version_id=%s AND review_status='automated_pass'",(parent['version_id'],)).fetchall()
            require(len(refs)==len(bounds)==1 and bounds[0]['evidence_ref_key']==REFERENCE,'Unique reviewed integration_parts regime/reference required')
            reviewed.append(dict(family=family,artifact_id=item['artifact_id'],version_id=item['version_id'],graph_sha256=item['graph_sha256'],
                approved_family_qualification=parent,original_history_sha256=item['original_history_sha256'],limitations=LIMITATIONS+scope['limitations'],
                reused_atoms=1,new_atoms=0,publication=dict(technical=DESCRIPTION,plain=PLAIN_DESCRIPTION,bound=bounds[0],reference=refs[0])))
    names=['execution_scope','revision_transaction','revision_import','revision_repeat','catalog_execution','database_gates']
    return dict(eligible_for_approval_transaction=True,approved=False,proposed_tier=3,review_source='automated',catalog_mutations=0,graphs=reviewed,
        evidence_sha256={'integration_parts_identity_'+name+'.json':sha(directory/('integration_parts_identity_'+name+'.json')) for name in names},
        reviewer_sha256=sha(__file__),parent_policy_sha256=sha(ROOT/'scripts/promote_integration_parts_execution.py'))


if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--source-directory',type=Path,required=True)
    report=review(p.parse_args().source_directory)
    (ROOT/'docs/reviews/integration_parts_identity_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,graphs=2,approved=False)))
