"""Version-bound automated Tier 3 review of the corrected first-wave identity."""
import json
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_first_wave_identity_revisions import ROOT,sha,require
from scripts.audit_first_wave_revision_scope import audit
from scripts.check_first_wave_identity_revision import check
from scripts.validate_first_wave_identity_database_gates import cases

DESCRIPTION='Compute symbolic and sampled position, acceleration and net force for a supplied position expression in one inertial Cartesian component, using positive constant mass and explicit acceleration/position kinematics. Corrected realization; the historical derivative parse and missing premise are preserved as provenance.'
PLAIN_DESCRIPTION='Given how position changes over time and a positive constant mass, calculate acceleration and force, optionally at chosen times. Use consistent SI units in an inertial frame. This differentiates the supplied motion; it does not simulate motion from a force.'
VALIDITY='Positive finite constant mass; one inertial Cartesian component; position twice differentiable throughout the evaluation domain; coefficients and sample times in consistent SI units. Acceleration is the second time derivative of position and net force is mass times acceleration. Sampled expressions must evaluate to finite real values. Historical derivative parsing is corrected and the kinematic premise is explicitly supplied. No numerical time integration, global smoothness certification, or human expert certification.'


def review(source):
    directory=ROOT/'docs/reviews'
    scope=audit(source)
    require(scope==json.loads((directory/'first_wave_identity_execution_scope.json').read_text()),'Fresh source scope changed')
    imported=json.loads((directory/'first_wave_identity_revision_import.json').read_text())
    repeat=json.loads((directory/'first_wave_identity_revision_repeat.json').read_text())
    transaction=json.loads((directory/'first_wave_identity_revision_transaction.json').read_text())
    execution=json.loads((directory/'first_wave_identity_catalog_execution.json').read_text())
    gates=json.loads((directory/'first_wave_identity_database_gates.json').read_text())
    require(len(imported['graphs'])==len(scope['graphs'])==1 and repeat['rows_created']==0 and repeat['applied'],'One staged repeatable revision required')
    require(transaction['passed'] and transaction['rollback_verified'] and transaction['injected_failure_rollback_verified']
        and transaction['stager_sha256']==sha(ROOT/'scripts/stage_first_wave_identity_revisions.py')
        and transaction['validator_sha256']==sha(ROOT/'scripts/validate_first_wave_identity_transaction.py'),'Staging rollback evidence changed')
    item=imported['graphs'][0];scoped=scope['graphs'][0]
    require(execution['passed'] and not execution['approved'] and execution['catalog_mutations']==0 and execution['version_id']==item['version_id']
        and execution['graph_sha256']==item['graph_sha256'],'Exact stored execution required')
    require(execution['validator_sha256']==sha(ROOT/'scripts/validate_first_wave_identity_execution.py')
        and execution['checker_sha256']==sha(ROOT/'scripts/check_first_wave_identity_revision.py'),'Execution/checker code changed')
    require(execution['independent_runner_cases']==18 and execution['independent_numeric_components']==270 and execution['maximum_ulp_error']<=4
        and execution['invalid_cases_rejected']==['zero_mass','negative_mass','singular_position','unevaluable_position'],'Independent analytic coverage incomplete')
    inherited=execution['inherited_execution']
    require(inherited['checks']==dict(provider_contracts=1,serialized_graph_cases=3,runner_rejections=1)
        and inherited['serialized_graph_sha256']==item['graph_sha256'],'Inherited graph coverage differs')
    for name,digest in inherited['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Execution dependency changed')
    require(gates['passed'] and gates['rollback_verified'] and gates['committed_catalog_mutations']==0
        and gates['validator_sha256']==sha(ROOT/'scripts/validate_first_wave_identity_database_gates.py')
        and gates['checker_sha256']==execution['checker_sha256'],'Database fault qualification changed')
    require(gates['graphs']==[dict(family=item['family'],version_id=item['version_id'],graph_sha256=item['graph_sha256'],rejected_faults=[c[0] for c in cases(item)])],'Exact seventeen corruption cases required')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(item['artifact_id'],)).fetchone()
        check(db,item['family'],state==dict(status='approved',is_publishable=True))
        refs=db.execute("SELECT ref_id,ref_key,title,url,source,verified,confidence,relevance_note FROM artifact_references WHERE artifact_id=(SELECT artifact_id FROM artifact_versions WHERE version_id=%s) AND ref_key='first-wave-dynamics-openstax-v1' AND verified",(scoped['approved_execution_version_id'],)).fetchall()
        require(len(refs)==1,'Qualified parent reference required')
    limitations=scope['limitations']+scoped['exclusions']
    names=['execution_scope','revision_transaction','revision_import','revision_repeat','catalog_execution','database_gates']
    return dict(eligible_for_approval_transaction=True,approved=False,proposed_tier=3,review_source='automated',catalog_mutations=0,
        graphs=[dict(family=item['family'],artifact_id=item['artifact_id'],version_id=item['version_id'],graph_sha256=item['graph_sha256'],
            approved_family_qualification=scoped['qualification'],original_history_sha256=item['original_history_sha256'],limitations=limitations,
            reused_atoms=1,new_atoms=0,publication=dict(technical=DESCRIPTION,plain=PLAIN_DESCRIPTION,
                bound=dict(validity_statement=VALIDITY,evidence_ref_key=refs[0]['ref_key']),reference=refs[0]))],
        evidence_sha256={'first_wave_identity_'+name+'.json':sha(directory/('first_wave_identity_'+name+'.json')) for name in names},
        reviewer_sha256=sha(__file__))


if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--source-directory',type=Path,required=True)
    report=review(p.parse_args().source_directory)
    (ROOT/'docs/reviews/first_wave_identity_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,approved=False)))
