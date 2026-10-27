"""Atomically publish the corrected Otto identity and its providers at Tier 3.

Default execution rolls back; --apply requires publication transaction gates.
"""
import argparse
import json
from uuid import UUID, NAMESPACE_URL, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from sciona.cdg_projection import build_published_cdg_projection
from sciona.stacking_complete_graph import build_complete_graph
from scripts.check_stacking_complete_catalog import ROOT, ORIGINAL_FQDN, qualification, check_candidate, require, sha
from scripts.import_residual_execution_drafts import ensure_row
from scripts.review_stacking_complete_publication import review
from scripts.stage_stacking_complete import RUNNER as STAGING_RUNNER

RUNNER='otto-decomposed-original-community.v1'
DESCRIPTION=('Train and evaluate a decomposed nonnegative-tabular nine-class stack with explicit population embedding, '
    '33 first-level entries, seven supplemental blocks, four-fold meta selection, full 250/600/250 model bags and '
    'probability blending. Composes 27 separately callable providers in 62 nodes. Application feature extraction, '
    'class meanings and observation availability are caller responsibilities. Automated Tier 3 synthetic qualification; '
    'fixed transductive population and provisioned uncached in-process runtime only.')
PLAIN_DESCRIPTION=('Build a nine-category prediction ensemble from caller-supplied numerical features and fold assignments. '
    'The graph exposes the feature, model-selection and blending steps for composition. This corrected version preserves '
    'the original competition intake as history. Synthetic checks establish computation, not predictive effectiveness '
    'in another application or historical winning performance.')
OPERATIONS={
    'source_entry_policy':'Emit the 33 source entry indices as explicit policy constants for model dispatch.',
    'prepare_first':'Validate and detach aligned first-level features, labels, five folds, identities, embedding and controls into an in-process context.',
    'fit_first_entry':'Fit one explicitly selected first-level entry and produce aligned out-of-fold and query feature blocks.',
    'fit_neighbor_bank':'Fit the source neighbor bank once for a representative control set and retain its aligned class probabilities.',
    'select_neighbor_entry':'Select one source neighbor variant from a bank owned by the same validated context and matching controls.',
    'collect_first':'Check and assemble all 33 first-level blocks without dropping their population alignment.',
    'prepare_supplemental':'Validate aligned numerical features, labels, five folds, embedding and distance/cluster controls for supplemental producers.',
    'raw_class_distances':'Compute the three shared source raw class-distance feature blocks with held-fold label exclusion.',
    'weighted_class_distances':'Compute fold-local TF-IDF class-distance features using explicit weighting and distance controls.',
    'embedded_class_distances':'Compute source class-distance features from the fixed population embedding with held-fold label exclusion.',
    'fit_reference_clusters':'Fit clustering state using the reference training population only.',
    'cluster_assignments':'Apply context-owned reference clustering state to training and query populations.',
    'nonzero_counts':'Count nonzero numerical features for each aligned training and query row.',
    'raw_neural_features':'Expose the source raw feature representation with its declared population alignment for neural meta inputs.',
    'collect_supplemental':'Validate and collect seven supplemental blocks and the raw neural view with consistent row identities.',
    'prepare_meta':'Validate aligned tree and neural feature layouts, labels, four-fold assignments and candidate controls for meta selection.',
    'select_xgboost':'Select XGBoost controls using the source four-fold evaluation and complete 250-model bags per fit.',
    'select_neural':'Select neural controls using the source four-fold evaluation and complete 600-model bags per fit.',
    'select_extratrees':'Select AdaBoost-ExtraTrees controls using the source four-fold evaluation and complete 250-model bags per fit.',
    'refit_meta':'Refit a selected meta family on the complete training population, enforcing selection/context ownership and full bag size.',
    'mean_probability_bag':'Average a finite normalized probability bag across model runs; arbitrary supported run counts and two or more classes, independent of class meaning.',
    'blend_meta':'Apply the explicit source geometric/arithmetic blending policy to three nine-class family means and normalize the complete scores.',
    'meta_result':'Package the blend and selection summaries while checking the three selected families.',
    'unpack_payload':'Validate and detach a complete source-policy payload into explicit feature, identity, fold and control ports.',
    'validate_neighbor_sharing':'Reject compiled neighbor-bank sharing when any entry controls differ from its designated representative.',
    'fit_population_embedding':'Fit a fixed transductive embedding over the declared training-plus-query population without labels; no out-of-sample transform.',
    'assemble_layout':'Assemble aligned first-level and supplemental blocks into the source tree and neural meta feature layouts.',
}


def targets(scope):
    result=[(scope['artifact_id'],scope['corrected_version_id'],None,DESCRIPTION,PLAIN_DESCRIPTION)]
    for fqdn,p in sorted(scope['providers'].items()):
        name=p['runtime_fqdn'].rsplit('.',1)[1]
        require(name in OPERATIONS,'Missing reviewed provider description')
        common=(' General probability operation; callers define category meaning.' if name=='mean_probability_bag' else
                ' Source-policy adapter for the nine-class stack; callers supply application features and identities.')
        text=OPERATIONS[name]+common+' Automated Tier 3 provisioned in-process synthetic qualification; no static shape-proof claim.'
        result.append((str(uuid5(NAMESPACE_URL,'sciona-provider-draft:'+fqdn)),p['version_id'],fqdn,text,OPERATIONS[name]+common))
    return result


def rollup(qualified):
    return dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',
        semantic_status='pass',developer_semantics_status='pass',review_status='approved',review_semantic_verdict='pass',
        review_developer_semantics_verdict='pass',trust_readiness='ready',review_limitations=qualified['limitations'],
        review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits',
        parity_coverage_level='positive_and_negative',parity_test_status='pass')


def audit_details(qualified,scope,fqdn,technical,plain):
    return dict(qualification=qualified,publication_tier=3,publisher_sha256=sha(__file__),
                original_version_id=scope['original_version_id'],original_content_hash=scope['original_content_hash'],
                provider=scope['providers'][fqdn] if fqdn else None,description=technical,plain_description=plain)


def topology(scope):
    return build_published_cdg_projection(artifact=dict(artifact_id=scope['artifact_id'],fqdn=ORIGINAL_FQDN),
        version=dict(version_id=scope['corrected_version_id'],content_hash=scope['graph_sha256']),
        cdg=build_complete_graph()).topo_hash


def check_publication(db,scope,specs,qualified):
    check_candidate(db,scope,specs,approved=True)
    for owner,version,fqdn,technical,plain in targets(scope):
        for kind,text in [('technical',technical),('dejargonized',plain)]:
            rows=db.execute('SELECT description_id::text,content,generated_by,reviewed FROM artifact_descriptions '
                            'WHERE artifact_id=%s AND kind=%s AND language=%s',(owner,kind,'en')).fetchall()
            require(rows==[dict(description_id=str(uuid5(UUID(version),RUNNER+':'+kind)),content=text,
                                generated_by=RUNNER,reviewed=False)],'Publication descriptions differ')
        for table,key in [('artifacts','artifact_id')]+([('atoms','atom_id')] if fqdn else []):
            require(db.execute('SELECT description FROM '+table+' WHERE '+key+'=%s',(owner,)).fetchone()
                    ==dict(description=technical),'Published selection description differs')
        expected=rollup(qualified)
        for table,key in [('artifact_audit_rollups','artifact_id')]+([('atom_audit_rollups','atom_id')] if fqdn else []):
            rows=db.execute('SELECT '+','.join(expected)+' FROM '+table+' WHERE '+key+'=%s',(owner,)).fetchall()
            require(rows==[expected],'Publication rollup differs')
        rows=db.execute('SELECT evidence_id::text,artifact_id::text,audit_type,passed,status,source_kind,details '
                        'FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s',(version,RUNNER)).fetchall()
        require(rows==[dict(evidence_id=str(uuid5(UUID(version),RUNNER)),artifact_id=owner,audit_type='semantic_audit',
            passed=True,status='completed',source_kind='automated',details=audit_details(qualified,scope,fqdn,technical,plain))],
            'Publication evidence differs')
        if fqdn:
            require(db.execute('SELECT count(*) AS n FROM catalog_atoms_served WHERE atom_id=%s',(owner,)).fetchone()['n']==1,
                    'Legacy provider selection differs')
    require(db.execute('SELECT verified_leaf_coverage,leaf_count,top_level_input_arity,top_level_output_arity,topo_hash '
                       'FROM artifacts WHERE artifact_id=%s',(scope['artifact_id'],)).fetchone()==dict(
                           verified_leaf_coverage=1.,leaf_count=62,top_level_input_arity=1,top_level_output_arity=1,
                           topo_hash=topology(scope)), 'Published graph metadata differs')


def promote(apply=False,fail_at=None):
    require(not apply or fail_at is None,'Failure injection requires rollback mode')
    directory=ROOT/'docs/reviews'; qualified=review(); scope,specs=qualification()
    require(qualified==json.loads((directory/'competition_otto_decomposed_publication_review.json').read_text()),
            'Retained publication review changed')
    if apply:
        gates=json.loads((directory/'competition_otto_decomposed_publication_transaction.json').read_text())
        require(gates['passed'] and gates['rollback_verified'] and gates['metadata_failure_rolled_back']
                and gates['activation_failure_rolled_back'] and gates['publisher_sha256']==sha(__file__)
                and gates['validator_sha256']==sha(ROOT/'scripts/validate_stacking_complete_publication_transaction.py')
                and gates['review_sha256']==sha(directory/'competition_otto_decomposed_publication_review.json'),
                'Current publication transaction qualification required')
    created=updated=0
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(STAGING_RUNNER,))
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s FOR UPDATE',
                         (scope['artifact_id'],)).fetchone()
        already=state==dict(status='approved',is_publishable=True)
        check_candidate(db,scope,specs,approved=already)
        if already:
            check_publication(db,scope,specs,qualified)

        def ensure(table,key,row):
            nonlocal created
            created+=int(ensure_row(db,table,key,row))

        for owner,version,fqdn,technical,plain in targets(scope):
            for kind,text in [('technical',technical),('dejargonized',plain)]:
                ensure('artifact_descriptions',dict(artifact_id=UUID(owner),kind=kind,language='en'),dict(
                    description_id=uuid5(UUID(version),RUNNER+':'+kind),artifact_id=UUID(owner),kind=kind,language='en',
                    content=text,generated_by=RUNNER,reviewed=False))
            for table,key in [('artifact_audit_rollups','artifact_id')]+([('atom_audit_rollups','atom_id')] if fqdn else []):
                ensure(table,{key:UUID(owner)},dict(**{key:UUID(owner)},**rollup(qualified)))
            evidence=uuid5(UUID(version),RUNNER)
            ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,
                artifact_id=UUID(owner),version_id=UUID(version),audit_type='semantic_audit',passed=True,
                status='completed',source_kind='automated',runner_version=RUNNER,
                details=Jsonb(audit_details(qualified,scope,fqdn,technical,plain))))
        if fail_at=='metadata':
            raise RuntimeError('injected publication metadata failure')
        if not already:
            for owner,version,fqdn,technical,plain in targets(scope):
                for table,key in [('artifacts','artifact_id')]+([('atoms','atom_id')] if fqdn else []):
                    updated+=db.execute('UPDATE '+table+" SET status='approved',is_publishable=true,description=%s WHERE "+key+'=%s',
                                        (technical,owner)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',
                                (scope['artifact_id'],)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',
                                (scope['corrected_version_id'],)).rowcount
            updated+=db.execute('UPDATE artifacts SET verified_leaf_coverage=1,leaf_count=62,top_level_input_arity=1,'
                                'top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s',
                                (topology(scope),scope['artifact_id'])).rowcount
        check_publication(db,scope,specs,qualified)
        if fail_at=='activation':
            raise RuntimeError('injected publication activation failure')
        require(review()==qualified,'Publication qualification changed during transaction')
        if not apply:
            db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,tier=3,
                artifact_id=scope['artifact_id'],version_id=scope['corrected_version_id'],graph_sha256=scope['graph_sha256'],
                served_providers_in_transaction=27,served_cdgs_in_transaction=1,original_history_preserved=True,
                publisher_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    result=promote(args.apply)
    suffix=('repeat' if result['already_approved'] else 'apply') if args.apply else 'rollback'
    (ROOT/'docs/reviews'/('competition_otto_decomposed_publication_'+suffix+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
