"""Read-only inventory distinguishing approval, serving and native eligibility."""
import json
from collections import Counter
from pathlib import Path
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.promote_arc_execution import sha


def inspect(db, identity):
    row=db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
        v.version_id::text,v.content_hash,v.trust_tier,
        artifact_is_publishable(a.artifact_id) AS native_eligible,
        EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served,
        EXISTS(SELECT 1 FROM artifact_descriptions d WHERE d.artifact_id=a.artifact_id AND d.kind='dejargonized' AND d.language='en') AS plain_description,
        EXISTS(SELECT 1 FROM artifact_references r WHERE r.artifact_id=a.artifact_id) AS reference,
        EXISTS(SELECT 1 FROM artifact_audit_rollups r WHERE r.artifact_id=a.artifact_id) AS audit_rollup,
        EXISTS(SELECT 1 FROM artifact_io_specs p WHERE p.artifact_id=a.artifact_id) OR a.top_level_input_arity>0 OR a.top_level_output_arity>0 AS boundary,
        EXISTS(SELECT 1 FROM artifact_cdg_nodes n JOIN artifact_versions nv USING(version_id) WHERE nv.artifact_id=a.artifact_id) AS nodes,
        EXISTS(SELECT 1 FROM artifact_cdg_bindings b JOIN artifact_versions bv USING(version_id) WHERE bv.artifact_id=a.artifact_id) OR a.verified_leaf_coverage>0 AS bindings_or_coverage
        FROM artifacts a LEFT JOIN artifact_versions v ON v.artifact_id=a.artifact_id AND v.is_latest
        WHERE a.artifact_id=%s''',(identity,)).fetchall()
    if len(row)!=1:
        raise ValueError('Unique latest catalog identity required')
    row=row[0]
    row['missing_native_requirements']=[k for k in ('plain_description','reference','audit_rollup','boundary','nodes','bindings_or_coverage') if row[k] is not True]
    for k in ('plain_description','reference','audit_rollup','boundary','nodes','bindings_or_coverage'):
        del row[k]
    row['failed_selected_audits']=db.execute('SELECT count(*) AS n FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed',(row['version_id'],)).fetchone()['n']
    return row


def audit():
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        originals=db.execute("""SELECT DISTINCT a.artifact_id,a.fqdn,e.version_id::text AS intake_version
            FROM artifacts a JOIN artifact_audit_evidence e USING(artifact_id)
            WHERE e.runner_version='competition-intake.v1' ORDER BY a.fqdn""").fetchall()
        competition=[]
        for original in originals:
            record=inspect(db,original['artifact_id'])
            record['intake_version']=original['intake_version']
            record['corrected_version']=record['version_id']!=original['intake_version']
            record['qualified_catalog_state']=(record['corrected_version'] and record['status']=='approved'
                and record['is_publishable'] and record['served'] and record['native_eligible']
                and record['trust_tier'] in (2,3) and record['failed_selected_audits']==0)
            if not record['qualified_catalog_state']:
                derivative=db.execute('SELECT artifact_id FROM artifacts WHERE fqdn=%s',(record['fqdn']+'.execution',)).fetchone()
                record['retained_derivative']=inspect(db,derivative['artifact_id']) if derivative else None
            competition.append(record)
        physics=[inspect(db,r['artifact_id']) for r in db.execute("SELECT artifact_id FROM artifacts WHERE artifact_kind='cdg' AND fqdn LIKE 'physics.pdg.%' ORDER BY fqdn").fetchall()]
    def summary(records):
        return dict(total=len(records),approved=sum(r['status']=='approved' for r in records),
            served=sum(r['served'] for r in records),native_eligible=sum(r['native_eligible'] for r in records),
            missing_native_requirements=dict(Counter(k for r in records for k in r['missing_native_requirements'])),
            with_failed_selected_audits=sum(r['failed_selected_audits']>0 for r in records))
    return dict(format='remaining-cdg-readiness.v1',read_only=True,catalog_mutations=0,
        competition_summary=dict(summary(competition),qualified_catalog_state=sum(r['qualified_catalog_state'] for r in competition)),
        physics_summary=summary(physics),competition=competition,physics=physics,
        auditor_sha256=sha(Path(__file__)),overall_goal_complete=False,
        limits=['Approval flags, served views and native predicates are separate checks.',
                'Native eligibility is necessary catalog metadata coverage, not proof of semantic correctness or full runtime qualification.',
                'Physics namespace includes original and derived graphs; no original-only completion count is inferred.',
                'Retained competition derivatives are discovered by the established .execution naming convention; other names are not ruled out.',
                'Catalog identities and software versions only; no runtime dataset contents or identifying metadata.'])


if __name__=='__main__':
    report=audit()
    (ROOT/'docs/reviews/remaining_cdg_readiness_post_arc.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('competition_summary','physics_summary','overall_goal_complete')}))
