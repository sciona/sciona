"""Read-only aggregate progress, distinguishing original intakes from derivatives."""
import hashlib
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

ROOT=Path(__file__).resolve().parents[1]


def audit():
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        physics=db.execute("SELECT status,count(*) AS artifacts FROM artifacts WHERE artifact_kind='cdg' "
                           "AND fqdn LIKE 'physics.pdg.%' GROUP BY status ORDER BY status").fetchall()
        served=db.execute("SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served "
                          "WHERE artifact_kind='cdg' AND fqdn LIKE 'physics.pdg.%'").fetchone()['n']
        competition=db.execute("""
            WITH intakes AS (
                SELECT DISTINCT artifact_id,version_id FROM artifact_audit_evidence
                WHERE runner_version='competition-intake.v1'
            )
            SELECT count(*) AS original_intake_versions,
                   count(DISTINCT i.artifact_id) AS original_intake_artifacts,
                   count(*) FILTER (WHERE a.status='approved' AND a.is_publishable
                       AND v.version_id<>i.version_id AND v.trust_tier=3
                       AND EXISTS (SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=i.artifact_id))
                       AS approved_served_corrected_originals_tier_3
            FROM intakes i JOIN artifacts a USING(artifact_id)
            LEFT JOIN artifact_versions v ON v.artifact_id=i.artifact_id AND v.is_latest
        """).fetchone()
    return dict(read_only=True,catalog_mutations=0,physics_artifact_status_counts=physics,
        physics_served_cdgs=served,competition=competition,overall_goal_complete=False,
        scope=['Physics counts include original and derived CDGs in the physics.pdg namespace.',
               'Competition counts refer to original intake identities; derived execution graphs are not counted as completed originals.',
               'These are catalog state totals, not substitutes for per-version qualification or a completion audit.'],
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    report=audit()
    (ROOT/'docs/reviews/cdg_promotion_post_mechanical_energy_totals.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
