"""Read-only current catalog inventory for the fully executed Bengali recipe."""
import hashlib
import json
from pathlib import Path

from scripts.import_flavours_provider_drafts import ROOT, connect
from sciona.grapheme_execution import build_grapheme_graph

SOURCE_VERSION = '0c0997e0-c5f7-509c-9068-dcf00a167f56'


def audit():
    graph = build_grapheme_graph()
    records = []
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        original = db.execute('SELECT a.artifact_id::text,a.fqdn FROM artifacts a '
            'JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s',
            (SOURCE_VERSION,)).fetchone()
        if original is None:
            raise ValueError('Original intake identity missing')
        names = [original['fqdn'], *sorted({n.matched_primitive for n in graph.nodes})]
        for name in names:
            canonical = db.execute('''SELECT a.artifact_id::text,a.artifact_kind,a.status,
                a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier,
                artifact_is_publishable(a.artifact_id) AS native_eligible,
                EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
                FROM artifacts a LEFT JOIN artifact_versions v ON v.artifact_id=a.artifact_id AND v.is_latest
                WHERE a.fqdn=%s''', (name,)).fetchall()
            legacy = []
            if name != original['fqdn']:
                legacy = db.execute('''SELECT a.atom_id::text,a.status,
                    v.version_id::text,v.content_hash,v.trust_tier,
                    atom_is_publishable(a.atom_id) AS native_eligible
                    FROM atoms a LEFT JOIN atom_versions v ON v.atom_id=a.atom_id AND v.is_latest
                    WHERE a.fqdn=%s''', (name,)).fetchall()
            if len(canonical) > 1 or len(legacy) > 1:
                raise ValueError('Ambiguous latest catalog identity')
            records.append(dict(fqdn=name, canonical=canonical, legacy=legacy))
        evidence = db.execute('''SELECT passed,status,source_kind,runner_version,count(*) AS records
            FROM artifact_audit_evidence WHERE artifact_id=%s
            GROUP BY passed,status,source_kind,runner_version ORDER BY runner_version,status,passed''',
            (original['artifact_id'],)).fetchall()
    providers = records[1:]
    return dict(format='bengali-catalog-readiness.v1', read_only=True, catalog_mutations=0,
        original=original, source_version_id=SOURCE_VERSION, records=records,
        original_evidence_summary=evidence,
        provider_counts=dict(required=len(providers),
            missing_canonical=sum(not r['canonical'] for r in providers),
            missing_legacy=sum(not r['legacy'] for r in providers),
            canonical_eligible=sum(bool(r['canonical'] and r['canonical'][0]['native_eligible']) for r in providers),
            legacy_eligible=sum(bool(r['legacy'] and r['legacy'][0]['native_eligible']) for r in providers)),
        original_ready_for_promotion=False,
        next_requirements=[
            'Resolve source licensing, upstream attribution and candidate initialization permissions; do not infer permissions from execution success.',
            'Reconcile all fourteen provider identities with complete port contracts and implementation hashes using rollback-tested draft imports.',
            'Review original source stages and corrections, then stage a complete corrected version preserving original history.',
            'Execute the stored corrected graph and prove publication rollback, idempotence and production selection before Tier 3 approval.',
        ],
        limits=['Current catalog flags are inventory only, not runtime, source or transfer qualification.',
                'No approval or catalog mutation is performed by this audit.'],
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    report = audit()
    (ROOT/'docs/reviews/bengali_catalog_readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['provider_counts']))
