#!/usr/bin/env python3
"""Apply exact-version automated Community approval to reviewed Riemannian providers."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid5
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.physics_ingest.pdg_evidence import _digest
from scripts.import_residual_execution_drafts import ensure_row
from scripts.import_riemannian_execution import validated_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--component', choices=['riemannian', 'relative_power', 'combined_feature', 'feng_knn', 'feng_xgb', 'feng_expanded'], default='riemannian')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report, _, _, _, _ = validated_graph(root, args.reference_dir, args.component)
    prefix = args.component.replace('_', '-')
    bundle = json.loads((root / 'docs/reviews' / (args.component + '_provider_community.json')).read_text())
    for name, digest in bundle['test_source_hashes'].items():
        if hashlib.sha256((root / 'tests' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('reviewed tests changed')
    reviews = {r['fqdn']: r for r in bundle['atoms']}
    required_providers, required_tests = {'riemannian': (8, 47), 'relative_power': (3, 15), 'combined_feature': (5, 27), 'feng_knn': (5, 39), 'feng_xgb': (1, 7), 'feng_expanded': (4, 29)}[args.component]
    if len(reviews) != required_providers or bundle['tests_passed'] != required_tests:
        raise ValueError('complete reviewed provider set required')
    created = 0
    with psycopg.connect(dotenv_values(root / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (prefix + '-provider-community.v1',))
        records = db.execute("SELECT e.*,a.fqdn,a.status AS artifact_status,v.content_hash,v.is_latest,v.trust_tier FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.runner_version=%s FOR UPDATE OF a,v FOR SHARE OF e", (prefix + '-provider-intake.v1',)).fetchall()
        if {r['fqdn'] for r in records} != set(reviews):
            raise ValueError('catalog provider set differs')
        for record in records:
            review = reviews[record['fqdn']]
            if review['publication_tier'] != 3 or review['review_source'] != 'automated' or review['review_status'] != 'approved' or review['review_required_actions'] or any(review[k] != 'pass' for k in ['structural_status', 'runtime_status', 'semantic_status', 'developer_semantics_status']):
                raise ValueError('review gates incomplete')
            if str(record['version_id']) != review['version_id'] or record['trust_tier'] != 3 or not record['is_latest'] or record['artifact_status'] not in {'draft', 'approved'}:
                raise ValueError('reviewed version/state differs')
            if not record['passed'] or _digest(record['details']) != review['intake_evidence_sha256'] or record['details']['source_sha256'] != review['source_sha256']:
                raise ValueError('intake evidence differs')
            legacy = db.execute('SELECT content_hash FROM atom_versions WHERE atom_id=%s AND version_id=%s AND is_latest FOR SHARE', (record['artifact_id'], record['version_id'])).fetchone()
            if not legacy or legacy['content_hash'] != record['content_hash']:
                raise ValueError('legacy provider version differs')
            executed = db.execute('SELECT * FROM artifact_audit_evidence WHERE evidence_id=%s AND passed FOR SHARE', (record['details']['execution_evidence_id'],)).fetchone()
            if not executed or _digest(executed['details']) != record['details']['execution_evidence_sha256'] or executed['details']['execution_report_sha256'] != _digest(report):
                raise ValueError('execution evidence differs')
            if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (record['version_id'],)).fetchone():
                raise ValueError('failed version audit requires resolution')
            for table in ['artifact_io_specs', 'atom_io_specs']:
                from psycopg import sql
                actual = db.execute(sql.SQL('SELECT direction,name,ordinal,type_desc,constraints,required,default_value_repr FROM {} WHERE version_id=%s').format(sql.Identifier(table)), (record['version_id'],)).fetchall()
                if sorted(actual, key=lambda r: (r['direction'], r['ordinal'])) != sorted(record['details']['interfaces'], key=lambda r: (r['direction'], r['ordinal'])):
                    raise ValueError('catalog interface differs from exact review')
            ref_id = 'barachant-seizure-00f937cc'
            url = 'https://github.com/alexandrebarachant/kaggle-seizure-prediction-challenge-2016/tree/00f937cc7710977dc812d9fc675864e2b8288658'
            title = 'Pinned competition source for Riemannian model branches'
            ensure_row(db, 'references_registry', {'ref_id': ref_id}, dict(ref_id=ref_id, ref_type='repository', title=title, url=url))
            for table, key in [('artifact_references', 'artifact_id'), ('atom_references', 'atom_id')]:
                ensure_row(db, table, {key: record['artifact_id'], 'ref_key': ref_id}, {key: record['artifact_id'], 'ref_id': ref_id,
                    'ref_key': ref_id, 'title': title, 'url': url, 'source': 'llm_extracted', 'verified': True, 'confidence': 'high',
                    'relevance_note': review['review_notes']})
            rollup = dict(overall_verdict='acceptable_with_limits', structural_status='pass', runtime_status='pass', semantic_status='pass',
                developer_semantics_status='pass', review_status='approved', review_semantic_verdict='pass', review_developer_semantics_verdict='pass',
                trust_readiness='ready', review_limitations=review['review_limitations'], review_required_actions=[], trust_blockers=[],
                acceptability_band='acceptable_with_limits', parity_coverage_level=('positive_path' if review['runtime_fqdn'].endswith(('.source_riemann_configuration', '.relative_power_configuration', '.combined_feature_configuration')) else 'positive_and_negative'), parity_test_status='pass')
            for table, key in [('artifact_audit_rollups', 'artifact_id'), ('atom_audit_rollups', 'atom_id')]:
                ensure_row(db, table, {key: record['artifact_id']}, {key: record['artifact_id'], **rollup})
            evidence_id = uuid5(record['version_id'], prefix + '-provider-community.v1')
            created += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence_id}, dict(evidence_id=evidence_id,
                artifact_id=record['artifact_id'], version_id=record['version_id'], audit_type='semantic_audit', passed=True,
                status='completed', source_kind='automated', runner_version=prefix + '-provider-community.v1',
                details=Jsonb({'publication_tier': 3, 'review': review, 'review_bundle_sha256': _digest(bundle),
                    'execution_evidence_id': str(executed['evidence_id']), 'content_hash': record['content_hash']})))
            db.execute("UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s", (record['artifact_id'],))
            db.execute("UPDATE atoms SET status='approved',is_publishable=true WHERE atom_id=%s", (record['artifact_id'],))
        # Check dependency closure after all reviewed versions are approved in this transaction.
        for record in records:
            for dep in db.execute("SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s AND dependency_role='logic_atom' AND NOT optional", (record['version_id'],)):
                if not db.execute("SELECT 1 FROM catalog_atoms_served a JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.content_hash=%s AND v.is_latest", (dep['dependency_artifact_fqdn'], dep['dependency_content_hash'])).fetchone():
                    raise ValueError('invoked dependency version not served')
            if not db.execute('SELECT 1 FROM catalog_atoms_served WHERE atom_id=%s', (record['artifact_id'],)).fetchone():
                raise ValueError('reviewed provider not served')
        if not args.apply:
            db.rollback()
    print(json.dumps({'applied': args.apply, 'new_approvals': created, 'served_providers_verified': len(records), 'trust_tier': 3}, indent=2))


if __name__ == '__main__':
    main()
