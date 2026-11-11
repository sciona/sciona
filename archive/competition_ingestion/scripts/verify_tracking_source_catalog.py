#!/usr/bin/env python3
"""Independent read-only verification of the approved tracking realization."""
import json
from pathlib import Path
import hashlib

import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values

from sciona.tracking_source_execution import SOURCE_VERSION, SOURCE_HASH, build_tracking_source_execution
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph


def verify(root):
    graph = build_tracking_source_execution()
    digest, _, _ = encode_execution_graph(graph)
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on') as db:
        targets = db.execute("SELECT a.artifact_id,a.artifact_kind,a.fqdn,a.status,a.is_publishable,v.version_id,v.content_hash,v.trust_tier,v.is_latest,e.details FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e ON e.version_id=v.version_id WHERE e.runner_version='tracking-source-community.v1' AND e.passed").fetchall()
        if len(targets) != 2 or {r['artifact_kind'] for r in targets} != {'atom', 'cdg'}:
            raise ValueError('Expected approved provider and graph')
        for row in targets:
            if row['status'] != 'approved' or not row['is_publishable'] or not row['is_latest'] or row['trust_tier'] != 3:
                raise ValueError('Target not approved/latest Tier 3')
            if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (row['artifact_id'],)).fetchone():
                raise ValueError('Canonical target not served')
            if row['details']['source_parity_claim'] is not False:
                raise ValueError('Unexpected source parity claim')
            for filename, sha in row['details']['evidence_sha256'].items():
                if hashlib.sha256((root/'docs/reviews'/filename).read_bytes()).hexdigest() != sha:
                    raise ValueError('Retained evidence changed')
            if row['artifact_kind'] == 'atom':
                if not db.execute('SELECT 1 FROM catalog_atoms_served WHERE atom_id=%s', (row['artifact_id'],)).fetchone():
                    raise ValueError('Legacy provider not served')
            else:
                if row['content_hash'] != digest:
                    raise ValueError('Graph hash mismatch')
                doc = db.execute('SELECT get_artifact_document(%s) AS d', (row['fqdn'],)).fetchone()['d']
                restored = _artifact_document_to_cdg(doc, version_id=str(row['version_id']), content_hash=digest, require_execution_envelope=True)
                if restored != graph:
                    raise ValueError('Catalog graph round trip differs')
            del row['details']
            del row['fqdn']
        original = db.execute('SELECT a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        if original != dict(status='draft', is_publishable=False, content_hash=SOURCE_HASH):
            raise ValueError('Original source state differs')
        count = db.execute("SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served WHERE artifact_kind='cdg'").fetchone()['n']
    return dict(read_only=True, targets=targets, original_source=original, served_cdgs=count,
                graph_round_trip=True, retained_evidence_hashes_verified=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    result = verify(root)
    (root/'docs/reviews/tracking_source_catalog_verification.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps(result, default=str))
