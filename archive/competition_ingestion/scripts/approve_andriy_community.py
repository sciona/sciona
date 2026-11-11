#!/usr/bin/env python3
"""Approve reviewed exact Andriy providers and graph atomically at Tier 3."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import dotenv_values
from sciona.andriy_execution import build_andriy_execution_graph
from sciona.cdg_projection import build_published_cdg_projection
from sciona.physics_ingest.pdg_evidence import _digest
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.audit_andriy_promotion_evidence import audit
from scripts.inventory_andriy_provider_closure import inventory
from scripts.import_residual_execution_drafts import ensure_row
from scripts.import_riemannian_execution import SOURCE_ID, SOURCE_VERSION, SOURCE_HASH


LIMITATIONS = [
    'Automated Tier 3 review only; no Tier 1 certification or Tier 2 usage claim.',
    'Three Andriy branches and their provider closure only; full eleven-model ensemble remains outside scope.',
    'Caller supplies three ordered populations, source-safe P/P1/I membership and CSP candidate enumeration; no membership inference.',
    'Documented resampling/Welch/AR and current numerical/R API adaptations; no historical-engine equivalence or predictive-quality claim.',
    'Source selectors require at least 300/200 used features. Small samples can fail; finite training and nondegenerate CSP are required.',
    'Full raw synthetic execution plus separate source references and contract tests; random-label raw scores do not establish discrimination.',
    'Prediction-batch normalization, branch-specific validity handling and rank semantics follow the declared contracts.',
]


def approve(root, reference_dir, library, apply=False):
    prerequisite = audit(root, reference_dir, library)
    if not prerequisite['component_execution_prerequisites_complete']:
        raise ValueError('Successful full raw graph evidence required')
    closure = inventory(root)
    reviews = {record['runtime_fqdn']: record for record in closure['providers']}
    tests = json.loads((root/'docs/reviews/andriy_provider_test_review.json').read_text())
    if tests['test_results'] != dict(tests=202, failures=0, errors=0, skipped=0):
        raise ValueError('Complete reviewed test suite required')
    for name, sha in tests['test_source_sha256'].items():
        if hashlib.sha256((root/'tests'/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Reviewed tests changed')
    if tests['provider_source_sha256'] != {name: r['provider_sha256'] for name, r in reviews.items()}:
        raise ValueError('Tested provider closure changed')
    raw = json.loads((root/'docs/reviews/andriy_raw_execution_parity.json').read_text())
    shapes = {'andriy_xgb_scores': [6], 'andriy_xgb_segment_model_probabilities': [6, 5],
        'andriy_glm_scores': [6], 'andriy_glm_segment_model_probabilities': [6, 3],
        'andriy_svm_scores': [6], 'andriy_svm_unmasked_window_probabilities': [114]}
    if raw['output_shapes'] != shapes or set(raw['score_spreads']) != {'xgb', 'svm', 'glm'} or any(v <= 0 for v in raw['score_spreads'].values()):
        raise ValueError('Full raw output gates differ')
    files = ['andriy_raw_execution_parity.json', 'andriy_provider_test_review.json',
        'andriy_raw_small_sample_failure.json', 'andriy_semantic_review.md']
    evidence_hashes = {name: hashlib.sha256((root/'docs/reviews'/name).read_bytes()).hexdigest() for name in files}
    graph = build_andriy_execution_graph()
    digest, _, _ = encode_execution_graph(graph)
    graph_id = uuid5(SOURCE_ID, 'andriy-execution.v1')
    graph_version = uuid5(graph_id, digest)
    graph_fqdn = 'cdg.competition.solution.kaggle.barachant_seizure_1st.andriy_execution'
    created = 0
    rollup = dict(overall_verdict='acceptable_with_limits', structural_status='pass', runtime_status='pass',
        semantic_status='pass', developer_semantics_status='pass', review_status='approved',
        review_semantic_verdict='pass', review_developer_semantics_verdict='pass', trust_readiness='ready',
        review_limitations=LIMITATIONS, review_required_actions=[], trust_blockers=[],
        acceptability_band='acceptable_with_limits', parity_coverage_level='positive_and_negative', parity_test_status='pass')
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('andriy-community.v1'))")
        rows = db.execute("SELECT e.*,a.fqdn,a.status AS artifact_status,v.content_hash,v.is_latest,v.trust_tier FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.runner_version='andriy-provider-intake.v1' FOR UPDATE OF a,v FOR SHARE OF e").fetchall()
        if len(rows) != 25 or {r['details']['runtime_fqdn'] for r in rows} != set(reviews):
            raise ValueError('Complete exact provider intake required')
        reference = db.execute("SELECT ref_id,title,url FROM references_registry WHERE ref_id='barachant-seizure-00f937cc' FOR SHARE").fetchone()
        if not reference:
            raise ValueError('Pinned public source reference missing')
        for row in rows:
            review = reviews[row['details']['runtime_fqdn']]
            if (not row['passed'] or row['artifact_status'] not in {'draft', 'approved'} or not row['is_latest']
                    or row['trust_tier'] != 3 or str(row['version_id']) != review['version_id']
                    or row['content_hash'] != review['content_hash'] or row['fqdn'] != review['catalog_fqdn']
                    or row['details']['source_sha256'] != review['provider_sha256']):
                raise ValueError('Reviewed provider version/state differs')
            for name, sha in row['details']['reference_report_sha256'].items():
                if hashlib.sha256((root/'docs/reviews'/name).read_bytes()).hexdigest() != sha:
                    raise ValueError('Provider intake references changed')
            legacy = db.execute('SELECT content_hash,trust_tier FROM atom_versions WHERE atom_id=%s AND version_id=%s AND is_latest FOR SHARE', (row['artifact_id'], row['version_id'])).fetchone()
            if not legacy or legacy['content_hash'] != row['content_hash'] or legacy['trust_tier'] != 3:
                raise ValueError('Legacy provider version differs')
            for table in ['artifact_io_specs', 'atom_io_specs']:
                from psycopg import sql
                actual = db.execute(sql.SQL('SELECT direction,name,ordinal,type_desc,constraints,required,default_value_repr FROM {} WHERE version_id=%s ORDER BY direction,ordinal').format(sql.Identifier(table)), (row['version_id'],)).fetchall()
                if actual != sorted(review['interfaces'], key=lambda p: (p['direction'], p['ordinal'])):
                    raise ValueError('Provider interfaces changed')
            expected = {(reviews[d]['catalog_fqdn'], reviews[d]['content_hash']) for d in review['dependencies']}
            deps = db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE', (row['version_id'],)).fetchall()
            if len(deps) != len(expected) or any(d['dependency_role'] != 'logic_atom' or d['optional'] for d in deps) or {(d['dependency_artifact_fqdn'], d['dependency_content_hash']) for d in deps} != expected:
                raise ValueError('Provider dependency closure changed')
            if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (row['version_id'],)).fetchone():
                raise ValueError('Unresolved failed provider audit')
            for table, key in [('artifact_references', 'artifact_id'), ('atom_references', 'atom_id')]:
                ensure_row(db, table, {key: row['artifact_id'], 'ref_key': reference['ref_id']}, {
                    key: row['artifact_id'], **reference, 'ref_key': reference['ref_id'], 'source': 'llm_extracted',
                    'verified': True, 'confidence': 'high', 'relevance_note': 'Pinned source Andriy implementation with documented current-runtime adaptations and automated Tier 3 evidence.'})
            for table, key in [('artifact_audit_rollups', 'artifact_id'), ('atom_audit_rollups', 'atom_id')]:
                ensure_row(db, table, {key: row['artifact_id']}, {key: row['artifact_id'], **rollup})
            evidence_id = uuid5(row['version_id'], 'andriy-provider-community.v1')
            created += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence_id}, dict(
                evidence_id=evidence_id, artifact_id=row['artifact_id'], version_id=row['version_id'],
                audit_type='semantic_audit', passed=True, status='completed', source_kind='automated',
                runner_version='andriy-provider-community.v1', details=Jsonb(dict(publication_tier=3,
                    review_source='automated', review=review, intake_evidence_sha256=_digest(row['details']),
                    evidence_sha256=evidence_hashes, limitations=LIMITATIONS))))
            db.execute("UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s", (row['artifact_id'],))
            db.execute("UPDATE atoms SET status='approved',is_publishable=true WHERE atom_id=%s", (row['artifact_id'],))
        for row in rows:
            if not db.execute('SELECT 1 FROM catalog_atoms_served WHERE atom_id=%s', (row['artifact_id'],)).fetchone():
                raise ValueError('Approved provider not served')
        intake = db.execute("SELECT e.*,a.status AS artifact_status,v.content_hash,v.trust_tier FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.version_id=%s AND e.runner_version='andriy-execution-draft-intake.v1' FOR UPDATE OF a,v FOR SHARE OF e", (graph_version,)).fetchone()
        if not intake or intake['artifact_id'] != graph_id or not intake['passed'] or intake['content_hash'] != digest or intake['trust_tier'] != 3 or intake['artifact_status'] not in {'draft', 'approved'}:
            raise ValueError('Exact graph intake missing')
        for name, sha in intake['details']['reference_report_sha256'].items():
            if hashlib.sha256((root/'docs/reviews'/name).read_bytes()).hexdigest() != sha:
                raise ValueError('Graph intake references changed')
        doc = db.execute('SELECT get_artifact_document(%s) AS d', (graph_fqdn,)).fetchone()['d']
        if _artifact_document_to_cdg(doc, version_id=str(graph_version), content_hash=digest, require_execution_envelope=True) != graph:
            raise ValueError('Catalog graph changed')
        source = db.execute("SELECT a.fqdn,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) JOIN artifact_audit_evidence e USING(version_id) WHERE v.version_id=%s AND e.runner_version='competition-intake.v1' AND e.passed FOR SHARE OF a,v,e", (SOURCE_VERSION,)).fetchone()
        deps = db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE', (graph_version,)).fetchall()
        if not source or source['content_hash'] != SOURCE_HASH or len(deps) != 1 or deps[0]['dependency_role'] != 'cdg' or deps[0]['dependency_artifact_fqdn'] != source['fqdn'] or deps[0]['dependency_content_hash'] != SOURCE_HASH or deps[0]['optional']:
            raise ValueError('Conceptual source provenance changed')
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s FOR SHARE', (graph_version,)).fetchall()
        if len(bindings) != len(graph.nodes):
            raise ValueError('Incomplete graph bindings')
        for node in graph.nodes:
            matches = [b for b in bindings if b['node_id'] == node.node_id]
            review = reviews[node.matched_primitive]
            if len(matches) != 1:
                raise ValueError('Ambiguous binding')
            b = matches[0]
            if b['bound_artifact_fqdn'] != review['catalog_fqdn'] or b['bound_version_content_hash'] != review['content_hash'] or b['status'] != 'active' or b['evidence_summary']['runtime_fqdn'] != node.matched_primitive or b['evidence_summary']['output_aliases_by_ordinal'] != [p.name for p in node.outputs]:
                raise ValueError('Exact binding differs')
        expected_io = []
        for direction, ports in [('input', graph.nodes[0].inputs),
                                 ('output', [p for node in graph.nodes[1:] for p in node.outputs])]:
            for ordinal, port in enumerate(ports):
                expected_io.append(dict(direction=direction, name=port.name, ordinal=ordinal,
                    type_desc=port.type_desc, constraints=port.constraints, required=port.required,
                    default_value_repr=port.default_value_repr))
        actual_io = db.execute('SELECT direction,name,ordinal,type_desc,constraints,required,default_value_repr FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal', (graph_version,)).fetchall()
        if actual_io != sorted(expected_io, key=lambda p: (p['direction'], p['ordinal'])):
            raise ValueError('Graph boundary differs')
        if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (graph_version,)).fetchone():
            raise ValueError('Unresolved failed graph audit')
        ensure_row(db, 'artifact_audit_rollups', {'artifact_id': graph_id}, dict(artifact_id=graph_id, **rollup))
        ensure_row(db, 'artifact_references', {'artifact_id': graph_id, 'ref_key': reference['ref_id']}, dict(
            artifact_id=graph_id, **reference, ref_key=reference['ref_id'], source='llm_extracted', verified=True,
            confidence='high', relevance_note='Three Andriy branches validated from full synthetic raw clips; complete eleven-model ensemble remains outside scope.'))
        evidence_id = uuid5(graph_version, 'andriy-execution-community.v1')
        created += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence_id}, dict(
            evidence_id=evidence_id, artifact_id=graph_id, version_id=graph_version, audit_type='semantic_audit',
            passed=True, status='completed', source_kind='automated', runner_version='andriy-execution-community.v1',
            details=Jsonb(dict(publication_tier=3, review_source='automated', content_hash=digest,
                intake_evidence_sha256=_digest(intake['details']), evidence_sha256=evidence_hashes,
                provider_versions={r['runtime_fqdn']: r['version_id'] for r in reviews.values()}, limitations=LIMITATIONS))))
        topo = build_published_cdg_projection(artifact={'artifact_id': str(graph_id), 'fqdn': graph_fqdn},
            version={'version_id': str(graph_version), 'content_hash': digest}, cdg=graph).topo_hash
        db.execute("UPDATE artifacts SET status='approved',is_publishable=true,verified_leaf_coverage=1,leaf_count=4,top_level_input_arity=7,top_level_output_arity=6,topo_hash=%s,source_symbol='andriy_three_branch_training_prediction',description=%s WHERE artifact_id=%s", (topo,
            'Train and execute three Andriy branches from raw clips across three explicitly ordered populations. Automated Tier 3 validation with documented current-runtime adaptations and source feature-selection gates; full eleven-model ensemble and predictive-quality claims remain outside scope.', graph_id))
        db.execute('UPDATE artifact_versions SET is_latest=(version_id=%s) WHERE artifact_id=%s', (graph_version, graph_id))
        if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (graph_id,)).fetchone():
            raise ValueError('Approved graph not served')
        if not apply:
            db.rollback()
    return dict(applied=apply, new_approval_evidence=created, served_provider_versions=25, served_component_cdgs=1, trust_tier=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(approve(Path(__file__).resolve().parents[1], args.reference_dir, args.library, args.apply), indent=2))
