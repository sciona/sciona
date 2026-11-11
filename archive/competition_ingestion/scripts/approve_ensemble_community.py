#!/usr/bin/env python3
"""Approve reviewed exact ensemble adapters and graph atomically at Tier 3."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid5
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from dotenv import dotenv_values
from sciona.ensemble_execution import build_ensemble_execution_graph
from sciona.cdg_projection import build_published_cdg_projection
from sciona.physics_ingest.pdg_evidence import _digest
from sciona.services.execution_graph_codec import encode_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.audit_ensemble_prerequisites import audit
from scripts.inventory_ensemble_adapters import inventory
from scripts.import_residual_execution_drafts import ensure_row
from scripts.import_riemannian_execution import SOURCE_ID, SOURCE_VERSION, SOURCE_HASH


LIMITATIONS = [
    'Automated Tier 3 review only; no Tier 1 certification or Tier 2 usage claim.',
    'Caller supplies explicit source-safe training membership, population order and common physical prediction identity across families.',
    'Current numerical and R API adaptations apply; no historical-engine equivalence or predictive-quality claim.',
    'Source feature-selection gates, finite-input and nondegenerate CSP requirements remain enforced.',
    'Synthetic raw execution and separate source references establish implementation behavior, not predictive accuracy.',
    'Full-vector ranks precede identity selection; equal weights add to caller baseline without clipping or reranking.',
]


def provider_closure(root, graph):
    import importlib
    import inspect
    from sciona.ghost.registry import REGISTRY
    from sciona.atoms.provider_inventory import ProviderRepo
    from sciona.atoms.supabase_seed import _parse_registered_atoms
    from scripts.inventory_andriy_provider_closure import registered_dependencies
    roots = {n.matched_primitive for n in graph.nodes}
    for runtime in roots:
        importlib.import_module(runtime.rsplit('.', 1)[0])
    registered = {entry['impl']: name for name, entry in REGISTRY.items()}
    dependencies, todo = {}, list(roots)
    while todo:
        runtime = todo.pop()
        if runtime in dependencies:
            continue
        deps = registered_dependencies(REGISTRY[runtime]['impl'], registered) - {runtime}
        dependencies[runtime] = sorted(deps)
        todo.extend(deps - dependencies.keys())
    records = {}
    for name in ['sciona-atoms', 'sciona-atoms-signal', 'sciona-atoms-ml']:
        directory = root.parent/name
        for spec in _parse_registered_atoms(repo=ProviderRepo(name, directory), artifact_root=directory/'src/sciona/atoms'):
            runtime = spec.import_module+'.'+spec.source_symbol
            if runtime not in dependencies:
                continue
            if runtime in records or Path(inspect.getsourcefile(inspect.unwrap(REGISTRY[runtime]['impl']))).resolve() != spec.file_path.resolve():
                raise ValueError('Ambiguous or mismatched provider source: '+runtime+' imported='+str(inspect.getsourcefile(inspect.unwrap(REGISTRY[runtime]['impl'])))+' parsed='+str(spec.file_path))
            records[runtime] = dict(runtime_fqdn=runtime, catalog_fqdn=spec.fqdn,
                version_id=str(spec.version_id), content_hash=spec.content_hash,
                provider_sha256=hashlib.sha256(spec.file_path.read_bytes()).hexdigest(),
                dependencies=dependencies[runtime])
    if set(records) != set(dependencies) or len(records) != 59 or sum(map(len, dependencies.values())) != 26:
        raise ValueError('Reviewed provider closure changed')
    return records



def approve(root, reference_dir, library, apply=False):
    prerequisite = audit(root, reference_dir, library)
    if not prerequisite['execution_prerequisites_complete']:
        raise ValueError('Successful full raw graph evidence required')
    closure = inventory(root)
    reviews = {record['runtime_fqdn']: record for record in closure['providers']}
    tests = json.loads((root/'docs/reviews/ensemble_test_review.json').read_text())
    if tests['test_results'] != dict(tests=24, failures=0, errors=0, skipped=0):
        raise ValueError('Complete reviewed test suite required')
    for name, sha in tests['test_source_sha256'].items():
        if hashlib.sha256((root/'tests'/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Reviewed tests changed')
    if tests['adapter_source_sha256'] != {name: r['provider_sha256'] for name, r in reviews.items()}:
        raise ValueError('Tested provider closure changed')
    files = ['ensemble_execution_parity.json', 'ensemble_test_review.json',
        'ensemble_alignment_parity.json', 'ensemble_semantic_review.md', 'ensemble_prerequisites.json']
    evidence_hashes = {name: hashlib.sha256((root/'docs/reviews'/name).read_bytes()).hexdigest() for name in files}
    graph = build_ensemble_execution_graph()
    digest, _, _ = encode_execution_graph(graph)
    graph_id = uuid5(SOURCE_ID, 'full-ensemble-execution.v1')
    graph_version = uuid5(graph_id, digest)
    graph_fqdn = 'cdg.competition.solution.kaggle.barachant_seizure_1st.full_ensemble_execution'
    all_providers = provider_closure(root, graph)
    created = 0
    rollup = dict(overall_verdict='acceptable_with_limits', structural_status='pass', runtime_status='pass',
        semantic_status='pass', developer_semantics_status='pass', review_status='approved',
        review_semantic_verdict='pass', review_developer_semantics_verdict='pass', trust_readiness='ready',
        review_limitations=LIMITATIONS, review_required_actions=[], trust_blockers=[],
        acceptability_band='acceptable_with_limits', parity_coverage_level='positive_and_negative', parity_test_status='pass')
    with psycopg.connect(dotenv_values(root/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('ensemble-community.v1'))")
        rows = db.execute("SELECT e.*,a.fqdn,a.status AS artifact_status,v.content_hash,v.is_latest,v.trust_tier FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.runner_version='ensemble-adapter-intake.v1' FOR UPDATE OF a,v FOR SHARE OF e").fetchall()
        if len(rows) != 7 or {r['details']['runtime_fqdn'] for r in rows} != set(reviews):
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
            expected = {(reviews[d]['catalog_fqdn'], reviews[d]['content_hash']) for d in review.get('dependencies', [])}
            deps = db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE', (row['version_id'],)).fetchall()
            if len(deps) != len(expected) or any(d['dependency_role'] != 'logic_atom' or d['optional'] for d in deps) or {(d['dependency_artifact_fqdn'], d['dependency_content_hash']) for d in deps} != expected:
                raise ValueError('Provider dependency closure changed')
            if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (row['version_id'],)).fetchone():
                raise ValueError('Unresolved failed provider audit')
            for table, key in [('artifact_references', 'artifact_id'), ('atom_references', 'atom_id')]:
                ensure_row(db, table, {key: row['artifact_id'], 'ref_key': reference['ref_id']}, {
                    key: row['artifact_id'], **reference, 'ref_key': reference['ref_id'], 'source': 'llm_extracted',
                    'verified': True, 'confidence': 'high', 'relevance_note': 'Pinned source ensemble semantics with explicit identity adapters and automated Tier 3 evidence.'})
            for table, key in [('artifact_audit_rollups', 'artifact_id'), ('atom_audit_rollups', 'atom_id')]:
                ensure_row(db, table, {key: row['artifact_id']}, {key: row['artifact_id'], **rollup})
            evidence_id = uuid5(row['version_id'], 'ensemble-adapter-community.v1')
            created += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence_id}, dict(
                evidence_id=evidence_id, artifact_id=row['artifact_id'], version_id=row['version_id'],
                audit_type='semantic_audit', passed=True, status='completed', source_kind='automated',
                runner_version='ensemble-adapter-community.v1', details=Jsonb(dict(publication_tier=3,
                    review_source='automated', review=review, intake_evidence_sha256=_digest(row['details']),
                    evidence_sha256=evidence_hashes, limitations=LIMITATIONS))))
            db.execute("UPDATE artifacts SET status='approved',is_publishable=true WHERE artifact_id=%s", (row['artifact_id'],))
            db.execute("UPDATE atoms SET status='approved',is_publishable=true WHERE atom_id=%s", (row['artifact_id'],))
        for runtime, record in all_providers.items():
            served = db.execute('SELECT v.version_id,v.content_hash FROM catalog_atoms_served a JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.is_latest FOR SHARE OF v', (record['catalog_fqdn'],)).fetchall()
            if len(served) != 1 or str(served[0]['version_id']) != record['version_id'] or served[0]['content_hash'] != record['content_hash']:
                raise ValueError('Exact provider closure not served: '+runtime)
            expected_deps = {(all_providers[d]['catalog_fqdn'], all_providers[d]['content_hash']) for d in record['dependencies']}
            deps = db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE', (record['version_id'],)).fetchall()
            if len(deps) != len(expected_deps) or any(d['dependency_role'] != 'logic_atom' or d['optional'] for d in deps) or {(d['dependency_artifact_fqdn'], d['dependency_content_hash']) for d in deps} != expected_deps:
                raise ValueError('Stored runtime dependencies differ: '+runtime)
            if runtime == 'sciona.atoms.audio_speech.atoms.audio_windows':
                continue  # Existing legacy provider; exact source covered by upstream execution audit.
            canonical = db.execute("SELECT 1 FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND a.status='approved' AND a.is_publishable AND v.version_id=%s AND v.content_hash=%s AND v.is_latest FOR SHARE OF a,v", (record['catalog_fqdn'], record['version_id'], record['content_hash'])).fetchone()
            if not canonical:
                raise ValueError('Exact canonical approval missing: '+runtime)
            intakes = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND passed AND runner_version=ANY(%s) FOR SHARE", (record['version_id'], ['riemannian-provider-intake.v1', 'relative-power-provider-intake.v1', 'combined-feature-provider-intake.v1', 'feng-knn-provider-intake.v1', 'feng-xgb-provider-intake.v1', 'feng-expanded-provider-intake.v1', 'andriy-provider-intake.v1', 'ensemble-adapter-intake.v1'])).fetchall()
            if len(intakes) != 1 or intakes[0]['details']['source_sha256'] != record['provider_sha256']:
                raise ValueError('Exact provider intake source differs: '+runtime)
            for table in ['artifact_io_specs', 'atom_io_specs']:
                actual = db.execute(sql.SQL('SELECT direction,name,ordinal,type_desc,constraints,required,default_value_repr FROM {} WHERE version_id=%s ORDER BY direction,ordinal').format(sql.Identifier(table)), (record['version_id'],)).fetchall()
                if actual != sorted(intakes[0]['details']['interfaces'], key=lambda p: (p['direction'],p['ordinal'])):
                    raise ValueError('Reused provider interface differs: '+runtime)
            if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (record['version_id'],)).fetchone():
                raise ValueError('Unresolved failed closure audit')
        intake = db.execute("SELECT e.*,a.status AS artifact_status,v.content_hash,v.trust_tier FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.version_id=%s AND e.runner_version='ensemble-execution-draft-intake.v1' FOR UPDATE OF a,v FOR SHARE OF e", (graph_version,)).fetchone()
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
            review = all_providers[node.matched_primitive]
            if len(matches) != 1:
                raise ValueError('Ambiguous binding')
            b = matches[0]
            if b['bound_artifact_fqdn'] != review['catalog_fqdn'] or b['bound_version_content_hash'] != review['content_hash'] or b['evidence_summary']['provider_version_id'] != review['version_id'] or b['status'] != 'active' or b['evidence_summary']['runtime_fqdn'] != node.matched_primitive or b['evidence_summary']['output_aliases_by_ordinal'] != [p.name for p in node.outputs]:
                raise ValueError('Exact binding differs')
        expected_io = []
        consumed = {(e.target_id, e.input_name) for e in graph.edges}
        producers = {e.source_id for e in graph.edges}
        for direction, ports in [('input', [p for n in graph.nodes for p in n.inputs if (n.node_id,p.name) not in consumed]),
                                 ('output', [p for n in graph.nodes if n.node_id not in producers for p in n.outputs])]:
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
            confidence='high', relevance_note='All eleven models executed from synthetic raw clips with exact pinned-source final blending and declared runtime adaptations.'))
        evidence_id = uuid5(graph_version, 'ensemble-execution-community.v1')
        created += ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence_id}, dict(
            evidence_id=evidence_id, artifact_id=graph_id, version_id=graph_version, audit_type='semantic_audit',
            passed=True, status='completed', source_kind='automated', runner_version='ensemble-execution-community.v1',
            details=Jsonb(dict(publication_tier=3, review_source='automated', content_hash=digest,
                intake_evidence_sha256=_digest(intake['details']), evidence_sha256=evidence_hashes,
                provider_versions={r['runtime_fqdn']: r['version_id'] for r in all_providers.values()}, limitations=LIMITATIONS))))
        topo = build_published_cdg_projection(artifact={'artifact_id': str(graph_id), 'fqdn': graph_fqdn},
            version={'version_id': str(graph_version), 'content_hash': digest}, cdg=graph).topo_hash
        db.execute("UPDATE artifacts SET status='approved',is_publishable=true,verified_leaf_coverage=1,leaf_count=155,top_level_input_arity=18,top_level_output_arity=1,topo_hash=%s,source_symbol='eleven_model_training_prediction_blend',description=%s WHERE artifact_id=%s", (topo,
            'Train and execute eleven models across three explicitly ordered populations and combine identity-aligned source ranks. Automated Tier 3 validation with full synthetic raw execution, source parity and documented current-runtime adaptations; no predictive-quality claim.', graph_id))
        db.execute('UPDATE artifact_versions SET is_latest=(version_id=%s) WHERE artifact_id=%s', (graph_version, graph_id))
        if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (graph_id,)).fetchone():
            raise ValueError('Approved graph not served')
        if not apply:
            db.rollback()
    return dict(applied=apply, new_approval_evidence=created, served_provider_versions=59, newly_approved_adapter_versions=7, served_component_cdgs=1, trust_tier=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(approve(Path(__file__).resolve().parents[1], args.reference_dir, args.library, args.apply), indent=2))
