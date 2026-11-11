"""Read-only reconciliation of the original intake with the full ensemble.

Structural coverage and retained evidence are not fresh runtime qualification.
Missing external runtime evidence leaves promotion explicitly ineligible.
"""
import argparse
import hashlib
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.competition_import import _canonical
from sciona.ensemble_execution import build_ensemble_execution_graph
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph
from scripts.approve_ensemble_community import provider_closure
from scripts.audit_andriy_promotion_evidence import audit as audit_andriy
from scripts.import_ensemble_adapter_drafts import verify_local_evidence
from scripts.import_riemannian_execution import SOURCE_ID, SOURCE_VERSION, SOURCE_HASH, validated_graph
from scripts.stage_remaining_legacy_physics_revisions import history

ROOT = Path(__file__).resolve().parents[1]
PARENT_VERSION = '65759b82-3166-5b2c-ae50-de7be3b3ab20'
COMPONENTS = ['riemannian', 'relative_power', 'combined_feature', 'feng_knn', 'feng_xgb', 'feng_expanded']
STAGES = {
    'windower': ('prepare_segment_windows', 'Correct overlapping-window claim: Alex uses nonoverlapping source windows. Feng and Andriy retain separate filtering, windowing and population contracts.'),
    'autocorrelation_covariance_matrix': ('channel_delay_correlations', 'Retain explicit delay list, subsampling, channel axes and positive-semidefinite input-to-tangent contract.'),
    'cross_frequency_coherence_matrix': ('frequency_band_coherence', 'Retain fixed source bands and frequency-by-frequency channel matrices; this is not a general physiological inference.'),
    'riemannian_mean_spd': ('channel_tangent_features', 'Replace unsupported iterative affine-invariant mean claim: autocorrelation uses identity/False; coherence uses logeuclid/True with whole-prediction-batch reference update.'),
    'tangent_space_projection': ('channel_tangent_features', 'Retain fixed shrinkage, matrix logarithm, weighted upper triangle and channel concatenation. Prediction-batch dependence is explicit; no global distance-isometry claim.'),
    'segment_max_aggregation': ('segment_probability_max', 'Maximum aggregation applies to the Alex branches. Feng and Andriy retain their own aggregation; historical performance numbers do not qualify this implementation.'),
    'ranked_prediction_blend': ('aligned_eleven_model_blend', 'Replace placeholder dictionary interface with eleven ordered vectors and opaque identities. Rank complete vectors before output selection, add equal-weight scores to caller baseline, without clipping or reranking.'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def region(node_id):
    for family in ['alex', 'feng']:
        for population in range(3):
            prefix = f'{family}_population{population}'
            if node_id == prefix:
                return 'population_routing'
            if node_id.startswith(prefix + '_'):
                return family + '_' + node_id[len(prefix)+1:].split('_')[0]
    if node_id.startswith('collect_'):
        return 'population_score_collection'
    if node_id in ['families', 'indices', 'andriy_ids']:
        return 'population_routing'
    if node_id in ['andriy_populations', 'andriy_xgb', 'andriy_svm', 'andriy_glm']:
        return 'andriy'
    if node_id in ['pack', 'blend']:
        return 'identity_aligned_blend'
    raise ValueError('Unmapped ensemble node')


def audit(reference_dir, library=None):
    graph = build_ensemble_execution_graph()
    digest, nodes, edges = encode_execution_graph(graph)
    require(len(nodes) == 155 and len(edges) == 523, 'Full ensemble cardinality changed')
    closure = provider_closure(ROOT, graph)
    regions = {}
    for node in graph.nodes:
        regions.setdefault(region(node.node_id), []).append(node.node_id)
    mapped = [n for values in regions.values() for n in values]
    require(len(mapped) == len(set(mapped)) == 155, 'Regions must partition the entire graph')
    scope = {stage: dict(nodes=[n.node_id for n in graph.nodes if n.matched_primitive.rsplit('.', 1)[1] == fn],
                        correction=correction) for stage, (fn, correction) in STAGES.items()}
    require(all(row['nodes'] for row in scope.values()), 'Unmapped original stage')
    evidence_hashes = {}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        intake = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='competition-intake.v1'", (SOURCE_VERSION,)).fetchall()
        require(len(intake) == 1, 'Unique original intake required')
        snapshot = intake[0]['details']['snapshot']
        require(hashlib.sha256(_canonical(snapshot).encode()).hexdigest() == SOURCE_HASH, 'Original snapshot changed')
        original_stages = snapshot['template']['stages']
        require(len(original_stages) == len(STAGES) and {s['stage_id'] for s in original_stages} == set(STAGES), 'Original stage set changed')
        original = db.execute('SELECT a.artifact_id::text,a.status,a.is_publishable,v.content_hash FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (SOURCE_VERSION,)).fetchone()
        require(original['artifact_id'] == str(SOURCE_ID) and original['content_hash'] == SOURCE_HASH, 'Original identity changed')
        original_history = history(db, SOURCE_VERSION)
        parent = db.execute('SELECT a.fqdn,a.status,a.is_publishable,v.content_hash,v.trust_tier,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE v.version_id=%s', (PARENT_VERSION,)).fetchone()
        require(parent and parent['status'] == 'approved' and parent['is_publishable'] and parent['is_latest'] and parent['trust_tier'] == 3 and parent['content_hash'] == digest, 'Qualified parent differs')
        document = db.execute('SELECT get_artifact_document(%s) AS d', (parent['fqdn'],)).fetchone()['d']
        require(_artifact_document_to_cdg(document, version_id=PARENT_VERSION, content_hash=digest, require_execution_envelope=True) == graph, 'Stored parent differs')
        bindings = db.execute('SELECT node_id,status,bound_artifact_fqdn,bound_version_content_hash,evidence_summary FROM artifact_cdg_bindings WHERE version_id=%s', (PARENT_VERSION,)).fetchall()
        require(len(bindings) == 155 and len({b['node_id'] for b in bindings}) == 155, 'Complete unique parent bindings required')
        by_id = {b['node_id']: b for b in bindings}
        for node in graph.nodes:
            record = closure[node.matched_primitive]; bound = by_id[node.node_id]
            require(bound['status'] == 'active' and bound['bound_artifact_fqdn'] == record['catalog_fqdn'] and bound['bound_version_content_hash'] == record['content_hash'], 'Exact parent provider binding differs')
            require(bound['evidence_summary']['provider_version_id'] == record['version_id'] and bound['evidence_summary']['runtime_fqdn'] == node.matched_primitive and bound['evidence_summary']['output_aliases_by_ordinal'] == [p.name for p in node.outputs], 'Provider binding semantics differ')
        provider_states = []
        for runtime, record in sorted(closure.items()):
            rows = db.execute('SELECT v.version_id::text,v.content_hash FROM catalog_atoms_served a JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.is_latest', (record['catalog_fqdn'],)).fetchall()
            require(rows == [dict(version_id=record['version_id'], content_hash=record['content_hash'])], 'Exact served provider changed')
            canonical = db.execute("SELECT 1 FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.fqdn=%s AND a.status='approved' AND a.is_publishable AND v.version_id=%s AND v.content_hash=%s AND v.is_latest", (record['catalog_fqdn'], record['version_id'], record['content_hash'])).fetchone()
            expected = {(closure[d]['catalog_fqdn'], closure[d]['content_hash']) for d in record['dependencies']}
            deps = db.execute('SELECT dependency_artifact_fqdn,dependency_content_hash,dependency_role,optional FROM artifact_dependencies WHERE dependent_version_id=%s', (record['version_id'],)).fetchall()
            require(len(deps) == len(expected) and {(d['dependency_artifact_fqdn'], d['dependency_content_hash']) for d in deps} == expected and all(d['dependency_role'] == 'logic_atom' and not d['optional'] for d in deps), 'Mandatory provider closure changed')
            provider_states.append(dict(runtime_fqdn=runtime, version_id=record['version_id'], canonical_approved=bool(canonical)))
        evidence = db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='ensemble-execution-community.v1' AND passed AND status='completed' AND source_kind='automated'", (PARENT_VERSION,)).fetchall()
        require(len(evidence) == 1, 'Unique parent approval required')
        evidence_hashes = evidence[0]['details']['evidence_sha256']
        for name, expected in evidence_hashes.items():
            require(sha(ROOT/'docs/reviews'/name) == expected, 'Retained approval evidence changed')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (PARENT_VERSION,)).fetchone(), 'Failed parent audit')
    checks = {}
    for component in COMPONENTS:
        _, _, component_digest, _, _ = validated_graph(ROOT, reference_dir, component)
        checks[component] = dict(retained_evidence_verified=True, graph_sha256=component_digest)
    verify_local_evidence(ROOT, reference_dir)
    missing_runtime = []
    if library is not None:
        for name, expected in json.loads((ROOT/'docs/reviews/andriy_r_contract_audit.json').read_text())['compiled_library_sha256'].items():
            path = library/name
            if not path.is_file() or sha(path) != expected:
                missing_runtime.append(name)
    if library is None or missing_runtime:
        checks['andriy'] = dict(retained_evidence_verified=False, reason='Historical compiled R qualification not satisfied', missing_or_changed_libraries=missing_runtime)
    else:
        checks['andriy'] = audit_andriy(ROOT, reference_dir, library)
    return dict(passed=True, read_only=True, catalog_mutations=0, eligible_for_approval=False,
        source_version_id=str(SOURCE_VERSION), source_content_hash=SOURCE_HASH, original_state=original,
        original_history_sha256=original_history, parent_version_id=PARENT_VERSION, parent_graph_sha256=digest,
        source_stage_mapping=scope, graph_regions=regions, nodes=155, edges=523,
        provider_versions=provider_states,
        canonical_provider_gaps=[p['runtime_fqdn'] for p in provider_states if not p['canonical_approved']],
        provider_closure_sha256=hashlib.sha256(_canonical(closure).encode()).hexdigest(),
        upstream_evidence_checks=checks, retained_parent_evidence_sha256=evidence_hashes,
        reusable_core='Windowing, PSD feature transformations, statistical features, model fitting, paired score collection and identity-aligned rank blending remain separate callable operations. Source configuration and population membership remain explicit adapters.',
        limitations=['Read-only source and structural reconciliation; no fresh full-graph execution or corrected-original approval.',
            'No empirical cross-domain predictive-quality claim. Caller establishes sampling, units, training membership and physical identity across families.',
            'Coherence reference updates use the entire prediction batch; changing that batch can change predictions.',
            'Original iterative-mean, overlapping-window and global metric-isometry claims require corrected descriptions.'],
        remaining=['Restore or requalify the exact R runtime before full execution.',
            'Qualify the legacy-only audio window provider in the canonical catalog; do not inherit the historical exception silently.',
            'Check complete parent/provider interfaces, validity and current runtime contracts.',
            'Stage a corrected original version preserving all eleven branches and original history.',
            'Verify stored full execution, corruption rejection, publication rollback, idempotency and served selection.'],
        auditor_sha256=sha(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--library', type=Path)
    args = parser.parse_args()
    report = audit(args.reference_dir, args.library)
    (ROOT/'docs/reviews/ensemble_original_intake_scope.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, eligible_for_approval=False, nodes=report['nodes'], providers=len(report['provider_versions']), catalog_mutations=0)))
