#!/usr/bin/env python3
"""Approve the reviewed two-branch Riemannian execution component at Tier 3."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid5
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from sciona.cdg_projection import build_published_cdg_projection
from sciona.physics_ingest.pdg_evidence import _digest
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.import_residual_execution_drafts import ensure_row
from scripts.import_riemannian_execution import validated_graph, SOURCE_VERSION, SOURCE_HASH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--component', choices=['riemannian', 'relative_power', 'combined_feature', 'feng_knn', 'feng_xgb', 'feng_expanded'], default='riemannian')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report, graph, digest, _, _ = validated_graph(root, args.reference_dir, args.component)
    relative = args.component == 'relative_power'
    combined = args.component == 'combined_feature'
    feng = args.component == 'feng_knn'
    feng_xgb = args.component == 'feng_xgb'
    expanded = args.component == 'feng_expanded'
    prefix = args.component.replace('_', '-')
    bundle = json.loads((root / 'docs/reviews/riemannian_provider_community.json').read_text())
    bundles = {'riemannian-provider-community.v1': bundle}
    if relative or combined:
        bundles['relative-power-provider-community.v1'] = json.loads((root / 'docs/reviews/relative_power_provider_community.json').read_text())
    if combined:
        bundles['combined-feature-provider-community.v1'] = json.loads((root / 'docs/reviews/combined_feature_provider_community.json').read_text())
    if feng or feng_xgb or expanded:
        bundle = json.loads((root / 'docs/reviews/feng_knn_provider_community.json').read_text())
        bundles = {'feng-knn-provider-community.v1': bundle}
    if feng_xgb:
        bundles['feng-xgb-provider-community.v1'] = json.loads((root / 'docs/reviews/feng_xgb_provider_community.json').read_text())
    if expanded:
        bundles['feng-expanded-provider-community.v1'] = json.loads((root / 'docs/reviews/feng_expanded_provider_community.json').read_text())
    for reviewed_bundle in bundles.values():
        for name, sha in reviewed_bundle['test_source_hashes'].items():
            if hashlib.sha256((root / 'tests' / name).read_bytes()).hexdigest() != sha:
                raise ValueError('reviewed tests changed')
    with psycopg.connect(dotenv_values(root / '.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (prefix + '-execution-community.v1',))
        rows = db.execute("SELECT e.*,a.fqdn,a.status AS artifact_status,v.content_hash,v.trust_tier FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.runner_version=%s FOR UPDATE OF a,v FOR SHARE OF e", (prefix + '-execution-intake.v1',)).fetchall()
        if len(rows) != 1:
            raise ValueError('unique component intake required')
        intake = rows[0]
        if not intake['passed'] or intake['artifact_status'] not in {'draft', 'approved'} or intake['trust_tier'] != 3 or intake['content_hash'] != digest or intake['details']['execution_report_sha256'] != _digest(report):
            raise ValueError('validated component version/state differs')
        doc = db.execute('SELECT get_artifact_document(%s) AS d', (intake['fqdn'],)).fetchone()['d']
        if _artifact_document_to_cdg(doc, version_id=str(intake['version_id']), content_hash=digest, require_execution_envelope=True) != graph:
            raise ValueError('catalog graph differs')
        source = db.execute("SELECT e.details,a.fqdn,v.content_hash FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) WHERE e.version_id=%s AND e.runner_version='competition-intake.v1' AND e.passed FOR SHARE OF e,a,v", (SOURCE_VERSION,)).fetchone()
        if not source or source['content_hash'] != SOURCE_HASH or _digest(source['details']) != intake['details']['source_intake_sha256']:
            raise ValueError('source provenance differs')
        dependencies = db.execute('SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s FOR SHARE', (intake['version_id'],)).fetchall()
        if len(dependencies) != 1 or dependencies[0]['dependency_artifact_fqdn'] != source['fqdn'] or dependencies[0]['dependency_content_hash'] != SOURCE_HASH or dependencies[0]['optional']:
            raise ValueError('mandatory source provenance missing')
        reviews = db.execute("SELECT e.*,a.fqdn FROM artifact_audit_evidence e JOIN artifacts a USING(artifact_id) JOIN artifact_versions v USING(version_id) JOIN atoms l ON l.atom_id=a.artifact_id JOIN atom_versions lv ON lv.version_id=v.version_id AND lv.atom_id=l.atom_id WHERE e.runner_version=ANY(%s) AND e.passed AND a.status='approved' AND a.is_publishable AND l.status='approved' AND l.is_publishable AND v.is_latest AND lv.is_latest AND v.content_hash=lv.content_hash AND v.trust_tier=3 FOR SHARE OF e,a,v,l,lv", (list(bundles),)).fetchall()
        if expanded:
            wanted = {n.matched_primitive for n in graph.nodes} | {'sciona.atoms.riemannian_bci.signal_processing.feng_filter.feng_resample_filter', 'sciona.atoms.riemannian_bci.signal_processing.feng_expanded_features.feng_expanded_features'}
            reviews = [r for r in reviews if r['details']['review']['runtime_fqdn'] in wanted]
            complete = len(reviews) == 6 and {r['details']['review']['runtime_fqdn'] for r in reviews} == wanted
        elif feng_xgb:
            wanted = {n.matched_primitive for n in graph.nodes} | {'sciona.atoms.riemannian_bci.signal_processing.feng_filter.feng_resample_filter', 'sciona.atoms.riemannian_bci.signal_processing.feng_fft.feng_fft_features'}
            reviews = [r for r in reviews if r['details']['review']['runtime_fqdn'] in wanted]
            complete = len(reviews) == 5 and {r['details']['review']['runtime_fqdn'] for r in reviews} == wanted
        elif relative or combined:
            wanted = {n.matched_primitive for n in graph.nodes} | {'sciona.atoms.riemannian_bci.signal_processing.segment_windows.window_signal_segments'}
            reviews = [r for r in reviews if r['details']['review']['runtime_fqdn'] in wanted]
            complete = len(reviews) == (11 if combined else 7) and {r['details']['review']['runtime_fqdn'] for r in reviews} == wanted
        else:
            complete = len(reviews) == (5 if feng else 8) and {r['fqdn'] for r in reviews} == {r['fqdn'] for r in bundle['atoms']}
        if not complete:
            raise ValueError('complete exact provider approvals required')
        for review in reviews:
            if review['details']['review_bundle_sha256'] != _digest(bundles[review['runner_version']]):
                raise ValueError('review bundle drift')
            for dep in db.execute("SELECT * FROM artifact_dependencies WHERE dependent_version_id=%s AND dependency_role='logic_atom' AND NOT optional FOR SHARE", (review['version_id'],)):
                if not db.execute('SELECT 1 FROM catalog_atoms_served a JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.content_hash=%s AND v.is_latest', (dep['dependency_artifact_fqdn'], dep['dependency_content_hash'])).fetchone():
                    raise ValueError('invoked dependency not served at reviewed version')
        bindings = db.execute('SELECT * FROM artifact_cdg_bindings WHERE version_id=%s FOR SHARE', (intake['version_id'],)).fetchall()
        if len(bindings) != len(graph.nodes):
            raise ValueError('incomplete graph binding set')
        for node in graph.nodes:
            matches = [b for b in bindings if b['node_id'] == node.node_id]
            if len(matches) != 1:
                raise ValueError('ambiguous node binding')
            binding = matches[0]
            matches = [r for r in reviews if r['fqdn'] == binding['bound_artifact_fqdn'] and r['details']['content_hash'] == binding['bound_version_content_hash']]
            if len(matches) != 1 or binding['status'] != 'active' or binding['evidence_summary']['runtime_fqdn'] != node.matched_primitive or matches[0]['details']['review']['runtime_fqdn'] != node.matched_primitive:
                raise ValueError('exact reviewed runtime binding missing')
            if binding['evidence_summary']['output_aliases_by_ordinal'] != [p.name for p in node.outputs]:
                raise ValueError('output binding aliases differ')
        if db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=ANY(%s) AND NOT passed', ([intake['version_id']] + [r['version_id'] for r in reviews],)).fetchone():
            raise ValueError('failed component/provider audit requires resolution')
        io = db.execute('SELECT direction,name FROM artifact_io_specs WHERE version_id=%s', (intake['version_id'],)).fetchall()
        expected_outputs = {'relative_power_segment_probabilities'} if relative else {'autocorrelation_segment_probabilities', 'coherence_segment_probabilities'}
        if expanded:
            expected_outputs = {'feng_expanded_knn_segment_probabilities', 'feng_glm_segment_probabilities'}
        if feng_xgb:
            expected_outputs = {'feng_xgb_segment_probabilities'}
        if feng:
            expected_outputs = {'feng_knn_segment_probabilities'}
        if combined:
            expected_outputs = {'combined_feature_segment_probabilities'}
        if {r['name'] for r in io if r['direction'] == 'input'} != {'training_segments', 'prediction_segments', 'segment_labels'} or {r['name'] for r in io if r['direction'] == 'output'} != expected_outputs or len(io) != 3 + len(expected_outputs):
            raise ValueError('component boundary differs')
        reference = db.execute("SELECT ref_id,title,url FROM references_registry WHERE ref_id='barachant-seizure-00f937cc' FOR SHARE").fetchone()
        if not reference:
            raise ValueError('source reference missing')
        ensure_row(db, 'artifact_references', {'artifact_id': intake['artifact_id'], 'ref_key': reference['ref_id']}, {
            'artifact_id': intake['artifact_id'], **reference, 'ref_key': reference['ref_id'], 'source': 'llm_extracted',
            'verified': True, 'confidence': 'high', 'relevance_note': ('Source expanded Feng KNN and GLM branches validated from full-duration synthetic signals through the serialized runner; complete ensemble remains outside scope.' if expanded else 'Source Feng XGBoost branch validated from full-duration synthetic signals through the serialized runner; complete ensemble remains outside scope.' if feng_xgb else 'Source Feng base KNN branch validated from full-duration synthetic signals through the serialized runner; complete ensemble remains outside scope.' if feng else 'Source combined-feature branch validated through the serialized runner at full source tree settings; complete eleven-model ensemble remains outside scope.' if combined else 'Source relative-power branch validated through the serialized runner at full source tree settings; complete eleven-model ensemble remains outside scope.' if relative else 'Two source Riemannian branches validated through the serialized runner at full source tree settings; complete eleven-model ensemble remains outside scope.')})
        limitations = ['Community Tier 3 automated review; no Tier 1 certification or Tier 2 usage claim.',
            'Two source model branches only. Nine other models and the complete eleven-model blend remain outside this component.',
            'Input segments must be finite multichannel signals at 400 Hz, at least 8000 samples each, with matching channel count and binary training labels.',
            'Undefined delay correlations or zero-power bands are rejected. Tangent projection applies explicit source shrinkage; prediction coherence reference depends on the whole prediction batch.',
            'Modern XGBoost/sklearn execution with synthetic numerical parity; no original training engine, predictive-performance or clinical-validity claim.',
            'Returns separate aligned segment-score vectors in prediction input order. Embedded draft metadata preserves the immutable validated graph snapshot.']
        if relative:
            limitations[1] = 'One relative-log-power model branch only. The complete eleven-model ensemble remains outside this component.'
            limitations[3] = 'Zero-power bands and undefined logarithms are rejected. Welch band means are normalized without band-width integration; feature axes must match across training and prediction.'
            limitations[5] = 'Returns one aligned segment-score vector in prediction input order. Embedded draft metadata preserves the immutable validated graph snapshot.'
        if combined:
            limitations[1] = 'One combined-feature model branch only. The complete eleven-model ensemble remains outside this component.'
            limitations[3] = 'Rejects undefined spectra, singular AR regressions and undefined fractal features. Preserves source statistic, AR coefficient-standard-error and fractal conventions; concatenates feature families within channels before vectorization.'
            limitations[5] = 'Returns one aligned segment-score vector in prediction input order. Embedded draft metadata preserves the immutable validated graph snapshot.'
        if feng:
            limitations[1] = 'One Feng base KNN model for one caller-selected population; complete eleven-model ensemble remains outside scope.'
            limitations[2] = 'Caller supplies ordered actual 600-second samples/channels clips with matching channel counts and both binary training classes; at least forty training windows required.'
            limitations[3] = 'Preserves source float32 loading, causal filter transients, undefined-log cleanup and separately fitted prediction-batch scaler. Predictions depend on the entire prediction batch.'
            limitations[4] = 'Modern NumPy/SciPy/pandas/sklearn synthetic source parity; no historical engine, predictive-performance or clinical-validity claim.'
            limitations[5] = 'Returns mean positive-class window probabilities per prediction segment in input order. Embedded draft metadata preserves the validated graph snapshot.'
        if feng_xgb:
            limitations[1] = 'One Feng XGBoost model for one caller-selected population; complete eleven-model ensemble remains outside scope.'
            limitations[2] = 'Caller supplies ordered actual 600-second samples/channels clips with matching channels and both binary training classes.'
            limitations[3] = 'Preserves source float32 loading and causal filter transients. Features are unscaled; NaNs remain missing values, infinities are rejected.'
            limitations[4] = 'Current XGBoost with 500 source-configured rounds; no historical engine, predictive-performance or clinical-validity claim.'
            limitations[5] = 'Returns mean positive-class window probabilities per prediction segment in input order. Embedded draft metadata preserves the validated snapshot.'
        if expanded:
            limitations[1] = 'Two expanded-feature Feng models for one caller-selected population; full eleven-model ensemble remains outside scope.'
            limitations[2] = 'Caller supplies ordered actual 600-second clips with 16 channels and both binary training classes; at least forty training windows required.'
            limitations[3] = 'Source C-order correlation arithmetic, float32 loading, KNN prediction-batch scaling and GLM training-scaler reuse are preserved. GLM rejects NaNs; both reject positive infinity.'
            limitations[4] = 'Current sklearn GLM LBFGS for unspecified source solver; no historical-engine, predictive-performance or clinical-validity claim.'
            limitations[5] = 'Returns two aligned mean window-probability vectors in prediction segment order. Embedded draft metadata preserves the validated graph snapshot.'
        rollup = dict(artifact_id=intake['artifact_id'], overall_verdict='acceptable_with_limits', structural_status='pass', runtime_status='pass',
            semantic_status='pass', developer_semantics_status='pass', review_status='approved', review_semantic_verdict='pass',
            review_developer_semantics_verdict='pass', trust_readiness='ready', review_limitations=limitations, review_required_actions=[],
            trust_blockers=[], acceptability_band='acceptable_with_limits', parity_coverage_level='positive_and_negative', parity_test_status='pass')
        ensure_row(db, 'artifact_audit_rollups', {'artifact_id': intake['artifact_id']}, rollup)
        evidence_id = uuid5(intake['version_id'], prefix + '-execution-community.v1')
        created = ensure_row(db, 'artifact_audit_evidence', {'evidence_id': evidence_id}, dict(evidence_id=evidence_id,
            artifact_id=intake['artifact_id'], version_id=intake['version_id'], audit_type='semantic_audit', passed=True, status='completed',
            source_kind='automated', runner_version=prefix + '-execution-community.v1', details=Jsonb({
                'publication_tier': 3, 'review_source': 'automated', 'content_hash': digest, 'execution_intake_sha256': _digest(intake['details']),
                'provider_approvals': {str(r['evidence_id']): _digest(r['details']) for r in reviews}, 'limitations': limitations})))
        topo = build_published_cdg_projection(artifact={'artifact_id': str(intake['artifact_id']), 'fqdn': intake['fqdn']},
            version={'version_id': str(intake['version_id']), 'content_hash': digest}, cdg=graph).topo_hash
        db.execute("UPDATE artifacts SET status='approved',is_publishable=true,verified_leaf_coverage=1,leaf_count=%s,top_level_input_arity=3,top_level_output_arity=%s,topo_hash=%s,source_symbol=%s WHERE artifact_id=%s", (len(graph.nodes), len(expected_outputs), topo, 'feng_expanded_training_prediction' if expanded else 'feng_xgb_training_prediction' if feng_xgb else 'feng_knn_training_prediction' if feng else 'combined_feature_training_prediction' if combined else 'relative_power_training_prediction' if relative else 'riemannian_two_branch_training_prediction', intake['artifact_id']))
        db.execute('UPDATE artifact_versions SET is_latest=(version_id=%s) WHERE artifact_id=%s', (intake['version_id'], intake['artifact_id']))
        if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (intake['artifact_id'],)).fetchone():
            raise ValueError('approved component not served')
        if not args.apply:
            db.rollback()
    print(json.dumps({'applied': args.apply, 'new_approval': created, 'approved_provider_versions': len(reviews), 'node_bindings': len(bindings), 'trust_tier': 3}, indent=2))


if __name__ == '__main__':
    main()
