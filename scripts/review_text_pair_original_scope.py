"""Read-only mapping of every original text-pair stage to the pending recipe."""
import ast
import json

from scripts.audit_decomposed_text_pair_readiness import audit, ROOT, sha
from scripts.import_flavours_provider_drafts import connect


STAGES = {
    'pair_normalization': dict(
        anchors=[('text_pair_normalization_policy', 'NormalizationPolicy.normalize_pairs'),
                 ('text_pair_normalization_policy', 'NormalizationPolicy.normalize_populations')],
        realization='Apply NFKC/casefold, existing spelling correction, word punctuation handling, explicit domain substitutions and one-pass synonym expansion; reject split collisions after normalization.',
        limits='Caller supplies a frozen vocabulary and mappings. Empty mappings express identity policy. No semantic correctness or external vocabulary fitting claim.'),
    'lexical_similarity_features': dict(
        anchors=[('text_pair_features', 'PairFeatures._features'), ('text_pair_bm25', 'PairBM25'),
                 ('bm25_scoring', 'bm25_score'), ('text_pair_complete_recipe', 'CompletePairFeatures._features')],
        realization='Retain character n-gram overlap, token Jaccard, TF-IDF cosine, normalized edit similarity, exact match and length ratio; append symmetric BM25 with training-only corpus statistics.',
        limits='The external boundary declares shared_text: two comparable representations, without distinct title/description subfields or implicit concatenation. BM25 query terms have unit weight, corpus documents are unique training texts, and positive IDF is explicit. Separate-field source fidelity still requires review before original promotion.'),
    'embedding_similarity_features': dict(
        anchors=[('text_pair_complete_recipe', 'CompletePairFeatures.fit'), ('text_pair_complete_recipe', 'Projection.transform')],
        realization='Select the listed LSA/SVD branch with reusable sparse randomized projection fit and transform; append latent cosine and Euclidean distance.',
        limits='Training-only TF-IDF and uncentered randomized projection. No pretrained embedding or topic-model branch claimed.'),
    'pair_context_features': dict(
        anchors=[('text_pair_complete_recipe', 'CompletePairFeatures.fit'), ('text_pair_features', 'PairFeatures._features')],
        realization='Select unlabeled pair graph degree, neighborhood overlap and pair frequency; exclude the current edge for training feature rows.',
        limits='Only candidate-pair co-occurrence, not semantic equivalence. Population isolation is normalized unordered pair identity, not entity-disjoint splitting; caller must meet any stronger application split requirements.'),
    'feature_ensemble': dict(
        anchors=[('text_pair_complete_recipe', 'fit_normalized'), ('text_pair_complete_recipe', 'CompleteFit.scores')],
        realization='Fit scaled logistic regression and a random forest over all thirteen features; combine ordered probability vectors with the reusable equal mean.',
        limits='Model family, equal weights and controls are explicit recipe policy. No benchmark accuracy claim.'),
    'metric_thresholding': dict(
        anchors=[('text_pair_complete_recipe', 'execute'), ('text_pair_complete_contract', 'prepare'),
                 ('text_pair_complete_contract', 'execute'), ('text_pair_complete_contract', 'execute_json')],
        realization='Select the listed relevance-score scaling branch: fit increasing isotonic regression on held-out raw scores and binary labels, bound to [0,1], clip outside calibration score range, then select an empirical binary-F1 threshold.',
        limits='Mapping and threshold share the calibration population; selection F1 is optimistic and not independent evaluation. Strict finite JSON distinguishes raw and scaled scores and emits ordered generic binary-relevance rows. No competition-specific format is claimed. Native stored-graph execution remains pending.'),
}


def anchor(module, symbol):
    path = ROOT / 'sciona' / (module + '.py')
    node = ast.parse(path.read_text())
    for part in symbol.split('.'):
        matches = [n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part]
        if len(matches) != 1:
            raise ValueError('Missing source anchor')
        node = matches[0]
    return dict(file=str(path.relative_to(ROOT)), symbol=symbol, line=node.lineno, file_sha256=sha(path))


def review():
    state = audit()
    if state['implementation_drift']:
        raise ValueError('Retained implementation changed')
    if set(STAGES) != {s['stage_id'] for s in state['source_stages']}:
        raise ValueError('Every original stage must be reviewed')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        candidates = db.execute('''SELECT a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier,
            artifact_is_publishable(a.artifact_id) AS native_eligible
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE v.is_latest AND a.artifact_kind='atom'
            AND a.fqdn ILIKE ANY(%s) ORDER BY a.fqdn''',
            (['%isotonic%', '%bm25%', '%clean_text%', '%levenshtein%'],)).fetchall()
    return dict(format='text-pair-original-scope.v1', read_only=True, catalog_mutations=0, approved=False,
        original=state['original'], source_version_id=state['source_version_id'],
        source_stage_review=[dict(source_stage=s,
            **{k: v for k, v in STAGES[s['stage_id']].items() if k != 'anchors'},
            source_anchors=[anchor(*item) for item in STAGES[s['stage_id']]['anchors']]) for s in state['source_stages']],
        additional_provider_catalog_candidates=candidates,
        reviewer_sha256=sha(ROOT / 'scripts/review_text_pair_original_scope.py'),
        remaining_requirements=[
            'Qualify the existing isotonic provider branch and source dependencies; a local registered implementation is not proof of catalog approval.',
            'Complete reusable primitive inventory and exact callable bindings, including BM25 and normalization dependencies.',
            'Qualify the strict finite serialization boundary in the native graph and resolve separate title/description field fidelity against the original source.',
            'Execute the full serialized/stored original graph and verify exact catalog interfaces, rollback, native eligibility and served selection.'],
        limits=['Scope mapping only; no full graph, submission format, provider approval or original-CDG approval claim.',
                'The binary relevance task is a declared realization of the generic topology, not every listed competition metric.'])


if __name__ == '__main__':
    result = review()
    (ROOT / 'docs/reviews/text_pair_original_scope.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(stages=len(result['source_stage_review']),
        provider_candidates=len(result['additional_provider_catalog_candidates']), approved=False)))
