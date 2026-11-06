"""Read-only catalog and inspected-source evidence for remaining sparse reuse."""
import hashlib
import json
from scripts.import_flavours_provider_drafts import ROOT, connect

CANDIDATES={
    'sciona.atoms.ml.model_selection.materialized_prediction':
        'Reuse average per label column: finite ordered vectors, equal weights, sum then divide. Recipe owns column alignment.',
    'sciona.atoms.ml.sklearn.feature_extraction.atoms':
        'dict_vectorizer_fit requires materialized records; transform allocates dense rows by vocabulary. Not a drop-in for iterable sparse CSR execution.',
    'sciona.atoms.ml.tabular.supervised_classification.atoms':
        'Logistic fit adds DataFrame preprocessing and fixed controls; prediction additionally requires targets. Does not preserve the existing sparse fit/predict boundary.',
    'sciona.atoms.ml.sklearn.linear_model.logistic_public_api_shell.atoms':
        'Public fit/prediction payload atoms validate dense arrays and package calls without executing the estimator. Not a fitted sparse estimator implementation.',
}


def review():
    paths={module:'../sciona-atoms-ml/src/'+module.replace('.','/')+'.py' for module in CANDIDATES}
    paths['mean_implementation']='sciona/materialized_prediction.py'
    paths['recipe']='sciona/video_multilabel_sharded_training.py'
    paths['reviewer']='scripts/review_video_sparse_reuse.py'
    def fingerprints():
        return {path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths.values()}
    before=fingerprints()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows=db.execute('''SELECT a.fqdn,l.import_module,l.source_symbol,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier,
            EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
            FROM artifacts a JOIN atoms l ON l.atom_id=a.artifact_id
            LEFT JOIN artifact_versions v ON v.artifact_id=a.artifact_id AND v.is_latest
            WHERE a.artifact_kind='atom' AND l.import_module=ANY(%s)
            ORDER BY a.fqdn,v.version_id''',(list(CANDIDATES),)).fetchall()
    if before!=fingerprints():raise ValueError('Inspected sources changed')
    return dict(format='video-sparse-reuse.v1',read_only=True,catalog_mutations=0,approved=False,
        inspected_candidates=CANDIDATES,catalog_candidates=rows,software_sha256=before,
        decision='Reuse registered vector mean columnwise. Continue inventory before implementing iterable CSR vocabulary and fitted sparse binary logistic boundaries.',
        limits=['Targeted source/catalog review, not an exhaustive proof that no alternative provider exists.',
                'Catalog status alone does not establish current source or dependency qualification.',
                'No training data, runtime paths or dataset metadata are included.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/video_sparse_reuse.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(candidate_records=len(result['catalog_candidates']),catalog_mutations=0)))
