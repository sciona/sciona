"""Review exact staged numerical providers for automated Tier 3 publication."""
import json
from scripts.stage_sparse_projection_providers import ROOT, PLAN, reviewed_plan, connect, sha

DESCRIPTIONS={
    'fit_sparse_projection':('Fit uncentered sparse randomized right singular vectors with explicit numerical controls.',
        'Learn a smaller set of numerical directions from a sparse table using the supplied fitting settings.'),
    'transform_sparse_projection':('Project sparse rows through supplied finite components without refitting.',
        'Express new rows using the supplied numerical directions, without learning new directions.'),
}
LIMITS=[
    'Automated Tier 3 only; no human Tier 1 review or usage-qualified Tier 2 designation.',
    'Float64 scipy CSR inputs only; dense inputs, other precisions and accelerators are not qualified.',
    'Sparse storage order and duplicates are retained. Canonicalizing TF-IDF storage changed downstream scores during investigation and is intentionally excluded.',
    'Explicit randomized range-finder controls and gesdd semantics apply; sign conventions match right-vector sklearn TruncatedSVD orientation. Degenerate subspaces and near-boundary predictions need separate qualification.',
    'Dense intermediate basis/projection and output factors can approach full matrix size. No constant-memory or arbitrary-rank scalability promise.',
    'Finite input can overflow intermediate arithmetic. Nonfinite factors/output reject; no universal numerical error bound is claimed.',
    'Unoptimized einsum avoids the reproduced layout-sensitive matmul warning in the tested basis lift. This is not a global replacement or a guarantee for all native operations.',
    'Caller owns training/query separation, feature ordering, fitted-factor provenance, numerical scale and aggregate concurrency.',
    'Concrete witnesses reject abstract execution; no symbolic correctness proof is claimed.',
    'Eighteen provider tests and six uncached native graph executions are source-pinned evidence. Current-platform dependency RECORD checks do not authenticate upstream publishers.',
    'Retained text-pair lifecycle comparison preserves predictions and calibration F1 exactly and scores/thresholds within documented tolerances; no historical accuracy or universal bitwise claim.',
    'This review does not approve the text-pair original or remedy its normalization, BM25 and output-scope gaps.',
]


def review():
    intake=reviewed_plan()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for record in intake['providers']:
            for prefix,key,status in [('atom','atom_id','flagged'),('artifact','artifact_id','draft')]:
                rows=db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                    FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''',
                    (record['atom_id'],)).fetchall()
                expected=dict(fqdn=record['catalog_fqdn'],status=status,is_publishable=False,
                    version_id=record['version_id'],content_hash=record['content_hash'],trust_tier=3)
                if rows!=[expected]:raise ValueError('Exact staged version required')
                ports=db.execute(f'''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
                    FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal''',(record['version_id'],)).fetchall()
                if ports!=record['interfaces']:raise ValueError('Exact staged interface required')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(record['version_id'],)).fetchone():
                raise ValueError('Unexpected registered provider dependency')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('failed','completed')",(record['version_id'],)).fetchone():
                raise ValueError('Failed completed audit exists')
            if db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s',(record['atom_id'],)).fetchone():
                raise ValueError('Unapproved provider unexpectedly served')
    providers=[]
    for record in intake['providers']:
        operation,plain=DESCRIPTIONS[record['runtime_fqdn'].rsplit('.',1)[1]]
        providers.append(dict(artifact_id=record['atom_id'],fqdn=record['catalog_fqdn'],
            runtime_fqdn=record['runtime_fqdn'],version_id=record['version_id'],content_hash=record['content_hash'],
            interfaces=record['interfaces'],operation=operation,plain_description=plain,
            validity_bounds=record['seed_fields']['description'],
            reuse='Generic numeric operation; domain populations and application policy are explicit caller responsibilities.',
            registered_provider_dependencies=[]))
    if reviewed_plan()!=intake:raise ValueError('Intake changed during review')
    return dict(format='sparse-projection-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=providers,
        intake_plan_sha256=sha(PLAN),graph_evidence_sha256=intake['review_sha256'],
        prior_provider_evidence_sha256=intake['prior_provider_review_sha256'],reviewer_sha256=sha(__file__),
        limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/sparse_projection_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),approved=False)))
