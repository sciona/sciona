"""Review exact staged numerical providers for automated Tier 3 publication."""
import json
from scripts.stage_video_numeric_providers import ROOT, PLAN, reviewed_plan, connect, sha

DESCRIPTIONS={
    'apply_column_thresholds':('Apply inclusive thresholds independently to score columns.','Turn each score into a yes or no using the supplied cutoff.'),
    'fit_empirical_f1_thresholds':('Fit independent empirical F1 thresholds with highest-threshold ties.','Choose a cutoff for each output using labeled examples supplied by the caller.'),
    'masked_attention_pool':('Reduce tensor sequences with masked softmax weights while preserving differentiation.','Combine valid entries using supplied importance scores and ignore padded entries.'),
    'sigmoid_context_gate':('Multiply a tensor matrix by sigmoid-transformed aligned gate logits.','Adjust each value using a supplied control score.'),
    'fit_numeric_vocabulary':('Fit sorted numeric feature vocabulary from an iterable and return sparse CSR rows.','Learn the feature names from supplied records and arrange their values in columns.'),
    'transform_numeric_vocabulary':('Transform numeric mapping iterables using a frozen sparse vocabulary.','Arrange new records using the learned columns, ignoring new feature names.'),
    'fit_sparse_binary_logistic':('Fit sparse binary L2 logistic regression with explicit LBFGS controls.','Learn to score two possible outcomes from labeled examples and supplied fitting settings.'),
    'predict_sparse_binary_probability':('Predict class-one probabilities from a fitted sparse binary logistic model.','Score new rows using an already fitted model, without needing their answers.'),
}
LIMITS=[
    'Automated Tier 3 only; no human Tier 1 review or usage-qualified Tier 2 designation.',
    'Exact per-operation shape, dtype, finite-value and fitted-state contracts apply; no arbitrary tensor or estimator compatibility is claimed.',
    'Training/calibration/query isolation, label-column ordering, model construction and optimization schedules remain caller/recipe responsibilities.',
    'Empirical F1 fitting requires both classes per column and does not calibrate probabilities or establish unbiased performance.',
    'Tensor tests cover CPU float64 differentiation and masking; other devices and precisions are not runtime-qualified by this evidence.',
    'Sparse vocabulary and logistic features remain CSR; output matrices, vocabulary and dense fitted coefficients consume resident memory.',
    'Logistic nonconvergence raises; fitting uses current installed sklearn semantics and does not promise historical or cross-platform bitwise reproduction.',
    'Concrete witnesses reject abstract execution; no symbolic correctness proof is claimed.',
    'Installed RECORD consistency verifies current package files, not publisher authenticity or clean-platform deployment.',
    'Synthetic numerical and complete lifecycle parity do not establish historical competition accuracy.',
    'Serial validation uses one native thread; callers remain responsible for aggregate concurrency.',
    'This review does not publish the recipe boundary or original CDG; those require separate contracts and publication evidence.',
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
    return dict(format='video-numeric-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=providers,
        intake_plan_sha256=sha(PLAN),graph_evidence_sha256=intake['review_sha256'],
        runtime_evidence_sha256=intake['runtime_review_sha256'],reviewer_sha256=sha(__file__),
        limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/video_numeric_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),approved=False)))
