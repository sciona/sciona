"""Review staged separate-field recipe adapters for automated Tier 3 publication."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from scripts.stage_text_pair_complete_adapters import (
    ROOT, PLAN, reviewed_plan, connect, sha, legacy_snapshot, served_snapshot, preserved,
)

# Recipe adapters intentionally retain policy; numerical providers are separate.
FEATURE_DEPS = ['levenshtein', 'score_bm25', 'transform_sparse_projection']
CONTRACTS = {
    'prepare_fields': dict(
        operation='Validate and copy a separate-field binary relevance request.',
        plain='Check the input rows and settings before fitting anything, keeping titles and descriptions separate.',
        bounds='Exact finite JSON-compatible version-2 request with binary_relevance task and title_description role. Nonempty training, calibration and query record pairs; each record has exactly string title and description fields. Training and calibration labels are integer 0/1 with both classes. Explicit vocabulary, domain substitutions and synonyms; explicit projection, BM25 and forest controls. Text limits and normalized full-record population collision rules apply. No entity-disjointness guarantee. Deep-copies the request; caller must not mutate the resulting payload before the next stage.',
        dependencies=['clean_text']),
    'normalize_fields': dict(
        operation='Normalize disjoint title/description populations using the declared frozen policy.',
        plain='Apply the chosen text rules to each field and reject repeated or overlapping training and calibration pairs.',
        bounds='Unmodified PreparedCompletePairs from the qualified prepare stage. Revalidates and copies the request. NFKC/casefold, spelling, tokenization, one-pass domain substitution and synonym expansion are explicit policy; titles and descriptions are never concatenated. Train/calibration unordered full-record pairs must be unique and all three populations disjoint; repeated query rows are allowed. Output holds copied payload and normalized rows in original order. Vocabulary and mapping adequacy remain caller responsibilities.',
        dependencies=['clean_text']),
    'fit_features': dict(
        operation='Fit training-only description representations and full-record pair context.',
        plain='Learn text features from training rows while keeping later rows out of the fitted vocabulary and statistics.',
        bounds='Unmodified Normalized state from the qualified normalization stage, with valid nonempty training rows and feasible projection controls. Fits unique training descriptions using word unigram/bigram TF-IDF and uncentered randomized sparse projection, and training-only BM25 corpus statistics. Builds candidate co-occurrence context from full title/description entities. Does not read labels, calibration or query text to fit. Returns execution-local mutable feature state; memory/runtime depend on caller population and vocabulary size.',
        dependencies=['fit_sparse_projection']),
    'fit_models': dict(
        operation='Fit the declared two-model binary relevance ensemble on thirteen training features.',
        plain='Train two models using only training labels and training features.',
        bounds='Unmodified Features state from the qualified feature-fit stage. Thirteen features retain separate title edit and description similarities; current training pair occurrence is excluded from graph context. Fit standardized logistic regression with max_iter=1000 and a random forest with explicit trees/depth/leaf/seed controls and n_jobs=1. Native thread pools are bounded to one. Convergence warnings fail execution. Caller owns the choice of model families and feature policy; calibration labels and query rows do not fit these models.',
        dependencies=FEATURE_DEPS),
    'predict': dict(
        operation='Fit calibration score mapping and empirical F1 cutoff, then predict ordered query rows.',
        plain='Use held-out labels to choose a rising score curve and decision cutoff, then apply both to new rows.',
        bounds='Unmodified Models state from the qualified model-fit stage. Average two class-1 score vectors equally. Fit increasing isotonic mapping bounded to [0,1], clipping outside calibration score range; then select the highest observed-score cutoff among equal empirical binary F1 values. Apply score >= cutoff to query rows without refitting text features or base models. Mapping and cutoff use the same calibration rows, so reported selection F1 is not independent accuracy. Returns raw and scaled scores, decisions and selection metrics; no probability-calibration guarantee.',
        dependencies=FEATURE_DEPS+['average', 'isotonic_regression_fit', 'isotonic_regression',
            'isotonic_regression_transform', 'fit_empirical_f1_thresholds', 'apply_column_thresholds']),
    'format_rows': dict(
        operation='Validate and emit finite ordered generic relevance rows and selection metrics.',
        plain='Return one result per query row, preserving order and repeated rows.',
        bounds='Unmodified Predictions state from the qualified prediction stage. Checks equal score/decision lengths, query count, finite JSON values, score ranges and inclusive-threshold decisions. Returns version-2 binary_relevance/title_description output with ordinal row indices, raw and scaled scores, binary decisions, cutoff and selection F1. This is a generic result table; competition-specific file formats and external identifiers remain caller-owned. Arbitrary fabricated intermediate records are not qualified.',
        dependencies=[]),
}


def review():
    intake = reviewed_plan()
    by_symbol = {r['seed_fields']['source_symbol']: r for r in intake['providers']}
    if set(by_symbol) != set(CONTRACTS):
        raise ValueError('Every staged provider needs explicit semantic review')
    test = 'tests/test_text_pair_complete_semantics.py'
    before = {test: sha(ROOT/test), 'scripts/review_text_pair_complete_adapter_semantics.py': sha(__file__)}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='synthetic-provider-semantics-') as temp:
        xml = Path(temp)/'tests.xml'
        run = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', test,
            '-W', 'error::RuntimeWarning', '--junitxml='+str(xml)], cwd=ROOT, env=env, capture_output=True, text=True)
        if run.returncode:
            raise ValueError('Semantic checks failed: '+run.stdout[-4000:]+run.stderr[-1000:])
        counts = {k: sum(int(s.get(k,'0')) for s in ET.parse(xml).getroot().iter('testsuite'))
                  for k in ('tests','failures','errors','skipped')}
    if counts != dict(tests=3,failures=0,errors=0,skipped=0):
        raise ValueError('Complete semantic cases required')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if (preserved(db)!=intake['preserved_sha256'] or legacy_snapshot(db,intake)!=intake['legacy_sha256']
                or served_snapshot(db,intake)!=intake['served_sha256']):
            raise ValueError('Existing legacy/source/served state changed')
        for r in intake['providers']:
            legacy=db.execute('''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier,
                a.import_module,a.source_symbol FROM atoms a JOIN atom_versions v USING(atom_id)
                WHERE a.atom_id=%s AND v.is_latest''',(r['atom_id'],)).fetchall()
            module,symbol=r['runtime_fqdn'].rsplit('.',1)
            if legacy != [dict(fqdn=r['catalog_fqdn'],status='flagged',is_publishable=False,
                    version_id=r['version_id'],content_hash=r['content_hash'],trust_tier=3,
                    import_module=module,source_symbol=symbol)]:
                raise ValueError('Exact staged legacy source required')
            legacy_ports=db.execute('''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
                FROM atom_io_specs WHERE version_id=%s ORDER BY direction,ordinal''',(r['version_id'],)).fetchall()
            if legacy_ports != r['interfaces']:
                raise ValueError('Exact staged legacy interface required')
            rows=db.execute('''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                FROM artifacts a JOIN artifact_versions v USING(artifact_id) WHERE a.artifact_id=%s AND v.is_latest''',
                (r['atom_id'],)).fetchall()
            if rows != [dict(fqdn=r['catalog_fqdn'],status='draft',is_publishable=False,
                    version_id=r['version_id'],content_hash=r['content_hash'],trust_tier=3)]:
                raise ValueError('Exact staged canonical version required')
            ports=db.execute('''SELECT direction,name,ordinal,type_desc,required,default_value_repr,constraints
                FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal''',(r['version_id'],)).fetchall()
            if ports != r['interfaces']:
                raise ValueError('Exact staged interface required')
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s',(r['version_id'],)).fetchone():
                raise ValueError('Unexpected staged dependency')
            if db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND passed=false AND status IN ('failed','completed')",(r['version_id'],)).fetchone():
                raise ValueError('Failed completed evidence exists')
    dependencies={r['runtime_fqdn'].rsplit('.',1)[-1]:r for r in intake['approved_dependencies']}
    providers=[]
    for r in intake['providers']:
        contract=CONTRACTS[r['seed_fields']['source_symbol']]
        providers.append(dict(artifact_id=r['atom_id'],fqdn=r['catalog_fqdn'],runtime_fqdn=r['runtime_fqdn'],
            version_id=r['version_id'],content_hash=r['content_hash'],legacy_existing=r['legacy_existing'],
            intake_interfaces=r['interfaces'],interfaces=[dict(p,constraints=contract['bounds']) for p in r['interfaces']],
            operation=contract['operation'],plain_description=contract['plain'],validity_bounds=contract['bounds'],
            reuse='Reusable separate-field binary relevance recipe stage across applications with compatible contracts; explicit text, population and model policies are not universal numerical semantics.',
            registered_provider_dependencies=[dependencies[d] for d in contract['dependencies']]))
    if reviewed_plan()!=intake or before!={p:sha(ROOT/p) for p in before}:
        raise ValueError('Qualified source/runtime changed during review')
    return dict(format='text-pair-complete-adapter-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=providers,
        tests=counts,semantic_source_sha256=before,intake_plan_sha256=sha(PLAN),
        graph_evidence_sha256=intake['review_sha256'],
        reviewer_sha256=sha(__file__),limitations=[
            'Automated Tier 3 only; no human Tier 1 review or Tier 2 designation.',
            'Recipe adapters carry explicit policy; independently reusable numerical/text primitives retain separate catalog identities.',
            'Concrete execution-local state only; arbitrary fabricated or concurrently mutated intermediate states are unqualified.',
            'Bounds are caller contracts, not a promise to reject every invalid object or finite arithmetic overflow.',
            'Full-record pair-disjoint populations are required; stronger entity/domain separation remains caller-owned.',
            'No named competition accuracy, probability calibration, submission-file compatibility or abstract-witness equivalence claim.',
            'Three semantic cases supplement the source-bound 115-test recipe qualification and three native graph executions.',
            'Exact registered dependency versions and reviewed metadata must be installed before activation.',
            'No adapter or original-CDG approval is performed by this review.'])



if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/text_pair_complete_adapter_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),tests=result['tests'],approved=False)))
