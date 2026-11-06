"""Review staged reusable providers for concrete automated Tier 3 publication."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from scripts.stage_text_pair_providers import (
    ROOT, PLAN, reviewed_plan, connect, sha, legacy_snapshot, served_snapshot, preserved,
)

CONTRACTS = {
    'isotonic_regression': dict(
        operation='Weighted monotone least-squares projection of an ordered numerical response vector.',
        plain='Adjust a sequence so it moves in one direction, keeping it as close as possible to the supplied values.',
        bounds='Nonempty finite float64 response vector. Optional aligned finite strictly positive weights. Increasing is a boolean. Optional finite lower/upper bounds must be ordered. Uses SciPy isotonic regression then clips to bounds; returns an owned vector in input order. Caller defines the order and numerical scale; no sorting by a separate predictor occurs.',
        dependencies=[]),
    'isotonic_regression_fit': dict(
        operation='Fit a piecewise-linear monotone mapping from numerical predictors to responses.',
        plain='Learn a curve that only rises or only falls, while allowing repeated input values and weighted observations.',
        bounds='Nonempty finite float64 X vector or single-column matrix and aligned finite float64 response vector. Optional aligned finite nonnegative weights with at least one positive value; zero-weight rows are excluded. Duplicate predictors are aggregated by weighted mean. Increasing is True, False or auto; auto requires a defined nonconstant rank correlation and may warn when direction is uncertain. Optional finite response bounds must be ordered. Out-of-bounds policy is nan, clip or raise. Returned threshold arrays are owned but mutable; callers must preserve them. No probability or domain interpretation is inherent.',
        dependencies=['isotonic_regression']),
    'isotonic_regression_transform': dict(
        operation='Interpolate numerical queries through supplied fitted isotonic thresholds without refitting.',
        plain='Apply a previously learned rising or falling curve to new input values.',
        bounds='Nonempty finite float64 query vector or single-column matrix and a valid state from the qualified fit provider. Threshold arrays must remain aligned, finite and sorted with strictly increasing X (or one threshold), monotone responses, and consistent endpoint bounds and policy. Multiple-threshold states clip, raise or return NaN outside the fitted range according to policy. A single-threshold state returns its constant response for every finite query, regardless of policy. No fitting or state mutation. Caller owns state provenance and population isolation; arbitrary fabricated states are not qualified.',
        dependencies=[]),
    'clean_text': dict(
        operation='Apply explicitly selected deterministic cleanup operations to one text string.',
        plain='Clean one text using chosen steps and, when requested, a caller-supplied spelling vocabulary.',
        bounds='String input and nonempty list drawn from html, url, unicode, lower and spell. Operations execute in that fixed order, irrespective of list order; duplicates do not repeat work. Unicode uses NFKC and lower uses lower(), not casefold(). Nonprintable non-whitespace characters are removed; whitespace is collapsed and stripped. Spell requires a nonempty string-key vocabulary of nonnegative integer counts, excluding booleans. ASCII alphabetic tokens use known original, then one-edit, then two-edit candidates; ties prefer higher count, shorter word, then lexicographically greatest candidate. Unknown tokens become lowercase. Vocabulary is not learned or modified. HTML/URL handling is regex cleanup, not a security sanitizer. Two-edit candidate generation may be expensive; caller bounds input size.',
        dependencies=[]),
    'levenshtein': dict(
        operation='Compute unit-cost Levenshtein distance between two strings.',
        plain='Count the fewest single-character insertions, deletions and replacements needed to turn one string into another.',
        bounds='Two Python strings, including empty strings. Distance compares Python code points without Unicode normalization; transposition is not a single edit. Returns a nonnegative integer no greater than the longer length. Dynamic programming uses O(m*n) time and O(min(m,n)) auxiliary space. Caller owns text normalization and practical size limits.',
        dependencies=[]),
    'score_bm25': dict(
        operation='Compute a BM25 score from aligned term and corpus counts with explicit numerical controls.',
        plain='Score a count vector using supplied frequency and length statistics, without choosing or learning those statistics.',
        bounds='Aligned count lists for distinct unit-weight query terms; Python integer counts in [0,2**53], positive corpus size, document frequencies at most corpus size, and total supplied term frequency at most document length. Positive finite average length and k1 at most 2**53; b in [0,1]. Uses positive log1p IDF and the explicit (k1+1) saturation scale. Zero frequencies contribute zero; empty query and empty document score zero when arithmetic controls are representable. Rejects nonfinite intermediate/output arithmetic. No text, token alignment, corpus fitting, labels or split policy is inferred.',
        dependencies=[]),
}


def review():
    intake = reviewed_plan()
    by_symbol = {r['seed_fields']['source_symbol']: r for r in intake['providers']}
    if set(by_symbol) != set(CONTRACTS):
        raise ValueError('Every staged provider needs explicit semantic review')
    test = 'tests/test_text_pair_provider_semantics.py'
    before = {test: sha(ROOT/test), 'scripts/review_text_pair_provider_semantics.py': sha(__file__)}
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
    if counts != dict(tests=8,failures=0,errors=0,skipped=0):
        raise ValueError('Complete semantic cases required')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        if (preserved(db)!=intake['preserved_sha256'] or legacy_snapshot(db,intake)!=intake['legacy_sha256']
                or served_snapshot(db,intake)!=intake['served_sha256']):
            raise ValueError('Existing legacy/source/served state changed')
        for r in intake['providers']:
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
    providers=[]
    for r in intake['providers']:
        contract=CONTRACTS[r['seed_fields']['source_symbol']]
        providers.append(dict(artifact_id=r['atom_id'],fqdn=r['catalog_fqdn'],runtime_fqdn=r['runtime_fqdn'],
            version_id=r['version_id'],content_hash=r['content_hash'],legacy_existing=r['legacy_existing'],
            intake_interfaces=r['interfaces'],interfaces=[dict(p,constraints=contract['bounds']) for p in r['interfaces']],
            operation=contract['operation'],plain_description=contract['plain'],validity_bounds=contract['bounds'],
            reuse='Reusable primitive; application semantics, populations and model policies remain caller-owned.',
            registered_provider_dependencies=[dict(artifact_id=by_symbol[d]['atom_id'],version_id=by_symbol[d]['version_id'],
                content_hash=by_symbol[d]['content_hash']) for d in contract['dependencies']]))
    if reviewed_plan()!=intake or before!={p:sha(ROOT/p) for p in before}:
        raise ValueError('Qualified source/runtime changed during review')
    return dict(format='text-pair-provider-semantics.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,providers=providers,
        tests=counts,semantic_source_sha256=before,intake_plan_sha256=sha(PLAN),
        graph_evidence_sha256=intake['review_sha256'],prior_provider_evidence_sha256=intake['prior_provider_review_sha256'],
        reviewer_sha256=sha(__file__),limitations=[
            'Automated Tier 3 only; no human Tier 1 review or Tier 2 designation.',
            'Validity bounds are caller contracts, not a claim that every invalid input or fabricated state is rejected.',
            'Float64 numeric use is qualified; optional automatic monotonic direction has its stated statistical limitations.',
            'Finite numerical inputs can still overflow; no universal error bound or calibration/accuracy guarantee.',
            'Concrete runtime evidence does not prove existing text/isotonic witnesses value-preserving; abstract execution is not qualified.',
            '115 prior tests, 12 native graph executions and eight semantic cases are source-bound current-platform evidence.',
            'Semantic metadata must replace draft placeholder constraints; fit must record its exact isotonic dependency before activation.',
            'Legacy metadata/state differs from new canonical drafts and must be handled explicitly during publication; no identity replacement.',
            'This review alone does not approve these providers or the full original text-pair CDG.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/text_pair_provider_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),tests=result['tests'],approved=False)))
