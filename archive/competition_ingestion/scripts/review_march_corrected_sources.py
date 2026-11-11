"""Qualify all original March stages against the complete stored reconstruction."""
import json
import subprocess
from sciona.competition_graph import load_competition_graph
from scripts import stage_march_corrected_intake as intake
from scripts.review_march_decomposed_providers import review as provider_review
from scripts.validate_decomposed_march_baseline import ROOT, sha, runtime


def review():
    plan,providers=intake.qualified_plan()
    historical='docs/reviews/competition_march_semantic_review.json'
    raw=subprocess.check_output(['git','show','ce82837e:'+historical],cwd=ROOT)
    if (ROOT/historical).read_bytes()!=raw: raise ValueError('Historical source review changed')
    old=json.loads(raw)
    if old['source_version_id']!=intake.ORIGINAL or old['source_hash']!=intake.ORIGINAL_HASH:
        raise ValueError('Historical source identity differs')
    for section,base in [('evidence_sha256',ROOT/'docs/reviews'),('auxiliary_sha256',ROOT)]:
        if any(sha(base/p)!=h for p,h in old[section].items()): raise ValueError('Historical evidence changed')
    with intake.connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        graph=intake.check_candidate(db,plan,providers)
        source=load_competition_graph(db,version_id=intake.ORIGINAL)
    descriptions={n.node_id:n.description for n in source.nodes}
    if set(descriptions)!=set(intake.CORRECTIONS): raise ValueError('Incomplete original stage coverage')
    phrases={'season_aggregation':'points per possession','graph_centralities':'PageRank',
             'elo_ratings':'chronologically','pairwise_concatenation':'every possible tournament matchup',
             'logistic_regression':'Logistic Regression model (or XGBoost'}
    if any(p not in descriptions[k] for k,p in phrases.items()): raise ValueError('Source computation meaning changed')
    if graph.metadata['source_corrections']!=intake.CORRECTIONS: raise ValueError('Stored source mapping differs')
    qualification=provider_review()
    expected=json.loads((ROOT/'docs/reviews/march_decomposed_provider_semantics.json').read_text())
    if qualification!=expected: raise ValueError('Current full implementation qualification changed')
    executed=json.loads((ROOT/'docs/reviews/march_corrected_intake_execution.json').read_text())
    if (executed['passed'] is not True or executed['stored_catalog_graph_executed'] is not True
            or executed['version_id']!=plan['version_id'] or executed['artifact_id']!=intake.IDENTITY
            or executed['serialized_graph_sha256']!=plan['graph_sha256']
            or (executed['nodes'],executed['edges'],executed['distinct_providers'])!=(10,16,9)
            or executed['runtime']!=runtime()): raise ValueError('Exact stored execution required')
    if executed['cases']!=[dict(seed=s,elo_scale=scale,nodes_executed=10,all_uncached=True,
            probability_bytes_equal=True,full_result_equal=True,nontrivial_orientation_predictions=True,
            predicted_matchups=6,all_ordered_pairs=True,orientation_symmetry=True)
            for s,scale in [(0,200.),(3,400.),(11,800.)]]:
        raise ValueError('Complete all-pairs execution evidence required')
    for section in ('code_sha256','evidence_sha256'):
        if any(sha(ROOT/p)!=h for p,h in executed[section].items()): raise ValueError('Stored execution evidence changed')
    paths=['docs/reviews/march_corrected_intake_execution.json','docs/reviews/march_corrected_intake_plan.json',
        'docs/reviews/march_approved_provider_state.json','docs/reviews/march_decomposed_provider_semantics.json',historical]
    return dict(format='march-corrected-source-review.v1',passed=True,approved=False,proposed_tier=3,
        verdict='acceptable_with_limits',catalog_mutations=0,source_version_id=intake.ORIGINAL,
        source_content_hash=intake.ORIGINAL_HASH,candidate_version_id=plan['version_id'],graph_sha256=plan['graph_sha256'],
        original_stages=5,source_stage_mapping=intake.CORRECTIONS,
        tests=qualification['tests'],complete_stored_cases=3,all_ordered_matchups_per_case=6,
        runtime=executed['runtime'],historical_runtime=old['dependencies'],
        evidence_sha256={p:sha(ROOT/p) for p in paths},reviewer_sha256=sha(__file__),
        limitations=qualification['limitations']+[
            'The original intake has only a generic benchmark description and Kaggle homepage citation; no historical winner or source-code parity is claimed.',
            'The explicitly permitted logistic branch is fully implemented. XGBoost is an alternative branch, not an omitted mandatory stage.',
            'PageRank remains a declared graph feature; the intake claim of true strength of schedule is unsupported and explicitly corrected.',
            'Prediction rows are caller-enumerated. To evaluate every possible matchup, callers must provide all desired ordered pairs; the stored execution exercises all six pairs among three synthetic teams. The implementation does not infer tournament membership or silently add rows.',
            'Current NumPy is 2.2.6 rather than historical 2.4.2. Current active requirements, actual imports, installed files and fresh full-reference comparisons pass; historical environment equivalence is not claimed.',
            'Preparation validates only root fields, finite JSON and nonempty populations. Downstream stages enforce detailed numerical and chronological contracts.',
            'Original history and derivative are preserved. Native publication, atomic activation and served selection remain separate gates.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/march_corrected_source_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,original_stages=5,tests=result['tests'],complete_stored_cases=3,approved=False)))
