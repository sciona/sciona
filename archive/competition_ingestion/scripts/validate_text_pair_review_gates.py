"""Reject weakened Text-pair evidence without changing files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_text_pair_execution as review


def main():
    original=Path.read_text
    checks=[]
    def failure(suffix,name,mutate):
        target=f'competition_text_pair_{suffix}.json'
        def changed(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path.name==target:
                document=json.loads(text);mutate(document);return json.dumps(document)
            return text
        with patch.object(Path,'read_text',changed):
            try:review.audit()
            except AssertionError:checks.append(name);return
        raise AssertionError('Weakened evidence accepted: '+name)
    failure('source_triage','wrong_source',lambda d:d.__setitem__('source_content_hash','0'*64))
    failure('source_triage','false_historical_scope',lambda d:d.__setitem__('source_scope','winning solution'))
    failure('runtime_tests','missing_boundary_tests',lambda d:d.__setitem__('serialized_boundary_tests',0))
    failure('runtime_tests','calibration_leakage',lambda d:d['checks'].__setitem__('calibration_labels_do_not_change_base_scores',False))
    failure('runtime_tests','missing_threshold_validation',lambda d:d['checks'].__setitem__('independent_f1_search',False))
    failure('runtime_tests','query_population_leakage',lambda d:d['checks'].__setitem__('query_population_independence',False))
    failure('graph_execution','missing_forest',lambda d:d['checks'].__setitem__('forest_trees',0))
    failure('graph_execution','wrong_provider',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('environment','wrong_dependency',lambda d:d['dependency_versions'].__setitem__('scikit-learn','0.0.0'))
    failure('environment','missing_notices',lambda d:d.__setitem__('license_sha256',{}))
    assert review.audit()['verdict']=='acceptable_with_limits'
    paths=['scripts/review_text_pair_execution.py','scripts/validate_text_pair_review_gates.py']
    report=dict(format='text_pair-review-negative-gates.v1',result='passed',
        checks=dict(injected_failures_rejected=len(checks),unaltered_audit_passes=True),
        rejected_scenarios=checks,sha256={p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits='Read-only review injections; no catalog mutations.')
    (review.ROOT/'docs/reviews/competition_text_pair_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
