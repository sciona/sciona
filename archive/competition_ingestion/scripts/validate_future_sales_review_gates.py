"""Reject weakened Future Sales evidence without changing files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_future_sales_execution as review


def main():
    original=Path.read_text
    checks=[]
    def failure(suffix,name,mutate):
        target=f'competition_future_sales_{suffix}.json'
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
    failure('source_triage','false_scope',lambda d:d.__setitem__('source_scope','winning solution'))
    failure('runtime_tests','clipping_order',lambda d:d.__setitem__('aggregation_before_clipping',False))
    failure('runtime_tests','future_leakage',lambda d:d.__setitem__('current_and_future_target_feature_invariance',False))
    failure('runtime_tests','missing_tpe',lambda d:d.__setitem__('model_based_tpe_sampling_exercised',False))
    failure('worker_execution','worker_parity',lambda d:d.__setitem__('isolated_matches_direct',False))
    failure('graph_execution','missing_refit',lambda d:d['checks'].__setitem__('refit_rows',0))
    failure('graph_execution','wrong_provider',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('environment','wrong_dependency',lambda d:d['dependency_versions'].__setitem__('lightgbm','0.0.0'))
    failure('environment','missing_notice',lambda d:d.__setitem__('license_sha256',{}))
    assert review.audit()['verdict']=='acceptable_with_limits'
    paths=['scripts/review_future_sales_execution.py','scripts/validate_future_sales_review_gates.py']
    report=dict(format='future_sales-review-negative-gates.v1',result='passed',
        checks=dict(injected_failures_rejected=len(checks),unaltered_audit_passes=True),
        rejected_scenarios=checks,sha256={p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits='Read-only review injections; no catalog mutations.')
    (review.ROOT/'docs/reviews/competition_future_sales_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
