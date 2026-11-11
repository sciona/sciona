"""Reject weakened uncertainty evidence using read-only in-memory injections."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from scripts import review_m5u_execution as review


def main():
    assert review.audit()['verdict']=='acceptable_with_limits'
    original=Path.read_text;checks=[]
    def failure(suffix,name,mutate):
        target=f'competition_m5_uncertainty_{suffix}.json'
        def changed(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path.name==target:
                document=json.loads(text);mutate(document);return json.dumps(document)
            return text
        with patch.object(Path,'read_text',changed):
            try:review.audit()
            except AssertionError:checks.append(name);return
        raise AssertionError('Weakened evidence accepted: '+name)
    failure('execution_source_review','wrong_source',lambda d:d.__setitem__('source_content_hash','0'*64))
    failure('execution_source_review','false_historical_scope',lambda d:d.__setitem__('scope','exact winner'))
    failure('execution_source_review','unsupported_distribution_recipe',lambda d:d['stage_mapping'].__setitem__('probability_distribution_fitting','fit a Gaussian'))
    failure('training_parameters','missing_source_schedule_comparison',lambda d:d['checks'].__setitem__('source_schedule_comparisons',0))
    failure('component_tests','missing_tests',lambda d:d['checks'].__setitem__('tests_passed',0))
    failure('training_schedule','reduced_training_limits',lambda d:d['levels']['1'].__setitem__('iterations',1))
    failure('full_pipeline','reduced_forecast_budget',lambda d:d['checks'].__setitem__('query_rows',1))
    failure('normalization_correction','undisclosed_correction',lambda d:d.__setitem__('policy','exact_source'))
    failure('graph_execution','missing_models',lambda d:d['checks'].__setitem__('models',1))
    failure('graph_execution','wrong_provider',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('environment','wrong_dependency',lambda d:d['dependency_versions'].__setitem__('lightgbm','0.0.0'))
    failure('environment','missing_notices',lambda d:d.__setitem__('license_sha256',{}))
    assert len(checks)==12 and review.audit()['verdict']=='acceptable_with_limits'
    paths=['scripts/review_m5u_execution.py','scripts/validate_m5u_review_gates.py']
    report=dict(format='m5u-review-negative-gates.v1',result='passed',catalog_mutations=0,
                checks=dict(injected_failures_rejected=12,unaltered_audit_passes=True),rejected_scenarios=checks,
                sha256={p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths},
                limits='In-memory read-only review injections; no catalog publication approval.')
    (review.ROOT/'docs/reviews/competition_m5_uncertainty_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
