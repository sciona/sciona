"""Reject weakened Connectomics evidence without changing files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_cornell_execution as review


def main():
    original=Path.read_text
    checks=[]
    def failure(suffix,name,mutate):
        target=f'competition_cornell_{suffix}.json'
        def changed(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path.name==target:
                document=json.loads(text);mutate(document);return json.dumps(document)
            return text
        with patch.object(Path,'read_text',changed):
            try:review.audit()
            except AssertionError:checks.append(name);return
        raise AssertionError('Weakened evidence accepted: '+name)
    failure('source_pins','wrong_commit',lambda d:d.__setitem__('commit','0'*40))
    failure('source_pins','wrong_license',lambda d:d['license'].__setitem__('spdx_id','UNKNOWN'))
    failure('loss_execution','wrong_gradient',lambda d:d['cases'][0].__setitem__('scalar_and_gradient_match',False))
    failure('population_execution','wrong_sampling',lambda d:d.__setitem__('exact_source_indices',False))
    failure('augmentation_runtime_execution','missing_resume',lambda d:d['cases'][0].__setitem__('exact_rng_resume',False))
    failure('schedule_execution','wrong_schedule',lambda d:d['cases'][0].__setitem__('max_abs_error',1.))
    failure('lifecycle_execution','reduced_models',lambda d:d.__setitem__('models_completed',1))
    failure('lifecycle_execution','missing_continuations',lambda d:d.__setitem__('continuation_phases',0))
    failure('graph_execution','wrong_provider',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('environment','wrong_dependency',lambda d:d['direct_runtime_versions'].__setitem__('numpy','0.0.0'))
    assert review.audit()['verdict']=='acceptable_with_limits'
    paths=['scripts/review_cornell_execution.py','scripts/validate_cornell_review_gates.py']
    report=dict(format='cornell-review-negative-gates.v1',result='passed',
        checks=dict(injected_failures_rejected=len(checks),unaltered_audit_passes=True),
        rejected_scenarios=checks,sha256={p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits='Read-only review injections; no catalog mutations.')
    (review.ROOT/'docs/reviews/competition_cornell_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
