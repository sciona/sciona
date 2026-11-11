"""Reject weakened OpenVaccine evidence without changing files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_openvaccine_execution as review


def main():
    original=Path.read_text
    checks=[]
    def failure(suffix,name,mutate):
        target=f'competition_openvaccine_{suffix}.json'
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
    failure('runtime','wrong_source_output',lambda d:d['families'][0].__setitem__('direct_source_output_exact',False))
    failure('tensorflow_loss','masked_gradient',lambda d:d.__setitem__('masked_gradients_zero',False))
    failure('sections','exception_state_loss',lambda d:d.__setitem__('post_update_exception_restores_model_optimizer',False))
    failure('ensemble','missing_clipping_fix',lambda d:d.__setitem__('symmetric_clipping_order_comparison',False))
    failure('training','missing_masked_validation',lambda d:d.__setitem__('partially_observed_validation',False))
    failure('teacher','missing_member',lambda d:d.__setitem__('trained_checkpoint_members',19))
    failure('folding_engine','wrong_binary',lambda d:d.__setitem__('binary_sha256','0'*64))
    failure('eternafold_parameters','wrong_parameters',lambda d:d.__setitem__('sha256','0'*64))
    failure('graph_execution','wrong_provider',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('graph_execution','reduced_models',lambda d:d['checks'].__setitem__('models',1))
    failure('environment','wrong_dependency',lambda d:d['dependency_versions'].__setitem__('numpy','0.0.0'))
    assert review.audit()['verdict']=='acceptable_with_limits'
    paths=['scripts/review_openvaccine_execution.py','scripts/validate_openvaccine_review_gates.py']
    report=dict(format='openvaccine-review-negative-gates.v1',result='passed',
        checks=dict(injected_failures_rejected=len(checks),unaltered_audit_passes=True),
        rejected_scenarios=checks,sha256={p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits='Read-only review injections; no catalog mutations.')
    (review.ROOT/'docs/reviews/competition_openvaccine_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
