"""Injected audit failures without mutating source, evidence or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_adversarial_execution as review


def main():
    original=Path.read_text
    def failure(suffix,mutate):
        target='competition_adversarial_'+suffix+'.json'
        def changed(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path.name==target:
                obj=json.loads(text);mutate(obj);return json.dumps(obj)
            return text
        with patch.object(Path,'read_text',changed):
            try:review.audit()
            except AssertionError:return
        raise AssertionError('weakened evidence accepted')
    failure('source_pins',lambda d:d.__setitem__('intake_content_hash','0'*64))
    failure('losses',lambda d:d.__setitem__('result','failed'))
    failure('lifecycle',lambda d:d['checks']['targeted_large'].__setitem__('complete_iterations',1))
    failure('graph_execution',lambda d:d['checks'].__setitem__('actual_full_models',1))
    failure('graph_execution',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('environment',lambda d:d['direct_runtime_versions'].__setitem__('torch','0.0.0'))
    good=review.audit();assert good['verdict']=='acceptable_with_limits'
    paths=['scripts/review_adversarial_execution.py','scripts/validate_adversarial_review_gates.py']
    report={'format':'adversarial-review-negative-gates.v1','result':'passed',
            'checks':{'injected_failures_rejected':6,'unaltered_audit_passes':True},
            'limits':'Read-only semantic audit injections. Transactional publication and served-catalog gates are separate and not yet exercised.',
            'sha256':{p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (review.ROOT/'docs/reviews/competition_adversarial_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
