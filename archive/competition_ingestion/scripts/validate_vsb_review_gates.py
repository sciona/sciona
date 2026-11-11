"""Reject weakened VSB evidence without changing evidence files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_vsb_execution as review


def main():
    # Establish the positive baseline first: missing evidence cannot count as
    # rejection of any injected fault.
    assert review.audit()['verdict']=='acceptable_with_limits'
    original=Path.read_text;checks=[]
    def failure(suffix,name,mutate):
        target=f'competition_vsb_{suffix}.json'
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
    failure('execution_source_review','false_historical_scope',lambda d:d.__setitem__('source_scope','exact winner'))
    failure('execution_source_review','wrong_blend',lambda d:d['stage_mapping'].__setitem__('weighted_blend','rank average'))
    failure('threshold_comparison','missing_threshold_comparison',lambda d:d.__setitem__('comparison_cases',0))
    failure('component_tests','missing_tests',lambda d:d.__setitem__('tests_passed',0))
    failure('full_training','reduced_training_limits',lambda d:d.__setitem__('round_ceiling',1))
    failure('graph_execution','missing_models',lambda d:d['checks'].__setitem__('models',1))
    failure('graph_execution','wrong_provider',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('environment','wrong_dependency',lambda d:d['dependency_versions'].__setitem__('lightgbm','0.0.0'))
    failure('environment','missing_notices',lambda d:d.__setitem__('license_sha256',{}))
    assert len(checks)==10 and review.audit()['verdict']=='acceptable_with_limits'
    paths=['scripts/review_vsb_execution.py','scripts/validate_vsb_review_gates.py']
    report=dict(format='vsb-review-negative-gates.v1',result='passed',catalog_mutations=0,
        checks=dict(injected_failures_rejected=10,unaltered_audit_passes=True),rejected_scenarios=checks,
        sha256={p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits='In-memory read-only review injections; no publication approval.')
    (review.ROOT/'docs/reviews/competition_vsb_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
