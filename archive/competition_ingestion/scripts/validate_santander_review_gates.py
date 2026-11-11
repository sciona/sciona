"""Reject weakened Santander evidence without changing evidence files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_santander_execution as review


def main():
    # Establish the positive baseline first: missing evidence cannot count as
    # rejection of any injected fault.
    assert review.audit()['verdict']=='acceptable_with_limits'
    original=Path.read_text;checks=[]
    def failure(suffix,name,mutate):
        target=f'competition_santander_{suffix}.json'
        def changed(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path.name==target:
                document=json.loads(text);mutate(document);return json.dumps(document)
            return text
        with patch.object(Path,'read_text',changed):
            try:review.audit()
            except AssertionError:checks.append(name);return
        raise AssertionError('Weakened evidence accepted: '+name)
    failure('source_review','wrong_source',lambda d:d.__setitem__('source_content_hash','0'*64))
    failure('source_review','false_historical_scope',lambda d:d.__setitem__('source_scope','exact winner'))
    failure('source_review','wrong_detector',lambda d:d['stage_mapping'].__setitem__('fake_data_identification','drop singleton rows'))
    failure('execution_boundary','missing_boundaries',lambda d:d.__setitem__('tests_passed',0))
    failure('runtime_tests','missing_tests',lambda d:d.__setitem__('tests_passed',0))
    failure('default_tail_execution','reduced_pseudo_counts',lambda d:d['selection_counts']['neural'].__setitem__('positive',1))
    failure('graph_execution','missing_models',lambda d:d['checks'].__setitem__('models',1))
    failure('graph_execution','wrong_provider',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('environment','wrong_dependency',lambda d:d['dependency_versions'].__setitem__('torch','0.0.0'))
    failure('environment','missing_notices',lambda d:d.__setitem__('license_sha256',{}))
    assert len(checks)==10 and review.audit()['verdict']=='acceptable_with_limits'
    paths=['scripts/review_santander_execution.py','scripts/validate_santander_review_gates.py']
    report=dict(format='santander-review-negative-gates.v1',result='passed',catalog_mutations=0,
        checks=dict(injected_failures_rejected=10,unaltered_audit_passes=True),rejected_scenarios=checks,
        sha256={p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits='In-memory read-only review injections; no publication approval.')
    (review.ROOT/'docs/reviews/competition_santander_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
