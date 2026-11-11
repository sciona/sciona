"""Reject weakened NAB evidence without changing files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_nab_execution as review


def main():
    original=Path.read_text
    checks=[]
    def failure(suffix,name,mutate):
        target=f'competition_nab_{suffix}.json'
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
    failure('source_triage','wrong_license',lambda d:d.__setitem__('license','UNKNOWN'))
    failure('scoring_probe','missing_source_bug',lambda d:d.__setitem__('threshold_below_minimum_returns_no_score',False))
    failure('scoring_runtime','wrong_score',lambda d:d.__setitem__('max_absolute_score_error',1.))
    failure('scoring_runtime','wrong_counts',lambda d:d.__setitem__('point_confusion_counts_exact',False))
    failure('runtime_tests','future_leakage',lambda d:d.__setitem__('prefix_and_future_mutation_invariance',False))
    failure('runtime_tests','label_leakage',lambda d:d.__setitem__('label_independent_detection',False))
    failure('graph_execution','reduced_models',lambda d:d['checks'].__setitem__('models_fitted',1))
    failure('graph_execution','wrong_provider',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('environment','wrong_dependency',lambda d:d['direct_runtime_versions'].__setitem__('numpy','0.0.0'))
    assert review.audit()['verdict']=='acceptable_with_limits'
    paths=['scripts/review_nab_execution.py','scripts/validate_nab_review_gates.py']
    report=dict(format='nab-review-negative-gates.v1',result='passed',
        checks=dict(injected_failures_rejected=len(checks),unaltered_audit_passes=True),
        rejected_scenarios=checks,sha256={p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits='Read-only review injections; no catalog mutations.')
    (review.ROOT/'docs/reviews/competition_nab_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
