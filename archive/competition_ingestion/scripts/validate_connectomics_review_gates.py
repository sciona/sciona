"""Reject weakened Connectomics evidence without changing files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_connectomics_execution as review


def main():
    original=Path.read_text
    checks=[]
    def failure(suffix,name,mutate):
        target=f'competition_connectomics_{suffix}.json'
        def changed(path,*args,**kwargs):
            text=original(path,*args,**kwargs)
            if path.name==target:
                document=json.loads(text);mutate(document);return json.dumps(document)
            return text
        with patch.object(Path,'read_text',changed):
            try:review.audit()
            except AssertionError:checks.append(name);return
        raise AssertionError('Weakened evidence accepted: '+name)
    failure('source_pins','wrong_source_commit',lambda d:d.__setitem__('commit','0'*40))
    failure('source_pins','unknown_license',lambda d:d['execution_license'].__setitem__('declared','UNKNOWN'))
    failure('source_execution','missing_historical_pca_check',lambda d:d.__setitem__('legacy_precision_inverse_covariance_check',False))
    failure('corrected_execution','reduced_grid',lambda d:d.__setitem__('grid_entries',2))
    failure('corrected_execution','ignored_thresholds',lambda d:d['checks'][0].__setitem__('distinct_preprocessed_inputs',2))
    failure('corrected_execution','reference_mismatch',lambda d:d['checks'][0].__setitem__('max_abs_covariance_reference_error',1))
    failure('graph_execution','missing_mode',lambda d:d['cases'].pop())
    failure('graph_execution','wrong_directivity',lambda d:d.__setitem__('independent_directivity_exact',False))
    failure('graph_execution','altered_provider',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('environment','wrong_dependency',lambda d:d['direct_runtime_versions'].__setitem__('numpy','0.0.0'))
    assert review.audit()['verdict']=='acceptable_with_limits'
    paths=['scripts/review_connectomics_execution.py','scripts/validate_connectomics_review_gates.py']
    report=dict(format='connectomics-review-negative-gates.v1',result='passed',
        checks=dict(injected_failures_rejected=len(checks),unaltered_audit_passes=True),
        rejected_scenarios=checks,sha256={p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits='Read-only review injections; no catalog mutations.')
    (review.ROOT/'docs/reviews/competition_connectomics_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
