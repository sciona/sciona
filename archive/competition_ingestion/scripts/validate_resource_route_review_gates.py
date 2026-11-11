"""Reject weakened Resource-route evidence without changing files or catalog."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import review_resource_route_execution as review


def main():
    original=Path.read_text
    checks=[]
    def failure(suffix,name,mutate):
        target=f'competition_resource_route_{suffix}.json'
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
    failure('runtime_tests','missing_exhaustive_comparison',lambda d:d.__setitem__('exhaustively_compared_synthetic_graphs',0))
    failure('runtime_tests','missing_stage',lambda d:d['stage_mapping'].pop('constraint_repair'))
    failure('graph_execution','missing_limit_case',lambda d:d['checks'].__setitem__('search_limit',False))
    failure('graph_execution','missing_infeasible_case',lambda d:d['checks'].__setitem__('infeasible',False))
    failure('graph_execution','missing_stage_execution',lambda d:d['checks'].__setitem__('actual_runner_nodes',4))
    failure('graph_execution','wrong_provider',lambda d:d.__setitem__('provider_sha256','0'*64))
    failure('environment','wrong_dependency',lambda d:d['graph_schema_dependency_versions'].__setitem__('pydantic','0.0.0'))
    failure('environment','wrong_environment',lambda d:d.__setitem__('python_version','0.0.0'))
    assert review.audit()['verdict']=='acceptable_with_limits'
    paths=['scripts/review_resource_route_execution.py','scripts/validate_resource_route_review_gates.py']
    report=dict(format='resource_route-review-negative-gates.v1',result='passed',
        checks=dict(injected_failures_rejected=len(checks),unaltered_audit_passes=True),
        rejected_scenarios=checks,sha256={p:hashlib.sha256((review.ROOT/p).read_bytes()).hexdigest() for p in paths},
        limits='Read-only review injections; no catalog mutations.')
    (review.ROOT/'docs/reviews/competition_resource_route_review_gates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
