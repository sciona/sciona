"""Qualify all five March adapters against complete execution and reusable contracts."""
import ast
import inspect
import json
from pathlib import Path
from scripts.stage_march_decomposed_providers import reviewed_plan, ROOT, sha, require
from scripts.review_sigmoid_provider_semantics import review as sigmoid_review
from scripts.validate_decomposed_march_graph_execution import validate as execute_graph
from sciona.atoms.ml import decomposed_march_execution as provider


def review():
    intake=reviewed_plan()
    # This qualification includes the complete 88-test extraction/reference suite
    # and verifies the installed numerical and registration dependency closure.
    calibration=sigmoid_review()
    execution=execute_graph()
    expected=json.loads((ROOT/'docs/reviews/decomposed_march_graph_execution.json').read_text())
    require(execution==expected,'Current complete graph differs from committed execution')
    paths=['scripts/review_march_decomposed_providers.py','scripts/stage_march_decomposed_providers.py',
        'scripts/decomposed_march_provider_contracts.py','sciona/decomposed_march_runtime.py',
        'sciona/decomposed_march_graph.py','sciona/march_contract.py','sciona/march_features.py','sciona/march_training.py']
    files={p:ROOT/p for p in paths}
    files['sciona-atoms-ml/src/sciona/atoms/ml/decomposed_march_execution.py']=Path(inspect.getfile(provider))
    imports={name:sorted({ast.unparse(n) for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n,(ast.Import,ast.ImportFrom))}) for name,path in files.items()}
    for name in ('sciona/decomposed_march_runtime.py','sciona-atoms-ml/src/sciona/atoms/ml/decomposed_march_execution.py'):
        require(not any('sciona.atoms.' in line for line in imports[name]),'Hidden registered operation in adapter')
    records=[]
    for item in intake['providers']:
        require(set(item['parameters'])=={p['name'] for p in item['interfaces'] if p['direction']=='input'},
                'Complete parameter semantic inventory required')
        records.append(dict(item,internal_registered_provider_dependencies=[]))
    require(len(records)==5,'Five qualified providers required')
    require(reviewed_plan()==intake,'Intake changed during qualification')
    return dict(format='march-decomposed-provider-semantics.v1',passed=True,approved=False,
        proposed_tier=3,verdict='acceptable_with_limits',catalog_mutations=0,synthetic_only=True,
        providers=records,graph_digest=execution['serialized_graph_sha256'],tests=calibration['tests'],
        graph_cases=execution['cases'],installed_packages=calibration['installed_packages'],
        software_sha256={name:sha(path) for name,path in files.items()},imports=imports,
        evidence_sha256={name:sha(ROOT/'docs/reviews'/name) for name in (
            'decomposed_march_graph_execution.json','decomposed_march_interfaces.json',
            'march_decomposed_provider_intake_plan.json','march_decomposed_provider_intake_gates.json')},
        resource_policy=dict(native_threads=1,serial_graph_cases=True,test_workers=1),
        limitations=[
            'Automated Tier 3 implementation qualification only; no human Tier 1 review or Tier 2 usage certification.',
            'The source describes a generic benchmark topology, not an identified historical winning solution. No competitive accuracy or true strength-of-schedule claim is made.',
            'Complete synthetic graph results match the unchanged reference. Matrix-only tests demonstrate mechanical reuse, not empirical cross-domain accuracy.',
            'Season feature aggregation and mirrored chronological pairing retain the source recipe. Dense standardized binary fitting supports arbitrary positive feature widths; sigmoid fit/predict and ordered sum remain independently bound generic providers.',
            'Intermediate provenance and population independence are caller responsibilities; shape and chronology checks do not authenticate them.',
            'Existing sigmoid fitting does not check optimizer success. Finite calibration coefficients do not establish convergence; base logistic fitting does treat ConvergenceWarning as an error.',
            'Installed package files match their own RECORD hashes; publisher authenticity, clean deployment, optional extras and other platforms remain unqualified.',
            'Provider publication transaction gates and stored corrected-original CDG execution/publication remain separate requirements.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/march_decomposed_provider_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=5,tests=result['tests'],graph_cases=len(result['graph_cases']),approved=False)))
