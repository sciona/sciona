"""Qualify complete decomposed forecasting providers without catalog mutations."""
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import math
import numpy as np
from sciona.atoms.ml.tabular.gradient_boosting import atoms
from scripts.review_rolling_provider_semantics import packages
from scripts.review_decomposed_future_sales_interfaces import review as interfaces
from scripts.validate_decomposed_future_sales_graph import validate as execute_graph
from scripts.audit_decomposed_future_sales_readiness import ROOT, sha

TESTS = ['tests/test_'+prefix+'future_sales_'+suffix+'.py'
         for prefix in ('', 'decomposed_') for suffix in ('contract','features','training')]
PINNED = {
    'docs/reviews/decomposed_future_sales_interfaces.json':'86ae033e',
    'docs/reviews/decomposed_future_sales_graph_execution.json':'aeabc8da',
    'docs/reviews/future_sales_original_scope.json':'8a970b49',
}
CANDIDATES = {
    'src/sciona/atoms/ml/tabular/gradient_boosting/atoms.py':
        'Reuse rolling_statistics through the graph callable edge. group_aggregate uses bincount sums, which differ from compensated signed totals under cancellation. target_encode uses all supplied targets and optional smoothing; it does not enforce period-prefix availability or produce the complete expanding feature block. Prefix adapters and reduction-order qualification would still be required.',
    'src/sciona/atoms/ml/tabular/available_features.py':
        'clip_truncate adds offsets and truncates to integers; bounded_targets preserves fractional clipped totals and appends an unknown future row.',
    'src/sciona/atoms/ml/model_selection/bayesian_hyperparameter_search/atoms.py':
        'propose_next_parameters is a single proposal wrapper, not a complete trial evaluation, early-stopping, selection and final-fit workflow. Its direct TPESampler import and sample_independent keyword interface are not qualified by this review.',
    'src/sciona/atoms/ml/future_sales_execution.py':
        'The approved monolithic prepare/execute providers remain preserved reference implementations. Reusing their whole execution would hide the decomposed feature, search and refit stages.',
}
LIMITS = [
    'Automated Tier 3 semantic qualification only; provider intake/publication and corrected original CDG approval remain separate pending actions.',
    'Nearby candidate review is bounded to the named implementations; it is not an exhaustive proof that no equivalent implementation exists anywhere in the catalog.',
    'Numeric lags, grouping means, bounds and model inputs are reusable. Root parsing, event role names and three-category assembly retain explicit recipe assumptions.',
    'Fixed entity population and zero absence, one unknown final period, and rolling one-step validation availability are required. Hyperparameter-selection scores are not unbiased generalization estimates.',
    'Complete synthetic computation and unchanged reference parity do not establish historical winning accuracy, source-data equivalence, clean deployment or fitted-weight provenance.',
    'Installed files match their own RECORD hashes; independent package publisher authentication is not established.',
    'Selection and feature checks validate structure, not same-invocation provenance; callers must preserve their bindings.',
]


def review():
    frozen = {}
    for path, commit in PINNED.items():
        raw=(ROOT/path).read_bytes()
        if raw != subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT):
            raise ValueError('Immutable evidence changed: '+path)
        frozen[path]=json.loads(raw)
    current=interfaces()
    original=frozen['docs/reviews/decomposed_future_sales_interfaces.json']
    if current['new_providers'] != original['new_providers']:
        raise ValueError('Previously reviewed providers changed')
    if len(current['existing_providers'])!=1 or current['existing_providers'][0]['canonical_mirror_required']:
        raise ValueError('Published canonical rolling dependency required')
    paths={p:ROOT/p for p in TESTS}
    graph_evidence=frozen['docs/reviews/decomposed_future_sales_graph_execution.json']
    for p in graph_evidence['code_sha256']: paths[p]=ROOT/p
    for p in original['code_sha256']: paths[p]=ROOT/p
    for suffix in ('contract','features','training','worker'):
        p='sciona/future_sales_'+suffix+'.py';paths[p]=ROOT/p
    paths['scripts/review_decomposed_future_sales_semantics.py']=Path(__file__)
    paths['scripts/review_rolling_provider_semantics.py']=ROOT/'scripts/review_rolling_provider_semantics.py'
    for p in CANDIDATES: paths['sciona-atoms-ml/'+p]=ROOT.parent/'sciona-atoms-ml'/p
    p='src/sciona/atoms/ml/decomposed_future_sales_execution.py'
    paths['sciona-atoms-ml/'+p]=ROOT.parent/'sciona-atoms-ml'/p
    before={key:sha(path) for key,path in paths.items()};installed=packages()
    values=np.array([1e16,1.,-1e16],dtype=np.float64)
    reduced=float(atoms.group_aggregate(values,np.array(['g']*3,dtype=object),'sum')[0])
    compensated=math.fsum(values)
    if (reduced,compensated)!=(0.,1.):raise ValueError('Candidate reduction behavior changed')
    all_mean=atoms.target_encode(np.array(['g','g'],dtype=object),np.array([2.,100.]),smoothing=0.)['g']
    prefix_mean=atoms.target_encode(np.array(['g'],dtype=object),np.array([2.]),smoothing=0.)['g']
    if (all_mean,prefix_mean)!=(51.,2.):raise ValueError('Candidate encoding behavior changed')
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='future-sales-semantic-') as directory:
        xml=Path(directory)/'tests.xml'
        run=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*TESTS,'--junitxml='+str(xml)],
            cwd=ROOT,env=environment,capture_output=True,text=True)
        if run.returncode:raise ValueError('Synthetic qualification failed: '+run.stdout[-3000:]+run.stderr[-1000:])
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        counts={key:sum(int(s.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=92,failures=0,errors=0,skipped=0):raise ValueError('Complete 92-test suite required')
    execution=execute_graph()
    if execution!=graph_evidence:raise ValueError('Full graph evidence differs')
    if before!={key:sha(path) for key,path in paths.items()} or installed!=packages():
        raise ValueError('Qualification software changed')
    if current!=interfaces():raise ValueError('Provider inventory changed')
    return dict(format='decomposed-future-sales-semantics.v1',passed=True,approved=False,proposed_tier=3,
        catalog_mutations=0,synthetic_only=True,providers=current['new_providers'],
        existing_providers=current['existing_providers'],tests=counts,execution=execution,
        candidate_review=CANDIDATES,candidate_checks=dict(cancellation_distinguishes_sum=True,prefix_availability_required=True),
        software_sha256=before,installed_packages=installed,evidence_sha256={p:sha(ROOT/p) for p in PINNED},
        resource_policy=dict(serial_tests=True,serial_workers=True,native_threads=1),limitations=LIMITS)


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/decomposed_future_sales_semantics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,providers=len(result['providers']),tests=result['tests'],graph_cases=len(result['execution']['cases']),approved=False)))
