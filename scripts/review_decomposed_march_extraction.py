"""Exercise full extraction and audit existing calibration providers read-only."""
import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from sciona.atoms.ml.sklearn.calibration.atoms import sigmoid_calibration_fit,sigmoid_calibration_predict
from sciona.atoms.ml.sklearn.calibration import state_models
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from scripts.validate_decomposed_march_baseline import ROOT,sha,runtime
from scripts.import_flavours_provider_drafts import connect

TESTS=['tests/test_decomposed_march_runtime.py','tests/test_march_features.py',
       'tests/test_march_training.py','tests/test_march_contract.py']
GENERIC=['sciona.atoms.ml.sklearn.calibration.sigmoid_calibration_fit',
         'sciona.atoms.ml.sklearn.calibration.sigmoid_calibration_predict']


def review():
    baseline=json.loads((ROOT/'docs/reviews/decomposed_march_baseline_execution.json').read_text())
    if any(sha(ROOT/p)!=digest for p,digest in baseline['code_sha256'].items()):
        raise ValueError('Qualified baseline changed')
    if runtime()!=baseline['runtime']:raise ValueError('Qualified runtime changed')
    tree=ast.parse((ROOT/'sciona/decomposed_march_runtime.py').read_text())
    if any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('sciona.atoms') for n in ast.walk(tree)):
        raise ValueError('Selected generic operations must remain external')
    paths=set(baseline['code_sha256'])|set(TESTS)|{'sciona/decomposed_march_runtime.py','scripts/review_decomposed_march_extraction.py'}
    before={p:sha(ROOT/p) for p in sorted(paths)}
    functions={'sigmoid_fit':sigmoid_calibration_fit,'sigmoid_predict':sigmoid_calibration_predict,'ordered_sum':ordered_weighted_sum}
    providers={name:Path(inspect.getfile(inspect.unwrap(fn))) for name,fn in functions.items()}
    providers['calibration_state']=Path(inspect.getfile(state_models))
    provider_before={name:sha(path) for name,path in providers.items()}
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/sciona-mpl-cache',PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='march-synthetic-extraction-tests-') as directory:
        xml=Path(directory)/'tests.xml'
        result=subprocess.run([sys.executable,'-m','pytest',*TESTS,'-q','--junitxml='+str(xml)],
            cwd=ROOT,env=environment,capture_output=True,text=True)
        if result.returncode:raise ValueError('Extraction tests failed: '+result.stdout[-4000:])
        suites=list(ET.parse(xml).getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts!=dict(tests=70,failures=0,errors=0,skipped=0):raise ValueError('Complete extraction suite required')
    catalogs={}
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            catalogs[prefix]=db.execute(f'''SELECT a.{key}::text AS identity,a.fqdn,a.status,a.is_publishable,
                v.version_id::text,v.content_hash,v.trust_tier,{prefix}_is_publishable(a.{key}) AS native_eligible,
                (SELECT count(*) FROM {prefix}_descriptions d WHERE d.{key}=a.{key}) AS descriptions,
                (SELECT count(*) FROM {prefix}_parameters p WHERE p.version_id=v.version_id) AS parameters
                FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE v.is_latest
                AND a.fqdn=ANY(%s) ORDER BY a.fqdn''',(GENERIC,)).fetchall()
    if len(catalogs['atom'])!=2:raise ValueError('Existing calibration identities required')
    if before!={p:sha(ROOT/p) for p in paths} or provider_before!={name:sha(path) for name,path in providers.items()}:
        raise ValueError('Implementation changed during qualification')
    runtime_after=runtime()
    if runtime_after!=baseline['runtime']:
        raise ValueError('Qualified runtime changed during qualification')
    return dict(format='decomposed-march-extraction.v1',status='passed',approved=False,catalog_mutations=0,
        synthetic_only=True,stored_graph_executed=False,tests=counts,code_sha256=before,
        generic_provider_sha256=provider_before,runtime=runtime_after,existing_calibration_catalog_state=catalogs,
        checks=dict(full_reference_cases=12,matrix_only_feature_widths=[1,3,8,23],
            reference_probability_bytes_equal=True,source_pairing_statements_preserved=True,
            fitted_sigmoid_coefficients_exact=True,calibration_does_not_refit_base=True,
            invalid_arrays_rejected_before_model_fit=True,pagerank_exhaustion_preserved=True,
            optimizer_convergence_warning_remains_error=True),
        resource_policy=dict(native_threads=1,test_workers=1,calibration_workers=1),
        reuse='Two existing generic sigmoid operations and the ordered numeric sum run outside the extracted recipe. Dense standardized binary score fitting has no season or box-score inputs and accepts arbitrary positive feature widths. Box-score aggregation, PageRank, Elo, chronology and mirrored pairing remain explicit recipe policy.',
        next_steps=['Preserve existing sigmoid atom identities and versions; qualify missing descriptions/parameters and their absent canonical artifact counterparts before serving.',
            'Review complete provider contracts and duplication checks before registering recipe adapters.',
            'Build and execute the full explicit graph with external sigmoid calibration, both predictions and ordered reduction; then qualify stored original publication.'],
        limits=['Tests execute the complete direct composition and unchanged reference, not a stored corrected CDG or new registered-provider graph.',
            'Matrix-only reuse demonstrates mechanical compatibility, not empirical cross-domain accuracy or authenticated population independence.',
            'Existing sigmoid providers are not approved for new serving by this report. Package-file integrity and complete publication gates remain required.'])


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/decomposed_march_extraction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],tests=report['tests'],checks=report['checks'],catalog_mutations=0)))
