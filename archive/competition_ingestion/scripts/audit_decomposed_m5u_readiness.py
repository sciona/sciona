"""Read-only uncertainty promotion preflight and exact generic reduction checks."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import numpy as np
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from sciona.m5u_harmonization import restore
from scripts.import_flavours_provider_drafts import connect

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = '8641dbb8-c742-5a35-b94a-1c0269c03984'
GENERIC = 'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reduction_check():
    rng = np.random.default_rng(61452)
    quantiles = (.005,.025,.165,.25,.5,.75,.835,.975,.995)
    level_ids = np.array([1,2,3,4,5,6,7,8,9,10,11,13,14,15])
    cases = 0
    for multiplicity in (1,2,5):
        levels = np.repeat(level_ids,multiplicity)
        for exponent in (-100,0,100):
            for signed in (False,True):
                for adjustment in (0.,.7,1.):
                    for _ in range(5):
                        values = rng.normal(size=(9,28,len(levels))) if signed else rng.uniform(.1,5.,(9,28,len(levels)))
                        values *= 10.**exponent
                        factors = np.repeat(rng.uniform(.1,10.,len(level_ids)),multiplicity)
                        saved = values.copy()
                        expected = restore(values,levels,factors,quantiles,adjustment=adjustment)
                        result = values.astype(float,copy=True)/factors[None,None,:]
                        selected = [level for level in np.unique(levels) if level<=9]
                        totals = np.stack([result[quantiles.index(.5)][:,levels==level].sum(axis=1) for level in selected])
                        target = ordered_weighted_sum(totals,np.ones(len(selected)))/len(selected)
                        if not np.array_equal(target,totals.mean(axis=0)):
                            raise ValueError('Generic reduction changes target mean')
                        for level,total in zip(selected,totals):
                            correction = (1-adjustment)+adjustment*target/total
                            result[:,:,levels==level] *= correction[None,:,None]
                        if result.tobytes()!=expected.tobytes() or values.tobytes()!=saved.tobytes():
                            raise ValueError('Generic restoration changes reference arithmetic or inputs')
                        cases += 1
    return dict(cases=cases,bitwise_reference_equality=True,inputs_unchanged=True,
        all_14_levels=True,quantiles=9,horizon=28,positive_and_signed_forecasts=True,
        adjustments=[0.,.7,1.],
        scope='Exact generated restoration comparisons at the complete recipe horizon. Preserve native-unit division, level-local sums, ascending level order, sum-before-division and correction multiplication. No arbitrary-shape NumPy reduction equivalence claimed.')


def audit():
    baseline = json.loads((ROOT/'docs/reviews/competition_m5_uncertainty_graph_execution.json').read_text())
    changed = [p for p,d in baseline['code_sha256'].items() if sha(ROOT/p)!=d]
    versions = dict(line.split('==') for line in (ROOT/'requirements/m5u-execution.txt').read_text().splitlines() if line)
    drift = {name:dict(recorded=version,installed=metadata.version(name))
             for name,version in versions.items() if metadata.version(name)!=version}
    comparisons = reduction_check()
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        catalog = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE v.is_latest AND (a.artifact_id=%s OR a.fqdn=%s) ORDER BY a.fqdn''',
            (ORIGINAL,'cdg.competition.solution.kaggle.m5_uncertainty_1st.execution')).fetchall()
        providers = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,artifact_is_publishable(a.artifact_id) AS native_eligible,
            (SELECT count(*) FROM artifact_descriptions d WHERE d.artifact_id=a.artifact_id) AS descriptions,
            (SELECT count(*) FROM artifact_parameters p WHERE p.version_id=v.version_id) AS parameters
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE v.is_latest AND a.fqdn IN (%s,%s,%s) ORDER BY a.fqdn''',
            ('sciona.atoms.ml.m5u_execution.m5u_prepare','sciona.atoms.ml.m5u_execution.m5u_execute',GENERIC)).fetchall()
    return dict(format='decomposed-m5u-readiness.v1',read_only=True,catalog_mutations=0,approved=False,
        catalog=catalog,providers=providers,prior_execution_changed_files=changed,runtime_drift=drift,
        synthetic_reduction=comparisons,
        source_sha256={p:sha(ROOT/p) for p in ('sciona/m5u_execution.py','sciona/m5u_pipeline.py','sciona/m5u_harmonization.py')},
        generic_provider_sha256=sha(inspect.getfile(ordered_weighted_sum)),auditor_sha256=sha(__file__),
        next_steps=[
            'Expose complete training and repeated inference before restoration while retaining the reference run wrapper and unchanged restoration oracle.',
            'Bind preparation, full training/inference, native-unit and median-total preparation, generic ordered sum, recipe correction and complete result packaging explicitly.',
            'Preserve all 14 levels, nine quantiles, full 4482 candidate fits, 126 models, 2730000 repeated synthetic queries and the 28-step horizon; do not enable speed modes.',
            'Qualify complete graph execution and current runtime, exact provider contracts and metadata, rollback gates and original-intake production selection.',
            'Preserve original/source histories and the existing executable derivative; retain the explicit aggregate-normalization correction and limits on historical parity.'],
        limits=['Numeric extraction readiness only; no new provider or original-CDG approval.',
                'Known synthetic inventory counts do not establish historical sample-volume or predictive accuracy equivalence.',
                'Generic sum has no hierarchy or quantile policy. Same-invocation adapters must retain those recipe constraints.'])


if __name__=='__main__':
    report = audit()
    (ROOT/'docs/reviews/decomposed_m5u_readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(synthetic_comparisons=report['synthetic_reduction']['cases'],
        changed_prior_files=report['prior_execution_changed_files'],runtime_drift=report['runtime_drift'],
        catalog=report['catalog'],providers=report['providers'])))
