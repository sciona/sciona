"""Read-only M5 promotion preflight using synthetic reduction comparisons."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
import numpy as np
from sciona.atoms.numerical.statistics.ordered_sum import ordered_weighted_sum
from scripts.import_flavours_provider_drafts import connect

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = '9c0a711a-dc75-5c7f-8711-b2879024e282'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    baseline = json.loads((ROOT/'docs/reviews/competition_m5_graph_execution.json').read_text())
    changed = [name for name, digest in baseline['code_sha256'].items() if sha(ROOT/name) != digest]
    versions = dict(line.split('==') for line in (ROOT/'requirements/m5-execution.txt').read_text().splitlines() if line)
    drift = {name: dict(recorded=version, installed=metadata.version(name))
             for name, version in versions.items() if metadata.version(name) != version}
    rng = np.random.default_rng(4321)
    comparisons = 0
    for count in (1, 2, 7, 70, 101):
        for exponent in (-100, 0, 100):
            for _ in range(10):
                # Generated family forecasts only; no real input or output is retained.
                values = np.ascontiguousarray(rng.random((6, count, 28))*10.**exponent)
                before = values.copy()
                actual = ordered_weighted_sum(values.reshape(6, -1), np.ones(6)).reshape(count, 28)/6
                expected = np.sum(values, axis=0)/6
                if not np.array_equal(actual, expected) or not np.array_equal(values, before):
                    raise ValueError('Ordered reduction changes reference arithmetic or caller input')
                comparisons += 1
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        state = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE v.is_latest AND (a.artifact_id=%s OR a.fqdn=%s) ORDER BY a.fqdn''',
            (ORIGINAL, 'cdg.competition.solution.kaggle.m5_accuracy_1st.execution')).fetchall()
        providers = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,artifact_is_publishable(a.artifact_id) AS native_eligible,
            (SELECT count(*) FROM artifact_descriptions d WHERE d.artifact_id=a.artifact_id) AS descriptions,
            (SELECT count(*) FROM artifact_parameters p WHERE p.version_id=v.version_id) AS parameters
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE v.is_latest AND a.fqdn IN (%s,%s,%s) ORDER BY a.fqdn''',
            ('sciona.atoms.ml.m5_execution.m5_prepare','sciona.atoms.ml.m5_execution.m5_execute',
             'sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum')).fetchall()
    return dict(format='decomposed-m5-readiness.v1',read_only=True,catalog_mutations=0,approved=False,
        catalog=state,providers=providers,prior_execution_changed_files=changed,runtime_drift=drift,
        synthetic_reduction=dict(cases=comparisons,exact_reference_equality=True,inputs_unchanged=True,
            shape_policy='Six contiguous float64 family matrices, flattened over series and horizon, unit coefficients, divide after sum.'),
        source_sha256={name:sha(ROOT/name) for name in ('sciona/m5_execution.py','sciona/m5_forecast.py','sciona/m5_pooled_training.py')},
        generic_provider_sha256=sha(inspect.getfile(ordered_weighted_sum)),auditor_sha256=sha(__file__),
        next_steps=[
            'Expose the full existing family forecasting loop before its final reduction; retain predict as a compatible full reference wrapper.',
            'Bind preparation, feature construction, all pooled training, all recursive/nonrecursive family forecasts, generic ordered reduction and result packaging explicitly.',
            'Preserve all six families, 3000 evaluated rounds per supplied pool and the full 28-day horizon. Qualify the complete 220-model synthetic inventory again after extraction.',
            'Refresh runtime qualification and repair missing provider metadata before approval; preserve original and derivative histories.',
            'Execute the stored corrected original and validate publication rollback and production selection.'],
        limits=['Synthetic numeric parity is a candidate reuse check, not full graph execution or promotion eligibility.',
                'No claim of historical winning-model accuracy, independent validation or completed original promotion.'])


if __name__ == '__main__':
    report = audit()
    (ROOT/'docs/reviews/decomposed_m5_readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(synthetic_comparisons=report['synthetic_reduction']['cases'],
        changed_prior_files=len(report['prior_execution_changed_files']),runtime_drift=report['runtime_drift'],
        provider_eligibility=[dict(fqdn=p['fqdn'],eligible=p['native_eligible']) for p in report['providers']])))
