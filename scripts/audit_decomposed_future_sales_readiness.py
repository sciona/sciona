"""Inspect original scope, preserved implementation and candidate reuse contracts."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import math
from pathlib import Path

import numpy as np
from sciona.atoms.ml.tabular.gradient_boosting.atoms import group_aggregate, rolling_statistics
from sciona.competition_graph import load_competition_graph
from scripts.import_flavours_provider_drafts import ROOT, connect

ORIGINAL = 'e6803752-90ef-5cab-8b30-5ebeec716c15'
SOURCE_HASH = 'ffde31b5d874a8d6fafa0103988b0b9a8e0f13a0169f39ff6dce4e30369423e2'
FQDN = 'cdg.competition.solution.kaggle.predict_future_sales_benchmark'
STAGES = ['time_aggregation', 'target_clipping', 'lag_feature_generation', 'mean_encoding', 'gbdt_modeling']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reuse_probes():
    # Generated cancellation case distinguishes compensated summation from bincount.
    values = np.array([1e16, 1., -1e16])
    generic = float(group_aggregate(values, np.zeros(3, dtype=np.int64), 'sum')[0])
    retained = math.fsum(values)
    if generic == retained or retained != 1.:
        raise ValueError('Aggregation numerical contract changed; reassess reuse')
    series = np.array([2., 4., 8., 16.])
    rolling = rolling_statistics(series, 2, ['mean'])[:, 0]
    reference = np.array([np.nan, np.nan, 3., 6.])
    # Generic rolling includes current data and partial startup windows. A lag
    # and full-window mask can adapt it; direct substitution would leak targets.
    adapted = np.concatenate(([np.nan], rolling[:-1]))
    adapted[:2] = np.nan
    if not np.array_equal(adapted, reference, equal_nan=True):
        raise ValueError('Shifted full-window rolling adapter differs')
    if np.array_equal(rolling, reference, equal_nan=True):
        raise ValueError('Expected rolling contract distinction disappeared')
    return dict(synthetic_only=True, compensated_sum_required=True,
                generic_group_sum_not_byte_equivalent=True,
                rolling_requires_prior_shift_and_full_window_mask=True,
                adapted_rolling_probe_equal=True,
                limits='Small contract probes only; complete extraction parity and broader duplicate search remain required.')


def audit():
    previous = json.loads((ROOT/'docs/reviews/competition_future_sales_graph_execution.json').read_text())
    changed = [name for name, digest in previous['code_sha256'].items() if sha(ROOT/name) != digest]
    provider = ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/future_sales_execution.py'
    if changed or sha(provider) != previous['provider_sha256']:
        raise ValueError('Previously executed implementation changed')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        original = db.execute('''SELECT a.artifact_id::text,a.fqdn,a.status,a.is_publishable,
            v.content_hash,v.is_latest FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE version_id=%s''', (ORIGINAL,)).fetchone()
        if not original or original['fqdn'] != FQDN or original['status'] != 'draft' or original['is_publishable'] or not original['is_latest'] or original['content_hash'] != SOURCE_HASH:
            raise ValueError('Original identity or state differs')
        graph = load_competition_graph(db, version_id=ORIGINAL)
        if {n.node_id for n in graph.nodes} != set(STAGES):
            raise ValueError('Original stage coverage differs')
        parents = db.execute('''SELECT a.artifact_id::text,a.fqdn,v.version_id::text,v.content_hash,v.trust_tier
            FROM artifact_dependencies d JOIN artifact_versions v ON v.version_id=d.dependent_version_id
            JOIN artifacts a ON a.artifact_id=v.artifact_id
            WHERE d.dependency_artifact_fqdn=%s AND d.dependency_content_hash=%s AND NOT d.optional
            AND v.is_latest AND a.status='approved' AND a.is_publishable
            AND EXISTS(SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id)
            ORDER BY a.fqdn''', (FQDN,SOURCE_HASH)).fetchall()
        if len(parents) != 1 or parents[0]['trust_tier'] != 3:
            raise ValueError('Unique approved Tier 3 reference required')
        candidates = db.execute('''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash
            FROM atoms a JOIN atom_versions v USING(atom_id) WHERE v.is_latest
            AND a.source_symbol=ANY(%s) ORDER BY a.fqdn''',
            (['group_aggregate','rolling_statistics','future_sales_prepare','future_sales_execute'],)).fetchall()
    historical = dict(line.split('==') for line in (ROOT/'requirements/future-sales-execution.txt').read_text().splitlines())
    return dict(format='decomposed-future-sales-readiness.v1',read_only=True,catalog_mutations=0,approved=False,
        original=original,source_version_id=ORIGINAL,source_stages=STAGES,reference=parents[0],
        source_scope='Generic benchmark topology; no identified historical winning implementation.',
        unresolved_source_scope=[
            'Original mean_encoding description explicitly lists entity embeddings, holiday effects and promotion effects. Existing derivative implements prior expanding means and cyclic period features only.',
            'Earlier derivative review calls these alternatives/examples, but the original mean_encoding wording does not explicitly say optional. Establish the intended complete contract and implement required computations before original promotion; existing derivative approval does not settle this discrepancy.',
            'Original target_clipping claims exact evaluation-metric replication; clipping alone does not establish that claim. Preserve numeric [0,20] policy and separately qualify the chosen loss and validation metric.'],
        unchanged_reference_code_sha256=previous['code_sha256'],provider_sha256=sha(provider),
        runtime_versions={name:dict(recorded=version,installed=metadata.version(name)) for name,version in historical.items()},
        reuse_candidates=candidates,reuse_probes=reuse_probes(),
        inspected_provider_sha256={fn.__name__:sha(inspect.getfile(inspect.unwrap(fn))) for fn in (group_aggregate,rolling_statistics)},
        auditor_sha256=sha(__file__),next_steps=[
            'Resolve the original mean-encoding scope against the retained source description before choosing extraction boundaries; do not silently inherit omitted computations from the derivative.',
            'Preserve compensated net aggregation before clipping, fixed active population, strict prior-only features, rolling one-step validation and the full configured TPE/early-stopping/refit lifecycle.',
            'Expose reusable numeric stages with explicit contracts; retain recipe-specific clipping bounds, categorical layout and observation-availability assumptions as declared policy.',
            'Preserve dedicated worker execution for native training; do not move LightGBM into the combined graph process without resolving the recorded crash.',
            'Qualify current environment and complete native graph execution without overwriting historical NumPy 2.4.2 evidence.',
            'Finish duplicate search, extraction parity, provider qualification, stored original execution and publication gates before approval.'],
        limits=['Readiness and contract probes are not publication approval or full execution qualification.',
                'A fixed entity universe assumes all entities are active for every observed period; absent transactions mean zero.',
                'Validation uses earlier validation-period observations and is not fixed-origin multi-period forecasting.',
                'Native worker isolation remains required; original crash cause has not been established.'])


if __name__ == '__main__':
    result = audit()
    (ROOT/'docs/reviews/decomposed_future_sales_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(approved=False,source_stages=len(result['source_stages']),
                         reference_unchanged=True,reuse_probes=result['reuse_probes'])))
