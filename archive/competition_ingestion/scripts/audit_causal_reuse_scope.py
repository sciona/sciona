"""Read-only inventory of reuse boundaries before original causal-CDG promotion.

This report is a decomposition worklist, not an approval or numerical test.
Only algorithm identities and source hashes are recorded; no datasets are read.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from sciona.causal_prediction_execution import build_raw_pair_causal_training_prediction_graph
from sciona.atoms.causal_inference.feature_primitives import selected_features
from sciona.atoms.ml.sklearn.ensemble.gradient_boosting import causal_training, causal_predictions
from sciona.services.execution_graph_codec import encode_execution_graph


def audit():
    graph = build_raw_pair_causal_training_prediction_graph()
    digest, _, _ = encode_execution_graph(graph)
    fqdn = 'cdg.competition.solution.kaggle.cause_effect_2nd'
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row) as db:
        db.execute('SET TRANSACTION READ ONLY')
        records = db.execute('''
            SELECT a.fqdn,a.status,a.is_publishable,v.version_id,v.content_hash,
                   v.trust_tier,
                   (SELECT count(*) FROM artifact_cdg_nodes n
                    WHERE n.version_id=v.version_id) AS nodes,
                   EXISTS(SELECT 1 FROM catalog_artifacts_served s
                          WHERE s.artifact_id=a.artifact_id) AS served
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.fqdn IN (%s,%s) AND v.is_latest ORDER BY a.fqdn
        ''', (fqdn, fqdn + '.execution')).fetchall()
    if len(records) != 2:
        raise ValueError('Expected one latest version for original and execution derivative')
    derivative = next(r for r in records if r['fqdn'].endswith('.execution'))
    if derivative['content_hash'] != digest or derivative['nodes'] != len(graph.nodes):
        raise ValueError('Local execution builder does not match latest catalog identity')
    sources = {
        module.__name__: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
        for module in (selected_features, causal_training, causal_predictions)
    }
    return {
        'scope': 'Read-only source/decomposition inventory; not promotion evidence',
        'catalog': records,
        'source_hashes': sources,
        'execution_graph_hash': digest,
        'selected_feature_order': list(selected_features.FEATURE_ORDER),
        'direct_graph_providers': {n.node_id: n.matched_primitive for n in graph.nodes},
        'decomposition_required': [
            {
                'node': 'features',
                'finding': 'One node extracts both populations, reverses observations and labels, and computes all 43 features.',
                'required': 'Expose typed pair validation/orientation, label augmentation, reusable statistics, and ordered feature assembly. Keep source-specific feature order in a recipe adapter.',
                'invariants': ['No prediction-population statistics or labels enter fitting',
                               'Adjacent forward/reverse rows and negated labels',
                               'Categorical correlation and HSIC dispatch preserved',
                               'Undefined feature regimes rejected, not silently imputed'],
            },
            {
                'node': 'training',
                'finding': 'One node partitions labels, computes weights and fits five classifiers.',
                'required': 'Expose partition/weight preparation and five separate generic weighted gradient-boosting fits.',
                'invariants': ['One-step uses all three labels and no sample weights',
                               'Direction excludes zero labels and uses unit weights',
                               'Dependence/left/right use source negative-class balancing',
                               'Retain all five fits and source default 500 estimators'],
            },
            {
                'node': 'classifiers',
                'finding': 'One node predicts five models and extracts four positive-class columns.',
                'required': 'Use five generic probability-prediction nodes and explicit labelled-column selection; retain role validation.',
                'invariants': ['One-step class set {-1,0,1}',
                               'Direction class set {-1,1}; other binary roles {0,1}',
                               'Direction output is P(+1), not a signed score',
                               'Identical feature column and observation orientation order'],
            },
        ],
        'promotion_gates': [
            'Decompose all three orchestration nodes while preserving source behavior',
            'Qualify generic providers independently of this competition recipe',
            'Run pinned-source feature and full five-model parity including 500-estimator case',
            'Check negative contracts, codec roundtrip and uncached production execution',
            'Publish exact provider versions and mandatory numerical dependencies',
            'Preserve original identity/history and validate transactional promotion guards',
            'Execute the production-selected stored graph before claiming original completion',
        ],
        'tier_policy': 'Automated Tier 3; no Tier 2 community-usage or Tier 1 human-review claim',
    }


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')
    report = audit()
    output = root / 'docs/reviews/causal_reuse_scope.json'
    output.write_text(json.dumps(report, default=str, indent=2) + '\n')
    print(json.dumps({'report': str(output.relative_to(root)), 'catalog': report['catalog']}, default=str, indent=2))
