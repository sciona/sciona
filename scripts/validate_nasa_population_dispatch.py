"""Exercise all source slots with interleaved synthetic queries and served core."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from threadpoolctl import threadpool_limits

from sciona.nasa_population_dispatch import partition_records, assemble_predictions
from sciona.nasa_workflow_inputs import prepare_training_inputs, prepare_prediction_inputs, attach_predictions
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.visualizer import runner
from scripts.plan_residual_classifier_draft import ROOT, plan
from scripts.validate_residual_classifier_database_gates import check_staged
from scripts.validate_residual_classifier_graph import source_level_tables
from scripts.validate_nasa_corrected_workflow import source_programs
from scripts.validate_residual_classifier_reuse import reference


def validate(source):
    controls, _, source_hashes = source_programs(source)
    slots = tuple(controls['list_airports'])
    training, prediction = {}, {}
    for index, airport in enumerate(slots):
        rng = np.random.default_rng(9021+index)
        _, vocabulary, training[airport] = source_level_tables(rng, airport, True)
        _, _, prediction[airport] = source_level_tables(rng, airport, False, vocabulary)
    def combine(tables):
        return {key:pd.concat([tables[airport][key] for airport in slots], ignore_index=True)
                for key in ['queries','estimates','stands','arrivals']}
    combined_train, combined_query = combine(training), combine(prediction)
    combined_query['queries'] = combined_query['queries'].sample(frac=1, random_state=713).reset_index(drop=True)
    train_parts, _ = partition_records(combined_train, slots)
    query_parts, identities = partition_records(combined_query, tuple(reversed(slots)))
    proposed = plan()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db, proposed, approved=True)
        if not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=%s', (proposed['artifact_id'],)).fetchone():
            raise ValueError('Numerical core is not served')
        document = db.execute('SELECT get_artifact_document(%s) AS d', (proposed['fqdn'],)).fetchone()['d']
    graph = _artifact_document_to_cdg(document, version_id=proposed['version_id'],
        content_hash=proposed['graph_sha256'], require_execution_envelope=True)
    results, expected_results, scenarios = {}, {}, []
    for index, airport in enumerate(query_parts):
        prepared, vocabulary, train_ids = prepare_training_inputs(train_parts[airport], airport)
        independent, independent_vocab, independent_ids = prepare_training_inputs(training[airport], airport)
        for name in ['features','targets','groups']:
            np.testing.assert_array_equal(prepared[name], independent[name])
        assert prepared['feature_names'] == independent['feature_names'] and vocabulary == independent_vocab
        pd.testing.assert_frame_equal(train_ids, independent_ids)
        query_matrix, query_ids = prepare_prediction_inputs(query_parts[airport], airport, prepared['feature_names'], vocabulary)
        isolated_queries = dict(prediction[airport], queries=query_parts[airport]['queries'])
        independent_matrix, independent_query_ids = prepare_prediction_inputs(isolated_queries, airport, prepared['feature_names'], vocabulary)
        np.testing.assert_array_equal(query_matrix, independent_matrix)
        pd.testing.assert_frame_equal(query_ids, independent_query_ids)
        cutoff = controls['mae_thresh_bad'] if airport in controls['bad_airports'] else controls['mae_thresh_good']
        payload = dict(prepared, prediction_features=query_matrix, train_fraction=.4, seed=42,
                       maximum_error=float(cutoff), threshold=.5, prediction_feature_name='pred_minutes_until_pushback')
        with threadpool_limits(limits=1):
            expected, _, _, _ = reference(payload)
        captured = {}
        def capture(directory, node, name, value):
            if node == 'final' and name == 'out_predictions':
                captured['predictions'] = value
        with tempfile.TemporaryDirectory(prefix='nasa-dispatch-') as directory:
            with threadpool_limits(limits=1), patch.object(runner, 'RUNS_DIR', Path(directory)), \
                    patch.object(runner, 'save_intermediate_value', side_effect=capture):
                outcome = asyncio.run(runner.CDGExecutionSession(None, 'synthetic-population-dispatch', str(index))
                                      .execute(payload, cdg=graph))
        if outcome['status'] != 'completed':
            raise ValueError('Dispatched numerical execution failed')
        np.testing.assert_array_equal(captured['predictions'], expected)
        results[airport] = attach_predictions(query_ids, captured['predictions'])
        expected_results[airport] = attach_predictions(query_ids, expected)
        scenarios.append(dict(slot=index, predictions=len(expected), source_cutoff=cutoff,
            isolated_and_combined_features_exact=True, independent_predictions_exact=True))
        print(json.dumps(scenarios[-1]), flush=True)
    output = assemble_predictions(identities, results)
    expected = pd.concat(expected_results.values(), ignore_index=True).set_index(['gufi','timestamp','airport'])
    expected = expected.loc[pd.MultiIndex.from_frame(identities)].reset_index()
    pd.testing.assert_frame_equal(output, expected)
    return dict(passed=True, approved=False, synthetic_only=True, catalog_mutations=0,
        served_core_version_id=proposed['version_id'], graph_sha256=proposed['graph_sha256'],
        slots=len(slots), predictions=len(output), original_query_order_preserved=True,
        scenarios=scenarios, source_ast_sha256=source_hashes,
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dispatcher_sha256=hashlib.sha256((ROOT/'sciona/nasa_population_dispatch.py').read_bytes()).hexdigest(),
        limitations=['Domain adapters and dispatch are runtime orchestration, not yet catalog-facing graph nodes.',
                    'Synthetic execution is not empirical competition performance or completion of both original NASA intakes.',
                    'Only queried slots execute here; persistent model-training lifecycle and empty-population policies remain explicit separate work.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-directory', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source_directory)
    (ROOT/'docs/reviews/competition_nasa_population_dispatch.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='scenarios'}))
