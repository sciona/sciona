"""Exact provider contracts and dependency qualification; no catalog mutations."""
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from scripts.stage_amex_decomposed_providers import reviewed_plan
from scripts.review_decomposed_amex import ROOT, require, sha, review as integration_review

TESTS = ('tests/test_ordered_weighted_sum.py','tests/test_decomposed_amex_graph.py',
         'tests/test_decomposed_amex_provider_intake.py','tests/test_decomposed_amex_review.py')
TEST_REPORT = ROOT/'docs/reviews/amex_decomposed_provider_tests.json'
REPORT = ROOT/'docs/reviews/amex_decomposed_provider_semantics.json'
PROVIDER_FILES = ('sciona-atoms/src/sciona/atoms/numerical/statistics/ordered_sum.py',
                  'sciona-atoms-ml/src/sciona/atoms/ml/decomposed_amex_execution.py')


def recipe(operation, plain, bounds, parameters, outputs, dependencies):
    return dict(operation=operation,plain_limits=plain,validity_bounds=bounds,parameters=parameters,
        outputs=outputs,dependencies=dependencies+['sciona.ghost.registry (registration only)'],
        reuse='Source recipe adapter with explicit runtime feature roles and graph ports. Supply coherent intermediates from the same invocation; this stage does not authenticate their origin. Fixed source controls are not generic numerical policy.')


CONTRACTS = {
 'populations':recipe(
    'Read checked input and prepare the ordered training and query groups.',
    'Use the unchanged checked input. Keep group order, labels and settings. No model is trained here.',
    'Exact Prepared type from the reviewed version-1 preparation boundary. Verify canonical JSON and SHA-256 fingerprint and revalidate the complete payload. Apply explicit numeric flooring, category rules and runtime time/month roles independently to training and query, then concatenate training before query. Preserve labels, training count, zero-fill column roles and seed.',
    {'prepared':'Canonical immutable Prepared object from amex_prepare; raw finite JSON and complete five-fold population requirements apply.'},
    'Dictionary containing joint numerics/categories/times/months populations, training_count, labels, zero_fill_columns and seed.',
    ['sciona.amex_execution.Prepared/prepare','sciona.amex_raw.preprocess','hashlib','json']),
 'features':recipe(
    'Build all input tables and sequences needed by the learners.',
    'Use the full prepared groups in their original order. Keep all nine groups of summary features and both sequence representations.',
    'Population dictionary from populations with aligned joint sequences and runtime roles. Build all nine source manual feature blocks, joint categorical/sequence vocabulary and raw numeric-plus-category row inputs. Preserve customer/month ranks, missing adjacency, timestamp rank windows, feature ordering and source quantization. Joint training/query operations are transductive; their use is not an unbiased validation claim.',
    {'population':'Coherent joint population dictionary produced by populations; ordering, label count and fill roles must correspond to the same invocation.'},
    'Dictionary with all nine blocks, encoded sequences and raw row-learner arrays, ordered training before query.',
    ['sciona.amex_manual.build','sciona.amex_sequence.encode_series','numpy']),
 'row_models':recipe(
    'Train five models using groups that keep each example sequence together.',
    'Use matching groups, features and labels. Run every training round and return predictions in the original order.',
    'Population and feature dictionaries from the same invocation. Five shuffled customer-stratified DART folds, seed supplied by population, source parameters and 4500 evaluated boosting rounds without active early stopping. Retained iterations may be lower for empty trees. Each training row receives one out-of-fold prediction; query predictions average five in-memory reloaded boosters. Row inputs are raw numeric and mapped category codes, not one-hot neural inputs.',
    {'population':'Training count, binary labels and bounded seed aligned with feature_values.',
     'feature_values':'Complete feature dictionary whose rows are aligned numeric/category matrices, at most thirteen rows per sequence.'},
    'Training/query per-sequence predictions, model count five, retained iterations, configured rounds and scope.',
    ['sciona.amex_row_model.fit','lightgbm','numpy','scikit-learn']),
 'assembly':recipe(
    'Join the summary features and row predictions in the required order.',
    'Use features and predictions from the same groups. Keep missing entries, padding and the source scaling rules.',
    'All nine manual blocks and complete row predictions from the same ordered population. Preserve BLOCK_ORDER, left-pad row predictions into thirteen slots, build tree features without those slots in manual, and build neural features using source greedy bin normalization plus zero-filled prediction slots. Undefined all-missing histograms and invalid shapes reject in the underlying helpers.',
    {'feature_values':'Feature dictionary containing all nine named source blocks for the same invocation.',
     'row_result':'Complete training_predictions and query_predictions lists from row_models, in matching group order.'},
    'Dictionary with manual feature matrix, thirteen prediction slots per group and assembled neural feature matrix.',
    ['sciona.amex_assembly.assemble','sciona.amex_sequence.prediction_slots']),
 'tree_models':recipe(
    'Train both sets of five downstream tree models.',
    'Use matching labels and feature tables. Train once without row prediction slots and once with them, keeping all rounds.',
    'Population and assembled dictionaries from the same invocation. Two five-fold DART variants, manual and manual_with_slots, each evaluating 4500 rounds with source parameters and no active early stopping. Split at training_count; stratify using source seed and binary labels. Retained iterations may be lower for empty trees. Preserve in-memory booster reload and all-round inference; return out-of-fold and averaged query predictions.',
    {'population':'Training count, binary labels and seed from the matching population.',
     'assembled':'Matching manual and slots matrices from assembly; source thirteen-slot boundary is retained.'},
    'Dictionary with both named variants, five models per variant, training/query probabilities, retained iterations and configured rounds.',
    ['sciona.amex_tree.fit','lightgbm','numpy','scikit-learn']),
 'neural_models':recipe(
    'Train both sets of five sequence models.',
    'Use matching sequences, features and labels. Keep all training epochs and reload the best eligible model before prediction.',
    'Population, encoded feature sequences and assembled neural features must originate from the same invocation. Five folds each for series and combined variants, hidden width128, ten epochs, batch256 with drop-last and right-padded masked sequences. Raw BCE, non-AMP CPU Adam rates .001 for epochs0-4, .0001 for5-8 and .00001 for9. Strict positive Amex-metric checkpoint improvement and readback validation; no eligible checkpoint rejects. Torch RNG is restored; no historical GPU/RNG parity claim.',
    {'population':'Matching training count, binary labels and bounded seed; every training fold must contain at least256 examples.',
     'feature_values':'Matching encoded sequences produced by features, in training-then-query order.',
     'assembled':'Matching neural feature matrix produced by assembly.'},
    'Both named variants with complete training/query probabilities, five fold histories and selected checkpoints; ten models, ten epochs and hidden width128 metadata.',
    ['sciona.amex_neural_training.fit','sciona.amex_network.Network','sciona.amex_metric.score','torch','numpy','scikit-learn']),
 'combination_inputs':recipe(
    'Order the four prediction vectors and supply the source weights.',
    'Use four equally sized vectors with values from zero to one. The weights are fixed by the recipe and are not rescaled.',
    'Manual tree, manual-with-slots tree, series neural and combined neural query probabilities in that exact order. Each must be a nonempty aligned finite one-dimensional real numeric vector in [0,1]. Convert each independently to float64 and stack. Emit coefficients [.30,.35,.15,.10] unchanged; no rank transform or normalization. Shape/range validation does not verify that supplied vectors came from trained models.',
    {'tree_result':'Both manual and manual_with_slots variant query_predictions from tree_models.',
     'neural_result':'Both series and combined variant query_predictions from neural_models for the same query groups.'},
    'Tuple of a fresh four-by-query float64 matrix and a four-element float64 coefficient vector, in declared output order.',
    ['numpy']),
 'result':recipe(
    'Encode the combined scores and label the complete recipe output.',
    'Use a nonempty vector of finite scores between zero and one. This step formats results; it does not check that models were trained.',
    'Nonempty one-dimensional float64 ndarray, finite and bounded in [0,1]. Return finite JSON-compatible scores plus fixed complete-recipe counts25 and row5/downstream_tree10/neural10, score kind literal_weighted_sum_0.9 and scope. The counts describe the bound complete graph; this encoder alone does not establish training provenance or score calibration.',
    {'scores':'Combined float64 score vector from the graph-bound ordered sum with the exact recipe branches and coefficients.'},
    'Finite JSON-compatible dictionary with scores, model-count metadata, score kind and scope.',
    ['numpy','json']),
 'ordered_weighted_sum':dict(
    operation='Multiply each numeric row by its coefficient and add the results in row order.',
    plain_limits='Use a nonempty table and one finite coefficient per row. Negative and zero coefficients are allowed. Do not rescale the coefficients. Reject overflow. Inputs are unchanged.',
    validity_bounds='Nonempty finite two-dimensional NumPy real numeric ndarray with dtype kind i/u/f and aligned finite one-dimensional numeric coefficients. Convert both to float64 before arithmetic and reject conversion overflow. Start with the first row product, then add remaining products left to right; reject any nonfinite product or partial sum. No dot product, pairwise sum, normalization or probability restriction. Preserve signed-zero first-product behavior. Return a fresh vector; input arrays are unchanged. Units, row and column meaning, and weight interpretation are caller policy.',
    parameters={'values':'Nonempty finite two-dimensional real numeric ndarray; rows are summands, columns are aligned output positions.',
                'coefficients':'Finite real numeric ndarray of shape (number of rows,); any signs and non-unit total are permitted.'},
    outputs='Fresh finite float64 vector with one element per input column.',
    dependencies=['numpy','sciona.ghost.registry (registration only)'],
    reuse='Domain independent linear combination for arbitrary aligned numeric quantities; no model, probability, competition or fixed coefficient meaning.'),
}


def run_tests():
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    with tempfile.TemporaryDirectory(prefix='amex-provider-tests-') as temporary:
        junit = Path(temporary)/'results.xml'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', *TESTS,
                                 '--junitxml='+str(junit)], cwd=ROOT, env=env, capture_output=True, text=True)
        require(result.returncode == 0, 'Provider tests failed: '+result.stdout[-2500:]+result.stderr[-1000:])
        suites = ET.parse(junit).getroot().findall('testsuite')
        counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        require(counts == dict(tests=79, failures=0, errors=0, skipped=0), 'Expected complete 79-test provider qualification')
    report = dict(passed=True, **counts, synthetic_only=True,
                  test_sha256={p:sha(ROOT/p) for p in TESTS},
                  runner_sha256=sha(Path(__file__)))
    TEST_REPORT.write_text(json.dumps(report, indent=2)+'\n')
    return report


def review():
    intake = reviewed_plan()
    integration = integration_review()
    tests = json.loads(TEST_REPORT.read_text())
    require(tests == dict(passed=True, tests=79, failures=0, errors=0, skipped=0, synthetic_only=True,
        test_sha256={p:sha(ROOT/p) for p in TESTS}, runner_sha256=sha(Path(__file__))),
        'Provider test evidence missing or stale')
    requirements = ROOT/'requirements/amex-execution.txt'
    dependencies = dict(line.split('==') for line in requirements.read_text().splitlines()
                        if line.strip() and not line.startswith('#'))
    require(all(importlib.metadata.version(k)==v for k,v in dependencies.items()), 'Runtime packages changed')
    files = {str(p.relative_to(ROOT)):p for p in (ROOT/'sciona').glob('amex_*.py')}
    files.update({str(p.relative_to(ROOT)):p for p in (ROOT/'sciona').glob('amex_*topology.json')})
    files.update({p:ROOT/p for p in ('sciona/decomposed_amex_pipeline.py',
        'sciona/decomposed_amex_graph.py',
        'sciona/visualizer/runner.py', 'sciona/services/execution_graph_codec.py',
        'requirements/amex-execution.txt')})
    files.update({p:ROOT.parent/p for p in PROVIDER_FILES})
    imports = {name:sorted({ast.unparse(n) for n in ast.walk(ast.parse(path.read_text()))
                           if isinstance(n, (ast.Import, ast.ImportFrom))})
               for name,path in files.items() if path.suffix=='.py'}
    direct = ['sciona/decomposed_amex_pipeline.py', *PROVIDER_FILES]
    require(not any('sciona.atoms.' in line for name in direct for line in imports[name]),
            'Unexpected hidden registered-provider import')
    records = []
    for record in intake['providers']:
        contract = CONTRACTS[record['runtime_fqdn'].rsplit('.',1)[1]]
        require(set(contract['parameters']) == {p['name'] for p in record['interfaces'] if p['direction']=='input'},
                'Parameter semantic coverage differs')
        records.append(dict(record, **contract, internal_registered_provider_dependencies=[],
            callable_dependency_contracts={}))
    evidence = ('amex_decomposed_provider_tests.json','amex_decomposed_semantic_review.json',
                'decomposed_amex_graph.json','decomposed_amex_pipeline.json','amex_decomposed_provider_intake_plan.json')
    return dict(format='amex-decomposed-provider-semantics.v1', passed=True, approved=False,
        review_source='automated', proposed_tier=3, verdict='acceptable_with_limits',
        providers=records, graph_digest=intake['graph_digest'],
        software_sha256={name:sha(path) for name,path in sorted(files.items())}, imports=imports,
        package_versions=dependencies, python_version=platform.python_version(),
        evidence_sha256={name:sha(ROOT/'docs/reviews'/name) for name in evidence},
        intake_stage_mapping=integration['source_stage_mapping'],
        limitations=[
            'Automated Tier 3 only; no Tier 1 human certification or Tier 2 usage qualification.',
            'Generic ordered summation is domain independent. Source weights and interpretation belong to the recipe or caller.',
            'Recipe intermediate dictionaries must come from one coherent invocation. Standalone stages do not authenticate their provenance; arbitrary mixing is unqualified.',
            'Complete synthetic CPU execution and exact reconstructed-reference agreement do not establish historical GPU parity, learned-weight parity or competitive accuracy.',
            'Independent reconstruction; original-source reuse grant remains unresolved. No original source code, private data or trained weights are embedded in these implementations.',
            'Joint vocabulary, month ranks and binning are transductive; downstream evaluation is not an unbiased nested-validation protocol.',
            'Source literal weights sum mathematically to .90 with left-to-right float64 arithmetic; outputs are uncalibrated weighted scores.',
            'Package versions are pinned; clean installation and deployment qualification are not claimed.',
            'Validation runs serially with native pools and Torch threads at one; deployment must configure its own thread limits.',
            'Software dependencies are recorded separately from registered atom dependencies. This review alone does not approve catalog records.'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-tests', action='store_true')
    if parser.parse_args().run_tests:
        run_tests()
    result = review()
    REPORT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), tests=79, approved=False)))
