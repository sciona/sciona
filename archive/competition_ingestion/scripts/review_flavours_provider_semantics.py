"""Bound automated Tier 3 semantics for the complete reusable provider closure."""
import importlib
import importlib.metadata as metadata
import inspect
import json

from scripts.import_flavours_provider_drafts import ROOT
from scripts.inventory_flavours_decomposed_providers import inventory
from scripts.plan_flavours_provider_intake import sha
from scripts.review_flavours_decomposed_bindings import review as binding_review


# Scope describes the mathematical operation independently of its original task.
# Recipe adapters remain explicitly specialized; their reusable dependencies are
# individually selectable and are not assigned the adapter's domain restriction.
GROUPS = [
    ('distribution_metrics', ['compute_cvm_mass_decorrelation','compute_ks_agreement','roc_auc_truncated_weighted'],
     'generic', 'Compare weighted score distributions, score/protected-variable ranks, and weighted truncated binary discrimination.',
     'Aligned finite populations; valid weights/classes and window sizes. Fixed threshold decisions belong to the calling graph. Quality selection and identity independence remain caller responsibilities.',
     ['flavours_evaluation_wiring_qualification','flavours_decomposed_evaluation_execution']),
    ('source_configuration', ['classifier_view_inputs','mass_regression_inputs','neural_ensemble_inputs'],
     'recipe_adapter', 'Expose the recovered recipe as explicit feature views, hyperparameters, seeds, folds and ensemble sizes.',
     'Specialized to the retained five classifier configurations, seven regression seeds and five-by-ten neural ensemble; reuse generic fit/partition atoms for other recipes.',
     ['flavours_mass_graph_execution','flavours_classifier_graph_execution','flavours_neural_graph_execution']),
    ('evaluation_routing', ['diagnostic_report','discrimination_inputs','evaluation_inputs','split_population_scores'],
     'recipe_adapter', 'Separate evaluation-only populations and metadata, split ordered scores, select quality rows and report metric failures faithfully.',
     'Caller establishes population identity and disjointness. Strict quality >0.4, KS <0.09 and CvM <0.002 belong to this recipe. Selected row count is materialized; static propagation is explicitly unsupported.',
     ['flavours_evaluation_wiring_qualification','flavours_decomposed_evaluation_execution']),
    ('physical_transform', ['positive_longitudinal_kinematics'],
     'physical_transform', 'Construct positive-branch longitudinal momentum and a lifetime-based mass proxy for aligned daughter momenta.',
     'Consistent caller-defined units, finite nonnegative momenta, transverse momentum no greater than momentum, and positive distance/lifetime. Positive longitudinal branch only; no exact relativistic invariant-mass claim. Daughter count is generic.',
     ['flavours_pipeline_input_qualification']),
    ('source_feature_views', ['assemble_source_feature_views','enrich_corrected_mass_views'],
     'recipe_adapter', 'Assemble base, restricted, proxy and regression feature matrices; expose corrected-mass, residual and ratio features.',
     'Caller establishes the 46-column base order, excluded column and consistent mass units. Corrected width is 50. The 1e-10 ratio offset carries caller mass units; it is not invariant to arbitrary unit changes.',
     ['flavours_pipeline_input_qualification','flavours_decomposed_evaluation_execution']),
    ('training_routing', ['classifier_neural_view_pair','regression_view_pair','training_prediction_inputs'],
     'recipe_adapter', 'Validate and route distinct training/query arrays into explicit regression, classifier and neural views.',
     'Retained physical input layout and binary labels; sufficient class counts in every neural outer partition. Row alignment is checked, but subject independence and appropriate feature meaning are caller contracts.',
     ['flavours_pipeline_input_qualification','flavours_decomposed_evaluation_execution']),
    ('fixed_score_blend', ['six_model_power_blend'],
     'fixed_numeric_rule', 'Combine six ordered probability roles through the retained nonlinear power blend.',
     'Finite aligned float32/float64 probabilities in [0,1] and exact role order. Output is an unnormalized ranking score, not a calibrated probability or a general ensemble optimizer.',
     ['flavours_score_blend_qualification','flavours_decomposed_evaluation_execution']),
    ('masked_crossfit', ['fit_masked_configured_binary','fit_masked_configured_regression','shuffled_kfold_masks'],
     'generic', 'Produce deterministic shuffled fold masks and fit explicitly configured models on selected rows.',
     'Boolean aligned masks and sufficient selected rows; binary fits need both classes. Partitioning uses current sklearn KFold. Excluded feature/target values do not affect fitting; row disjointness does not imply subject independence.',
     ['configured_crossfit_qualification','flavours_mass_graph_execution','flavours_classifier_graph_execution']),
    ('masked_scaling', ['apply_standardization','fit_masked_standardization'],
     'generic', 'Fit dense finite per-feature mean/variance normalization on selected rows and replay numeric scaler state.',
     'Float64 matrices, ordered feature identities and consistent units. Constant-feature scale is one. The caller chooses fitting rows; the recovered neural recipe scales the complete outer training selection.',
     ['masked_standardization_qualification','flavours_neural_composition_qualification']),
    ('nested_partition', ['stratified_inner_masks'],
     'generic', 'Split an explicit outer selection into stratified inner fit and validation masks.',
     'Aligned integer labels and enough selected rows per class. Multiclass labels are supported. Labels outside the outer selection do not affect splitting; group-safe partitioning remains a separate caller requirement.',
     ['nested_stratified_masks_qualification']),
    ('prediction_accumulation', ['finalize_prediction_mean','masked_prediction_contribution','merge_prediction_sums'],
     'generic', 'Accumulate selected prediction sums and integer coverage counts, then compute an exactly covered mean.',
     'Float32/float64 contributions become float64 sums; ordered accumulation, nonnegative integer counts, finite selected values and exact expected coverage. Counts do not prove unique models or independent training.',
     ['prediction_accumulation_qualification','flavours_mass_graph_execution']),
    ('weighted_contributions', ['divided_prediction_contribution','finalize_prediction_sum'],
     'generic', 'Divide each selected contribution before summation and validate coverage without a second division.',
     'Float64 predictions, explicit positive finite divisor and exact coverage. The caller establishes weights; divide-before-sum rounding is intentional.',
     ['flavours_neural_graph_execution']),
    ('neural_diagnostics', ['masked_neural_loss','masked_neural_predictions'],
     'generic', 'Predict selected rows and compute selected stable softmax cross entropy without optimizer feedback.',
     'Matching numeric model state and ordered standardized features. Excluded prediction zeros are placeholders and must retain their selection mask. Diagnostic loss does not train, stop or select a model.',
     ['flavours_neural_graph_execution']),
    ('adagrad_binary', ['fit_masked_adagrad_binary','predict_adagrad_binary'],
     'generic', 'Train and replay a configurable ReLU hidden layer with two-class softmax and sequential AdaGrad updates.',
     'Finite float64 standardized features, binary int64 labels, ordered feature identity, explicit seeds/batches/epochs and numeric state. No historical Lasagne equivalence or predictive-quality guarantee; no early stopping.',
     ['numpy_adagrad_binary_qualification','flavours_neural_composition_qualification']),
    ('native_scaler', ['standard_scaler_fit','standard_scaler_partial_fit','standard_scaler_transform'],
     'generic_with_limited_modes', 'Fit and apply native standard-scaler state used by the reusable selected-row scaler.',
     'This approval evidence covers dense finite float64, centering/scaling enabled, sample_weight=None, and partial_fit with state=None. Sparse, weighted, incremental-update, missing-value and other API modes are not qualified by this review.',
     ['masked_standardization_qualification','flavours_neural_composition_qualification']),
    ('configured_tree_models', ['fit_configured_binary_model','fit_configured_regression_model'],
     'generic', 'Fit configured CPU binary or regression boosted trees and return the existing numeric/native model-state ABI.',
     'Nonempty float64 matrices with NaN missing features permitted and infinities rejected; aligned finite targets, ordered feature names, explicit supported parameters and full requested rounds. Serial execution with one native thread; concurrent fitting and other backend versions need qualification.',
     ['configured_xgboost_model_io_qualification','flavours_mass_graph_execution','flavours_classifier_graph_execution']),
    ('tree_prediction', ['predict_binary_model','predict_regression_model'],
     'generic', 'Replay compatible binary/regression tree model state on independent ordered feature rows.',
     'Exact reviewed XGBoost state format/backend and feature names/order; float64 inputs permit NaN missing features and reject infinities. Output is float32; binary single-return catalog name probabilities is an explicit graph alias. Existing approvals and evidence remain preserved.',
     ['configured_xgboost_model_io_qualification','flavours_decomposed_evaluation_execution']),
]

LIMITATIONS = [
    'Automated Community Tier 3 only: use at your own risk, no certification, warranty or liability coverage. No Tier 2 community-usage qualification or Tier 1 human certification.',
    'Qualified provisioned in-process CPU runtime and catalog selection only. Clean installation, deployment and binary/package redistribution are not established by this review.',
    'Generic atoms may be composed across disciplines within their numeric contracts. Recipe adapters retain explicit source-specific layout and configuration limits.',
    'Synthetic numerical execution establishes computational behavior and reuse, not empirical performance in another domain or historical engine equivalence.',
    'Model/scaler state and all caller data remain private runtime values; this review publishes no model weights or dataset records.',
    'The full recipe preserves 35 regression fits, 30 classifier fits, 50 neural fits, 111458 tree rounds and 150000 neural epochs. Out-of-fold diagnostics are not nested with upstream mass-feature fitting.',
    'Failed diagnostics remain failed. Quality-selected row counts require materialized execution rather than static shape propagation.',
]


def verify_report(name):
    path=ROOT/'docs/reviews'/(name+'.json')
    report=json.loads(path.read_text())
    if report.get('passed') is not True or report.get('synthetic_only') is not True:
        raise ValueError('Passing synthetic evidence required: '+name)
    tests=report.get('test_results',{})
    if tests.get('tests',0)<1 or any(tests.get(k,0) for k in ('failures','errors','skipped')):
        raise ValueError('Incomplete numerical tests: '+name)
    sources=report.get('source_sha256',{})
    for relative,digest in sources.items():
        if sha(ROOT/relative)!=digest:
            raise ValueError('Qualification source drift: '+relative)
    for package,version in report.get('runtime_versions',{}).items():
        if metadata.version(package)!=version:
            raise ValueError('Qualified runtime changed: '+package)
    # These two earlier reports used explicit fields instead of a source map.
    provider_root=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml'
    if name=='configured_crossfit_qualification':
        pins={provider_root/'model_selection/configured_crossfit.py':report['provider_sha256'],
              ROOT/'tests/test_configured_crossfit.py':report['tests_sha256']}
        pins.update({provider_root/p:d for p,d in report['dependencies_sha256'].items()})
    elif name=='configured_xgboost_model_io_qualification':
        pins={provider_root/'xgboost/configured_model_io.py':report['provider_sha256'],
              provider_root/'xgboost/model_io.py':report['reused_model_io_sha256'],
              ROOT/'sciona/flavours_classifiers.py':report['source_configuration_sha256'],
              ROOT/'tests/test_configured_xgboost_model_io.py':report['tests_sha256']}
    else:
        pins={}
        if not sources:raise ValueError('Missing qualification source pins')
    if any(sha(p)!=d for p,d in pins.items()):raise ValueError('Qualification source drift: '+name)
    return report,sha(path)


def review():
    bindings=binding_review()
    current=inventory()
    groups={}
    reports={}
    for group,names,reuse,meaning,bounds,evidence in GROUPS:
        for name in names:
            if name in groups:raise ValueError('Ambiguous semantic assignment')
            groups[name]=(group,reuse,meaning,bounds,evidence)
        for name in evidence:
            if name not in reports:reports[name]=verify_report(name)
    execution=reports['flavours_decomposed_evaluation_execution'][0]
    if execution['total_fits']!=115 or not execution['exact_retained_metrics_and_threshold_flags']:
        raise ValueError('Full evaluated execution evidence incomplete')
    providers=[]
    qualified_sources=set(execution['provider_sha256'].values())|set(execution['source_sha256'].values())
    for record in current['providers']:
        module,name=record['runtime_fqdn'].rsplit('.',1)
        group,reuse,meaning,bounds,evidence=groups.pop(name)
        function=getattr(importlib.import_module(module),name)
        if (sha(inspect.getsourcefile(inspect.unwrap(function)))!=record['provider_sha256']
                or record['provider_sha256'] not in qualified_sources):
            raise ValueError('Provider source changed')
        providers.append(dict(runtime=record['runtime_fqdn'],catalog_fqdn=record['catalog_fqdn'],
            version_id=record['version_id'],content_hash=record['content_hash'],source_sha256=record['provider_sha256'],
            semantic_group=group,reuse_class=reuse,operation=meaning,validity_bounds=bounds,
            evidence=evidence,proposed_tier=3,verdict='acceptable_with_limits'))
    if groups or len(providers)!=39:raise ValueError('Incomplete provider semantic coverage')
    attribution_path=ROOT/'docs/reviews/competition_flavours_reference_scope_review.json'
    attribution=json.loads(attribution_path.read_text())
    notices={name:digest for name,digest in attribution['sha256'].items() if name.startswith('docs/licenses/')}
    if attribution['source_attribution_status']!='conditional_pass' or len(notices)!=3:
        raise ValueError('Retained source attribution missing')
    if any(sha(ROOT/name)!=digest for name,digest in notices.items()):
        raise ValueError('Retained source or dependency notice changed')
    return dict(passed=True,approved=False,catalog_mutations=0,review_source='automated',proposed_tier=3,
        provider_semantic_review_passed=True,publication_transaction_qualified=False,
        providers=providers,limitations=LIMITATIONS,binding_review=bindings,
        evidence_sha256={k:v[1] for k,v in sorted(reports.items())},reviewer_sha256=sha(__file__),
        attribution=dict(review_sha256=sha(attribution_path),retained_notice_sha256=notices,
            scope='Independent implementation and catalog metadata; retained attribution and dependency notices. No third-party notebook license or binary redistribution rights inferred.'),
        remaining_gates=['Provider database corruption checks and atomic publication rollback/idempotence.',
                         'Publish exact providers with these validity bounds and verify serving.',
                         'Stage, execute and promote the decomposed CDG under its original identity.'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/flavours_provider_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],providers=len(result['providers']),proposed_tier=3,
        approved=False,publication_transaction_qualified=False)))
