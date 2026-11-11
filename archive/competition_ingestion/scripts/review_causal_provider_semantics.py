"""Bound automated Tier 3 semantic review for the exact 32-provider closure."""
import importlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.review_causal_decomposed_bindings import review as binding_review
from scripts.inventory_causal_decomposed_providers import inventory,ROOT
from scripts.plan_flavours_provider_intake import sha

GROUPS=[
 ('typed_discretization',['discretize_and_bin'],'generic_with_limited_modes',
  'Discretize explicit numerical or categorical samples using the retained normalization rule.',
  'Finite nonconstant real sample vectors and explicit Numerical/Categorical/Binary types. Qualified source calls use ffactor=2 or 3, maxdev=3 and norm=True; the conditional-similarity branch also uses norm=False on explicitly normalized numerical samples with ffactor=2. Other settings require separate qualification.'),
 ('spacing_statistics',['knn_entropy_estimator','uniform_divergence'],'generic_with_limited_modes',
  'Compute spacing-based entropy and divergence statistics for an explicitly typed variable.',
  'Finite nonconstant vectors in defined numerical regimes, with native neighbor defaults. No density-consistency or causal-direction guarantee; alternative neighbor settings are not qualified here.'),
 ('pair_statistics',['hsic_independence_test','igci_asymmetry_score','normalized_error_probability','polyfit_nonlinearity_asymmetry','polyfit_residual_error'],'generic_with_limited_modes',
  'Measure pair dependence, spacing asymmetry, binned prediction error and polynomial fit asymmetry.',
  'Finite aligned vectors in defined regimes. Retain native default bandwidth/subsampling, binning and polynomial-degree conventions. Graph evidence covers scalar-variable pairs; multivariate HSIC and nondefault modes are not qualified. Scores alone do not establish causality.'),
 ('conditional_statistics',['conditional_distribution_similarity','conditional_noise_kurtosis_variance','conditional_noise_skewness_variance','discretized_conditional_entropy_variance'],'generic_with_limited_modes',
  'Measure variation of conditional distribution shape, moments or discretized entropy across conditioning bins.',
  'Retain each exact native default, including the strict count > minc condition and discretized-response entropy definition. Finite nonconstant typed sample pairs and defined outputs only. Zero for insufficient qualifying bins is a source convention, not evidence of independence or causality.'),
 ('paired_scores',['left_right_decomposed_prediction','symmetrized_prediction_fusion','two_stage_independence_direction'],'generic_with_limited_modes',
  'Combine supplied paired score vectors by subtraction, antisymmetry and multiplicative gating.',
  'Finite aligned score vectors in adjacent forward/reverse order; retain symmetrize=True where available. These functions do not train classifiers despite older narrative docstrings. Pair ordering is supplied by the caller; antisymmetry is algebraic, not a physical or causal certification.'),
 ('weighted_scores',['weighted_ensemble_combination'],'generic_with_limited_modes',
  'Compute an explicitly weighted combination of aligned score arrays.',
  'This evidence covers three finite score vectors and explicit nonnegative weights summing to one via the validated caller. The native weights=None stacked-output mode and other weighting regimes are not qualified here.'),
 ('source_score_ports',['three_system_causal_ensemble'],'recipe_adapter',
  'Route three explicit source score roles into the generic weighted combination.',
  'Finite nonempty aligned antisymmetric vectors in [-1,1], tolerance 1e-8, and three nonnegative weights summing to one within 1e-8. Source role order is fixed; no calibration or predictive-quality claim.'),
 ('class_probability_scores',['binary_positive_class_probability','causal_direction_probability_score'],'generic',
  'Select the positive-class probability or subtract negative from positive probability using explicit labels.',
  'Finite normalized probability rows and exact integer class sets {0,1}/{-1,1} for binary selection or {-1,0,1} for signed scoring. Direction selection remains P(+1), not a signed probability conversion. Labels establish column meaning, not causal validity.'),
 ('batch_statistics',['typed_pair_statistic'],'generic_with_limited_modes',
  'Evaluate one declared typed statistic independently and serially for each observation pair.',
  'Explicit supported statistic names, finite nonconstant real vectors of at least three samples, equal lengths within each pair, and exact binary cardinality. Variable lengths between pairs are allowed. Fixed scalar conventions include 0.7 entropy correction, arithmetic adjusted MI and categorical dispatch. Undefined results are rejected; no labels or population-wide fitting occur.'),
 ('feature_recipe',['assemble_selected_columns','feature_statistic_configuration','route_feature_populations'],'recipe_adapter',
  'Expose source statistic choices, independent population ports and fixed 43-column assembly.',
  'The source column order is intentionally fixed. All numerical statistics are upstream nodes; assembly uses generic finite aligned stacking. Caller supplies population meaning and alignment; routing itself does not prove population independence.'),
 ('orientation',['orient_typed_pairs','augment_pair_labels'],'recipe_adapter',
  'Validate and orient observation pairs and interleave signed forward/reverse labels.',
  'Explicit recognized types, nonconstant finite vectors, at least three samples, exact binary cardinality and matching within-pair lengths. Integer labels are restricted to {-1,0,1}, converted to int64 before negation. No prediction labels are consumed; abstract witnesses certify shapes only.'),
 ('model_roles',['validate_causal_model_roles','prepare_causal_training_partitions'],'recipe_adapter',
  'Expose role-specific labels, weights and row selections and validate the five fitted classifier roles.',
  'Adjacent reversed labels, all three training classes, source negative-class balancing, direction rows excluding zero labels with unit weights and one-step weights=None. Exact classifier label sets retained. Abstract-only direction cardinality and fitted class-role validation are unsupported; concrete execution is qualified.'),
 ('generic_models',['gradient_boosting_class_probabilities','fit_weighted_gradient_boosting'],'generic',
  'Fit one explicitly configured weighted classifier and predict labelled probabilities from its in-memory state.',
  'Dense finite nonempty real features, aligned integer labels with at least two classes, and None or strictly positive finite aligned weights. Arbitrary multiclass labels are tested. Prediction requires the fitted feature width and compatible sklearn classifier; no state deserialization. Serial fitting with one native thread. No predictive-quality or historical-library claim.'),
 ('empirical_numerics',['empirical_entropy','mixed_power_moment','stack_feature_columns'],'generic',
  'Compute empirical symbol entropy, mixed power moments and explicitly ordered feature stacking.',
  'Finite real nonempty arrays, aligned lengths where required, nonnegative finite entropy correction and nonnegative integer powers. Entropy uses exact discrete scalar/row symbols; moments do not normalize inputs. Float64 conversion precedes powers; nonfinite arithmetic is rejected. Stacking neither infers column meanings nor proves row identity.'),
]

TESTS=['tests/test_causal_selected_features.py','tests/test_causal_prediction_execution.py',
       'tests/test_causal_training.py','tests/test_weighted_gradient_boosting.py',
       'tests/test_causal_decomposed_training.py','tests/test_causal_decomposed_prediction.py',
       'tests/test_causal_feature_building_blocks.py','tests/test_causal_decomposed_features.py',
       'tests/test_causal_discretization_modes.py']


def review():
    bindings=binding_review();current=inventory()
    groups={}
    for group,names,reuse,operation,bounds in GROUPS:
        for name in names:
            if name in groups:raise ValueError('Duplicate semantic coverage')
            groups[name]=dict(group=group,reuse_class=reuse,operation=operation,validity_bounds=bounds)
    if set(groups)!={p['runtime_fqdn'].rsplit('.',1)[1] for p in current['providers']}:
        raise ValueError('Semantic groups do not exactly cover provider closure')
    evidence_path=ROOT/'docs/reviews/causal_full_decomposition_source_parity.json'
    evidence=json.loads(evidence_path.read_text())
    if [(c['seed'],c['estimators_per_fit'],c['uncached_nodes']) for c in evidence['cases']]!=[(731,12,114),(732,12,114),(733,500,114)]:
        raise ValueError('Complete pinned-source cases required')
    for library,version in evidence['library_versions'].items():
        if importlib.metadata.version(library)!=version:raise ValueError('Numerical runtime drift')
    env=dict(os.environ)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
    env['PYTHONPATH']=str(ROOT);env['MPLCONFIGDIR']='/private/tmp/sciona-mpl'
    with tempfile.TemporaryDirectory(prefix='sciona-causal-semantic-') as temp:
        junit=Path(temp)/'results.xml'
        completed=subprocess.run([sys.executable,'-m','pytest','-q',*TESTS,'--junitxml='+str(junit)],cwd=ROOT,env=env,capture_output=True,text=True)
        if completed.returncode:raise RuntimeError(completed.stdout+completed.stderr)
        suites=list(ET.parse(junit).getroot().iter('testsuite'))
        counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
        if counts['tests']<70 or counts['failures'] or counts['errors'] or counts['skipped']:raise ValueError('Complete passing semantic tests required')
    providers=[]
    for p in current['providers']:
        module,name=p['runtime_fqdn'].rsplit('.',1)
        function=getattr(importlib.import_module(module),name)
        providers.append(dict(runtime_fqdn=p['runtime_fqdn'],catalog_fqdn=p['catalog_fqdn'],
            version_id=p['version_id'],content_hash=p['content_hash'],provider_sha256=p['provider_sha256'],
            native_defaults={n:repr(v.default) for n,v in inspect.signature(function).parameters.items() if v.default is not inspect.Parameter.empty},
            **groups[name]))
    return dict(passed=True,approved=False,catalog_mutations=0,proposed_tier=3,review_source='automated',
        graph_digest=current['graph_digest'],provider_count=len(providers),providers=providers,
        execution_evidence_sha256=sha(evidence_path),binding_review=bindings,test_results=counts,
        test_source_sha256={path:sha(ROOT/path) for path in TESTS},reviewer_sha256=sha(__file__),
        limitations=['Automated Tier 3 only; no Tier 2 community-usage evidence or Tier 1 human review claimed',
            'Numerical parity and synthetic tests establish computational behavior, not causal validity or cross-domain prediction quality',
            'Generic operations are reusable across disciplines within their bounds; recipe adapters retain explicit source conventions',
            'Stored artifacts have not yet passed publication/transaction qualification or original-CDG promotion',
            'In-process provisioned runtime only; clean-install distribution and historical library reproduction are not qualified'])


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/causal_provider_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'providers','binding_review','test_source_sha256','limitations'}},indent=2))
