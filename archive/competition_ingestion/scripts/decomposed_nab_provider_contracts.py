"""Separate domain-independent threshold semantics from NAB recipe policy."""


def recipe(operation, plain, bounds, parameters, outputs, dependencies):
    return dict(operation=operation, plain_limits=plain, validity_bounds=bounds,
        parameters=parameters, outputs=outputs,
        dependencies=dependencies+['sciona.ghost.registry (registration only)'],
        reuse='Generic anomaly recipe adapter with explicit caller controls. Inputs must belong to one invocation; structural validation does not authenticate computation provenance. Benchmark labels are used only in evaluation, never forest fitting or thresholds.')


CONTRACTS = {
    'forest_scores': recipe(
        'Fit every prior-window isolation forest and score the current window.',
        'Supply checked ordered numbers and explicit model settings. Training uses earlier observations only. Results stay private.',
        'Require the exact Prepared type and revalidate its payload. Accept finite univariate real values within float32 forest range. For each index from window_size-1 onward, fit only windows ending at indices from max(window_size-1,index-fit_history) through index-1 once min_fit_windows exist. Use every configured tree, max_samples capped at training population size, all features, no bootstrap, automatic contamination, n_jobs=1 and seed (seed+index) modulo 2**32. Score the current window using negative score_samples, requiring a finite score in [0,1]. Warmup scores remain None; models_fitted counts all fits. No benchmark labels enter fitting. No persistent model serialization or historical detector parity is claimed.',
        {'prepared':'Same-invocation Prepared payload from nab_prepare, with complete explicit detector controls and private ordered observations.'},
        'Dictionary with anomaly_scores aligned to every input position and models_fitted integer. No thresholding is hidden in this stage.',
        ['sciona.decomposed_nab_runtime.forest_scores','sciona.nab_contract','numpy','scikit-learn']),
    'threshold_inputs': recipe(
        'Expose finite forest scores and explicit rolling threshold controls.',
        'Use the complete forest result from the same run. Only initial warmup slots are removed; the recipe supplies multiplier three.',
        'Revalidate Prepared and exact forest field set. Require one score per observation; the first min(population length,window_size-1+min_fit_windows) entries are None and all later entries are Python float values in [0,1]. Require integer models_fitted equal to the number of non-warmup entries. Return a fresh score-list slice, threshold_history, min_threshold_scores, std_ddof and multiplier 3.0. No averaging, standard deviation or threshold comparison occurs here.',
        {'prepared':'Same-invocation checked Prepared payload.', 'forest':'Complete forest_scores dictionary for that payload.'},
        'Tuple of finite score list, integer history, integer minimum, integer ddof and Python float 3.0.',
        ['sciona.decomposed_nab_runtime.threshold_inputs','sciona.nab_contract','numpy']),
    'detection': recipe(
        'Restore warmup slots and compare each score strictly against its prior threshold.',
        'Use scores and thresholds from the same run. Warmup produces no detections; equality with a threshold is not a detection.',
        'Revalidate the complete forest result. Thresholds are a list aligned to the non-warmup score population, with None for the initial min_threshold_scores slots and finite Python floats thereafter. Require exact population lengths. Prepend forest warmup None values, copy scores, and set each flag only when a threshold exists and score is strictly greater. Thresholds may exceed [0,1]. This stage checks structure and performs comparison; it does not recompute or authenticate supplied thresholds.',
        {'prepared':'Same-invocation checked Prepared payload.', 'forest':'Complete forest result.',
         'thresholds':'Aligned prior-only threshold list from the graph-bound generic operation.'},
        'Dictionary containing full anomaly_scores, thresholds, boolean detections and models_fitted.',
        ['sciona.decomposed_nab_runtime.detection','sciona.nab_contract','numpy']),
    'result': recipe(
        'Evaluate complete detections with corrected benchmark scoring and normalization.',
        'Supply the complete detection result and evaluation windows. Scores measure this supplied population, not generalization accuracy.',
        'Require exact detection fields, complete score/threshold populations, Python boolean flags and consistency with strict threshold comparison. Convert flags to numeric zero/one and evaluate at threshold 0.5 with caller-supplied disjoint inclusive index windows, positive cost weights and probation fraction. Retain source probation cap, per-point confusion counts, best reward per window and preceding-window false-positive penalties. Below-minimum thresholds select the final source sweep state; singleton post-window penalty uses one row. Perfect normalization counts all supplied windows while the null baseline counts scorable windows. Return None when normalization denominator is zero; do not clip normalized scores. Reject scoring overflow. Return the detection object by reference and a new evaluation dictionary.',
        {'prepared':'Same-invocation checked Prepared payload containing evaluation-only windows, costs and probation.',
         'detected':'Complete same-invocation detection output with coherent flags and thresholds.'},
        'Dictionary containing detection and evaluation, including raw/null/perfect/normalized scores, counts and scored population sizes.',
        ['sciona.decomposed_nab_runtime.result','sciona.nab_contract','sciona.nab_scoring.evaluate','numpy']),
    'prior_location_scale_threshold': dict(
        operation='Compute prior-only rolling mean plus an explicit multiple of standard deviation.',
        plain_limits='Use finite numbers and choose the history length, minimum count and multiplier. The current value is excluded. Empty input is allowed; inputs stay unchanged.',
        validity_bounds='Values must be a Python list of finite Python int/float values, excluding bool, convertible to finite float64. History, minimum and ddof are Python integers excluding bool, with 1 <= minimum <= history and 0 <= ddof < minimum. Multiplier is a finite Python int/float, excluding bool, and may be zero or negative. At index i reduce values[max(0,i-history):i] using NumPy float64 mean and std(ddof), then add multiplier times std. Return None until minimum prior observations exist. Empty input returns an empty list. Reject conversion and intermediate reduction overflow, even when a later multiplication could cancel it. No missing-value policy, comparison, unit, detector, category or fixed multiplier is prescribed.',
        parameters={'values':'Finite Python real observation list; no bool or missing entries.',
                    'history':'Positive Python integer limiting prior observations in each reduction.',
                    'minimum':'Positive Python integer no greater than history; emit None before this many prior values.',
                    'ddof':'Python integer in [0,minimum), subtracted from sample count in variance denominator.',
                    'multiplier':'Finite Python real scale, including zero or negative values; application chooses meaning.'},
        outputs='Fresh list matching input length; initial None entries followed by finite Python float thresholds.',
        dependencies=['numpy','math (Python standard library)','sciona.ghost.registry (registration only)'],
        reuse='Domain-independent rolling numeric reduction for ordered measurements, estimates or scores. Units, interpretation, history policy, multiplier and downstream comparison belong to the caller.'),
}
