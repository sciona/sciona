"""Explicit Santa recipe contracts; the existing numeric atom stays generic."""


def recipe(operation, plain, bounds, parameters, outputs, dependencies):
    return dict(operation=operation,plain_limits=plain,validity_bounds=bounds,
        parameters=parameters,outputs=outputs,
        dependencies=dependencies+['sciona.ghost.registry (registration only)'],
        reuse='Replay-trained two-regressor recipe adapter. Intermediate values must belong to the same invocation; structural checks do not authenticate their provenance. Hidden replay thresholds supply training labels only. Historical winning-agent equivalence and competitive accuracy are not established.')


CONTRACTS = {
    'fit': recipe(
        'Generate causal replay features and fit both held-out-selected threshold regressors.',
        'Supply complete checked replays and explicit controls. Both models train with one native thread. Private replays and fitted models stay private.',
        'Require exact Prepared type, matching configuration fingerprint and canonical payload revalidation. Training, validation and query game identities must be disjoint. Generate ten float32 features before each observed round for each eligible player perspective, with separate seed-derived population RNG streams and independent one-in-five arm-row sampling. Reject sampled populations exceeding max_rows. Hidden initial thresholds become labels only. Fit two regression GBDTs sequentially: raw threshold and 1.02 raised to threshold. Preserve .05 learning rate, .9 feature fraction, .5 bagging fraction every five rounds, deterministic column-wise training and num_threads=1. Use caller seed, leaf count, iteration cap and stopping patience. Held-out RMSE drives early stopping independently for each model; require finite validation predictions and positive transformed predictions. Retain both boosters, best iteration, independently computed held-out RMSE, per-iteration metric history and sampled row counts. Decision seed and query observations do not enter fitting. These explicit reconstruction controls are not authenticated historical notebook parameters.',
        {'prepared':'Same-invocation canonical Prepared object from santa_prepare with private replay populations and explicit controls.'},
        'ThresholdModels containing normal and transformed LightGBM boosters, evaluation histories and training/validation row counts. Model objects are process-local; portable persistence is not qualified.',
        ['sciona.decomposed_santa_runtime.fit','sciona.santa_execution','sciona.santa_boosting',
         'sciona.santa_replay_training','sciona.santa_agent_features','sciona.santa_threshold_posterior','numpy','lightgbm']),
    'predictions': recipe(
        'Predict both initial-threshold families using each fitted model’s selected iteration.',
        'Use both models and visible query history from the same run. Opponent rewards and hidden thresholds are not query inputs.',
        'Revalidate canonical Prepared and require exact ThresholdModels type. Rebuild the visible BanditHistory from own and opponent actions and own binary rewards. Construct both 100-by-10 float32 feature arrays; they differ in the posterior-mean feature. Predict each family with its booster best_iteration and num_threads=1. This stage passes predictions onward; the following blend-input stage checks finite values, exact lengths and transformed positivity. Structural model type does not prove that it was fitted on this payload.',
        {'prepared':'Same-invocation canonical Prepared including the visible active query history.',
         'models':'Complete ThresholdModels returned by fit for this invocation.'},
        'Dictionary with normal and transformed prediction arrays, one value per action in each family.',
        ['sciona.decomposed_santa_runtime.predictions','sciona.santa_execution','sciona.santa_boosting',
         'sciona.santa_agent_features','sciona.santa_threshold_posterior','numpy','lightgbm']),
    'blend_inputs': recipe(
        'Invert the exponential prediction and expose the recipe’s ordered blend inputs.',
        'Keep the transformed prediction first and the raw prediction second. Negative coefficients are intentional; do not clip or normalize them.',
        'Revalidate Prepared. Require exactly normal/transformed dictionary keys. Convert predictions to float64 arrays of shape (100,), require finite entries and strictly positive transformed values. Rebuild visible history to obtain the step. Compute the existing normalized sigmoid schedule: 1.2 times the difference between logistic(-10*step/2000) and logistic(-10), divided by .5 minus logistic(-10). Return rows log(transformed)/log(1.02) then normal, and coefficients ratio then 1-ratio. The initial raw coefficient is negative. This adapter performs no weighted reduction. The interpolation is retained reconstruction policy; current source prose supports its endpoints but does not establish the exact formula.',
        {'prepared':'Same-invocation canonical Prepared with active query history.',
         'predicted':'Complete same-invocation dictionary of both model prediction families.'},
        'Tuple of a new float64 matrix with two rows and one hundred columns, and a new two-element float64 coefficient vector. No input mutation.',
        ['sciona.decomposed_santa_runtime.blend_inputs','sciona.santa_execution','sciona.santa_agent_features',
         'sciona.santa_threshold_posterior','numpy','math (Python standard library)']),
    'result': recipe(
        'Apply shared selection decay, choose an action, and retain complete fit diagnostics.',
        'Supply the same-run numeric blend and models. Seeded ties and the all-nonpositive fallback are explicit recipe choices.',
        'Revalidate Prepared and require exact ThresholdModels type. Convert initial blended values to finite float64 shape (100,). Rebuild query history and multiply by .97 raised to the sum of own and opponent pulls for each action. Reject nonfinite scores. Generate a default_rng(decision_seed) permutation of all one hundred actions; choose the first maximum in that order when any score is positive, otherwise retain action zero. Return action, full score list, schedule ratio, query step, model count two, sampled row counts and each model evaluation excluding iteration_rmse. Require strict finite JSON serialization. The model evaluation dictionaries are retained by value but their authenticity is a same-invocation precondition, not established by type checks. No benchmark strength or expected-reward evaluation is implied.',
        {'prepared':'Same-invocation canonical Prepared containing visible history and decision seed.',
         'models':'Complete same-invocation ThresholdModels with row counts and evaluation dictionaries.',
         'initial':'Same-invocation ordered weighted-sum result with one finite value per action.'},
        'Strict JSON result with selected action, all one hundred decayed scores, schedule ratio, query step, two-model count and fitting diagnostics.',
        ['sciona.decomposed_santa_runtime.result','sciona.santa_execution','sciona.santa_boosting',
         'sciona.santa_agent_features','sciona.santa_threshold_posterior','numpy']),
}
