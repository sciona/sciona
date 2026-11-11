"""Bounded Porto recipe contracts; ordered numeric reduction remains generic."""


def recipe(operation, plain, bounds, parameters, outputs, dependencies):
    return dict(operation=operation, plain_limits=plain, validity_bounds=bounds,
        parameters=parameters, outputs=outputs,
        dependencies=dependencies+['sciona.ghost.registry (registration only)'],
        reuse='Independent binary classification recipe accepting caller-defined numeric populations and column roles across domains. Five neural branches and one tree are recipe policy. Joint feature preparation and unlabeled query participation are transductive; no out-of-sample transform or independent generalization estimate is provided. Intermediates must come from the same invocation; structural checks do not authenticate provenance. Historical winning-ensemble equivalence is unproven.')


PREPARED = 'Canonical same-invocation Prepared returned by porto_prepare, containing private populations and explicit preparation, five neural branch and tree controls.'
CHECKED = 'Require exact Prepared type, string configuration, matching SHA-256 fingerprint and canonical payload revalidation. '
CONTRACTS = {
    'features': recipe('Prepare joint feature populations for all neural branches and the tree.',
        'Use checked training and query rows with explicit column roles. Query features help define the transform. Labels do not enter it.',
        CHECKED+'Stack training then query rows. Require nonempty finite real matrices of equal width, disjoint unique integer column roles, and at least one retained feature. Drop designated columns, retain other numeric columns in original order, and append one-hot columns using sorted joint categories in caller categorical-column order. Preserve declared binary values and one-hot columns. On each other column use average-tie midpoint ranks over the joint population, inverse error function and mean centering. Keep sentinel values literal. Split copied raw tree and normalized DAE views back into original row order. This is a population transform, not a fitted transform for future rows.',
        {'prepared': PREPARED},
        'Dictionary with tree_training, tree_query, dae_training, dae_query, prepared_width and binary_output_columns. No labels or model objects are returned.',
        ['sciona.decomposed_porto_runtime.features','sciona.porto_execution','sciona.porto_preparation','sciona.porto_transforms','numpy','scipy']),
    'fit': recipe('Fit every configured neural branch and the prepared-feature tree.',
        'Train five full neural branches, then one tree. Each neural branch learns from unlabeled joint rows before learning from training labels. Use every supplied training epoch.',
        CHECKED+'Require transformed to be a dictionary with exactly tree_training, tree_query, dae_training, dae_query, prepared_width and binary_output_columns. Same-invocation feature provenance is a precondition. For each of five explicit configurations, fit a separate CPU denoising autoencoder on stacked normalized training/query rows without labels, using same-column swap noise with self-donors allowed, ReLU hidden layers, a linear decoder and clean-target mean squared error. Use explicit hidden widths, extraction layers, positive epochs/batch size, learning rate and decay in (0,1], noise/momentum in [0,1), bounded integer seed, SGD and per-epoch decay. Concatenate selected hidden activations only, then train the supervised ReLU classifier on labeled training features for every configured epoch using binary cross entropy, SGD, explicit L2/momentum, input/hidden dropout and inverted or unscaled training dropout. Validate finite float32 inputs, losses, gradients and predictions. Fit a single deterministic one-thread binary LightGBM on raw prepared training features with caller rounds, leaves, rates, sampling and regularization; the library may stop splitting before requested rounds. Require both binary classes and aligned populations. Retain full classifier loss histories, branch counts and DAE metrics; no historical settings are inferred.',
        {'prepared': PREPARED, 'transformed':'Complete same-invocation features result with aligned raw and normalized training/query views.'},
        'Dictionary with five-row neural_probabilities, tree_probabilities, neural_records, actual tree_iterations, models=6, dae_models=5 and prepared_width. Neural records retain training_loss, learned_width, DAE population/count/clean-error metrics and reference/query row counts. No fitted model objects are returned.',
        ['sciona.decomposed_porto_runtime.fit','sciona.porto_execution','sciona.porto_neural_branch','sciona.porto_autoencoder','sciona.porto_neural','sciona.porto_tree_blend','numpy','torch','lightgbm']),
    'blend_inputs': recipe('Expose five ordered neural probability rows and unit coefficients.',
        'Use five aligned neural vectors and one tree vector from this run. Each value must be a finite probability. Keep neural addition separate from recipe averaging.',
        'Require a fitted dictionary with neural_probabilities of shape (5,N), N positive, and tree_probabilities of shape (N,). Require real integer or floating arrays with finite values in [0,1]. Return a fresh C-contiguous float64 copy of the five neural rows in supplied order and a five-element vector of ones. This step validates the tree vector but does not add it, sum neural rows, normalize, rank, or reorder scores. Shape alignment alone does not establish query identity.',
        {'fitted':'Same-invocation complete fitted ensemble containing five aligned neural probability rows and the tree vector.'},
        'Tuple values, coefficients: fresh float64 neural matrix and unit vector. Caller arrays are unchanged.',
        ['sciona.decomposed_porto_runtime.blend_inputs','numpy']),
    'result': recipe('Add the tree after neural reduction and divide by six.',
        'Use the neural sum and tree scores from this run. Add the tree only after the five neural rows have been added, then divide once. Retain all query probabilities.',
        CHECKED+'Revalidate all five neural vectors and the tree through blend_inputs. Require neural_sum to be a finite real vector aligned to the complete query population and tree shape. Compute (neural_sum + tree_probabilities)/6 in that order, preserving the original floating-point evaluation. Require finite results in [0,1] and strict JSON serialization. Read models, dae_models, prepared_width and learned_width from fitted metadata without independently authenticating those counts or recomputing the supplied neural sum. Same-invocation provenance is required.',
        {'prepared': PREPARED, 'fitted':'Same-invocation fit output with complete probabilities and neural records.',
         'neural_sum':'Same-invocation generic ordered sum of the five neural rows using unit coefficients.'},
        'Strict JSON dictionary containing probabilities, models, dae_models, prepared_width, learned_widths, score_kind=equal_probability_mean and the explicit independent-configuration/transductive scope.',
        ['sciona.decomposed_porto_runtime.result','sciona.porto_execution','numpy','json (Python standard library)']),
}
