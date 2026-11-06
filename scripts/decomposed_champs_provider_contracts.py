"""Explicit numerical reuse and recipe-policy boundaries for CHAMPS providers."""


def recipe(operation, plain, bounds, parameters, outputs, dependencies):
    return dict(operation=operation, plain_limits=plain, validity_bounds=bounds,
        parameters=parameters, outputs=outputs,
        dependencies=dependencies + ['sciona.ghost.registry (registration only)'],
        reuse='Source recipe adapter. Chemistry, identities, model selection, training schedules and retained count remain recipe policy. Inputs must belong to the same graph invocation; adapters do not authenticate provenance.')


CONTRACTS = {
    'train_models': recipe(
        'Execute complete chemistry, training and inference for all thirteen configured graph transformers.',
        'Supply the checked population and all thirteen training schedules. Provision the reviewed software locally. Checkpoints are temporary and outputs remain private.',
        'Require exact Prepared type from champs_prepare, with coherent private populations, model options and selection. Use hash-verified Bosch source, corrected packing and chunked evaluation. Fit vocabulary/scaling before the source split; reserve eight supervised vocabulary slots. Train all thirteen full architectures for caller-declared positive epoch counts, preserving supported optimizer/scheduler options, full batches, checkpoint selection and complete state reload. Infer with restored preprocessing, unscale per subtype and round each model prediction to six decimals. Unsupported motifs, absent vocabulary, degenerate scaling, invalid model options and missing type metrics can fail. Full selection keeps last epoch; validation selection requires all eight finite metrics. Remove temporary checkpoints on success or failure. Historical pretrained/accuracy/duration parity is not claimed.',
        {'prepared': 'Coherent unchanged Prepared object from the checked preparation provider for this invocation; all thirteen model schedules are mandatory.'},
        'Exact dictionary containing version, execution_version, model_predictions, coupling_types, models_completed and training. Model predictions are opaque-keyed maps; reports and predictions are private runtime state.',
        ['sciona.decomposed_champs_runtime.train_models', 'sciona.champs_contract', 'sciona.champs_source_runtime',
         'sciona.champs_preprocessing', 'sciona.champs_lifecycle', 'sciona.champs_training',
         'sciona.champs_prediction', 'sciona.champs_population', 'torch', 'numpy', 'pandas',
         'rdkit', 'openbabel-wheel', 'networkx', 'scipy', 'Pinned Bosch CHAMPS software manifest']),
    'ensemble_inputs': recipe(
        'Align all model predictions and select the source nine models per coupling category.',
        'Use all thirteen finite prediction maps with exactly matching record keys. The adapter supplies the selected values and the fixed retained count.',
        'Exact training-result field set is required. Prediction model keys equal MODEL_ORDER and each map covers exactly coupling_types keys. Keys are nonempty strings, categories are supported source categories, numeric values exclude bool and must fit finite float64. All model values are checked, including discarded ones. Apply the fixed source mask to produce nine selected values per record, in input record order. Empty inputs are accepted only with all thirteen empty maps. No averaging or sorting is hidden here; retained count five is explicit graph data.',
        {'training_result': 'Complete same-invocation output of train_models; model_predictions and coupling_types remain coherent.'},
        'Tuple of opaque record-key tuple, fresh column-contiguous float64 matrix with nine rows, and Python integer five.',
        ['sciona.decomposed_champs_ensemble.ensemble_inputs', 'sciona.champs_ensemble.MODEL_ORDER',
         'sciona.champs_ensemble.MODEL_MASK', 'sciona.champs_ensemble.COUPLING_TYPES', 'numpy']),
    'result': recipe(
        'Attach reduced predictions to the complete training result.',
        'Use keys and finite reduced values from the same graph run. Preserve model completion and training reports.',
        'Exact training-result field set is required. Keys must be a tuple of unique nonempty strings in exactly coupling_types iteration order. Values must be a finite float64 ndarray of matching one-dimensional shape. Return a fresh keyed prediction mapping and preserve version, execution_version, models_completed and training by reference. Empty coherent keys/values are accepted. This checks representation and key alignment, not whether values equal a central mean or whether any models were trained.',
        {'training_result': 'Complete same-invocation output of train_models.',
         'keys': 'Opaque key tuple from ensemble_inputs for the same invocation.',
         'values': 'Aligned finite float64 result vector from the graph-bound numerical mean.'},
        'Source-shaped dictionary with version, execution_version, predictions, models_completed and training.',
        ['sciona.decomposed_champs_ensemble.result', 'numpy']),
    'central_rank_mean': dict(
        operation='Average a caller-sized central rank interval independently in each numeric column.',
        plain_limits='Use finite real values and retain between one and all observations. Each column is independent. Empty columns are allowed; inputs stay unchanged.',
        validity_bounds='Two-dimensional NumPy array with at least one row and real integer/unsigned/floating dtype; reject booleans, complex, objects and nonfinite inputs. Retained is a Python int in [1, rows], excluding bool and NumPy scalar integers. Convert to a fresh column-contiguous float64 matrix, sort ascending per column, and apply NumPy mean to the slice starting at floor((rows-retained)/2) with exactly retained rows. Odd discarded counts remove one more upper-tail observation. Reject float64 conversion or reduction overflow, including intermediate sum overflow. Zero columns yield an empty float64 vector. No category, identity, model count, retained count, normalization or units are prescribed.',
        parameters={'values': 'Finite real two-dimensional ndarray, at least one row, optionally zero columns.',
                    'retained': 'Explicit Python integer count from one through the number of rows.'},
        outputs='Fresh finite float64 vector with one result per input column, or empty for zero columns.',
        dependencies=['numpy', 'sciona.ghost.registry (registration only)'],
        reuse='Domain-independent robust numeric aggregation. Rows may represent compatible repeated measurements, estimates or samples; their meaning, units and selected rank interval belong to the caller.'),
}
