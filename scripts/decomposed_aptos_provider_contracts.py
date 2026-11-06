"""Explicit reuse boundaries for the four new APTOS graph providers."""


def recipe(operation, plain, bounds, parameters, outputs, dependencies):
    return dict(operation=operation, plain_limits=plain, validity_bounds=bounds,
        parameters=parameters, outputs=outputs,
        dependencies=dependencies+['sciona.ghost.registry (registration only)'],
        reuse='Source recipe adapter. Image families, target roles, training settings and fixed cuts are recipe policy. Intermediate dictionaries must come from the same invocation; standalone adapters do not authenticate their origin.')


CONTRACTS = {
    'train_scores': recipe(
        'Train all eight image models in both stages and return ensemble scores.',
        'Supply all required image groups, model files and settings. Train every model, check saved predictions and remove temporary checkpoints.',
        'Population contains nonempty disjoint base, average, grouped and pseudo roles. Base and average labels are 0..4; grouped labels are 0..3. Queries may overlap pseudo inputs but never supplied-label roles. Exact image coverage, nonempty uint8 RGB arrays. Four reviewed pretrained families with two replicas each; native family resolutions, trainable GeM and scalar head. First-stage budgets are positive five-epoch multiples; all eight fits precede the teacher mean. Stage two continues each own checkpoint for ten epochs with fresh constant-rate Adam and SmoothL1. Preserve source target refinement, augmentation and partial batches. Exact checkpoint prediction replay and cleanup on success or failure. RNG state changes. No historical training or accuracy parity claim.',
        {
            'plan': 'Population produced by the existing checked population provider; all roles are required and must remain coherent.',
            'query_keys': 'Nonempty unique opaque query keys; may overlap pseudo keys but not supplied-label groups.',
            'images': 'Private runtime mapping covering exactly all population and query keys with nonempty uint8 RGB HWC arrays.',
            'references': 'All four family names mapped to local safetensors paths and reviewed SHA256 digests; external artifact permissions apply.',
            'first_stage_epochs': 'Exactly eight family/replica tuple keys with integer budgets at least five and divisible by five.',
            'seeds': 'Exactly eight family/replica tuple keys with distinct nonnegative 32-bit integer seeds.',
            'batch_size': 'Positive integer; final partial training batches are retained.',
            'learning_rate': 'Finite positive real number, excluding bool; constant Adam rate in each stage.',
            'lower_deviation': 'Explicit finite nonnegative lower refinement deviation, checked by second_stage_targets.',
            'upper_deviation': 'Explicit finite nonnegative upper refinement deviation, checked by second_stage_targets.',
            'tie_policy': 'Explicit upper or lower equality policy, validated before training and forwarded for final indexing.',
            'work_root': 'Existing private parent directory; the implementation removes its temporary checkpoint subdirectory.'},
        'Dictionary containing float64 scores, tie policy, all sixteen fit histories, both prediction maps, second-stage targets and replay/cleanup flags. Runtime results may contain private information and must not be published.',
        ['sciona.decomposed_aptos_pipeline.train_scores', 'sciona.aptos_population',
         'sciona.aptos_pipeline', 'sciona.aptos_initialization', 'sciona.aptos_models',
         'sciona.aptos_training', 'sciona.aptos_ensemble', 'numpy', 'torch',
         'torchvision', 'Pillow', 'pretrainedmodels', 'safetensors']),
    'cut_inputs': recipe(
        'Expose ensemble scores and the recipe cut points as graph inputs.',
        'Use finite scores from the training result and an explicit equality rule. Return copies with the four fixed cut points.',
        'Input dictionary must contain a nonempty finite one-dimensional float64 scores array and tie_policy equal to upper or lower. Return scores.copy(), float64 cuts [.7,1.5,2.5,3.5] and the policy. This boundary does not establish training provenance or calibrate thresholds.',
        {'training_result': 'Coherent output from train_scores for the same invocation.'},
        'Tuple of fresh scores, fresh four-element cut vector and equality-policy string, in that order.',
        ['sciona.decomposed_aptos_pipeline.cut_inputs', 'sciona.aptos_ensemble.CUT_POINTS', 'numpy']),
    'result': recipe(
        'Attach class indices to the complete training result.',
        'Use matching scores and integer indices from the same graph run. Keep all training details and copy the indices.',
        'Exact training-result field set is required. Scores and policy satisfy cut_inputs. Ordinals must be an integer ndarray of matching shape with values 0..4. Remove tie_policy and attach a copy of ordinals; other values remain shared. This checks representation, not whether indices equal the recipe cuts, whether flags are true or whether models were trained; complete graph qualification supplies those guarantees.',
        {'training_result': 'Complete same-invocation train_scores result.',
         'ordinals': 'Aligned integer vector from the graph-bound cut operation using the recipe scores, cuts and equality policy.'},
        'Complete result dictionary with scores, copied ordinal indices, fit histories, predictions, targets and replay/cleanup flags.',
        ['sciona.decomposed_aptos_pipeline.result', 'numpy']),
    'ordered_cut_indices': dict(
        operation='Assign each numeric value an index among ordered cut points.',
        plain_limits='Use finite vectors and strictly increasing cuts. Choose which side receives equality. Empty cuts are allowed. Inputs stay unchanged.',
        validity_bounds='One-dimensional native float32 or float64 NumPy arrays, both finite; values nonempty, cuts optionally empty and strictly increasing. Use numpy.searchsorted without rounding, clipping or casting. Upper means side=right; lower means side=left. Return a fresh integer vector from zero through the number of cuts. Mixed float32/float64 inputs follow NumPy comparison semantics. No labels, probability bounds, fixed thresholds or calibration are imposed.',
        parameters={'values': 'Nonempty finite one-dimensional float32 or float64 ndarray.',
            'cut_points': 'Finite one-dimensional float32 or float64 ndarray of strictly increasing cut points; may be empty.',
            'tie_policy': 'Explicit upper or lower string controlling exact equality at a cut.'},
        outputs='Fresh integer ndarray aligned with values, with indices in [0, number of cuts].',
        dependencies=['numpy', 'sciona.ghost.registry (registration only)'],
        reuse='Domain independent ordered interval indexing for any compatible numeric quantities. Units, thresholds, category names and downstream interpretation belong to the caller.'),
}
