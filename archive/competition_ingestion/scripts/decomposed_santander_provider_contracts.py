"""Bounded Santander recipe contracts; numeric combination remains generic."""


def recipe(operation, plain, bounds, parameters, outputs, dependencies):
    return dict(operation=operation, plain_limits=plain, validity_bounds=bounds,
        parameters=parameters, outputs=outputs,
        dependencies=dependencies+['sciona.ghost.registry (registration only)'],
        reuse='Independent supervised neural/tree pseudo-label recipe. Intermediate populations and model records must come from the same invocation. Structural checks do not authenticate their provenance. Global encoding and query-dependent preparation prevent independent generalization claims. Historical winner equivalence is unproven.')


PREPARED = 'Canonical same-invocation Prepared returned by santander_prepare, containing private populations and explicit lifecycle controls.'
NEURAL = ('Regenerate supervised class-presence categories and substituted features, then scale the neural populations jointly with query rows. '
    'Train every seed in each of ten stratified folds for fifteen epochs, using the full shared feature-group attention network. '
    'Retain class-preserving triplet augmentation, repeated epoch permutations, Adam updates with explicit decoupled decay and cosine scheduling. '
    'Restore the epoch with strictly best validation AUROC, verify its score, predict every query row, and average average-tie ranks across all fold/seed predictions. '
    'Return mean ranks, every model probability vector and fold/seed checkpoint histories. Validation scores use globally prepared features and are not independent estimates.')

CONTRACTS = {
    'initial': recipe('Train all initial neural folds and retain their query ranks.',
        'Use checked private populations and all supplied seeds. Each fold runs the full training schedule. These ranks are not probabilities.',
        'Require exact Prepared type, matching fingerprint and canonical payload revalidation. Convert population values to floating arrays. '+NEURAL,
        {'prepared': PREPARED},
        'Dictionary containing mean_ranks, stacked model_probabilities, model records and validation_scope; no model objects are returned.',
        ['sciona.decomposed_santander_runtime.initial','sciona.santander_execution','sciona.santander_training','sciona.santander_preparation','sciona.santander_network','numpy','torch','scikit-learn','scipy']),
    'selections': recipe('Select both complete hard pseudo-label tails and expand the labeled populations.',
        'Keep all requested high and low tails. Reject tied selection boundaries. Selected query rows also stay in the reference population.',
        'Revalidate Prepared. Require an initial-result dictionary containing mean_ranks aligned to all query rows. Call expanded_population for neural positive/negative counts 5000/3000 and tree counts 2700/2000. Both selections complete before any branch refit. Preserve retain_selected policy, finite aligned floating populations, binary labels, disjoint high/low selections and rejection of ambiguous ties. Append selected query rows and hard labels to original training rows; copy the full query population. No tail shrinking or hidden fitting.',
        {'prepared': PREPARED, 'initial_models':'Same-invocation initial neural result with mean_ranks aligned to the entire query population.'},
        'Neural and tree dictionaries each containing expanded training, labels, copied full query, selected_positions and selected_labels.',
        ['sciona.decomposed_santander_runtime.selections','sciona.santander_execution','sciona.santander_retraining.expanded_population','sciona.santander_pseudo_labels','numpy']),
    'branches': recipe('Regenerate features and refit every neural and tree fold.',
        'Supply both selected populations from this run. Train the neural branch then the tree branch. Query rows remain in their original order.',
        'Revalidate Prepared and require a dictionary with exactly neural and tree keys. Each selected population must supply training, labels and query accepted by the unchanged learner. '+NEURAL+' Then regenerate tree features for the tree-specific expanded population. For every stratified fold and seed, keep original fit rows and append sixteen positive and four negative shuffled copies, preserving feature triplets within class. Train deterministic single-thread LightGBM with explicit caller controls, held-out AUROC early stopping and selected-iteration verification. Predict all query rows, retain finite probabilities and average average-tie model ranks. Intermediate selection provenance is a precondition, not reauthenticated by this adapter.',
        {'prepared': PREPARED, 'selected':'Same-invocation complete selections output containing both expanded populations.'},
        'Dictionary with neural and tree learner results, each containing mean_ranks, model_probabilities, model records and validation_scope.',
        ['sciona.decomposed_santander_runtime.branches','sciona.santander_execution','sciona.santander_training','sciona.santander_tree','sciona.santander_encoding','sciona.santander_augmentation','numpy','torch','lightgbm','scikit-learn','scipy']),
    'blend_inputs': recipe('Rank the two branch scores and expose the ordered recipe weights.',
        'Use aligned query scores from both branches. Tied scores receive average ranks. The weights remain separate from the reusable numeric sum.',
        'Require exactly neural and tree result keys. Read their mean_ranks as nonempty finite real one-dimensional arrays of equal shape. Rank each vector with scipy average ties, then stack neural first and tree second. Return coefficients [2.1,1.0]. This adapter neither adds the rows nor divides by 3.1. Shape equality does not establish query identity or ordering; same-invocation alignment is required.',
        {'fitted':'Same-invocation branch results with aligned finite mean_ranks.'},
        'Tuple of a fresh two-row rank matrix and a two-element coefficient vector. Caller arrays are unchanged.',
        ['sciona.decomposed_santander_runtime.blend_inputs','numpy','scipy.stats.rankdata']),
    'result': recipe('Normalize the weighted sum and retain full selection and model counts.',
        'Divide only after the numeric rows have been added. Keep all query scores and the counts from the same run. Scores are ranks, not probabilities.',
        'Revalidate Prepared. Require exactly neural and tree keys in selected and fitted dictionaries. Require a finite real weighted vector aligned to the full query population. Divide by 3.1 after the ordered weighted sum, preserving original floating operation order. Count selected binary labels, expanded labeled rows, retained query rows and each branch model-record list including initial neural records. Return all scores, counts, rank_blend marker, retained-reference policy and global-encoding validation limitation. Require strict finite JSON serialization. Count records and selection provenance are not authenticated here.',
        {'prepared': PREPARED, 'initial_models':'Same-invocation initial result containing its complete models record list.',
         'selected':'Same-invocation selected populations with labels, query and selected_labels.',
         'fitted':'Same-invocation neural/tree results with complete models record lists.',
         'weighted':'Same-invocation generic ordered sum of ranked neural/tree rows with coefficients 2.1 and 1.'},
        'Strict JSON dictionary containing scores, selection_counts, model_counts, score_kind, reference_policy and validation_scope.',
        ['sciona.decomposed_santander_runtime.result','sciona.santander_execution','numpy','json (Python standard library)']),
}
