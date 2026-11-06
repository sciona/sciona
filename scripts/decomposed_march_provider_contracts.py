"""Explicit semantics for the five March adapters; reusable numeric stages stay generic."""

CONTRACTS = {
    'features': dict(
        operation='Compute complete season box-score efficiencies, loser-to-winner PageRank and chronological scaled Elo.',
        parameters={'prepared':'Prepared from march_prepare; its deep-copied strict finite JSON payload is revalidated before feature work.'},
        validity_bounds='Require explicit finite feature and optimizer controls. Preserve the original possessions formula and ratios of summed totals, per-season rating resets, stable custom-scale Elo expectation, directed weighted PageRank with explicit tolerance and iteration budget, sorted game order, unique game chronology and rejection of tied completed games. PageRank exhaustion propagates as an error. Team vectors are float64 with five components: offensive efficiency, defensive efficiency, turnover rate, PageRank and Elo. No training or calibration occurs.',
        outputs='Dictionary containing seasons, training, calibration, prediction and regularization/max_iterations/tolerance/seed. Raw games are removed.',
        plain_limits='Build five team features from completed games using explicit settings. These features retain the season and box-score recipe.',
        reuse='Season feature policy is domain-specific. Numeric model fitting and calibration are separate and do not require these raw inputs.'),
    'pairings': dict(
        operation='Validate chronological populations and form mirrored fitting/calibration matrices plus both prediction orientations.',
        parameters={'featured':'Complete feature packet with nonempty season/team states, finite float64 five-component vectors, explicit last game order and all three target populations.'},
        validity_bounds='Deep-copy and validate exact packet fields. Require nonnegative Python integer season/team/order identifiers, distinct teams known to their season, exact target row fields and binary fitting/calibration outcomes. Every target follows its season feature cutoff. Labeled (season,order) identities are unique. All training seasons precede all calibration seasons, which precede all prediction seasons. Require at least two original fitting and two calibration matchups. Concatenate A, B and A-B for each orientation; append reversed rows and complemented labels in source order. Reject invalid optimizer controls. Supplied features and chronology must be trustworthy; structural checks do not authenticate provenance.',
        outputs='Exact dictionary train_x/train_y/calibration_x/calibration_y/forward_x/reverse_x plus four optimizer controls. Feature matrices have 15 columns under this recipe; fitting/calibration labels are mirrored int64.',
        plain_limits='Build ordered pairs and their reversals, keeping fitting, calibration and prediction seasons separate.',
        reuse='Pair construction retains the five-feature season recipe. The following numeric fitting stage accepts other feature widths independently.'),
    'fit_scores': dict(
        operation='Fit StandardScaler and binary LogisticRegression on training rows only and expose scores for external calibration and prediction.',
        parameters={'arrays':'Exact matrix packet: train_x/train_y/calibration_x/calibration_y/forward_x/reverse_x and regularization/max_iterations/tolerance/seed.'},
        validity_bounds='Prevalidate every input before fitting. Matrices are nonempty finite float64 rank-two arrays with a shared arbitrary positive feature width; forward and reverse shapes match. Labels are aligned int64 rank-one arrays with both 0 and 1 represented and at least two rows per labeled population. Copy arrays before use. Seed is a Python integer in [0,2**32); iteration budget is a positive Python integer; C and tolerance are finite positive Python int/float excluding bool. Fit StandardScaler then LogisticRegression using lbfgs and explicit controls. Promote ConvergenceWarning to an error. Require classes [0,1]. Evaluate three separate decision_function calls for calibration, forward and reverse; require finite rank-one float64 scores. No base refit during external calibration. Caller establishes independent populations and shared feature meaning; this stage does not prove either.',
        outputs='Tuple: calibration scores, float64 binary calibration targets, None sample weights, forward scores, reverse scores, and counts(training_rows,calibration_rows,predicted_matchups). Model objects are not returned.',
        plain_limits='Fit a binary model on numeric training rows and produce scores for separate calibration and prediction rows. All rows must use the same feature meaning.',
        reuse='Domain-independent dense matrix fitting with arbitrary positive feature widths, tested at 1, 3, 8 and 23 columns. No season, team or box-score inputs are required. Logistic solver and scaling policy remain explicit.'),
    'projection_inputs': dict(
        operation='Prepare the orientation policy as numeric rows and coefficients for an external ordered sum.',
        parameters={'forward':'Nonempty finite float64 rank-one forward probabilities in [0,1].',
                    'reverse':'Matching-shape finite float64 reverse probabilities in [0,1], calibrated with the same state.'},
        validity_bounds='Require identical vector shapes and strict dtype/range checks. Return rows [forward,ones,reverse] and coefficients [1,1,-1] as fresh arrays. Do not sum or rescale here. The graph must preserve both prediction orientations and shared calibration state.',
        outputs='Tuple of float64 matrix (3,N) and float64 coefficients (3,), consumed by generic ordered_weighted_sum.',
        plain_limits='Prepare forward plus one minus reverse, in that arithmetic order. Pair orientation is the caller recipe.',
        reuse='Works for compatible paired binary predictions across domains; assumes reversing a pair complements the target probability.'),
    'result': dict(
        operation='Divide the external ordered orientation total by two and restore the complete JSON result.',
        parameters={'total':'Nonempty finite float64 rank-one total in [0,2] from the same-invocation ordered sum.',
                    'counts':'Exact training_rows/calibration_rows/predicted_matchups dictionary from fit_scores.'},
        validity_bounds='Require Python integer counts excluding bool, at least two training/calibration rows, and predicted_matchups equal to total length. Multiply total by .5 and convert probabilities to a list; require strict JSON serialization. Bounds and counts do not authenticate execution or same-invocation provenance.',
        outputs='Strict JSON probabilities plus training_rows, calibration_rows and predicted_matchups.',
        plain_limits='Return paired prediction probabilities and row counts. The input total must belong to this invocation.',
        reuse='Compatible paired binary prediction aggregation across domains; preserves the original result naming and half-sum policy.'),
}
