"""Domain-neutral catalog contracts for materialized training lifecycle atoms.

Domain adapters retain their explicit graph contracts separately; these generic
contracts must not inherit source-specific population names, units or bounds.
"""
from sciona.architect.models import IOSpec


def contracts(symbol):
    definitions = {
        'fit_review.absolute_error': (
            [('observed','np.ndarray','Nonempty finite real vector of observed scalar targets.'),
             ('predicted','np.ndarray','Finite real vector with identical shape and target units; no broadcasting.')],
            [('score','float','Mean absolute error in the caller-declared target units.')]),
        'fit_review.accept': (
            [('model','object','Nonmissing opaque model object; backend qualification is separate.'),
             ('score','float','Finite scalar score in caller-declared units; lower is better.'),
             ('ceiling','float','Finite acceptance ceiling in the same units; equality is rejected. Negative scores and ceilings are permitted.')],
            [('model','object','Same object identity, returned only when score is strictly below ceiling.')]),
        'fit_review.importance': (
            [('model','object','Fitted native model supporting ordered feature names and single-thread feature importance.')],
            [('names','list','Ordered feature names from the fitted model.'),
             ('values','np.ndarray','Aligned one-dimensional native feature importance weights; normalization is separate.')]),
        'fit_review.normalize': (
            [('names','list','Unique nonempty ordered string feature names.'),
             ('values','np.ndarray','Aligned finite nonnegative weights with strictly positive total.')],
            [('importance','pd.DataFrame','Columns feature and importance; weights sum to one, descending order, original positional indices retained.')]),
        'graph_fallbacks.emit': (
            [('value','object','Materialized branch output, with type and units established by the branch contract.'),
             ('receive','object','Controller-owned callable invoked once with value; never deserialized from model state.')],
            [('result','object','Same branch value after delivery to its controller.')]),
        'graph_fallbacks.predict': (
            [('primary','dict','Complete hash-bound executable graph descriptor with declared output node and port.'),
             ('baseline','dict','Complete hash-bound fallback graph descriptor; validated even for empty queries.'),
             ('inputs','dict','Explicit runtime branch inputs; no discovery of datasets or models.'),
             ('count','int','Nonnegative positional row count shared by both branch outputs.'),
             ('constant','float','Finite scalar last-resort output in the same caller-declared units as both branches.')],
            [('values','np.ndarray','Finite aligned real vector. Primary then baseline execute lazily and sequentially; ordinary exceptions fall through. Empty returns float64 without executing branches.'),
             ('route','str','Exactly primary, baseline, constant or empty. Malformed graph descriptors and process-control exceptions propagate.')]),
        'labeled_populations.concatenate': (
            [('tables','list','Nonempty ordered DataFrames with unique columns; caller establishes availability.'),
             ('labels','list','Aligned unique nonempty string population labels.'),
             ('columns','list','Aligned nonempty unique ordered column-name lists; each includes order_column. Different selections form a union with pandas missing values.'),
             ('label_column','str','New nonempty label column absent from all tables and distinct from order_column.'),
             ('order_column','str','Present in each selected schema; values share comparable type and units.')],
            [('table','pd.DataFrame','Owned selected/labeled concatenation sorted by order_column using quicksort with reset index. Equal-key order is not guaranteed.')]),
        'materialized_prediction.average': (
            [('vectors','list','Nonempty ordered list of finite real vectors with identical shape and physical units.')],
            [('values','NDArray[np.float64]','Owned positional arithmetic mean using caller order; rejects nonfinite sum or incompatible shapes.')]),
        'materialized_prediction.predict': (
            [('model','object','Fitted native regressor with ordered feature_names_ and predict(thread_count=1).'),
             ('frame','pd.DataFrame','Feature names and order exactly match the fitted schema; caller supplies training-compatible units, encodings and availability.')],
            [('values','NDArray[np.float64]','Owned finite scalar predictions aligned to frame rows in model target units.')]),
        'model_banks.assemble_bank': (
            [('models','dict','Unique nonempty string model names mapped to nonmissing opaque objects.'),
             ('population_bindings','dict','Explicit named populations mapped to nonempty local slot-to-model-name mappings; no implicit normalization.'),
             ('shared_bindings','dict','Shared slot-to-model-name mapping disjoint from local slots. All named models must be referenced; shared slots require populations.')],
            [('bank','dict','Owned population/slot containers preserving model identities and aliases; backend validation and serialization are separate.')]),
        'named_regression.temporal_split': (
            [('frame','pd.DataFrame','Named feature table with nonempty unique columns; positional row alignment is authoritative.'),
             ('targets','np.ndarray','Aligned scalar target vector, finite on eligible rows, in caller-declared units.'),
             ('times','pd.DatetimeIndex','Aligned nonmissing timezone-naive times on one declared clock.'),
             ('eligible','np.ndarray','Boolean vector aligned to frame; explicitly determines eligible training and validation rows.'),
             ('cutoff','object','Nonmissing timezone-naive pandas-compatible timestamp on the times clock. Equality belongs to validation.'),
             ('offsets','np.ndarray','Aligned target-unit offsets, finite on eligible rows; subtracted before fitting.'),
             ('categorical_columns','list','Unique explicit names in frame; categorical columns are not inferred from domain naming.')],
            [('x_train','pd.DataFrame','Owned eligible rows strictly before cutoff, preserving original order and index.'),
             ('y_train','pd.Series','Aligned float64 target minus offset, retaining frame index.'),
             ('x_valid','pd.DataFrame','Owned eligible rows at or after cutoff; both splits must be nonempty.'),
             ('y_valid','pd.Series','Aligned float64 validation target minus offset in target units.'),
             ('categorical_indices','list','Ordered categorical column positions in the unchanged feature schema.')]),
        'named_regression.fit': (
            [('x_train','pd.DataFrame','Nonempty unique ordered training feature schema with fixed units and categorical encodings.'),
             ('y_train','pd.Series','Finite positional target vector; Series index must equal training frame index.'),
             ('x_valid','pd.DataFrame','Nonempty validation frame with identical ordered feature schema and units.'),
             ('y_valid','pd.Series','Finite aligned validation targets in training target units, with matching row index.'),
             ('categorical_indices','list','Unique integer indices within the feature schema.'),
             ('parameters','dict','Explicit CatBoost CPU constructor settings. thread_count, allow_writing_files, train_dir, devices and device_config overrides are rejected.'),
             ('early_stopping_rounds','int','Strictly positive integer patience; validation selects the best model.')],
            [('model','object','Fitted CatBoostRegressor with one native thread, no backend file writes, explicit validation and best-model selection. Empirical performance is not guaranteed.')]),
        'native_model_state.pack_bank': (
            [('bank','dict','Population slot maps containing fitted CatBoost regressors; shared object identities and typed slots are preserved.')],
            [('model_state','dict','JSON-compatible native model-bank envelope, encoding each distinct object once with integrity hashes and exact backend/schema metadata. Private runtime state; hashes are not authentication.')]),
        'native_model_state.unpack_bank': (
            [('state','dict','Model-bank envelope with valid integrity, references, typed slots and exact supported backend version; treat payload as private runtime state.')],
            [('bank','dict','Restored native model objects with shared identities and slot types; schema and all references verified.')]),
        'policy_state.bind': (
            [('model_state','dict','JSON-compatible finite model-state dictionary; backend validation is separate.'),
             ('policy','dict','Explicit JSON-compatible finite policy dictionary; caller defines units, schemas and availability semantics.')],
            [('state','dict','Owned canonical JSON payload with format and SHA256 binding model state to policy. Integrity only, not authentication.')]),
        'policy_state.unbind': (
            [('state','dict','Exact supported envelope fields, format and verified SHA256 of the canonical finite JSON payload.')],
            [('model_state','dict','Owned restored model-state dictionary; backend decoding remains separate.'),
             ('policy','dict','Owned restored policy dictionary; application-specific policy validation remains separate.')]),
    }
    return tuple([IOSpec(name=n,type_desc=t,constraints=c).model_dump(mode='json') for n,t,c in side]
                 for side in definitions[symbol])
