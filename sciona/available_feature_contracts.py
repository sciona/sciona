"""Explicit catalog port contracts for reusable available-feature providers."""
import inspect

from sciona.architect.models import IOSpec


def contracts(name, implementation):
    """Return ordered callable inputs and named materialized outputs."""
    descriptions={
        'observed_at':'One-dimensional int64 observation times on the shared caller-declared clock, aligned to history rows.',
        'issued_at':'One-dimensional int64 forecast issue times; only issues at or before each query are available.',
        'valid_at':'One-dimensional int64 forecast valid times aligned to issue/value rows; issue-valid pairs must be unique.',
        'actual_at':'Int64 actual-event times on the observation clock; events cannot count before both actual and observed time.',
        'estimates':'One-dimensional int64 estimated event times, aligned to history identifiers; estimate_valid governs missing placeholders.',
        'estimate_times':'One-dimensional int64 estimated event times aligned to estimate identifiers and availability.',
        'estimate_observed_at':'One-dimensional int64 estimate observation times on the event/query clock.',
        'entity_ids':'Nonempty string identifiers aligned to estimate history rows; repeated identities are allowed.',
        'event_ids':'Nonempty string identifiers aligned to event rows; events are counted independently without implicit deduplication.',
        'estimate_ids':'Nonempty string identifiers aligned to estimate history rows.',
        'query_ids':'Nonempty string query identifiers; order and duplicates are preserved.',
        'record_ids':'Unique nonempty string identifiers aligned to authoritative snapshot rows; caller establishes snapshot availability.',
        'estimate_valid':'Boolean vector aligned to estimate history; false values mask placeholder estimates.',
        'countable':'Boolean event-by-channel mask aligned to actual_at; false entries exclude missing identities, categories or actual times.',
        'states':'Observation-by-channel string matrix with no missing strings; comma and literal substring semantics are explicit.',
        'vocabulary':'Unique ordered nonempty literal tokens supplied by the caller; never learned from query rows.',
        'suffixes':'Unique ordered nonempty suffixes defining match priority; no domain suffixes are inferred.',
        'change_windows':'Nonempty positive int64 grid-row counts; a full available window is required before its mask is true.',
        'grid_step':'Positive integer spacing in the shared clock; query grid must be strictly increasing and regular.',
        'lookbacks':'Nonempty positive int64 widths on the shared clock. Membership is (query-width, query].',
        'rounding_step':'Positive integer observation ceiling-grid spacing on the shared clock; overflow is rejected.',
        'ticks_per_output_unit':'Positive integer clock ticks per output time unit; applies consistently to time differences.',
        'ticks_per_unit':'Finite positive clock ticks per fallback output unit.',
        'history_window':'Positive integer clock width; historical issue membership is (query-width, query].',
        'lead_bands':'Nonempty list of (lower, upper] integer lead bounds; lower=None means unbounded below.',
        'contrast_leads':'Int64 matrix with two lead values per row; subtract right forecast from left within the latest issue only.',
        'revision_lags':'Nonempty positive int64 issue-row lags for the same valid time; missing revisions remain unavailable.',
        'holiday_midnights':'Sorted unique nonmissing timezone-naive holiday midnights covering the query horizon; caller chooses calendar rules.',
        'attributes':'DataFrame aligned positionally to unique record_ids; columns must be unique; caller establishes authoritative snapshot availability.',
        'vocabularies':'Explicit unique string vocabularies. Entity keys identify attribute columns; identifier keys are primary, numeric and category.',
        'fallback':'Explicit nonempty string category fallback; no inference-row category fitting occurs.',
        'delimiter':'Explicit nonempty identifier field delimiter.',
        'minimum_parts':'Integer minimum identifier field count of at least two; all parsed fields must be nonempty.',
        'primary_part':'Nonnegative primary-token field index below minimum_parts.',
        'category_part':'Nonnegative category field index below minimum_parts.',
        'reference_parts':'Nonempty ordered list of field indices below minimum_parts, concatenated to parse the reference time.',
        'reference_format':'Explicit pandas strptime format producing nonmissing timezone-naive reference times.',
        'seconds_per_unit':'Finite positive seconds per output elapsed-time unit; query and reference clocks must agree.',
        'queries':'Query DataFrame with unique columns and nonmissing declared join keys. Preserve original row order, duplicates and index.',
        'feature_tables':'Ordered (DataFrame, key-list) pairs. Keys must be unique/nonmissing on feature side; non-key collisions are rejected.',
        'frame':'Feature DataFrame with unique columns; metadata is excluded by explicit feature_columns.',
        'feature_columns':'Nonempty unique ordered feature names present in frame and completely covered by disjoint fill policies.',
        'numeric_fills':'Finite fixed per-column fill values in matching feature units; never fitted to query rows.',
        'categorical_fills':'String fills for categorical columns; finite integral categories normalize to strings.',
        'integer_dtype':'Explicit signed NumPy integer dtype. Truncation and overflow are checked, not wrapped.',
        'missing':'Finite fallback output value within [lower, upper], in the declared prediction units.',
    }
    inputs=[]
    for parameter in inspect.signature(implementation).parameters.values():
        key=parameter.name
        annotation=parameter.annotation
        typ=getattr(annotation,'__name__',str(annotation))
        if typ=='ndarray':
            typ='NDArray[np.bool_]' if key in ['estimate_valid','countable'] else 'NDArray[str]' if key.endswith('_ids') or key=='states' else 'NDArray[np.float64]' if key in ['values','offset'] else 'NDArray[np.int64]'
        elif typ in ['DataFrame','DatetimeIndex']:typ='pd.'+typ
        if key=='query_times':
            constraint=('Nonmissing timezone-naive query timestamps on the declared reference/calendar clock.' if 'DatetimeIndex' in typ
                else 'One-dimensional int64 query times on the shared clock; no future observations may contribute. Configuration queries form a regular increasing grid.')
        elif key=='values':
            constraint=('Aligned forecast-row by field float64 matrix; NaN means missing and infinity is rejected. Fields use explicit linear scalar units; circular statistics are not inferred.' if name=='forecast_features'
                else 'Finite real prediction vector aligned exactly with offset, in the same output units; no broadcasting or input mutation.')
        elif key=='offset':constraint='Aligned finite additive offset vector in prediction units.' if name=='clip_truncate' else 'Finite scalar subtracted after conversion to fallback output units.'
        elif key in ['lower','upper']:
            constraint='Ordered exact integer clipping bounds within the supported float64 integer range, in prediction units.' if name=='clip_truncate' else 'Finite ordered clipping bound in fallback output units; missing must lie within bounds.'
        else:constraint=descriptions[key]
        if name=='event_counts' and key=='actual_at':constraint+=' Shape is event rows by channels; masked placeholders are ignored.'
        inputs.append(IOSpec(name=key,type_desc=typ,constraints=constraint).model_dump())
    if name=='available_indices':
        outputs=[('indices','NDArray[np.int64]','Original row indices aligned to queries; -1 denotes unavailable. Equal-time ties select last input row.'),
            ('available','NDArray[np.bool_]','Query-aligned availability mask; false indices must not be dereferenced.')]
    elif name=='event_delays':
        outputs=[('count','NDArray[np.int64]','Query-by-window event counts.'),('value_count','NDArray[np.int64]','Query-by-window available delay counts.'),
            ('mean','NDArray[np.float64]','Query-by-window mean delay in declared output units; NaN for no available values.'),
            ('maximum','NDArray[np.float64]','Query-by-window maximum delay in declared output units; NaN for no available values.')]
    else:
        result_type={'configuration_features':'dict','estimate_features':'dict','forecast_features':'dict','calendar_features':'dict',
            'entity_attributes':'pd.DataFrame','identifier_features':'pd.DataFrame','assemble_tables':'pd.DataFrame','prepare_features':'pd.DataFrame',
            'event_counts':'NDArray[np.int64]','estimate_fallback':'NDArray[np.float64]','clip_truncate':'NDArray[np.int64]'}[name]
        detail={'configuration_features':'Category/count/change arrays with state_valid and window_valid masks; unavailable placeholders are not observations.',
            'estimate_features':'Query-aligned remaining/change features and query-by-window sum/max/sample-std arrays in declared output units; unavailable features are NaN.',
            'forecast_features':'Query-aligned historical, band, contrast and revision arrays in input channel units; missing values stay NaN and issue_available is explicit.',
            'calendar_features':'Eight query-aligned calendar arrays; holiday membership uses date, countdown uses the next midnight and whole elapsed days.',
            'event_counts':'Int64 array shaped queries by channels by lookback windows; tied events are counted independently.',
            'clip_truncate':'Owned int64 vector after offset addition, clipping and truncation toward zero; no rounding-to-nearest.',
            'estimate_fallback':'Owned float64 query vector; fractional estimates retained, missing groups use declared fallback.',
            'prepare_features':'Owned DataFrame in exact declared column order and original query index; numeric output has requested signed dtype, categories are strings.',
            'assemble_tables':'Owned query-preserving DataFrame with missing joins retained for a separate explicit imputation policy.',
            'entity_attributes':'Positional query-aligned DataFrame with explicit category fallbacks; unknown passthrough values remain missing.',
            'identifier_features':'Positional query-aligned lexical and time features; time features require a known authoritative identity; fallback derivation order is preserved.'}[name]
        outputs=[('result',result_type,detail)]
    return inputs,[IOSpec(name=key,type_desc=typ,constraints=detail).model_dump() for key,typ,detail in outputs]
