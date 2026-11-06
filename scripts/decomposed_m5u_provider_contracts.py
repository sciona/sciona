"""Explicit uncertainty recipe contracts around a domain-independent reduction."""


def recipe(operation, plain, bounds, parameters, outputs, dependencies):
    return dict(operation=operation, plain_limits=plain, validity_bounds=bounds,
        parameters=parameters, outputs=outputs,
        dependencies=dependencies+['sciona.ghost.registry (registration only)'],
        reuse='Grouped nonnegative time-series uncertainty forecasting with caller-defined hierarchy roles, category partitions and calendar covariates. Applicability across domains requires the declared hierarchy relationships, sufficient calendar history, nonnegative inputs and known future covariates. Fourteen levels, nine quantiles, a 28-step horizon, sampling schedules and restoration strength are recipe policy. The independently bound ordered weighted sum is domain independent. Intermediate arrays must come from one invocation; structural checks do not authenticate provenance. No historical competition accuracy or winning-model equivalence is claimed.')


CONTRACTS = {
    'fit_forecast': recipe(
        'Train every quantile model and forecast the complete hierarchy before unit restoration.',
        'Accept validated history and hierarchy configuration. Run the full grouped search and repeated forecasting schedule. Return all quantiles and future steps without the final hierarchy adjustment.',
        'Require exact Prepared type, string configuration, matching SHA-256 fingerprint and canonical revalidation through prepare. Forward caller minimum_day and seed without speed overrides. Run aggregate controller -1 for levels 1 through 11, followed by separate base controllers 13,14,15. Preserve retained_base_reference normalization for aggregates and source normalization for base partitions. Source calendar cleaning must retain the forecast origin and sufficient year groups. Build complete rolling populations, sample each level with its factor-dependent full schedule, fit all nine quantiles using one bag and the latest outer holdout, and execute nested grouped randomized search with pinned parameter choices. Native LightGBM and search workers are one; models run serially. The latest outer group is excluded from training, but this is not proof of unbiased future accuracy. Run every repeated-query budget for all 28 horizons, preserving scaling, fold weighting and inverse scaling. Reject excessive query allocations rather than silently reducing them. Require all 14 levels and unique contiguous hierarchy coverage; sort outputs by hierarchy index. All 126 final models are required. Candidate-fit and query counts depend on the caller population; synthetic evidence counts are not universal constants. No restoration, quantile sorting, clipping or historical-weight loading occurs here.',
        {'prepared':'Canonical same-invocation Prepared from m5u_prepare, with private nonnegative history, prices, hierarchy roles, three category partitions, calendar, minimum_day and seed.'},
        'Dictionary containing unadjusted float64 forecasts shaped (9,28,N), positive per-series factors, adjustment=.7, levels, quantiles, hierarchy_indices, per-level training/query reports, normalization and scope. Private intermediates remain runtime values.',
        ['sciona.decomposed_m5u_runtime.fit_forecast','sciona.m5u_execution','sciona.m5u_pipeline','sciona.m5u_preparation','sciona.m5u_population','sciona.m5u_training','sciona.m5u_sampling','sciona.m5u_targets','sciona.m5u_bagging','sciona.m5u_inference','numpy','pandas','scikit-learn','lightgbm']),
    'restoration_inputs': recipe(
        'Restore native units and expose ordered median totals for generic reduction.',
        'Divide by each level multiplier. Compute median totals for levels one through nine in ascending order, and supply one coefficient per level.',
        'Require exact dictionary; forecasts convertible to exact float64 array (9,28,N), N positive, finite; aligned integer levels containing exactly 1..11,13,14,15; aligned finite positive real factors with one unique factor per level; exact quantile sequence (.005,.025,.165,.25,.5,.75,.835,.975,.995); and Python float adjustment=.7. Signed forecasts are allowed. Copy forecasts then divide by factors. For each ascending level 1..9, sum median forecasts over its series separately at each horizon. Reject nonfinite native values or totals and any zero total. Return a fresh unit coefficient vector. No cross-level addition or normalization is performed. Shape checks do not establish provenance or validate other metadata.',
        {'forecasted':'Complete same-invocation fit_forecast output before restoration.'},
        'Tuple (state, values, coefficients): state holds native forecasts, copied levels, ascending selected levels and copied median totals; values is float64 (9,28); coefficients is float64 (9,) of ones. Values and state totals are independent arrays.',
        ['sciona.decomposed_m5u_runtime.restoration_inputs','sciona.m5u_training (quantile and level constants)','numpy']),
    'restore': recipe(
        'Apply the fixed hierarchy adjustment using the independently summed median totals.',
        'Divide the supplied level sum by nine. Adjust levels one through nine toward that target with strength .7. Leave the other levels in their restored native units.',
        'Require exact dictionary state with finite float64 native forecasts (9,28,N), N positive, aligned integer levels containing exactly all 14 required levels, selected levels exactly 1..9 in order, finite nonzero float64 totals (9,28), and finite float64 supplied total (28,). Copy native forecasts; target=total/9. For each selected level compute (1-.7)+.7*target/level_total and multiply every quantile and series of that level by the corresponding horizon correction. Reject nonfinite final output. Signed inputs and outputs are supported; there is no clipping, rounding, quantile sorting or monotonicity guarantee. Levels 10,11,13,14,15 receive no additional correction. The adapter does not recompute the supplied sum or verify that stored totals match native forecasts; same-invocation provenance is a precondition.',
        {'state':'Same-invocation state from restoration_inputs.',
         'total':'Same-invocation generic ordered weighted sum of the nine median-total rows with unit coefficients.'},
        'Fresh finite float64 forecast array (9,28,N), retaining quantile, horizon and hierarchy ordering; caller arrays are not mutated.',
        ['sciona.decomposed_m5u_runtime.restore','sciona.m5u_training (level constants)','numpy']),
    'result': recipe(
        'Package every restored forecast and its reconstruction metadata as strict JSON.',
        'Check the forecast structure and restored array, then return all quantiles, hierarchy levels and future steps with the training reports and stated scope.',
        'Revalidate forecasted through the same full shape, level, multiplier, quantile and adjustment checks used by restoration_inputs. Require restored values convertible to exact float64 with identical (9,28,N) shape and all finite entries. Convert forecast, levels and hierarchy indices to lists, quantiles to a list, retain reports, normalization and scope, set horizon=28 and require serialization with allow_nan=False. Reports, hierarchy indices, normalization and scope are passed through rather than independently authenticated. No sum, model execution or restoration is recomputed. Same-invocation provenance is required.',
        {'forecasted':'Complete same-invocation fit_forecast output including metadata.',
         'restored':'Complete same-invocation restore output in unchanged hierarchy order.'},
        'Strict JSON dictionary with forecast, levels, hierarchy_indices, quantiles, horizon=28, reports, normalization and scope. Forecast arrays are runtime outputs and must not be included in public evidence for private datasets.',
        ['sciona.decomposed_m5u_runtime.result','sciona.m5u_training (quantile and level constants)','numpy','json (Python standard library)']),
}
