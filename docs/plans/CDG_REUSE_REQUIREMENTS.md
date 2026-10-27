# Reusability requirements for CDG promotion

Cross-disciplinary composition is a primary objective of this project. Source
reproduction establishes provenance and behavior, but does not by itself prove
that a CDG is reusable. Apply this review alongside the existing publication,
execution, dependency, license and trust-tier checks.

For each candidate, identify:

- General computational operations and their explicit input/output contracts,
  including shape, dtype, units, alignment, timing and state assumptions.
- Domain-specific feature extraction, interpretation and output formatting that
  belong in adapters rather than in the general computation.
- The assumptions under which the structure transfers to another application,
  and the cases in which it does not.
- For historical prediction, distinguish when an event occurred from when its
  observation became available. Declare which upstream adapter enforces the
  availability cutoff; a lookback on event time alone does not establish this.
- Executable synthetic examples from materially different domains when reuse
  is claimed. Distinguish constructed examples from empirical validation.
- Selection information expressed through computational contracts and
  applicability, with competition provenance retained independently.

An opaque prepare/execute wrapper around an entire competition solution does
not establish compositional reuse. Review existing wrappers for meaningful
decomposition without discarding their source evidence or served behavior.

Generalizing a source operation must not silently change its behavior. State
the supported numeric contract and distinguish the reusable core, original
domain adapter, and any intentionally extended behavior. A reusable subgraph
does not establish completion of the original full competition workflow.

The conditional residual correction work is an initial example: separate
calibration and application atoms with explicit arrays, probabilities and
thresholds; domain joins, model training and submission quantization remain
outside those operations. It is approved at automated Tier 3 for the qualified provisioned in-process runner.
HTTP deployment, clean installation and the complete original NASA workflow
remain outside that qualification. See the conditional correction publication
report for rollback, idempotency and served-execution evidence.

Adaptive history statistics is another approved Tier 3 example. Its callers
provide event times, measurement values, query time, lookback durations and a
minimum count. Airport-specific joins and policy constants stay outside the
atom. The served graph executes synthetic manufacturing and energy examples,
including empty histories, with explicit count, mean and standard-deviation
outputs. An empty selected window returns zero count and undefined statistics;
consumers must inspect count before using the other outputs. Qualification is
for the provisioned in-process runner, and does not cover strict JSON transport
of NaN values or completion of the original NASA workflow. See
`docs/reviews/adaptive_history_served_verification.json`.

The grouped residual-classifier numerical workflow is approved at automated
Tier 3 with 16 shared atoms and 25 computational nodes. Fresh served-catalog
execution matches independent synthetic manufacturing-duration and
energy-consumption calculations, including group splits, residual filtering,
calibration offsets and final predictions. The caller supplies the feature
schema, group identities and target units; nearest-integer output quantization
is part of the contract. Domain extraction, observation availability and
multi-population dispatch remain adapter responsibilities. This provisioned
in-process qualification neither completes the original competition intake nor
qualifies package redistribution. See
`docs/reviews/residual_classifier_served_verification.json` and its publication
review for the full scope and limitations.

The corrected single-airport domain graph now composes 12 approved adapter
atoms with those 16 shared numerical atoms. Its 40 nodes expose feature
branches and joins separately. A required checkpoint validates observed array
metadata through the numerical witnesses after joins and before model fitting.
This is automated Tier 3 qualification for the provisioned in-process runner;
it excludes workflows requiring whole-graph static table simulation. Fresh
served execution matches 640 synthetic source-oracle predictions across ten
source configurations. The original intake and multi-airport graph composition
remain separate work; the later paired lifecycle qualification is described below. See
`docs/reviews/nasa_domain_served_verification.json` and the domain publication
review for the scope and limitations.

The separate training and inference graphs are now approved at automated Tier 3.
They reuse 27 approved atoms and add three explicit state boundaries. Training
accepts no future query records; inference takes complete saved state and raw
query records, recomputes domain features, and performs no fitting. The shared
numerical atoms remain independently reusable; airport vocabulary, identity and
minute-unit semantics remain explicit adapter responsibilities. Fresh served
execution matched all 640 synthetic source predictions in ten separate inference
processes. Learned state remains private runtime material. This approval does
not complete multi-airport composition or the original competition intake, and
retains the documented static-table and provisioned-runtime limits. See
`docs/reviews/nasa_lifecycle_served_verification.json` and the lifecycle publication
review.

The explicit population graph family is also approved at automated Tier 3: one
ten-population training graph and ten inference variants for one through ten
active populations. Four generic keyed routing atoms preserve caller-owned
values; five airport adapters partition records, pair saved states and restore
query identity order. Each branch retains the visible lifecycle operations,
reusing 30 approved atoms. Count selection performs validation and compilation,
without model execution; unqueried populations do not execute inference branches.
Fresh served execution matched 3,520 synthetic predictions across all ten active
counts with refitting disabled. This does not approve unsupported descriptions
or placeholder interfaces in the original intake. See
`docs/reviews/nasa_population_served_verification.json` and the population
publication review for runtime limits and provenance.

The original third-place intake now selects a corrected executable revision at
automated Tier 3. Its 485 nodes reuse the same 39 approved atoms; no new provider
was added. The original snapshot and graph records remain preserved as history.
Corrected descriptions replace unsupported feature claims and placeholder
interfaces. Fresh served catalog selection and the production candidate
converter selected the corrected version, and execution matched 640 synthetic
source predictions. Repeat publication created and updated zero rows. The
entrypoint requires ten training and ten queried populations; the separately
approved lifecycle family supplies sparse inference. HTTP transport, semantic
search ranking and empirical effectiveness are outside this qualification.
See `docs/reviews/nasa_corrected_intake_served_verification.json` and the
corrected-intake publication review. Other competition and physics intakes
remain part of the open promotion objective.

The period-to-frequency source artifact now also selects an executable Tier 3
revision. It reuses one approved provider across periodic motion, oscillating
circuits and recurring signals, with explicit seconds-to-hertz dimensions and
finite-positive input/output contracts. Its two historical proof versions remain
unchanged. Fresh source replay reconciled the original parsed equations with the
normalized source equations; numerical execution evaluates their terminal
relation. Fresh served selection and execution passed 256 independent synthetic
measurement cases, scalar/vector/matrix checks and nine invalid-domain cases.
This does not estimate periods from signals. Repeat publication changed zero
rows. See `docs/reviews/period_frequency_revision_served_verification.json` and
its publication review for the exact scope and runtime limits.

The two-component series-resistance source artifact now selects a Tier 3
execution revision using its existing approved provider. It retains the
four-step source proof and historical review records, and adds a version-bound
automated review of the applicable regime: two ohmic components in one series
branch, common nonzero current, additive voltage drops and SI units. Numeric
guards enforce finite nonnegative resistances and finite nonzero current;
callers establish the physical topology and ohmic behavior. Fresh served
execution passed 256 independent circuit cases, broadcasting and scalar-output
checks, and ten invalid-input cases. Repeat publication changed zero rows.
Reuse is appropriate across circuits satisfying that model; other additive
domains need their own dimensioned adapters. See
`docs/reviews/series_revision_served_verification.json` and its publication review.

The Schwarzschild source artifact now selects an executable Tier 3 revision
using its existing dimensioned provider. Explicit SI mass, gravitational
constant and light speed inputs produce a mass-shaped length array. Original
and replay proof histories, including failed historical evidence, are preserved.
Fresh served selection and execution passed 208 numerical values over five
runner cases against a 100-digit Decimal oracle, with at most one ULP observed
error under the qualified three-ULP tolerance. Repeat publication changed zero
rows, and rollback after activation was verified. Reuse here means an explicit
physical length-scale computation with reusable numerical machinery; no claim
of validation in unrelated domains is made. Horizon interpretation requires
caller-established Schwarzschild assumptions. Black-hole classification,
collapse simulation and general-relativistic field solving are outside scope.
See `docs/reviews/schwarzschild_revision_served_verification.json` and its
publication review for the exact scope and provisioned-runtime limits.

The original quadratic artifact now selects an explicitly corrected Tier 3
execution revision using one existing approved provider. It returns both ordered
real roots, retaining repeated and zero roots. Caller-owned coefficients carry
consistent polynomial units; the provider does not infer physical applicability
or choose an admissible root. Fresh served execution passed 305 polynomials
against an independent high-precision reference and 256 synthetic cases across
constant-acceleration and quadratic-revenue/linear-cost models. Both roots,
independent model residuals and signed coefficient rescaling were verified.
This establishes constructed computational reuse, not empirical model accuracy.
Three original/source discrepancies and four source arithmetic counterexamples
remain explicit historical evidence; no source-parity claim is made. Nineteen
catalog corruption cases were rejected, post-activation rollback passed and
repeat publication changed zero rows. See
`docs/reviews/quadratic_revision_served_verification.json` and its publication
review for scope and provisioned-runtime limits.

The fixed Euler identity source artifact now selects an executable Tier 3
revision using its existing approved proof provider. It has no inputs and
returns a JSON-safe certificate containing the initial identity, four exact
transformations and Equality(0,0). Reuse is limited to consumers that require
this fixed certificate; the artifact does not claim parameterized phasor
computation, generic theorem proving or physical simulation. All five original
expressions and four interpreted steps were checked. The source imaginary-unit
variable label and historical failed replay remain explicit, and literal source
parity remains false. Two fresh served executions produced identical exact
certificates. Nineteen corruption cases, post-activation rollback and zero-write
repeat publication passed. See
`docs/reviews/euler_revision_served_verification.json` and its publication review.

Five legacy pairwise graph identities now select executable Tier 3 revisions
for series resistance, period-to-frequency, Schwarzschild length scale,
corrected real quadratic roots and the fixed Euler proof certificate. Each
reuses its existing approved family provider; none adds a new atom. Complete
source-step pair counts, rules, feeds and dependency edges connect the legacy
representations to their qualified source projections. Historical split graphs
and failed evidence remain preserved. Fresh served selection and family-oracle
execution passed for all five, with inherited correction, interpretation and
application limits. Seventy legacy corruption cases, cross-family rollback and
rollback after all five activations passed; repeat publication changed zero
rows. This reuses qualified implementations across catalog representations and
does not enlarge their computational or physical scope. See
`docs/reviews/legacy_physics_served_verification.json` and its publication review.

The circular two-body source identity now selects an executable Tier 3 revision
using its existing provider. Inputs expose separation, both masses and the
scalar gravitational constant with SI dimensions and identical array shapes.
The isolated Newtonian circular two-point-mass model is a caller prerequisite;
the graph neither classifies orbits nor treats separation as a barycentric
radius. All thirteen selected source equations and six premises were verified
under explicit corrections, with the original stale binding and repaired source
history preserved. Fresh served execution passed 208 numerical values within
one ULP of an independent 180-digit reference. Nineteen corruption cases,
post-activation rollback and zero-write repeat publication passed. Reuse is
limited to systems satisfying this model; no unrelated-domain transfer claim
is inferred from the generic numerical inputs. See
`docs/reviews/two_body_revision_served_verification.json` and its publication review.

The momentum source identity now selects a Tier 3 execution revision reusing
its existing recoil provider. It returns a real 3D vector difference and squared
Euclidean norm for same-frame SI momentum inputs. Recoil interpretation depends
on caller-established conservation premises. Seven source equations and the
explicit reconstruction of one incomplete dot product preserve the full source
scope without claiming literal source-rule parity. The new source validator
checks immutable version identity independently of publication status; the old
validator and evidence remain unchanged. Fresh served execution matched all
84 synthetic vectors exactly against independent Decimal arithmetic. Nineteen
corruption cases, activation rollback and zero-write repeat publication passed.
The graph does not solve energy conservation or complete scattering events.
See `docs/reviews/momentum_revision_served_verification.json` and its review.

The projectile source identity now selects a Tier 3 corrected execution revision
reusing its approved trajectory provider. Six explicit SI inputs produce elapsed
time and height under the ideal no-drag constant-acceleration model, including
either horizontal direction and zero gravity. Two missing equations were
recovered from pinned public source bytes; initial-velocity and gravity-power
corrections remain explicit. Complete source dependencies and immutable source
history are retained. Fresh served execution matched all 29 synthetic points
exactly; nineteen corruption cases, activation rollback and zero-write repeat
publication passed. Physical applicability, collision handling and terrain remain
caller concerns. This does not claim empirical trajectory accuracy or reuse
outside the stated regime. See
`docs/reviews/projectile_revision_served_verification.json` and its review.

The constant-acceleration original identity now selects a Tier 3 execution
revision reusing its existing provider. Three explicit SI inputs return signed
final velocity, displacement and mean velocity for one-dimensional constant
acceleration. All 23 source steps and 24 expressions match the qualified
reconstruction; the missing equation, damaged squared-velocity AST and quotient
domains remain explicit. Fresh served execution covers 29 synthetic states in
six runner cases with exact output agreement; all 49 existing tests pass.
Zero time and zero acceleration use independently proved extensions, and
reversal does not turn displacement into path length. Reuse requires the stated
physical contract; unrelated-domain transfer has not been demonstrated.
Original source history is preserved. Nineteen corruption cases, publication
rollback including post-activation failure, and zero-write repeat publication
passed. See
`docs/reviews/constant_acceleration_revision_served_verification.json` and its review.

The parallel-resistance original identity now selects a Tier 3 execution revision
reusing its existing provider. Two positive finite resistances and a signed
common voltage return equivalent resistance, both branch currents and total
current with explicit SI units. All eight source steps and ten equations match
the qualified reconstruction, including one recovered Ohms-law equation used
by three bindings. Common-node topology and consistent current orientation are
caller prerequisites. Zero voltage follows the constitutive model independently
of the source cancellation step; a zero-voltage measurement does not identify
resistance from zero divided by zero. Fresh served execution covers 29 synthetic
states in six cases with exact agreement for all four outputs. All 38 tests,
nineteen corruption cases and publication rollback checks passed; repeat
publication wrote nothing. Source history remains preserved. No reactive,
nonlinear, negative, ideal short/open circuit or unrelated-domain transfer claim
is made. See `docs/reviews/parallel_resistance_revision_served_verification.json`
and its publication review.

The first-place NASA Phase 1 source needs a distinct corrected workflow from
the completed third-place intake. A fresh pinned-software AST audit establishes
21 training nodes: ten residual fits, ten local direct-target fits and one
global direct-target fit. The 31 model output slots include ten alias pairs:
the residual trainer returns the same fitted object twice. A reusable state
workflow must preserve this aliasing rather than independently retraining each
slot. The training prediction pipeline also requests ten model references not
produced under those names; external catalog aliases or corrected wiring remain
to be established. These findings supersede treating 31 model slots as 31
independent fits, but do not establish runtime or full source closure. No new
approval was applied. See
`docs/reviews/competition_nasa_first_training_closure.json`.

First-place submission arithmetic now has a source-independent offset/clip/truncate
operation with aligned finite-array inputs, explicit integer bounds and target-unit
responsibility. Nine synthetic contract tests pass. Isolating the exact pinned
submission function with constructed predictor outputs matches 1,848 predictions
across seven ensemble/override routing cases. The fixtures distinguish 209 cases
where postponing branch clipping and truncation until after averaging would be
incorrect. Source bounds and population routing stay outside the reusable
operation. This is numerical postprocessing evidence, not native model training,
feature extraction, CDG registration or full-source approval. See
`docs/reviews/competition_nasa_first_prediction_arithmetic.json` and
`docs/reviews/prediction_postprocessing_tests.json`.

First-place native training now executes both pinned trainers on synthetic
numeric inputs with CatBoost 1.2.10. Each fit uses 16,384 training rows and
8,192 validation rows, retaining the source 20,000-tree ceiling and 60-round
early stopping. Residual/direct fits retained 208/241 trees; saved-model
predictions round-trip exactly. The residual function returns the same fitted
object in both model slots and finite normalized feature importances. Training,
prediction and feature-importance threads are explicitly bounded to one; native
file logging is disabled. This is current-backend evidence, not historical
CatBoost 1.1.1 parity, categorical-feature coverage, all 21 population fits or
full-source approval. See
`docs/reviews/competition_nasa_first_native_training.json`.

First-place global training-table construction now has exact source comparison
for 3,840 synthetic rows across three cases and ten ordered populations. A
reusable concatenation operation receives explicit per-population selected
columns, labels and an ordering column, preserves caller tables, and retains
source pandas quicksort behavior. Different selected feature sets union with
missing values; neither imputation nor stable tie ordering is inferred. Six
contract tests pass. Source substring selection remains adapter logic rather
than a generic feature-selection rule. This closes the selected-table assembly
check, not upstream raw-feature extraction or categorical/global native fitting.
See `docs/reviews/competition_nasa_first_global_population.json` and
`docs/reviews/population_tables_tests.json`.

The complete first-place training topology now executes on synthetic selected
features: ten residual fits, ten local direct fits and one global direct fit,
serially with one native thread. Every local fit includes a categorical feature;
the global model additionally includes population identity. Source training
settings retain the 20,000-tree ceiling and 60-round early stopping. Ten residual
alias pairs and the shared global model yield 31 model slots from 21 independent
fits. Forty saved-model prediction checks pass, and all ten population submission
routes match independently composed postprocessing on 1,280 predictions. The
global fit uses 163,840 training rows and 81,920 validation rows, all synthetic.
This qualifies current-backend training/state/inference composition on already
built features, not raw-feature extraction, fallback behavior, historical backend
parity or CDG publication. See
`docs/reviews/competition_nasa_first_population_training.json`.

First-place fallback validation now covers primary success, model failure,
missing model state and double failure using synthetic inputs. The source
baseline skips missing estimates when selecting grouped last values and keeps
fractional outputs. A nondefault-index counterexample changes four of six
expected outputs through merge/subtraction misalignment. The reusable corrected
baseline now aligns by row position, accepts explicit grouping/time roles, unit
conversion and bounds, preserves caller inputs, and matches all six corrected
outputs. Five contract tests pass, including duplicate caller indices. The
source constant fallback mutates its input and both exception handlers are bare;
final adapter review must state input ownership and exception scope explicitly.
These tests do not qualify complete raw-feature extraction or publication. See
`docs/reviews/competition_nasa_first_fallbacks.json`.

The first-place feature boundary has four synthetic counterexamples: whole-table
forward/backward fill makes missing features depend on unrelated query rows,
and unchecked int16 conversion wraps out-of-range values. Seven raw-feature
families remain to be qualified; third-place adapters cannot be assumed to have
the same semantics. Added a reusable fixed-imputation integer conversion with
explicit per-feature fill values, query-batch independence, range checking and
integer precision guards. Ten contract tests pass. The corrected workflow still
needs a reviewed training/inference fill policy and integration; no source-parity
or approval claim is made for this alternative. See
`docs/reviews/competition_nasa_first_feature_boundaries.json` and
`docs/reviews/fixed_feature_imputation_tests.json`.

The first-place estimated-time extractor's previously flagged ungrouped shifts
are followed by an explicit entity-boundary reset. Synthetic runtime checks
confirm that this boundary is correct: perturbing another entity and adding a
future observation do not affect earlier query features. Thirty-six rolling
statistics match independent references; distinct-time input shuffling preserves
outputs. An off-grid/null-bin fixture verifies upward time rounding and last
nonmissing estimate selection. The extractor retains history rows, so an explicit
keyed join back to query rows is part of the execution contract. This resolves
the suspected cross-entity difference defect for tested cases; it does not
qualify all duplicate/window/timezone regimes or replace the remaining feature
family work. See `docs/reviews/competition_nasa_first_estimate_features.json`.

First-place stand-time features now pass 54 independent checks covering six
right-closed, left-open windows, exact window boundaries, upward observation-time
rounding and duplicate query times. A synthetic future-observed estimate changes
twelve source delay aggregates because the first-estimate lookup has no query-time
filter. This establishes an explicit input-availability requirement, not evidence
of leakage in original competition inputs. Count includes events with missing
delay values while mean/maximum exclude those values; preserve both populations.
The existing adaptive-history atom cannot replace this computation directly: its
windows are open at both ends and adaptively widen, and it returns different
statistics. A fixed-window computation with explicit as-of eligibility is needed
for the corrected adapter. See
`docs/reviews/competition_nasa_first_standtime_features.json`.

A reusable fixed-window statistics operation now implements (query-width, query]
with independent event/value availability times, event count, nonmissing value
count, mean and maximum. Integer time units, grouping, observation rounding and
row identity remain explicit caller contracts. Eight tests cover missing values,
future availability, exact boundaries and extreme integer-time subtraction.
All 54 source-compatible stand-time statistics match independent references and
the new operation. In the future-estimate counterexample, it preserves event
counts while excluding unavailable values from twelve aggregate outputs. The
raw-feature adapter still needs to construct correct per-query estimate values
and availability times; this operation alone does not close that source scope
or confer catalog approval. See
`docs/reviews/competition_nasa_first_standtime_features.json`.

The corrected event-delay adapter now composes the reusable fixed-window kernel
with per-query estimate availability. It accepts generic entity identifiers,
integer observation/actual/estimate times, explicit estimate-validity masks,
rounding width, output-unit conversion and window lengths. It excludes actual
events not yet available, keeps counts for events with unavailable estimates,
and applies estimates only from their observation time onward. Equal estimate
observation times use explicit input order. Seven contract tests and all 54
source-compatible stand-time statistics pass. This closes the numerical event
adapter for those regimes; source-field conversion, remaining feature families,
full graph integration and publication are still outstanding. See
`docs/reviews/competition_nasa_first_standtime_features.json` and
`tests/test_event_delay_windows.py`.

A reusable calendar operation now accepts explicit timezone-naive query times
and a caller-supplied sorted holiday schedule. Six contract tests pass; 16,376
feature values across 2,047 unique synthetic queries match the pinned source.
Date-based holiday membership and exact-timestamp countdown are intentionally
different: a holiday afternoon counts toward the next holiday midnight. Missing
schedule coverage raises an explicit error; the source empty-minimum failure is
retained as evidence. Regional holiday rules and original configuration were
not qualified by the synthetic calendar injection. Duplicate-query projection,
locale and clock interpretation remain adapter responsibilities. See
`docs/reviews/competition_nasa_first_calendar_features.json`.

Configuration/runway source checks now cover 96 event-count comparisons and
seven synthetic category cases. Four earlier configuration values change when
a future observation changes; sixteen tied-timestamp counts differ from explicit
window membership because source marker-row ordering affects rolling counts.
The fixed-window operation avoids that ordering dependence. Added a reusable
as-of index lookup with explicit missing masks and last-input tie selection;
five tests pass, and seven synthetic configuration-grid lookups select no future
observations. Source substring category matching and side-priority are documented,
not silently replaced by exact token parsing. Original category configuration,
corrected feature composition and full publication remain outstanding. See
`docs/reviews/competition_nasa_first_configuration_features.json`.

All seven first-place submission feature families now have synthetic source
probes, with their qualification gaps consolidated in
`docs/reviews/competition_nasa_first_feature_closure_progress.json`. Weather
checks cover eight independent forecast contrasts/band means and demonstrate
six earlier features depending on a later issue. Entity checks establish signed
time arithmetic, fallback-derived categories, five lookup columns dependent on
duplicate-record order, and malformed-identifier failure. Explicit issue/valid
times, reviewed vocabularies and unique authoritative entity records or as-of
lookup policies are required. These probes do not qualify complete adapters.
Training-side feature parity and integrated corrected raw-input execution remain
required before original-intake approval.

Training/submission source comparison now verifies identical complete ASTs for
all seven feature functions. The returned training feature pipeline contains
70 feature nodes plus ten perimeter nodes. Two auxiliary branches are defined
but excluded from the returned pipeline, and the corresponding master-function
arguments are unused. Ten shared configuration definition ASTs also match in a
read-only comparison; configuration contents are not included in reports. This
supports shared corrected implementations instead of duplicate training and
inference adapters. It does not establish runtime calendar/vocabulary coverage,
external catalog wiring or full raw-feature execution. See
`docs/reviews/competition_nasa_first_train_inference_features.json` and the
updated feature-closure progress report.

The generic configuration-state component now composes as-of lookup with
explicit ordered token/suffix vocabularies, grid spacing and change windows.
It matches 2,900 pinned-source feature values when state is available and
keeps unavailable earlier rows masked instead of borrowing future observations.
Fourteen component/as-of tests pass, including synthetic machine-mode transfer,
empty history, prior state, duplicate-time selection and input nonmutation.
This retains source substring/count conventions as explicit behavior; it does
not qualify original vocabulary values or the full training/inference adapter.
The configuration evidence and closure-map hashes have been refreshed.

Event-count composition now uses explicit per-channel countability and actual
event times with the fixed-window operation. Seventeen component/kernel tests
pass, including tied-row permutation, duplicate query preservation, future-event
exclusion, missing-value masks and synthetic machine-transition reuse. All 96
corrected synthetic counts match independent integer membership checks; the
original source still has sixteen tied-marker mismatches. The helper contains
no aviation field names or window constants. Raw-field masks, full master-table
assembly and version-bound catalog qualification remain outstanding.

Estimated-time source validation now reproduces a query-population dependency:
adding an earlier same-entity query changes six existing standard-deviation
features because that query contributes a zero-change sample. The reusable
estimate-history component evaluates history bins plus each current query
independently. Nine tests cover independent statistics, future/entity isolation,
query order/duplicates, rounded availability, null estimates, explicit units and
synthetic manufacturing reuse. Thirty-six rolling values match pinned source
under one-query-per-entity conditions, with remaining-time parity also verified.
This is an explicit correction, requiring retraining with the same sample policy;
it does not qualify existing trained weights or complete raw-field adapters.
See competition_nasa_first_estimate_features.json for current evidence.

Entity attribute projection now uses an explicit unique authoritative snapshot
with caller-provided vocabularies and fallback. All fifteen synthetic attribute
values match the pinned source for unique records; ambiguous duplicates are
rejected rather than resolved by input order. Seven tests cover machine-snapshot
reuse, unknown entities, query order/duplicates, missing snapshots, categorical
storage and vocabulary contracts. The component does not infer snapshot cutoff
availability. Structured identifier parsing and complete raw-feature integration
remain outstanding, and no new catalog approval is implied.

Structured identifier features now take explicit delimiter, field positions,
reference-time format, vocabularies, fallback and output time units. Together
with unique entity lookup they reproduce all 45 synthetic values across the
source entity feature columns. Fifteen combined tests pass, including a
manufacturing identifier format, duplicate queries, malformed identifiers,
missing authoritative records and time-unit conversion. Source fallback order
is retained explicitly, including categories derived from the fallback string.
These checks complete synthetic column coverage, not production configuration
or snapshot availability qualification; full workflow integration remains.

The generic forecast-history component separates issue availability from valid
time and takes explicit windows, lead bands, paired contrasts and revision lags.
Six tests pass for independent statistics, future-issue isolation, row/query
order, missing horizons, duplicate rejection, null histories and demand-forecast
clock rescaling. Seventy-two lead-feature comparisons match pinned source; a
dense shared-valid-time fixture additionally matches all 162 source features.
The correction removes padding observations and backward fills, advances history
windows at query time and leaves missing horizons/revisions unavailable. These
policies require consistent retraining. Categorical unknown mappings and full
raw-input workflow composition remain pending; component evidence is not approval.

All seven corrected feature families now compose in one synthetic validation
through an explicit many-to-one join operation and fixed integer imputation.
The assembled table has 251 features, including 243 numerical features. Query
order, repeated queries and original indices survive; evaluating each query
individually matches batch evaluation. Sixteen join/imputation tests pass, with
ambiguous feature keys, null keys and overlapping columns rejected. This is a
generic composition check with explicit synthetic fill values, not the complete
source-specific raw adapter, production configuration or native retraining.
See competition_nasa_first_corrected_feature_assembly.json.

The source-specific raw-field adapter now composes all seven generic feature
families and produces all 260 source feature names under a synthetic runtime
policy. Direct pinned-source execution matches 294 nonweather values; weather
column coverage is complete and numerical corrections retain separate forecast
evidence. Four adapter tests verify per-query/batch equality, query preservation,
future perturbation isolation, snapshot cutoff enforcement, unknown weather
category handling and input nonmutation. Runtime policies contain vocabularies
and calendar schedules; no original dataset contents are embedded. Production
policy qualification, explicit final imputation, native retraining, fallbacks and
catalog publication remain pending. See competition_nasa_first_raw_feature_adapter.json.

Model input preparation now requires explicit feature order and disjoint,
complete numeric/categorical fill policies. Numeric conversion checks overflow;
categories normalize integral values to stable strings. Twenty-four policy and
imputation tests pass. All 260 raw-adapter features produce complete model inputs
under a synthetic policy, and JSON policy round-trip plus isolated query execution
match batch execution exactly. Metadata stays outside the model matrix. Actual
production fill/vocabulary policy qualification and native retraining remain
pending; this evidence does not qualify historical weights.

The complete 21-fit source topology now executes from corrected synthetic raw
features: ten residual fits, ten direct fits and one shared global fit, preserving
31 state slots and ten residual aliases. Forty serialized prediction checks pass.
Each population has 192 unique raw queries repeated 128 times to exercise the
source large-leaf setting; 1,280 submission arithmetic comparisons cover the final
repeated query per population (ten unique queries), not independent performance
observations. Native threads were limited to one throughout. Twenty corrected
fallback/wrapper tests also pass, covering observed-as-of estimates, fractional
fallbacks, malformed output, process-control propagation and source override
arithmetic. Final-wrapper tests use controlled predictors, so native wrapper and
stored-CDG execution remain explicit outstanding gates. No new approval applied.

Final native wrapper qualification now passes all 21 source fits, forty saved
model checks, 640 distinct held-out query comparisons across ten populations,
ten duplicate-order checks and twenty injected fallback cases. Twenty-one local
synthetic model checkpoints have verified hashes and are reusable for graph
validation. All thirteen generic feature provider wrappers are cold-discovered
by the standard loader and match their implementations on materialized inputs.
The complete related component/adapter suite passes 137 tests. Static symbolic
propagation remains explicitly unsupported. Database identity/port staging,
full/sparse graph dispatch, stored-graph verification and Tier 3 publication are
still pending; local registration alone is not catalog approval.

Thirteen generic feature providers are now staged as non-publishable catalog
drafts with 83 input and 17 output ports mirrored in canonical/legacy catalogs.
The staging transaction created 265 rows; its repeat created zero. Dry rollback
and failures after writes 1, 132 and 265 preserved all original/target rows. Fresh
SQL checks confirm thirteen canonical drafts, thirteen legacy flagged atoms,
100 ports per catalog and zero served rows. Original source hash is unchanged.
Contracts declare time units, availability masks, row shapes, missing behavior
and unsupported static propagation. This is draft staging, not Tier 3 approval;
publication/runtime gates and complete original CDG graph work remain pending.

Post-staging checks now reject all 87 injected catalog faults, with transaction
rollback preserving the staged records and original source. Thirteen individual
provider graphs built from their actual stored ports pass codec round-trip and
production-executor checks for all seventeen outputs on synthetic fixtures.
Only intermediate-value persistence is replaced by output capture. This verifies
materialized provider dispatch; complete workflow graph execution and Tier 3
publication remain pending. Compute ran serially with native thread pools set
to one, within the shared four-core limit. See
available_feature_provider_database_gates.json and
available_feature_provider_graph_execution.json.

The thirteen materialized feature providers are now approved and served at
automated Tier 3. Their scoped runtime passed thirteen graph executions with a
compatible 43-package closure and matching retained notice versions. Publication
created 66 metadata rows and changed 26 approval flags atomically; dry rollback
and injected failures after mutations 1, 33, 66, 67 and 92 preserved all catalog
state. Fresh read-only verification confirms exact version hashes, ports,
runtime bindings, immutable review evidence and served metadata in both catalog
representations. The original first-place workflow remains a non-publishable
draft. Full training/inference CDG composition and population dispatch remain
required; this atom approval does not stand in for original-workflow closure.
See available_feature_provider_publication_review.json,
available_feature_provider_publication_gates.json and
available_feature_provider_served.json.

First-place model state handoff now uses a domain-neutral explicit slot bank:
local aliases retain native object identity and all populations share one global
model. All 21 saved synthetic models passed checkpoint integrity checks; 640
predictions matched direct wrapper invocation. Ten single-population subsets
selected the same objects, and unknown populations are rejected explicitly.
Eight contract tests cover ownership, missing/unused models and conflicting
bindings, using a manufacturing-style naming example. Source inspection confirms
one population per prediction call; multi-population prediction dispatch is not
a source requirement. Serialized full-CDG state handoff remains pending. See
competition_nasa_first_model_handoff.json.

The model-slot handoff now runs through a codec-roundtripped two-node graph with
standard-loader-discovered reusable providers. Twenty full/sparse native-model
graph cases and one independent application example passed; the selected models
produced 640 predictions equal to direct invocation. Four malformed bindings
were rejected with their exact underlying exceptions. Runtime native objects
retain shared/alias identity across the graph edge. This serializes graph
topology, not the model objects; intermediate persistence is captured for the
check. Full training/inference graph composition and provider catalog approval
for these two new operations remain pending. See
competition_nasa_first_model_slot_graph.json.

All seven first-place raw feature families now execute in an eleven-node,
73-edge graph: two domain preparation/projection adapters surround nine reused
generic operations, including identity-preserving table assembly. A codec
round-trip and production executor passed twelve synthetic cases, including
ten population fixtures and duplicate ordering, with exact complete-frame
agreement against the unchanged qualified adapter. An unavailable entity
snapshot was rejected. The domain adapter contains only field conversion and
column projection; reusable numerical computations remain separate graph
nodes. Model input preparation, training, prediction and fallback routing still
need full graph composition, and the new adapters require catalog binding and
publication. See competition_nasa_first_decomposed_feature_graph.json.

The decomposed primary prediction graph now contains 24 nodes and 105 edges,
covering raw feature extraction, fixed model input policy, four native model
calls, local offset/clipping, source population overrides, equal-weight
averaging and final clipping/output formatting. Twenty codec-roundtripped
production-executor cases match the qualified wrapper exactly, including
dtypes, for 640 distinct queries and 30 duplicate-row comparisons across ten
synthetic populations. All 21 retained native models were integrity checked;
native prediction used one thread. Seven new generic prediction/ensemble tests
pass. Baseline/constant fallback routing, the empty-query short circuit,
training graph assembly and new-provider catalog approval remain pending.
See competition_nasa_first_primary_graph.json.

Complete single-population inference now has a three-node control graph with
explicit hash-bound primary (25 nodes) and baseline (2 nodes) branch envelopes.
The generic async controller invokes existing production graph sessions
sequentially and returns terminal values through an in-memory receiver, without
global execution hooks or changes to the existing executor. Six synthetic
controlled-predictor cases pass for primary, override, baseline, constant,
empty and duplicate-baseline behavior. Actual intermediate persistence ran;
fourteen session traces establish the expected lazy branch count. Executed
branch definitions match the serialized outer graph metadata. Eight generic
guard tests cover malformed results, corrupt graph hashes, empty execution and
propagation of KeyboardInterrupt, SystemExit and cancellation. Native primary
prediction remains covered by the separate 670-comparison report. Training
graph assembly, native state persistence and catalog binding/publication remain
pending. See competition_nasa_first_guarded_graph.json.

Training now has separate reusable temporal-split and CPU CatBoost-fit
operations, with source eligibility and parameter policy isolated in domain
adapters. Seven tests pass, including a small native fit and rejection of
thread/device/file-policy overrides. Six residual/direct operand-graph cases
match pinned source feature frames, selected rows, target values, category
indices and fixed model settings. Cases cover cutoff equality, zero-target
exclusion, duplicate indices, integer targets and reversed row order. Targets
normalize explicitly to float64. The comparator stops at the split node and
uses a recording source backend; it is not source-sized native graph fitting.
Single-candidate score acceptance, residual feature importance and complete
21-fit graph assembly remain pending. See competition_nasa_first_fit_contracts.json.

Post-fit graphs now match pinned source score expressions for all 21 saved
native synthetic models and exact normalized importance tables for all ten
residual models; both residual output slots retain the same model identity.
Six generic metric/acceptance/importance tests pass. Undefined zero-total
importance is rejected explicitly. Complete direct/residual single-fit graphs
are assembled. The full prepared-population training graph contains 202 nodes
and 646 edges, including global selected-table concatenation, 21 fits, source
validation/importance and the model bank. A recording-fit wiring run exercised
all nodes and verified ten residual aliases, 30 local slots, one shared global
model and correct temporal population membership. This wiring run is not native
training qualification; source-sized native graph execution and raw training
input composition remain pending. See competition_nasa_first_fit_review.json
and competition_nasa_first_population_training_wiring.json.

The complete 202-node/646-edge prepared-population training graph has now run
all 21 native fits under the source model settings, with one thread per fit.
Every model was saved as it completed. All ten residual importance outputs and
alias pairs passed, with one shared global model. New graph-trained models
match the prior qualified source execution on 2,560 individual native prediction
values and 640 final predictions; the frozen local source closure remained
unchanged throughout the run. The synthetic training matrices contain repeated
queries to exercise the original large-leaf policy, not independent empirical
observations. Raw feature preparation currently precedes the training graph.
See competition_nasa_first_native_training_graph.json.

Portable native regression and model-bank envelopes now pass nine contract
tests and an all-21-model JSON round-trip with 672 exact prediction comparisons.
The envelope verifies payload integrity, backend version, ordered feature names
and categorical indices, and preserves integer slot keys, ten residual aliases
and the shared global object. No model payload is included in the committed
evidence report. Raw training graph integration, training-to-inference policy
binding and new-provider/catalog publication remain pending. See
competition_nasa_first_native_state.json.

Raw training preparation is now integrated into a 334-node/1,477-edge graph.
Its 132 preparation nodes execute through the production runner and produce
exactly the ten prepared population tables used by the qualified 202-node
native training graph: 245,760 synthetic rows with identical column order,
dtypes, values, indices and targets. Qualification-source hashes were checked
before comparison. Three target-attachment tests cover duplicate indices,
row misalignment and metadata exclusion. This is compositional qualification
of separately executed raw-preparation and native-training sections, joined by
exact full-frame equality; no second 21-fit run is claimed. Training/inference
policy binding and catalog publication remain pending. See
competition_nasa_first_raw_training_graph.json.

Training/inference policy binding now passes through the production executor.
The training graph has 337 nodes; the inference graph has six outer nodes plus
explicit primary and baseline branches. Encoding the qualified native bank,
JSON round-tripping its bound state, and restoring it for inference produced
640 exact native query matches, all three fallback/empty routes and three
rejected integrity or policy-drift faults. Training was not repeated: the state
tail uses the previously qualified models. Feature order, fills, vocabularies
and calendar policy are explicit runtime state, with no private payload in
the evidence report. See competition_nasa_first_policy_state_graphs.json.

A fresh read-only catalog inventory follows both lifecycle graphs and their
nested branch descriptors, including the controller-injected emitter. It finds
47 required providers: 28 reusable operations and 19 domain adapters. Twelve
required feature operations are already served; 35 lifecycle dependencies have
neither their current versions nor served entries in the catalog. The thirteenth
previously approved feature provider is not a direct node dependency of these
graphs. Reusability remains explicit in this separation; graph-local contracts
must be reviewed into canonical provider contracts before staging. No catalog
mutation or workflow approval is claimed. See
competition_nasa_first_workflow_providers.json.

Candidate semantic ports are now frozen for all 35 unserved dependencies:
16 reusable lifecycle operations and 19 explicit domain adapters. Generic ports
declare caller-selected target units, population labels, feature schemas,
temporal cutoffs, score bounds and policy state, without inheriting source
constants. Every graph use was checked against callable input order and named
output order/types. One intentional specialization is explicit: the source
formatter accepts both integer primary and fractional fallback vectors,
preserving dtype. Fifty-five existing lifecycle tests pass, including native
training, model state, graph routing and population bindings. These are candidate
contracts; runtime/license review, catalog staging and publication remain
pending. See competition_nasa_first_lifecycle_contracts.json.

The explicit supplemental CPU profile pins CatBoost 1.2.10 alongside current
source manifests. Its provisioned closure contains 54 compatible packages;
notices for 11 additional packages are retained, including the upstream
CatBoost notice pinned to the peeled release commit. No binary redistribution
or clean-install qualification is claimed. With optional server implementations
excluded, restored policy-bound native execution again passes 640 exact query
comparisons and fallback/policy fault checks. See
competition_nasa_first_lifecycle_runtime.json.

All 35 lifecycle providers are now staged as non-publishable Tier 3 candidate
drafts in canonical and legacy catalog tables, with version-bound ports and
qualification evidence. The transaction created 895 rows. Dry-run rollback and
injected failures at writes 1, 447 and 895 preserved the original and target
rows. A repeated apply created zero rows, and a fresh read-only verification
confirmed exact identities, versions, ports, runtime pointers and audit records;
none of these drafts is served or approved. Original intake publication remains
pending, as do stored-graph execution, database corruption gates and final
version-bound review. See competition_nasa_first_lifecycle_staging.json and
competition_nasa_first_lifecycle_staging_gates.json.

The staged lifecycle versions pass 219 actual rollback-only catalog corruption
checks, covering hashes, tiers, contracts, runtime pointers, evidence, latest
flags, publishability and original provenance. Fresh verification after rollback
passes. All 16 new reusable operations also execute through graphs built from
stored ports in 38 synthetic energy/water-demand cases, including two bounded
native fits, alternate target units, model aliases, JSON state restoration and
all fallback routes. This establishes computational reuse, not predictive
performance. See competition_nasa_first_lifecycle_database_gates.json and
competition_nasa_first_lifecycle_stored_reuse.json.

All 19 source adapters now execute using stored lifecycle contracts as well.
Raw preparation retains exact full-frame equality on 245,760 synthetic rows;
training wiring executes its 21 explicitly recorded fit calls; policy-bound
inference restores qualified native models and matches 640 predictions. All
35 staged providers are encountered in these sections. Topology is assembled
locally and codec-roundtripped with stored version bindings, so this does not
claim retrieval of a published original workflow. Native source-sized fitting
retains its separate qualification and is not repeated here. Final review,
publication and original-intake lifecycle binding remain pending. See
competition_nasa_first_lifecycle_stored_adapters.json.

All 35 lifecycle provider versions are now approved and served at automated
Tier 3: 16 reusable operations and 19 explicitly source-specific adapters.
Version-bound review checks current implementation hashes, exact stored-graph
coverage, generic transfer evidence, runtime dependencies and rollback gates.
The publication transaction created 176 rows and made 246 mutations; failures
after mutations 1, 88, 176, 177 and 246 rolled back completely. A repeated apply
made zero mutations. Fresh read-only verification confirms approved/ready
canonical and legacy served entries, exact ports and runtime pointers, retained
references and immutable approval evidence. This approves provider versions,
not the original competition CDG, which remains draft and non-publishable.
Original-intake lifecycle binding, publication and served-converter execution
are still required. See competition_nasa_first_lifecycle_publication.json,
competition_nasa_first_lifecycle_publication_gates.json and
competition_nasa_first_lifecycle_served.json. Historical draft planners and
draft-only execution validators retain their original scope; publication review
uses the frozen staged cohort so approval does not change its membership.

The original first-place workflow now has a complete 343-node/1,496-edge
candidate, combining unchanged qualified training and inference sections with
one explicit portable-state edge. The production executor traverses every
outer node. Before replaying each of 21 qualified native checkpoints, the
validator compares every feature frame, target Series, category index and fit
policy against the independently rebuilt qualified operands. State encoding,
decoding, scoring, importance and inference execute normally; 64 final native
predictions match exactly. This is compositional evidence, not 21 new fits.
The graph exposes six root inputs and both saved-state and prediction outputs.
See competition_nasa_first_complete_graph.json.

The corrected original-intake plan retains artifact identity
eb1f8b8c-c1cd-59fc-bb55-69b3044af9e7 and proposes version
ec4d158b-ae7e-5832-ab5c-e344f525c6ec. It reuses exactly 47 approved providers,
including nested branch and controller dependencies, with no new atoms.
Read-only catalog checks verify the approved provider contracts and immutable
original intake snapshot; original failed preflight evidence is explicitly
retained and included in the historical row hashes. Staging, original-workflow
stored execution and publication remain pending. See
competition_nasa_first_corrected_intake_plan.json.

The corrected original version is now staged in the catalog. Its transaction
created 2,240 rows; rollback checks after actual writes 3, 1,120 and 2,240 left
the original selection state and historical records intact. Repeating the
committed import created zero rows. Fresh verification checks all 343 direct
bindings, 47 approved provider dependencies, nested branch/controller evidence,
six root inputs and two declared outputs. The original remains latest and the
corrected candidate remains non-publishable, non-latest Tier 3 draft.

The version-selected production converter retrieves the exact graph, which now
also passes a complete 343-node execution with 21 checked checkpoint replays
and 64 exact native predictions. Catalog state and original history are checked
again after execution. This retains the earlier compositional fitting scope;
no new 21-fit run is claimed. Original-workflow corruption gates, final Tier 3
review, publication transaction and fresh served execution remain pending.
See competition_nasa_first_corrected_intake_draft_import.json,
competition_nasa_first_corrected_draft_verification.json and
competition_nasa_first_corrected_catalog_execution.json.

The corrected original first-place CDG is now approved and served at automated
Tier 3. Sixty-one rollback-only corruption checks cover root contracts, direct
bindings, every approved dependency, nested evidence, original failed preflight
and the explicit state-handoff edge. Publication rollback was verified both
after actual writes and after latest-version activation. The committed
transaction created four review/description rows and updated three selection
rows; a repeat made no changes. The original version and failed preflight are
preserved, while ec4d158b-ae7e-5832-ab5c-e344f525c6ec is now selected.

Fresh served SQL selection and the production candidate converter return the
343-node corrected graph with complete binding coverage and corrected
descriptions. Executing that selected graph again passed all 21 fit-operand
replays and 64 exact native predictions. This remains the documented
compositional training qualification; no empirical-effectiveness claim is made.
See competition_nasa_first_corrected_publication.json,
competition_nasa_first_corrected_publication_transaction_gates.json and
competition_nasa_first_corrected_served.json. This completes the current
first-place corrected-intake promotion, not the overall competition and physics
backlog objective.

The post-publication backlog refresh finds two served corrected original
competition intakes out of 142. Physics currently has 70 approved CDG artifacts
and 74 drafts. The older physics served_cdgs count includes every domain and
must not be reported as a physics-only count. Its five apparently uncovered
families are also an audit-scope artifact: it searches current draft projections
and omits preserved source projections whose artifact has since been approved.

A new read-only preserved-lineage audit checks all five against historical
versions of approved physics artifacts. Every family has one actionable reuse
candidate with identical inference contracts, identical bound expression
versions, active bindings, one currently served corrected version and mandatory
provenance to its original projection. These are the remaining legacy identities
for derivations 820976, 000015, 681943, 918264 and 332170. No new approval is
implied by this match. Next work should verify the full corrected parent scopes
and reuse those approved graphs under the legacy identities with immutable
history, rather than duplicate their implementations. See
physics_remaining_preserved_lineage.json. Remaining draft physics graphs and
the competition backlog remain in scope after this five-family batch.

All six pinned public PDG source files were restored from commit
cab0ffe9dad8614eaf168fd48b3e4a54dd4c1e8d after the upstream directory rename;
their bytes exactly match the historical ingestion hashes. Fresh reviews of
two-body period, momentum, parallel resistance, projectile trajectory and
constant acceleration pass against those source bytes and existing approval
evidence. The five legacy representations preserve every grouped source rule,
feed, expression pairing and pair multiplicity of their approved projections.
Four families have complete implied dependency edges. Constant acceleration has
the same two omitted expression-dependency edges in both historical forms;
these are explicitly recorded, and only the independently qualified corrected
execution is reused. No literal historical topology-validity claim is made.

Five corrected revisions under the legacy identities are now staged, reusing
the five approved family providers. The atomic transaction created 65 rows;
full rollback and failure after the first complete family plus three writes
preserved all histories. Repeat staging created zero rows. Fresh read-only
checks verify exact stored graphs, ports, provider bindings, mandatory source
and approved-parent provenance, original latest versions and unserved draft
status. Stored execution and final publication gates remain pending. See
remaining_legacy_physics_execution_scope.json,
remaining_legacy_physics_revision_transaction.json,
remaining_legacy_physics_revision_import.json and
remaining_legacy_physics_draft_verification.json.

The five remaining legacy revisions are now approved and served at automated
Tier 3. Stored execution passed 28 full runner cases across the five families,
and all 70 injected database corruption checks were rejected with rollback.
Publication rollback passed after cross-family writes and after all five
activations. The applied transaction created 30 review/publication rows and
updated 15 selection rows; a repeat created and updated zero rows. Fresh served
SQL selection, latest-version lookup, the production candidate converter and
execution of each selected graph all passed. Original versions and historical
source defects remain preserved, including the two constant-acceleration
expression-dependency omissions; those historical graphs are not certified.

No new atoms were introduced. The revisions reuse the existing five qualified
providers with their explicit physical regimes and input contracts; approval
does not establish unrestricted transfer to another domain. Cross-disciplinary
reuse remains a project requirement and must respect those applicability limits.
See remaining_legacy_physics_publication_review.json,
remaining_legacy_physics_database_gates.json,
remaining_legacy_physics_publication_transaction.json,
remaining_legacy_physics_publication.json,
remaining_legacy_physics_publication_repeat.json and
remaining_legacy_physics_served_verification.json.

The fresh physics-only catalog count is 75 approved/served CDGs and 69 draft
CDGs (physics_post_remaining_legacy_status.json). The wider competition and
physics promotion objective remains incomplete. Computation in this batch was
sequential with native thread pools capped at one; the older sixteen-worker
Wheat process group remains suspended under the shared four-core maximum.

The next complete inventory accounts for all 69 physics drafts exactly once:
34 remote legacy/projected pairs and one independent first-wave graph, covering
35 source families. Every family has a currently served derived implementation
with an exact source-version dependency. This is reuse lineage, not approval of
the draft identities or proof that the derived operation covers the full source.
Thirty-two pairs have identical grouped source scope. Wave interference
(539398) has one extra relation in its projection; orbit radius (282755) has one
changed inference contract. Sixty representations have complete stored
expression-dependency topology, eight have explicit edge differences (four
pairs), and the independent first-wave graph lacks this source-step identity
format. Missing/extraneous edges are recorded individually in
physics_remaining_reuse_scope.json. No draft has complete stored expression
gates under the existing readiness audit. Corrected, explicitly qualified
versions remain the intended promotion route; historical defective source
versions are retained. The old reconciliation's served_cdgs field is still an
all-domain count and must not be used as a physics count.

The independent first-wave dynamics parent was freshly revalidated: exact
source versions and the invalid historical derivative were confirmed, 18
symbolic examples and a missing-premise counterexample passed, and the served
parent's exact provider, approval hashes, source dependency, ports and all three
serialized execution cases passed. The original first-wave identity is still
draft. See physics_first_wave_revision_parent_revalidation.json. Next work is
to prepare its immutable-source review and original-identity corrected revision;
existing historical validators explicitly assume draft source status and must
remain preserved when adding approval-compatible checks. The other 34 families
and full competition backlog remain in scope.

The first-wave original-identity corrected revision is staged as
bc667b9b-f816-597c-b3df-f308a506de6b, graph digest
34c0be00a4946cc6b944c934b6aa682faaea85c38bd6f62320d06338667f7132.
It reuses the existing qualified dynamics provider, preserves original version
693697cf-a97c-53bd-8d6b-19dc3c7697c1 and includes mandatory dependencies on that
source and approved parent 83cb2c13-5dde-5b20-be61-2306f5aa321d. No new atoms
were introduced. Eight rows were created; repeat staging created zero rows.
Full rollback and an injected failure after seven actual writes were verified.

New immutable-source review checks exact historical equations and source nodes,
original reviewer hashes, corrected premises, eighteen symbolic cases and the
missing-premise counterexample without assuming the artifact remains draft.
Historical reviewers and their evidence files remain unchanged. Parent checks
verify current served graph, provider source, ports, evidence and mandatory
source dependency inside the caller's database transaction. Scope keeps the
explicit kinematic premise, positive constant mass and inertial-coordinate/SI
assumptions. Global smoothness, numerical integration and unconstrained domain
transfer are not claimed.

The stored revision passed the inherited three serialized cases and an
independent analytic oracle: eighteen production-runner cases over six position
functions and three masses, 270 numeric components, maximum one ULP error.
Zero/negative mass, singular evaluation and unevaluable formal position were
rejected. Publication review, activation rollback gates and fresh served
verification remain required before this original identity can be approved.
See first_wave_identity_execution_scope.json,
first_wave_identity_revision_transaction.json,
first_wave_identity_revision_import.json,
first_wave_identity_revision_repeat.json and
first_wave_identity_catalog_execution.json. The overall competition/physics
goal remains active, with the other original-identity revisions still in scope.

The first-wave original identity is now approved and served at automated Tier 3.
Publication created six description/review/reference/bound rows and updated
three activation rows. Repeat publication made zero changes. Rollback was
verified after partial publication writes and after latest-version activation.
Fresh served SQL selection, get_artifact_document and the production candidate
converter return bc667b9b-f816-597c-b3df-f308a506de6b with its exact provider,
original history and explicit validity regime. Executing that selected graph
again passed all inherited and independent analytic cases. See
first_wave_identity_publication_review.json,
first_wave_identity_publication_transaction.json,
first_wave_identity_publication.json,
first_wave_identity_publication_repeat.json and
first_wave_identity_served_verification.json. The physics-only authoritative
count is now 76 approved/served CDGs and 68 drafts, recorded separately in
physics_post_first_wave_identity_status.json.

Next work is the variance family (000014), represented by the remaining legacy
and projected originals. Its approved corrected finite-distribution provider
was freshly revalidated against pinned source symbols/rules: six runner cases,
eleven synthetic distributions and zero ULP error, with exact equality to the
retained execution report. Served parent graph/provider selection and retained
approval hashes also passed. This remains a finite normalized nonnegative
weighted-distribution specialization, with an explicitly corrected source
moment term; it does not establish continuum quadrature accuracy or unbiased
sample variance. See variance_identity_parent_revalidation.json. Prepare
immutable source checks and both original-identity revisions while preserving
those scope limits. The other physics families and competition backlog remain
part of the full objective; this milestone does not complete the goal.

Both variance original identities now have staged corrected revisions:
projected version 800c3790-22d8-5a24-be5f-c77daa8b4bf1 and legacy version
8145cdc4-fd1a-5d10-bed5-7b22fb279131. They reuse the single existing qualified
finite-distribution provider. Fresh immutable source validation reproduces the
retained five-equation reconstruction, with only the new validator identity
differing; historical source validators remain unchanged. Complete grouped
steps, inference contracts, bindings, expression-pair multiplicities and
expression-dependency edges match between the original representations.

The transaction created 21 rows. Full rollback and failure after thirteen
actual writes across the two identities were verified; repeat staging created
zero rows. All 34 corruption gates passed, including exact parent binding,
approval and mandatory provenance faults in the same transaction, followed by
fresh intact reads. Each stored revision passed the unchanged independent raw
moment oracle: six production-runner cases and eleven synthetic distributions,
with zero ULP error. These are two executions of the same eleven-distribution
coverage, not twenty-two distinct distributions. Original versions remain
latest/draft and unserved; publication review, activation rollback and served
verification are still pending. See variance_identity_execution_scope.json,
variance_identity_revision_transaction.json,
variance_identity_revision_import.json,
variance_identity_revision_repeat.json,
variance_identity_database_gates.json and
variance_identity_catalog_execution.json. The full competition/physics goal
remains active, including the remaining original source identities.

Both variance original identities are now approved and served at automated
Tier 3. Publication created twelve description/reference/review/regime rows
and updated six activation rows. Cross-identity and post-activation rollback
passed; repeat publication made no changes. Fresh served SQL selection,
get_artifact_document and production candidate conversion returned both exact
corrected graphs. Execution of each selected graph again passed all six runner
cases and eleven-distribution independent raw-moment comparisons with zero ULP
error. Original source versions and the malformed moment term remain intact.
See variance_identity_publication_review.json,
variance_identity_publication_transaction.json,
variance_identity_publication.json,
variance_identity_publication_repeat.json and
variance_identity_served_verification.json. The authoritative physics-only
count is now 78 approved/served and 66 draft CDGs, recorded in
physics_post_variance_identity_status.json. The full goal remains incomplete.

A Git checkpoint inventory found the last commits in matcher, physics providers
and ML providers dated 2026-09-14. Current unstaged work includes eight modified
tracked matcher files and 867 untracked matcher files, 110 untracked physics
provider files and 1515 untracked ML provider files. These counts include all
untracked files, not just source; generated caches and artifacts must be
classified and excluded where appropriate. The user's commit/push request
remains relevant: inspect the changes and confidential-data boundaries before
staging a concrete checkpoint, then continue the complete competition/physics
promotion backlog. Do not blanket-stage the untracked trees.

Provider checkpoint review found the apparent 110 untracked physics files were
entirely generated build output; ML had 1480 generated build files, five package
metadata files and thirty actual new source files. Build/package metadata are
now ignored. Physics cleanup commit 8984cae was pushed to origin on
kaggle-ingest-batch-1. ML commit 80b8d2a was pushed to origin/main and preserves
thirty reusable lifecycle/calibration/model-state/feature providers and explicit
public-software competition adapters, plus ignore rules. Both provider working
trees are clean. No provider implementation bytes were changed by checkpointing.

The ML source review checked literal data/credential/path indicators, inspected
the fixed public-software population/model-slot contract, and excluded all build
outputs and runtime artifacts. Staged whitespace checks passed. A sequential
single-thread focused run passed 150 tests over adaptive histories, residual
correction, prediction features, population masks, native model IO, training,
domain boundaries, state and lifecycle graphs. This is focused validation, not
a clean-install claim. The ML wrappers depend on matcher helpers that remain
in the pending matcher checkpoint; matcher source/tests/scripts and generated
review evidence must still be reviewed and committed/pushed. The full promotion
backlog remains active after that checkpoint.

Matcher checkpoint review covers the pending source, validation scripts,
synthetic tests, aggregate/structural qualification reports and upstream license
notices. The remaining 37 focused test files passed 287 tests; combined with the
15 provider-focused files, 437 tests passed. Staged Python/JSON syntax was
checked, staged bytes matched the reviewed working copies, and credential-pattern
scans were clear. Runtime-file references were inspected as synthetic outputs
or public competition-software interfaces. Verbatim license whitespace and two
publication-hash-bound source whitespace exceptions are intentionally retained.
See cdg_checkpoint_validation.json. This checkpoint does not add new publication
claims or certify a clean installation/full test suite. Resume the remaining
competition and physics original-identity promotions after pushing it.

Both integration-by-parts originals (derivation 000009) are now approved and
served at automated Tier 3. Exact source grouping, expression-pair multiplicity,
bindings and implied dependency edges match. The immutable source reviewer
reproduces the retained four-equation parametrized differential reconstruction,
while original source validators and incomplete AST histories remain unchanged.
The existing provider is reused under each original identity; no new atoms.

Staging created 25 rows, full rollback and cross-identity failure after sixteen
writes passed, and repeat staging made no changes. All 34 candidate/parent
corruption checks passed with rollback. Each stored graph passed the same six
symbolic cases: derivative verification, residual integral retention, portable
AST string transport and cache preservation. Publication created twelve review,
description, reference and regime rows and updated six selection rows; partial
and post-activation rollback passed, and repeat publication made no changes.
Fresh served SQL, document lookup, candidate conversion and execution passed
for both corrected identities. Scope remains scalar commutative C1 functions
on a caller-established common connected real interval, with an independent
constant and explicit pole/branch exclusions; no numerical quadrature or
automatic domain certification is claimed.

See integration_parts_identity_execution_scope.json,
integration_parts_identity_database_gates.json,
integration_parts_identity_catalog_execution.json,
integration_parts_identity_publication_transaction.json,
integration_parts_identity_publication.json,
integration_parts_identity_publication_repeat.json and
integration_parts_identity_served_verification.json. The fresh physics-only
count is 80 approved/served and 64 draft CDGs, recorded in
physics_post_integration_parts_identity_status.json. The full competition and
physics objective remains active; this checkpoint completes only this pair.

Both sine double-angle originals (derivation 000016) are now approved and
served at automated Tier 3. Fresh immutable source review checked all eight
source equations, including three recovered from pinned public Cypher bytes,
and preserved the explicit imaginary-unit interpretation. Grouped inference
scope, bindings, pair multiplicities and dependency edges match between originals.
The same approved real-radian provider is reused; no new atoms were introduced.

Staging created seventeen rows; full rollback, failure after eleven writes
across the identities and zero-change repeat staging passed. All 34 corruption
checks passed. Each stored graph passed six production-runner cases over the
same 29 numeric states with zero ULP error. Publication rollback passed after
cross-identity writes and after both latest versions activated. Publication
created twelve review/description/reference/regime rows and updated six
activation rows; repeat publication made no changes. Fresh served SQL,
document retrieval, production candidate conversion and stored execution passed.
Finite real-radian float64 conversion and shape preservation remain explicit;
no degree or complex-input semantics or unqualified domain transfer are claimed.

See sine_double_angle_identity_execution_scope.json,
sine_double_angle_identity_database_gates.json,
sine_double_angle_identity_catalog_execution.json,
sine_double_angle_identity_publication_transaction.json,
sine_double_angle_identity_publication.json,
sine_double_angle_identity_publication_repeat.json and
sine_double_angle_identity_served_verification.json. The authoritative physics
count is 82 approved/served and 62 drafts, recorded in
physics_post_sine_double_angle_identity_status.json. The wider competition
backlog remains part of the active objective; physics progress does not close it.

Otto original-identity review now explicitly rejects copying the existing
prepare/execute wrapper as sufficient compositional reuse. The complete source
workflow must expose its reusable operations and source-specific adapters before
promotion. The draft probability_ensemble module separates bag averaging from
geometric/arithmetic combination, with explicit weights and arbitrary categorical
class counts. The separate ensemble_otto_adapter retains nine classes and the
250/600/250 source bag sizes. It is outside the immutable historical otto_*
implementation inventory, is not imported by served providers, and must receive
its own provider/graph qualification. Historical implementation files are unchanged.

Synthetic three-class machine-fault and five-class land-cover examples check
the numeric operation against an 80-digit Decimal oracle; they demonstrate
constructed transfer of the probability contract, not empirical effectiveness.
Source-adapter parity, class permutations, row independence, input preservation,
invalid inputs and zero-support behavior are covered in
tests/test_probability_ensemble.py. These draft operations do not complete the
33-entry stacking workflow or authorize any new catalog selection. A fresh
runtime audit also found a missing native dependency from the retained closure;
competition_otto_original_identity_scope.json records the failure. Repair or
explicit requalification of the dependency closure remains required alongside
decomposition, full execution and publication gates. No new approval is claimed.

The native-environment follow-up checked all 273 historical library records:
266 still match and seven old Homebrew version paths are absent. Their installed
replacements have different bytes, so historical runtime evidence remains
historical. Fresh candidate traversal resolves 275 files, including two added
Brotli libraries. Candidate runtime checks loaded all 14 expected Python
dependencies, the patched CPU Lasagne/Theano pair, three R packages and Java.
The Java identity probe explicitly limits active processors to one and uses
serial garbage collection; numerical pools were limited to one throughout.

New notice evidence covers all 19 Homebrew packages in that closure, with
55 distinct notices and 132 origins. The new GLib source archive was checked
against the installed formula's source hash before retaining its notices.
An independent consistency review checks current native bytes, dependency
targets, script/evidence hashes, installed formulas and retained notices across
the candidate reports. See competition_otto_native_drift.json and
competition_otto_candidate_{native_links,runtime_identity,native_notices,
runtime_review}.json. The historical reports were not replaced. These checks
clear candidate runtime investigation for further qualification; they do not
qualify model fits, a decomposed graph or a new catalog approval. The complete
competition and physics promotion objective remains active.

Otto's meta stage now has a draft 12-node, 19-edge CDG with visible preparation,
three candidate-selection nodes, three complete-reference refit nodes, three
reusable probability means, source-policy blending and result assembly. Eight
draft provider functions implement these boundaries. The general mean operation
supports other categorical class/run counts; four folds, nine classes and
250/600/250 model bags remain explicit Otto adapter policy. Prepared inputs are
detached from caller-owned values, and refits reject selections from a different
prepared population or altered selected controls. Witnesses only express stage
kinds; runtime checks remain mandatory and static shape proof is not claimed.

All 39 focused checks passed, including exact historical-result and fitting-operand
parity with deterministic test learners, invalid-input rejection before fitting,
short/invalid bag rejection, selection ownership, codec round-trip and actual
production-runner wiring. Separate native execution then completed all 27 bags:
8,800 tuning fits and 1,100 refits across the three real learner families. All
12 nodes executed on synthetic meta features, probabilities normalized, strict
JSON output passed, and code/provider/runtime hashes matched before and after.
Numerical pools were restricted to one thread and graph execution was sequential.

See competition_otto_decomposed_meta_execution.json and
scripts/validate_stacking_meta_graph.py. This evidence qualifies the tested
meta-stage computation only, with 36 synthetic training rows and nine queries;
it does not execute or qualify the 33 first-level producers or supplemental
feature construction. The objects are qualified only within one uncached,
in-process graph execution. New providers and the graph remain unapproved.
First-level/supplemental decomposition, full-stack execution and all version-bound
publication gates remain necessary for the original Otto intake. Other competition
and physics backlog items remain part of the same open goal.

The first-level source adapters now expose all 33 entries in a draft graph.
For the retained synthetic policy, 37 nodes execute one shared ten-column KNN
bank; a separate test uses 38 nodes and two distinct control groups. Six provider
functions are reused across entry nodes, with explicit source-index policy
outputs rather than one provider per entry. Controls, input populations, fold
assignments and all source dispatch arguments match the historical implementation
in the orchestration tests. Wrong population ownership, incompatible neighbor
controls, missing entries and misaligned collection outputs are rejected.

Fresh native graph execution completed all 33 entries on the full synthetic
first-level fixture, including the complete H2O and 120-model neural families.
All 37 nodes completed, emitting the declared 293 columns over 1,350 synthetic
training rows and nine query rows. The five-feature embedding interpretation
and unresolved historical 297-column claim remain explicit. Code/provider and
candidate-runtime evidence hashes matched before and after execution. Numerical
pools were set to one thread; Java used one active processor, serial garbage
collection and one non-tiered compiler thread. See
competition_otto_decomposed_first_execution.json and
scripts/validate_stacking_first_graph.py. No model state or feature values are
retained in that report.

The supplemental stage also has a draft nine-node graph with separate raw,
fold-local TF-IDF and embedding distances, reference-only cluster fitting and
assignment, nonzero counts, raw neural features and aligned collection. The
53 focused checks across the first-level and supplemental work passed. These
include actual native execution of all supplemental nodes on constructed inputs,
exact comparison with the retained implementation, independent nearest-distance
calculations, held-fold label isolation and collection corruption checks.
Supplemental node execution is exercised by
tests/test_stacking_supplemental_operations.py; it is not yet a separately
published catalog qualification. Witnesses remain stage-kind checks only.

The first-level, supplemental and meta components now have explicit graphs,
but they have not yet been qualified together as the complete corrected Otto
workflow. Composition, complete graph execution, provider qualification and
version-bound publication/rollback/selection gates remain required. No original
intake or new provider approval is claimed by this checkpoint. The full
competition and physics promotion objective remains open.

### Complete Otto composition checkpoint

The complete draft graph now connects 62 nodes and 198 edges through 27
separately callable providers. Source-specific nine-class, feature-layout and
fold policies remain explicit; general probability operations retain their
independent contracts. Qualification covers the default shared-neighbor-bank
composition, not alternate compiled groupings or arbitrary application domains.

The full native run passed all 33 first-level entries, seven supplemental
blocks and 9,900 meta fits. Its final blend exactly matched the retained
formula and passed an independent 80-digit Decimal calculation. All nine
synthetic queries were classified correctly. The 76 focused checks passed;
learner stand-ins in wiring tests are distinct from the retained full native
execution evidence. See competition_otto_decomposed_complete_execution.json.
No real dataset values or identifying metadata are included.

The read-only scope reviewer passed its local implementation, runtime,
historical evidence, provider-contract and witness-flow checks, but could not
complete the catalog lineage check on October 27, 2026: the configured local
database refused connections, including outside the filesystem sandbox. No
scope-pass report or catalog approval was produced. Corrected-version staging,
stored-graph corruption and rollback checks, Tier 3 publication, idempotence
and fresh served selection/execution remain required. The original intake and
historical evidence remain unchanged.

The shared four-core ceiling remains mandatory. Numerical pools run with one
thread and compute jobs run sequentially; the older 16-worker job remains
suspended. The full competition and physics promotion objective remains open.

### Complete Otto draft staging

The local catalog connection was restored by starting its existing database
container with a one-CPU quota and CPU affinity. The complete read-only scope
review then passed, including the original source snapshot and immutable
history. The database service and one-thread foreground scripts remain within
the shared four-core budget.

The pre-catalog review rejected 21 injected evidence faults, with the unchanged
baseline reaching an explicit database stop sentinel before and after those
checks. This offline gate does not simulate catalog success. The separate real
database transaction checks staged and rolled back 862 rows, verified exact
catalog graph reconstruction, and injected failures after writes 1, 431 and
862. Snapshots across 16 canonical and legacy tables remained identical after
each rollback.

The qualified draft transaction was applied under the original Otto artifact
identity. Corrected version 98596890-736d-5a34-a31f-5c61c24ba3e7 derives from the
preserved original intake and contains 62 nodes, 198 edges and exact bindings
to 27 new draft providers. Repeating the operation created zero rows. The
original version stays latest; the corrected version and providers are not
served or approved. Evidence is retained in the
competition_otto_decomposed_complete_scope, review_gates, draft_transaction,
draft_import and draft_repeat JSON reports under docs/reviews.

Stored-catalog corruption qualification, atomic Tier 3 publication and its
rollback/idempotence checks, and fresh served selection/execution remain open.
The complete competition and physics backlog is still the goal; this is one
original identity's staging checkpoint, not completion of that goal.

### Corrected Otto Tier 3 publication

The corrected original Otto identity and all 27 decomposed providers are now
approved and served at automated Tier 3. Original intake rows remain unchanged;
latest selection advances to version 98596890-736d-5a34-a31f-5c61c24ba3e7.
The 62-node, 198-edge graph preserves explicit source policies and caller-owned
feature/class semantics. General probability averaging remains independently
callable across supported class and run counts; other adapters state their
nine-class source constraints rather than implying unrestricted reuse.

The stored-contract checker rejected 288 rollback-only corruption cases,
including canonical and legacy identity, version, tier, implementation, port
and evidence mutations for every provider. Original history, graph edges,
bindings, provenance and latest-selection faults were also rejected. Publication
rollback was verified across 17 tables after all metadata writes and after full
activation. Applying publication created 139 rows and updated 58; repeating it
created and updated zero rows. Fresh served SQL selection and the production
candidate converter selected the exact corrected graph, descriptions and tier.

Full native execution of that freshly served candidate is still in progress
at this checkpoint. scripts/verify_stacking_complete_served.py reuses the
complete native validator with only graph acquisition and the success-report
destination redirected; no numerical learner, count, fixture or oracle is
replaced. Existing native qualification evidence stays unchanged. The served
execution report must be inspected before calling this item's verification
complete. The first launch stopped before training because its restricted PATH
omitted pg_config discovery; adding the existing PostgreSQL client directory
allowed the current run to begin. This did not change numerical qualification.

HTTP transport, search ranking, clean installation and empirical effectiveness
remain outside this provisioned in-process synthetic qualification. Competition
and physics promotion beyond this identity remains the active objective.

### Otto served completion and mechanical-energy originals

The full fresh Otto served run has now passed all 62 nodes, 33 first-level
entries, seven supplemental blocks and 9,900 native meta fits. The retained
blend comparison, independent 80-digit numerical oracle and all nine synthetic
query classifications passed. Served selection was identical before and after
execution, and the earlier native qualification report stayed byte-identical.
See competition_otto_decomposed_served_execution.json. This completes the
outstanding served-run check for this corrected identity, not the full backlog.

Both original mechanical-energy identities now select corrected Tier 3
execution versions using the existing approved constant-force provider; no new
atom was added. Fresh source comparison covered all 23 steps, grouped relation
multiplicities, bindings and dependency edges. Four missing equations, the
malformed initial-energy identifier and interval-average velocity semantics
remain explicit corrections, with original source and failed evidence preserved.
Temporary public source files were restored from the original repository commit
and all six files matched the retained ingestion hashes before use.

The numerical contract is one-dimensional Newtonian motion with positive
constant mass, constant signed conservative force, nonnegative elapsed time and
time-independent U=-F*x with its zero at x=0. Five dimensioned inputs produce
position, velocity, kinetic/potential/total energy and signed work. Zero force,
zero duration and reversal are supported. Exact rational intermediates follow
float64 input conversion; independently rounded outputs do not imply an exact
floating-point energy sum. Callers establish physical applicability; variable
force, dissipation, relativistic motion and time-dependent potentials are out
of scope.

Each stored and subsequently served revision passed six runner cases covering
29 synthetic states against unchanged independent high-precision references,
with zero observed ULP error. All 38 database corruption cases were rejected,
including parent regime and staged-evidence mutations. Staging and publication
rollback checks passed, including failure after activation; repeat staging and
publication changed zero rows. Fresh served SQL selection, production candidate
conversion, descriptions, validity bounds and execution passed for both originals.
See mechanical_energy_identity_served_verification.json and the associated
source, transaction, execution and corruption evidence.

The read-only aggregate audit now reports 84 approved/served physics.pdg CDGs
and 60 drafts (including original and derived entries). Separately, 3 of 142
competition original intake identities select approved served corrected Tier 3
versions. Derived competition graphs are not counted as completed originals.
These totals are catalog state, not a substitute for per-version qualification;
the remaining competition and physics promotion objective is still open.

### Brewster-angle original identities

Both original Brewster-angle source identities now select corrected Tier 3
versions using the existing approved provider. No new atom was needed. Fresh
source comparison covered all ten steps, grouped relation multiplicities,
bindings and dependency edges, with the fourteen public source equations
rechecked against their retained hashes. Reciprocal-ratio, degree/radian,
malformed-angle-expression and renaming-direction corrections remain explicit;
original histories and historical failed evidence are preserved.

The contract accepts positive incident and transmitted refractive indices in
identically shaped nonempty arrays. It returns independently rounded incidence
and refraction angles in radians from the normal. Physical applicability is a
planar lossless isotropic nonmagnetic interface with p-polarization. Equal indices
use the documented pi/4 convention, not a unique reflection-zero claim. Rounded
near-grazing angles do not certify exact Fresnel equality. No lossy, anisotropic,
magnetic or negative-index extension is claimed.

Each stored and served revision passed six runner cases covering 29 synthetic
states against unchanged independent high-precision references, with zero
observed ULP error. All 38 corruption cases were rejected, including altered
parent regime and staged evidence. Staging and paired-publication rollback
passed, including failure after activation. Repeat staging and publication
made zero changes. Fresh served SQL selection, production candidate conversion,
descriptions, validity bounds and execution passed for both originals. See
brewster_angle_identity_served_verification.json and its associated evidence.

The aggregate catalog audit now reports 86 approved/served physics.pdg CDGs and
58 drafts, including original and derived entries. Competition corrected-original
coverage remains 3 of 142 intakes. The next source pair, wave interference, has
a retained projected-only relation and differing binding inventory; it requires
explicit reconciliation rather than the equal-scope pair assertion used here.
The complete competition and physics promotion goal remains open.

### Wave-interference source reconciliation and original promotion

Both original wave-interference identities now select corrected Tier 3
executions using their existing approved provider. Their historical graphs are
not equivalent: the legacy representation contains 26 source steps and the
projection contains 27. The legacy graph omits malformed source step 6988426,
which binds a phase feed as an equation and places the polar-Z equation among
its feeds. The pinned source confirms the malformed relationships. All 26
shared grouped contracts and pair multiplicities agree; the exact differences
are that step's relation, one phase-feed binding and two downstream dependencies.
See wave_interference_original_reconciliation.json. No original graph is
certified as a valid literal proof.

The already-reviewed reconstruction restores the polar expression and both
variable substitutions. The corrected legacy version explicitly recovers its
omitted step; the projected version corrects its retained malformed step.
Separate provenance-bound versions preserve both original histories and failed
evidence. Parent/provider identity, bindings, ports, evidence and physical
validity bounds were freshly checked before staging.

Reuse is bounded to two normalized scalar waves in a common mode/polarization
and intensity scale, with fixed finite nonnegative magnitudes not both zero and
a finite relative phase in radians. Outputs distinguish fixed-phase intensity,
incoherent mean, constructive/destructive extrema, cross term and ratio.
Coherence does not imply constructive alignment. The mean requires vanishing
averaged cosine, and the factor-two ratio additionally requires equal amplitudes
and constructive phase. Propagation, arbitrary polarization, stochastic
amplitudes, partial coherence and absolute SI calibration remain out of scope.

Both stored and served corrected versions passed six runner cases covering 29
synthetic states each against unchanged 2000-digit complex-field references,
with zero observed ULP error. All 38 corruption cases and staging/publication
rollback checks passed, including rollback after paired activation. Repeat
staging and publication changed zero rows. Fresh served selection, production
candidate conversion, descriptions, bounds and execution passed for both.
See wave_interference_identity_served_verification.json and associated reports.

The aggregate audit now reports 88 approved/served physics.pdg CDGs and 56
drafts, counting original and derived entries. Competition corrected-original
coverage remains 3 of 142 intakes. The overall promotion objective remains open.

### Escape-speed original source reconciliation

The two escape-speed originals agree on all 18 grouped source steps, relation
multiplicities, dependencies and bound equation versions. Fresh source review
and six runner cases covering 29 synthetic states match the retained evidence,
with zero observed ULP error. The existing derived realization still selects
the unchanged approved infall provider. See
escape_speed_original_reconciliation.json. This checkpoint makes no catalog
changes and does not approve either original identity.

Reuse must preserve the positional output aliases: positive escape threshold,
negative inward counterpart velocity, required positive launch energy and
negative potential energy. The inward counterpart is not outward escape
velocity. The model requires a fixed exterior spherical Newtonian field,
negligible test mass and asymptotic rest at infinity; callers establish the
outward trajectory's clearance. Before original promotion, qualify exact
parent/provider ports and bounds, stage corrected versions, and complete stored
execution, corruption, rollback and fresh served execution checks. Existing
hash-bound source and execution validators remain unchanged.
