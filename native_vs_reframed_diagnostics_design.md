# Design Note: Native-Grid vs Reframed-Grid Diagnostics

## Status

This is an architectural concern, not a small cleanup.

The `melcher25_bistable` proposal surfaced a broader ambiguity in the current
`CCModel` / `CCOutput` design: some integrations are naturally produced on one
time grid, then presented or exported on another. Today the code supports that
mechanically, but it does not yet establish a clear convention for where
diagnostics should be computed or how users should interpret them.

The main risk is that derived metrics can silently change when a trajectory is
reframed onto a different time axis.

## Problem

`CCOutput` currently stores:

- `model_time`: the solver-native time axis
- `time`: the user-facing time axis, which may be replaced by
  `reframe_time_axis`
- `state_variables`: aligned to `time`

This is useful, but it leaves one critical question unresolved:

When a diagnostic depends on trajectory geometry or event timing, should it be
defined on the native solver grid or on the reframed grid?

That distinction matters for:

- threshold crossings
- event counts
- waiting times
- onset / termination times
- residence times
- extrema timing
- durations above or below a threshold
- any classification with hysteresis

These quantities are not generally invariant under interpolation or resampling.

## Example That Surfaced The Issue

The uploaded bistable helper integrates an SDE on a fine Euler-Maruyama grid
and then derives stadial / interstadial state transitions from the simulated
trajectory.

The proposed ClimateCritters adaptation also wanted to return an evenly spaced
series on a coarser public grid for downstream analysis convenience.

That creates two distinct products:

1. the native simulated path
2. a reframed representation of that path

If event statistics are computed after reframing, interpolation can:

- remove short threshold crossings
- introduce new apparent crossings
- shift onset times
- alter waiting times
- change event counts

For a helper advertised as producing synthetic series with known event
structure, that is a substantive definition change, not just formatting.

## Terminology

To avoid ambiguity, use the following terms consistently:

- `native grid`
  The time axis actually used by the solver. In `CCOutput`, this is
  `model_time`.

- `reframed grid`
  A user-requested time axis produced by `reframe_time_axis` or via
  `integrate(..., output_time=...)`.

- `native diagnostics`
  Diagnostics defined from the solver-native trajectory.

- `reframed diagnostics`
  Diagnostics defined from the resampled trajectory on the user-facing grid.

These are not interchangeable concepts.

## Recommendation

### 1. Treat native diagnostics as the default for dynamics-sensitive metrics

For quantities that depend on event timing, crossing structure, or path
geometry, compute them on the native solver grid unless there is an explicit
reason not to.

This should be the default for:

- event counts
- waiting times
- threshold-based state classification
- transition detection
- durations in regime
- local-extrema timing

### 2. Treat reframing as a presentation or analysis step

Regular spacing is still important and often the right thing for:

- plotting
- spectral methods that assume uniform sampling
- ML / statistics pipelines that want fixed cadence
- comparison across runs on a common grid
- export to tools that expect evenly spaced time axes

But those benefits do not make the reframed grid the canonical source of truth
for all diagnostics.

### 3. If both are useful, store both explicitly

When a model helper needs native and evenly spaced products, do not collapse
them into a single unnamed interpretation.

Instead, expose both with distinct names, for example:

- `states_native`
- `states_reframed`
- `n_events_native`
- `waiting_times_native`

If only one is exposed, the docstring should state which grid defines it.

## Repo-Level Convention

The repo should adopt the following convention:

1. `CCOutput.model_time` is the authoritative native integration grid.
2. `CCOutput.time` is the current presentation grid.
3. Reframing may change `time` and `state_variables`, but must not be assumed
   to preserve all derived metrics.
4. Diagnostics that are sensitive to interpolation should be computed from the
   native trajectory unless explicitly labeled otherwise.
5. Helpers that export regular-grid series should document whether attached
   diagnostics are native-grid or reframed-grid quantities.

## Guidance For `melcher25_bistable`

For the bistable helper specifically, the recommended behavior is:

1. Integrate on the fine Euler-Maruyama grid.
2. Compute threshold classification and event metrics on the native simulated
   path.
3. Optionally return a reframed evenly spaced `Series` for downstream use.
4. If the reframed series also needs a classified state array, compute that
   separately and label it clearly as reframed-grid output.

That leads to a cleaner separation:

- native path diagnostics preserve the simulation-defined event structure
- reframed series support downstream methods that need uniform spacing

## Suggested API Direction

This does not require an immediate large refactor, but it does suggest a
meaningful design direction.

### Near-term, model/helper level

For helper functions like `generate_bistable_series(...)`:

- compute event diagnostics before reframing
- attach them with explicit naming
- document the difference in the return object

### Medium-term, `CCOutput` level

Consider extending `CCOutput` with first-class support for native and reframed
diagnostics, such as:

- `native_diagnostic_variables`
- `reframed_diagnostic_variables`

or a lighter-weight metadata convention recording the grid provenance for each
diagnostic.

This is not required to ship `melcher25_bistable`, but the current ambiguity
will likely recur in future models and helper utilities.

## Testing Implications

Tests should distinguish between:

- correctness on the native solver grid
- correctness after reframing to a regular grid

For models or helpers with threshold-based metrics, add tests that verify:

- native-grid metrics are reproducible under fixed seeds
- reframing preserves length and alignment as intended
- reframed event metrics, if exposed, are treated as a separate product

## Bottom Line

Evenly spaced outputs are important and worth supporting directly.

But event-sensitive diagnostics should not silently inherit the reframed grid as
their definition. The codebase should treat native-grid diagnostics and
reframed-grid diagnostics as separate, explicitly named concepts.
