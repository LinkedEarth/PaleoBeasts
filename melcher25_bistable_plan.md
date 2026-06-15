# Plan: `melcher25_bistable` model_critter

Based on review of `CCModel` (`core/ccmodel.py`), the solver utilities (`utils/solver.py`), and the
`Model3` (g24) model_critter, plus the uploaded `bistable.py` and `synthetic_series-2.py`.

## 1. New file: `climatecritters/model_critters/melcher25_bistable.py`

Port `BistableModel` from the uploaded `bistable.py`, adapted to `CCModel` conventions:

- Import `from ..core.ccmodel import CCModel` (not `..core.pbmodel.PBModel`).
- Fix the constructor signature — `CCModel.__init__` takes
  `(variable_name, state_variables=None, non_integrated_state_vars=None, diagnostic_variables=None)`,
  with **no** `forcing` first argument. New signature:

  ```python
  def __init__(self, var_name='melcher25_bistable', sigma=0.2, gamma=1.5, alpha=0.0,
               state_variables=None, diagnostic_variables=None, *args, **kwargs):
      if state_variables is None:
          state_variables = ['db', 'B']
      if diagnostic_variables is None:
          diagnostic_variables = []
      super().__init__(var_name, state_variables=state_variables,
                       diagnostic_variables=diagnostic_variables, *args, **kwargs)
  ```

  This drops the `forcing=None` arg entirely, consistent with `Stommel`, `Stocker2003BipolarSeesaw`,
  and `Model3`, which all attach forcings post-hoc via `register_forcing`.

- Keep `uses_post_history = True`.
- Keep `param_values = {'sigma': sigma, 'gamma': gamma, 'alpha': alpha}` and `self.params = ()`.
- Keep the class-level physical constants `b0`, `q0`, `q1`, `tau` (calibrated from NGRIP, not in
  `param_values`) unchanged.
- Port `dydt` and `sde_noise` unchanged — both are pure functions of `(t, y)` via
  `get_param_value`, so they're already compatible with `_build_forced_dydt` and the
  Euler-Maruyama solver.
- Port `compute_stability_thresholds` as a `@staticmethod`, unchanged.

## 2. Module-level helper functions (same file, g24-style)

`Model3` (g24.py) keeps its helper functions (`calc_df`, `calc_f`, `vc_func`) as module-level
functions alongside the model class — follow the same pattern here:

- `_classify_states(db, stadial_threshold, interstadial_threshold)` — private hysteresis
  classifier, ported as-is.
- `classify_bistable_states(signal, alpha)` — thin wrapper calling
  `BistableModel.compute_stability_thresholds` + `_classify_states`.
- `generate_bistable_series(...)` — ported with a fix to the output-grid issue (see below).

`__all__ = ['BistableModel', 'generate_bistable_series', 'classify_bistable_states']`.

### Output-grid fix in `generate_bistable_series`

The original calls:

```python
output = model.integrate(
    t_span=(...), y0=[1.0, 0.0], method='euler_maruyama',
    dt=dt / 10, kwargs={'random_seed': seed, 'si': dt},
)
```

`ccmodel.integrate()` only pops/honors `'si'` for `method='rk4'`. For `euler_maruyama`, `'si'`
is silently ignored, so the output ends up on the `dt/10` grid (~50k points for `n_steps=5000`),
not the documented ~`n_steps`-point series.

**Fix**: build `t_arr = np.linspace(0, (n_steps - 1) * dt, n_steps)` as before (used for the
gamma/alpha interpolation callables and for `t_span` bounds), integrate at `dt/10` for SDE
accuracy with `kwargs={'random_seed': seed}` (drop the dead `'si'` key), and pass
`output_time=t_arr` to `integrate()`. This reframes the fine `dt/10` trajectory back onto the
`n_steps`-point `t_arr` grid via `CCOutput.reframe_time_axis` (linear-interpolation fallback,
since fixed-step `Solution` objects expose `.t`/`.y` but not `.sol`).

Then build the `pyleo.Series` from `output.time` / `output.state_variables['db']` (length now
`== n_steps`) and attach `n_events`, `waiting_times`, `states`, `stadial_threshold`,
`interstadial_threshold` as in the original.

`pyleo` import: do it locally inside `generate_bistable_series` (matching the lazy-import
pattern in `CCOutput.to_pyleo`), rather than as a top-level module import.

## 3. Register the module

Add `from .melcher25_bistable import *` to `climatecritters/model_critters/__init__.py`,
alongside the existing model imports.

## 4. New test file: `climatecritters/tests/test_signal_models_melcher25_bistable.py`

Following the `test_signal_models_stocker2003.py` pattern:

- `dydt` / `sde_noise` sanity checks — correct shapes, expected sign behavior near `b0`.
- `compute_stability_thresholds` against the Eq. 7 formula for a couple of `alpha` values
  (scalar and array input).
- `integrate(method='euler_maruyama', dt=..., kwargs={'random_seed': 0})` produces finite
  `db`/`B` with `state_variables.dtype.names == ('db', 'B')`.
- `generate_bistable_series(n_steps=..., seed=0, return_states=True)`:
  - output length `== n_steps`
  - `states` is binary (0/1)
  - `n_events` / `waiting_times` are internally consistent
  - same seed → same trajectory (reproducibility)
- `classify_bistable_states` hysteresis behavior on a synthetic signal that crosses both
  thresholds.

## Open item

`pyleo` is imported at module scope in the original upload. Other model_critters don't import
pyleoclim directly, but `generate_bistable_series` needs it to build the annotated `Series`
with extra attributes. Plan is to do a local import inside the function only — flag if a
top-level import is preferred instead.
