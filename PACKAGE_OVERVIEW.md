# PaleoBeasts Package Overview

## Architecture (Text Diagram)

```
notebooks/
  -> use models + utilities

paleobeasts/
  core/
    forcing.py   -> Forcing class (time-dependent inputs)
    pbmodel.py   -> PBModel base class (integrate, reframe, params)

  signal_models/
    ebm.py       -> EBM (0D energy balance model)
    lorenz.py    -> Lorenz-63
    lorenz96.py  -> Lorenz-96 (single + two-scale)
    g24.py       -> Ganopolski 2024 Model3
    methane_d13c.py -> Two-box CH4 / d13C model

  utils/
    solver.py        -> define_t_eval, euler_method
    forcing_utils.py -> periodic forcing builders
    noise.py         -> surrogate/noise helpers
    resample.py      -> downsample utilities
    func.py          -> derivative helpers
    constants.py     -> physical constants
```

This document summarizes the contents of the `core`, `signal_models`, and `utils` packages and notes where they are used in the notebooks.

## Core (`paleobeasts/core`)

### `forcing.py`
**`Forcing` class**
- Wraps time-dependent forcing as either a callable function or an interpolated array.
- Supports interpolation for array data (`cubic` via `CubicSpline`, or `linear` via `interp1d`).
- `get_forcing(t)` returns forcing values at time `t`.
- `from_csv(...)` helper loads known datasets (e.g., `vieira_tsi`, `insolation`) or user-provided CSVs.

Notebook usage:
- `ebm_demo.ipynb` (EBM forcing from function, array, and CSV).
- `lorenz_demo.ipynb` (forcing utilities + EBM forcings).
- `lorenz63_demo.ipynb` (constant/periodic forcing).
- `lorenz96_demo.ipynb` (forcing for Lorenz-96).
- `model_noise_demo.ipynb` (forcing-driven model output).
- `Ganopolski2024_demo.ipynb` (forcing fed into Model3).

### `pbmodel.py`
**`PBModel` base class**
- Defines the common model API (`integrate`, `to_pyleo`, `reframe_time_axis`).
- Stores state variables (structured array), diagnostic variables, and time.
- Supports adaptive solvers via `solve_ivp` and fixed-step Euler via `utils.solver.euler_method`.
- **Time-varying parameter support**:
  - `param_values` dictionary maps parameter names to constants, callables, or `Forcing` objects.
  - `get_param(name, t, state)` resolves parameters at runtime.
  - `set_param(name, value)` updates both the attribute and `param_values` to avoid mismatch.

Notebook usage:
- All model notebooks rely on the `integrate(...)` API.
- `lorenz63_demo.ipynb`, `lorenz96_demo.ipynb`, `model_solver_choice_G24.ipynb`, `L96-two-scale-description.ipynb` use `reframe_time_axis(...)` / `define_t_eval(...)`.

## Signal Models (`paleobeasts/signal_models`)

### `lorenz.py`
**`Lorenz63`**
- Classic Lorenz-63 system with optional forcing (scalar or 3-vector).
- Parameters `sigma`, `rho`, `beta` can be constants, callables, or `Forcing` objects.

Notebook usage:
- `lorenz63_demo.ipynb` (unforced, forced, time-varying parameters example).
- `lorenz_demo.ipynb` (time-varying `rho` example).

### `lorenz96.py`
**`Lorenz96`**
- Single-scale L96 with `n` state variables and forcing `F`.
- `F` can be constant, callable, or `Forcing` (unless external forcing is provided).

**`Lorenz96TwoScale`**
- Two-time-scale L96 system (slow `X_k`, fast `Y_{j,k}`).
- Parameters `F`, `h`, `b`, `c` can be constants/callables/Forcing.
- Includes `run(si, total_time, y0, dt, method)` convenience method returning `(X, Y, t)`.

Notebook usage:
- `lorenz96_demo.ipynb` (single-scale L96).
- `L96-two-scale-description.ipynb` (two-scale reproduction using `Lorenz96TwoScale`).

### `ebm.py`
**`EBM` (0D energy balance model)**
- State variable: surface temperature `T`.
- Forcing interpreted as incoming solar radiation (`S0`).
- Parameters `C`, `albedo`, `OLR`, `merid_diff` can be constants, callables, or `Forcing` objects.
- Diagnostics include `albedo`, `absorbed_SW`, `OLR`, `solar_incoming`.

Notebook usage:
- `ebm_demo.ipynb` (function/array/TSI forcing + time-varying params example).
- `lorenz_demo.ipynb` (EBM forcing examples).

### `g24.py`
**`Model3` (Ganopolski 2024)**
- Ice volume model with two regimes (glacial/deglaciation).
- Parameters `f1`, `f2`, `t1`, `t2`, `vc`, and `dfdt` are resolved via `get_param(...)`.
- Supports time-varying parameters (constants, callables, or `Forcing`).

Notebook usage:
- `Ganopolski2024_demo.ipynb`
- `model_solver_choice_G24.ipynb`
- `model_noise_demo.ipynb`

### `methane_d13c.py`
**`MethaneD13C`**
- Two-box Northern/Southern Hemisphere methane model with explicit `12CH4` and `13CH4`.
- Supports four source categories (`biogenic`, `pyrogenic`, `geological`, `fossil`) and three sink categories (`oh`, `soil`, `stratosphere`).
- Includes helpers for source-mixing inversion and synthetic validation without external files.

Notebook usage:
- `methane_d13c_demo.ipynb`

### `insolation.py`
Standalone insolation utilities (e.g., `daily_insolation`, `instant_insolation`) adapted from climlab.
Not explicitly imported in notebooks.

### `multifractal_toy.py`
Standalone multifractal time series generator (example script style).
Not used by notebooks.

## Utils (`paleobeasts/utils`)

### `forcing_utils.py`
**`create_periodic_forcing_function(periods_powers, desired_amplitude, y0)`**
- Builds a composite sine forcing from multiple periods/powers.

Notebook usage:
- `ebm_demo.ipynb`
- `lorenz_demo.ipynb`

### `solver.py`
**`define_t_eval(t_span, delta_t|num_points)`**
- Convenience function for constructing a time grid.
**`euler_method(...)`**
- Fixed-step Euler integrator used by `PBModel` when `method='euler'`.

Notebook usage:
- `lorenz63_demo.ipynb`
- `lorenz96_demo.ipynb`
- `model_solver_choice_G24.ipynb`
- `L96-two-scale-description.ipynb`

### `resample.py`
**`downsample(series, ...)`**
- Randomly downsample time series using several distributions.

Notebook usage:
- `downsample_demo.ipynb`
- `model_noise_demo.ipynb`
- `model_solver_choice_G24.ipynb`
- `lorenz_demo.ipynb` (downsampling usage)

### `noise.py`
**`from_series(...)`** and **`from_param(...)`**
- Wraps Pyleoclim surrogate generation for adding noise.

Notebook usage:
- `noise_demo.ipynb`
- `model_noise_demo.ipynb`

### `func.py`
**`make_derivative_func(...)`**
- Utility for generating a derivative function from data.

Notebook usage:
- `Ganopolski2024_demo.ipynb`

### `constants.py`
**`sigma`**
- Stefan–Boltzmann constant used by EBM OLR defaults.

Notebook usage:
- Not directly imported in notebooks (used internally by `EBM`).

### `plotting_utils.py`
Helper plotting functions for solver/forcing comparisons.
Not used in notebooks.

## Notebook Coverage Summary

The notebooks primarily demonstrate model usage through the `PBModel` integration API and the `Forcing` class. In short:
- **Forcing API**: `ebm_demo.ipynb`, `lorenz_demo.ipynb`, `lorenz63_demo.ipynb`, `lorenz96_demo.ipynb`, `model_noise_demo.ipynb`, `Ganopolski2024_demo.ipynb`
- **Time grid utilities**: `lorenz63_demo.ipynb`, `lorenz96_demo.ipynb`, `model_solver_choice_G24.ipynb`, `L96-two-scale-description.ipynb`
- **Noise/downsampling**: `noise_demo.ipynb`, `model_noise_demo.ipynb`, `downsample_demo.ipynb`
- **Specific model demos**:
  - EBM: `ebm_demo.ipynb`
  - Lorenz-63: `lorenz63_demo.ipynb`
  - Lorenz-96 (single-scale): `lorenz96_demo.ipynb`
  - Lorenz-96 (two-scale): `L96-two-scale-description.ipynb`
  - Ganopolski 2024 Model3: `Ganopolski2024_demo.ipynb`
  - Methane / d13C two-box model: `methane_d13c_demo.ipynb`
