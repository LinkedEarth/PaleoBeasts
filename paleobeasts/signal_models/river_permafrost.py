"""Minimal 1D riverbed permafrost thermal model for PaleoBeasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.forcing import Forcing
from ..core.pbmodel import PBModel
from ..utils.solver import Solution, flux_divergence, validate_layer_thicknesses


SECONDS_PER_DAY = 86400.0


def build_nonuniform_layer_thicknesses(
    total_depth=70.0,
    fine_depth=9.0,
    dz_fine=0.25,
    dz_coarse=1.0,
):
    """Create a two-zone vertical grid with fine shallow layers."""
    total_depth = float(total_depth)
    fine_depth = float(fine_depth)
    dz_fine = float(dz_fine)
    dz_coarse = float(dz_coarse)

    if total_depth <= 0.0:
        raise ValueError("total_depth must be > 0.")
    if fine_depth < 0.0 or fine_depth > total_depth:
        raise ValueError("fine_depth must satisfy 0 <= fine_depth <= total_depth.")
    if dz_fine <= 0.0 or dz_coarse <= 0.0:
        raise ValueError("Grid spacings must be > 0.")

    thicknesses = []
    depth = 0.0
    while depth < total_depth - 1e-12:
        target_dz = dz_fine if depth < fine_depth - 1e-12 else dz_coarse
        dz = min(target_dz, total_depth - depth)
        thicknesses.append(float(dz))
        depth += dz
    return np.asarray(thicknesses, dtype=float)


def cell_center_depths(layer_thicknesses):
    dz = validate_layer_thicknesses(layer_thicknesses)
    return np.cumsum(dz) - 0.5 * dz


def apparent_heat_capacity(
    temperature,
    volumetric_heat_capacity,
    latent_heat_volumetric,
    freezing_temperature=0.0,
    mushy_zone_half_width=0.25,
):
    """Rectangular apparent heat capacity centered on the freezing point."""
    temperature = np.asarray(temperature, dtype=float)
    base = np.full_like(temperature, float(volumetric_heat_capacity), dtype=float)
    if mushy_zone_half_width <= 0.0:
        return base
    boost = float(latent_heat_volumetric) / (2.0 * float(mushy_zone_half_width))
    phase_mask = np.abs(temperature - float(freezing_temperature)) <= float(mushy_zone_half_width)
    return base + boost * phase_mask.astype(float)


@dataclass(frozen=True)
class GridSummary:
    layer_thicknesses: np.ndarray
    depths: np.ndarray
    total_depth: float


class RiverPermafrost1D(PBModel):
    """Minimal 1D vertical thermal diffusion model with inundation switching.

    The switching control is intentionally generic:
    - ``S``: stage series with threshold-based inundation detection
    - ``inundated``: explicit 0/1 (or bool) inundation series

    ``Q`` is accepted only as a backward-compatible alias for the stage driver.
    """

    def __init__(
        self,
        forcing=None,
        var_name="river_permafrost_1d",
        layer_thicknesses=None,
        total_depth=70.0,
        fine_depth=9.0,
        dz_fine=0.25,
        dz_coarse=1.0,
        thermal_conductivity=2.5,
        volumetric_heat_capacity=2.4e6,
        latent_heat_volumetric=2.5e8,
        freezing_temperature=0.0,
        mushy_zone_half_width=0.25,
        stage_threshold=1.0,
        Tg=None,
        Tw=None,
        S=None,
        inundated=None,
        Q=None,
        run_mode="inundated",
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if layer_thicknesses is None:
            layer_thicknesses = build_nonuniform_layer_thicknesses(
                total_depth=total_depth,
                fine_depth=fine_depth,
                dz_fine=dz_fine,
                dz_coarse=dz_coarse,
            )
        self.layer_thicknesses = validate_layer_thicknesses(layer_thicknesses)
        self.depth = cell_center_depths(self.layer_thicknesses)
        self.n_layers = int(self.layer_thicknesses.size)
        self.total_depth = float(np.sum(self.layer_thicknesses))
        self.fine_depth = float(fine_depth)
        self.dz_fine = float(dz_fine)
        self.dz_coarse = float(dz_coarse)

        if state_variables is None:
            state_variables = [f"T{i}" for i in range(self.n_layers)]
        if diagnostic_variables is None:
            diagnostic_variables = [
                "top_boundary_temperature",
                "inundated",
                "active_layer_depth",
                "S",
                "Tg",
                "Tw",
            ]

        super().__init__(
            forcing=forcing,
            variable_name=var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        run_mode = str(run_mode).lower()
        if run_mode not in {"dry", "inundated"}:
            raise ValueError("run_mode must be 'dry' or 'inundated'.")

        self.thermal_conductivity = thermal_conductivity
        self.volumetric_heat_capacity = volumetric_heat_capacity
        self.latent_heat_volumetric = latent_heat_volumetric
        self.freezing_temperature = freezing_temperature
        self.mushy_zone_half_width = mushy_zone_half_width
        self.stage_threshold = stage_threshold
        self.Tg = self._coerce_forcing(Tg)
        self.Tw = self._coerce_forcing(Tw)
        self.S = self._coerce_forcing(S if S is not None else Q)
        self.inundated = self._coerce_forcing(inundated)
        self.run_mode = run_mode

        self.param_values = {
            "thermal_conductivity": thermal_conductivity,
            "volumetric_heat_capacity": volumetric_heat_capacity,
            "latent_heat_volumetric": latent_heat_volumetric,
            "freezing_temperature": freezing_temperature,
            "mushy_zone_half_width": mushy_zone_half_width,
            "stage_threshold": stage_threshold,
        }
        self.params = ()

    def uses_post_history(self):
        return True

    @staticmethod
    def _coerce_forcing(spec):
        if spec is None:
            return None
        if hasattr(spec, "get_forcing"):
            return spec
        if callable(spec):
            return Forcing(spec)
        arr = np.asarray(spec, dtype=float)
        if arr.ndim != 1:
            raise ValueError("Array forcing specs must be one-dimensional.")
        return Forcing(arr, interpolation="linear")

    def grid_summary(self):
        return GridSummary(
            layer_thicknesses=self.layer_thicknesses.copy(),
            depths=self.depth.copy(),
            total_depth=float(self.total_depth),
        )

    def _forcing_value(self, spec, t):
        if spec is None:
            raise ValueError("Required forcing was not provided.")
        return float(spec.get_forcing(self.time_util(t)))

    def _inundated_flag(self, t):
        if self.run_mode == "dry":
            return 0.0
        if self.inundated is not None:
            return float(self._forcing_value(self.inundated, t) > 0.5)
        stage_val = self._forcing_value(self.S, t)
        threshold = float(self.get_param("stage_threshold", t, np.zeros(self.n_layers)))
        return float(stage_val > threshold)

    def top_boundary_temperature(self, t):
        tg = self._forcing_value(self.Tg, t)
        tw = self._forcing_value(self.Tw, t)
        inundated = self._inundated_flag(t)
        return tw if inundated > 0.5 else tg

    def effective_heat_capacity(self, temperature):
        return apparent_heat_capacity(
            temperature=temperature,
            volumetric_heat_capacity=float(self.volumetric_heat_capacity),
            latent_heat_volumetric=float(self.latent_heat_volumetric),
            freezing_temperature=float(self.freezing_temperature),
            mushy_zone_half_width=float(self.mushy_zone_half_width),
        )

    def _active_layer_depth_from_profile(self, profile):
        profile = np.asarray(profile, dtype=float)
        positive = np.flatnonzero(profile > float(self.freezing_temperature))
        if positive.size == 0:
            return 0.0

        last = int(positive[-1])
        if last == self.n_layers - 1:
            return float(self.total_depth)
        t_top = float(profile[last])
        t_bottom = float(profile[last + 1])
        z_top = float(self.depth[last])
        z_bottom = float(self.depth[last + 1])
        if np.isclose(t_top, t_bottom):
            return z_top
        frac = (float(self.freezing_temperature) - t_top) / (t_bottom - t_top)
        frac = float(np.clip(frac, 0.0, 1.0))
        return z_top + frac * (z_bottom - z_top)

    def dydt(self, t, x):
        temp = np.asarray(x, dtype=float).reshape(self.n_layers)
        conductivity = float(self.get_param("thermal_conductivity", t, temp))
        heat_capacity = self.effective_heat_capacity(temp)
        top_temp = float(self.top_boundary_temperature(t))

        face_flux = np.zeros(self.n_layers + 1, dtype=float)
        face_flux[0] = -conductivity * (temp[0] - top_temp) / np.maximum(0.5 * self.layer_thicknesses[0], 1e-12)
        for i in range(self.n_layers - 1):
            spacing = 0.5 * (self.layer_thicknesses[i] + self.layer_thicknesses[i + 1])
            face_flux[i + 1] = -conductivity * (temp[i + 1] - temp[i]) / np.maximum(spacing, 1e-12)
        face_flux[-1] = 0.0

        tendency_kelvin_per_second = flux_divergence(face_flux, self.layer_thicknesses) / heat_capacity
        tendency_kelvin_per_day = tendency_kelvin_per_second * SECONDS_PER_DAY
        return tendency_kelvin_per_day.tolist()

    def _max_stable_timestep_days(self):
        conductivity = float(self.thermal_conductivity)
        heat_capacity = float(self.volumetric_heat_capacity)
        min_dz = float(np.min(self.layer_thicknesses))
        if conductivity <= 0.0:
            raise ValueError("thermal_conductivity must be > 0.")
        if heat_capacity <= 0.0:
            raise ValueError("volumetric_heat_capacity must be > 0.")
        return 0.45 * heat_capacity * min_dz ** 2 / conductivity / SECONDS_PER_DAY

    def integrate(self, t_span=None, y0=None, method="RK45", kwargs=None, run_name=None):
        if method != "euler":
            return super().integrate(t_span=t_span, y0=y0, method=method, kwargs=kwargs, run_name=run_name)

        kwargs = {} if kwargs is None else dict(kwargs)
        if "dt" not in kwargs:
            raise AssertionError("Please provide a time step for the Euler method.")

        dt_output = float(kwargs["dt"])
        dt_internal_max = self._max_stable_timestep_days()
        n_substeps = max(1, int(np.ceil(dt_output / dt_internal_max)))
        dt_internal = dt_output / n_substeps

        y0_arr = self.validate_initial_state(y0)
        self.t_span = t_span
        self.y0 = y0_arr
        self.solution = None
        self.method = method
        self.kwargs = dict(kwargs)
        self.kwargs["dt"] = dt_output
        self.time = None
        self._reset_noise_overlays()

        if len(self.state_variables_names) > 0:
            dtype = [(var, float) for var in self.state_variables_names]
            self.dtypes = dtype
        else:
            self.dtypes = [type(val) for val in y0_arr]

        t0, tf = map(float, t_span)
        n_steps = int(round((tf - t0) / dt_output)) + 1
        time = t0 + np.arange(n_steps, dtype=float) * dt_output
        if not np.isclose(time[-1], tf):
            time[-1] = tf

        history = np.zeros((n_steps, y0_arr.size), dtype=float)
        history[0] = y0_arr
        state = y0_arr.astype(float).copy()

        for i in range(1, n_steps):
            t_segment = time[i - 1]
            dt_segment = float(time[i] - time[i - 1])
            local_dt = dt_segment / n_substeps
            for _ in range(n_substeps):
                state = state + np.asarray(self.dydt(t_segment, state), dtype=float) * local_dt
                t_segment += local_dt
            history[i] = state

        solution = Solution(time, history)
        self.solution = solution
        self.run_name = run_name if run_name is not None else f"{method}, dt={dt_output}"
        self.post_integrate(solution.t, solution.y)

    def populate_diagnostics_from_history(self, time, history):
        diagnostics = {name: [] for name in self.diagnostic_variables}
        for t, row in zip(np.asarray(time, dtype=float), np.asarray(history, dtype=float)):
            tg = self._forcing_value(self.Tg, t)
            tw = self._forcing_value(self.Tw, t)
            stage_val = self._forcing_value(self.S, t) if self.S is not None else float(self._inundated_flag(t))
            top_temp = self.top_boundary_temperature(t)
            diagnostics["Tg"].append(tg)
            diagnostics["Tw"].append(tw)
            diagnostics["S"].append(stage_val)
            diagnostics["top_boundary_temperature"].append(top_temp)
            diagnostics["inundated"].append(self._inundated_flag(t))
            diagnostics["active_layer_depth"].append(self._active_layer_depth_from_profile(row))
        self.diagnostic_variables = {k: np.asarray(v, dtype=float) for k, v in diagnostics.items()}


__all__ = [
    "SECONDS_PER_DAY",
    "GridSummary",
    "RiverPermafrost1D",
    "apparent_heat_capacity",
    "build_nonuniform_layer_thicknesses",
    "cell_center_depths",
]
