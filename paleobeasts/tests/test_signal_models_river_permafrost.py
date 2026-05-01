"""Tests for paleobeasts.signal_models.river_permafrost."""

import numpy as np

from paleobeasts.core import Forcing
from paleobeasts.signal_models import RiverPermafrost1D
from paleobeasts.signal_models.river_permafrost import (
    apparent_heat_capacity,
    build_nonuniform_layer_thicknesses,
)


class TestRiverPermafrostGrid:
    def test_nonuniform_grid_sums_to_target_depth_t0(self):
        dz = build_nonuniform_layer_thicknesses(total_depth=70.0, fine_depth=9.0, dz_fine=0.25, dz_coarse=1.5)
        assert np.isclose(np.sum(dz), 70.0)

    def test_nonuniform_grid_transitions_from_fine_to_coarse_t1(self):
        dz = build_nonuniform_layer_thicknesses(total_depth=12.0, fine_depth=3.0, dz_fine=0.25, dz_coarse=1.0)
        assert np.allclose(dz[:12], 0.25)
        assert np.all(dz[12:] >= 0.75)


class TestRiverPermafrostPhaseChange:
    def test_apparent_heat_capacity_peaks_near_freezing_t0(self):
        base = 2.4e6
        latent = 2.0e8
        ceff = apparent_heat_capacity(
            temperature=np.array([-1.0, 0.0, 1.0]),
            volumetric_heat_capacity=base,
            latent_heat_volumetric=latent,
            freezing_temperature=0.0,
            mushy_zone_half_width=0.25,
        )
        assert ceff[1] > ceff[0]
        assert ceff[1] > ceff[2]

    def test_apparent_heat_capacity_recovers_background_away_from_phase_window_t1(self):
        base = 2.4e6
        ceff = apparent_heat_capacity(
            temperature=np.array([-4.0, 4.0]),
            volumetric_heat_capacity=base,
            latent_heat_volumetric=2.0e8,
            freezing_temperature=0.0,
            mushy_zone_half_width=0.25,
        )
        assert np.allclose(ceff, base)


class TestRiverPermafrostBoundarySwitching:
    def _model(self, run_mode="inundated", stage_value=2.0):
        return RiverPermafrost1D(
            layer_thicknesses=np.array([0.5, 0.5, 1.0]),
            Tg=Forcing(lambda t: -5.0),
            Tw=Forcing(lambda t: 2.0),
            S=Forcing(lambda t: stage_value),
            stage_threshold=1.0,
            run_mode=run_mode,
        )

    def test_stage_below_threshold_uses_ground_temperature_t0(self):
        model = self._model(stage_value=0.2)
        assert np.isclose(model.top_boundary_temperature(0.0), -5.0)

    def test_stage_above_threshold_uses_water_temperature_t1(self):
        model = self._model(stage_value=2.0)
        assert np.isclose(model.top_boundary_temperature(0.0), 2.0)

    def test_dry_mode_ignores_inundation_switching_t2(self):
        model = self._model(run_mode="dry", stage_value=4.0)
        assert np.isclose(model.top_boundary_temperature(0.0), -5.0)

    def test_explicit_inundation_series_uses_water_temperature_t3(self):
        model = RiverPermafrost1D(
            layer_thicknesses=np.array([0.5, 0.5, 1.0]),
            Tg=Forcing(lambda t: -5.0),
            Tw=Forcing(lambda t: 2.0),
            inundated=Forcing(lambda t: 1.0),
            run_mode="inundated",
        )
        assert np.isclose(model.top_boundary_temperature(0.0), 2.0)


class TestRiverPermafrostModelBehavior:
    def test_constant_boundary_condition_relaxes_toward_stable_profile_t0(self):
        model = RiverPermafrost1D(
            layer_thicknesses=np.array([0.5, 0.5, 1.0, 1.0]),
            Tg=Forcing(lambda t: -2.0),
            Tw=Forcing(lambda t: -2.0),
            S=Forcing(lambda t: 0.0),
            run_mode="dry",
            latent_heat_volumetric=0.0,
        )
        model.integrate(t_span=(0.0, 200.0), y0=[-10.0, -10.0, -10.0, -10.0], method="euler", kwargs={"dt": 0.1})
        final_profile = np.array([model.state_variables[name][-1] for name in model.state_variables.dtype.names], dtype=float)
        assert np.all(np.isfinite(final_profile))
        assert np.all(final_profile > -10.0)
        assert final_profile[0] > -2.2
        assert np.max(np.abs(np.diff(final_profile))) < 0.35

    def test_zero_flux_bottom_yields_small_deep_trend_under_uniform_profile_t1(self):
        model = RiverPermafrost1D(
            layer_thicknesses=np.array([0.5, 0.5, 1.0, 1.0]),
            Tg=Forcing(lambda t: -4.0),
            Tw=Forcing(lambda t: -4.0),
            S=Forcing(lambda t: 0.0),
            run_mode="dry",
            latent_heat_volumetric=0.0,
        )
        tendency = np.asarray(model.dydt(0.0, np.array([-4.0, -4.0, -4.0, -4.0])), dtype=float)
        assert np.isclose(tendency[-1], 0.0, atol=1e-10)

    def test_inundated_run_warms_deeper_layers_more_than_dry_t2(self):
        common = dict(
            layer_thicknesses=np.array([0.25, 0.25, 0.5, 1.0, 1.0]),
            Tg=Forcing(lambda t: -6.0 if t < 10.0 else -2.0),
            Tw=Forcing(lambda t: 3.0),
            S=Forcing(lambda t: 2.0 if t < 10.0 else 0.0),
            stage_threshold=1.0,
            latent_heat_volumetric=0.0,
        )
        dry = RiverPermafrost1D(run_mode="dry", **common)
        wet = RiverPermafrost1D(run_mode="inundated", **common)
        y0 = [-8.0] * 5
        dry.integrate(t_span=(0.0, 30.0), y0=y0, method="euler", kwargs={"dt": 0.05})
        wet.integrate(t_span=(0.0, 30.0), y0=y0, method="euler", kwargs={"dt": 0.05})

        deep_name = wet.state_variables.dtype.names[-1]
        assert wet.state_variables[deep_name][-1] > dry.state_variables[deep_name][-1]
