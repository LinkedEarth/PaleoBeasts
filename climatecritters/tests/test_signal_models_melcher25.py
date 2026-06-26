"""Tests for climatecritters.model_critters.melcher25."""

import numpy as np
import pytest

from climatecritters.model_critters.melcher25 import (
    Melcher25, classify_bistable_states,
)


class TestSignalModelsMelcher25Integrate:
    @pytest.mark.parametrize('y0', [[1.0, 0.0], [0.6, 0.0]])
    @pytest.mark.parametrize('method', ['heun_maruyama', 'euler_maruyama', 'milstein'])
    def test_integrate_t0(self, y0, method):
        model = Melcher25(sigma=0.2, gamma=1.5, alpha=-0.4)
        output = model.integrate(
            t_span=(0, 12), y0=y0, method=method, dt=0.012,
            kwargs={'random_seed': 0, 'si': 0.12},
        )
        assert model.state_variables.dtype.names == ('db', 'B')
        assert 'states' in output.diagnostic_variables
        assert np.all(np.isfinite(output.state_variables['db']))
        assert np.all(np.isfinite(output.state_variables['B']))
        assert set(np.unique(output.diagnostic_variables['states'])) <= {0.0, 1.0}

    def test_integrate_with_deterministic_method_t1(self):
        """uses_post_history=True models should also work with non-SDE methods."""
        model = Melcher25(alpha=-0.4)
        output = model.integrate(t_span=(0, 1.2), y0=[1.0, 0.0], method='euler', dt=0.012)
        assert np.all(np.isfinite(output.state_variables['db']))
        assert model.stadial_threshold is not None


class TestSignalModelsMelcher25Thresholds:
    def test_thresholds_set_after_integrate_t0(self):
        model = Melcher25(alpha=-0.4)
        model.integrate(
            t_span=(0, 12), y0=[1.0, 0.0], method='heun_maruyama', dt=0.012,
            kwargs={'random_seed': 1, 'si': 0.12},
        )
        assert model.stadial_threshold is not None
        assert model.interstadial_threshold is not None
        assert model.stadial_threshold < model.interstadial_threshold

    def test_thresholds_match_compute_stability_thresholds_t1(self):
        model = Melcher25(alpha=-0.4)
        model.integrate(
            t_span=(0, 12), y0=[1.0, 0.0], method='heun_maruyama', dt=0.012,
            kwargs={'random_seed': 1, 'si': 0.12},
        )
        expected_stadial, expected_interstadial = model.compute_stability_thresholds(-0.4)
        assert model.stadial_threshold == expected_stadial
        assert model.interstadial_threshold == expected_interstadial

    def test_callable_alpha_thresholds_match_scalar_t2(self):
        """Regression test: populate_diagnostics_from_history resolves a
        constant-valued callable alpha to the same thresholds as passing the
        same constant directly (exercises the branch reworked alongside
        compute_stability_thresholds's own mean/float reduction)."""
        const_model = Melcher25(alpha=-0.4)
        tv_model = Melcher25(alpha=lambda t: -0.4)

        t_span, y0, dt = (0, 12), [1.0, 0.0], 0.012

        const_model.integrate(t_span=t_span, y0=y0, method='heun_maruyama', dt=dt,
                               kwargs={'random_seed': 7, 'si': 0.12})
        tv_model.integrate(t_span=t_span, y0=y0, method='heun_maruyama', dt=dt,
                            kwargs={'random_seed': 7, 'si': 0.12})

        assert tv_model.stadial_threshold == const_model.stadial_threshold
        assert tv_model.interstadial_threshold == const_model.interstadial_threshold

    def test_array_alpha_thresholds_use_mean_t3(self):
        """compute_stability_thresholds should reduce an array-like alpha to its mean."""
        model = Melcher25()
        alpha_arr = np.array([-0.2, -0.6])
        stadial, interstadial = model.compute_stability_thresholds(alpha_arr)
        expected_stadial, expected_interstadial = model.compute_stability_thresholds(np.mean(alpha_arr))
        assert stadial == expected_stadial
        assert interstadial == expected_interstadial


class TestSignalModelsMelcher25ClassifyStandalone:
    def test_classify_bistable_states_matches_model_t0(self):
        model = Melcher25(alpha=-0.4)
        output = model.integrate(
            t_span=(0, 12), y0=[1.0, 0.0], method='heun_maruyama', dt=0.012,
            kwargs={'random_seed': 3, 'si': 0.12},
        )
        db = output.state_variables['db']
        states_from_model = output.diagnostic_variables['states']
        states_standalone = classify_bistable_states(db, alpha=-0.4)
        assert np.array_equal(states_from_model, states_standalone)

    def test_classify_bistable_states_hysteresis_t1(self):
        """A signal that dips below the stadial threshold and rises above the
        interstadial threshold should flip states with hysteresis (no chatter
        for values between the two thresholds)."""
        model = Melcher25(alpha=-0.4)
        stadial, interstadial = model.compute_stability_thresholds(-0.4)
        mid = 0.5 * (stadial + interstadial)
        signal = np.array([interstadial + 0.1, mid, stadial - 0.1, mid, interstadial + 0.1])
        states = classify_bistable_states(signal, alpha=-0.4)
        assert list(states) == [0, 0, 1, 1, 0]


class TestSignalModelsMelcher25SDENoise:
    def test_zero_sigma_is_deterministic_t0(self):
        """sigma=0 should make euler_maruyama reduce to the deterministic drift,
        independent of the random seed."""
        model_a = Melcher25(sigma=0.0, alpha=-0.4)
        model_b = Melcher25(sigma=0.0, alpha=-0.4)
        t_span, y0, dt = (0, 12), [1.0, 0.0], 0.012
        out_a = model_a.integrate(t_span=t_span, y0=y0, method='euler_maruyama', dt=dt,
                                   kwargs={'random_seed': 1, 'si': 0.12})
        out_b = model_b.integrate(t_span=t_span, y0=y0, method='euler_maruyama', dt=dt,
                                   kwargs={'random_seed': 999, 'si': 0.12})
        assert np.allclose(out_a.state_variables['db'], out_b.state_variables['db'])

    def test_sde_noise_shape_and_scale_t1(self):
        model = Melcher25(sigma=0.3)
        diffusion = model.sde_noise(0.0, [1.0, 0.0])
        assert diffusion.shape == (2,)
        assert np.allclose(diffusion, 0.3)


class TestSignalModelsMelcher25TimeVaryingParams:
    def test_time_varying_params_match_constants_t0(self):
        model_const = Melcher25(gamma=1.5, alpha=-0.4)
        model_tv = Melcher25(
            gamma=lambda t: 1.5,
            alpha=lambda t, x: -0.4,
        )
        t_span, y0, dt = (0, 1.2), [1.0, 0.0], 0.012
        model_const.integrate(t_span=t_span, y0=y0, method='euler', dt=dt)
        model_tv.integrate(t_span=t_span, y0=y0, method='euler', dt=dt)

        const_last = np.array([model_const.state_variables['db'][-1], model_const.state_variables['B'][-1]])
        tv_last = np.array([model_tv.state_variables['db'][-1], model_tv.state_variables['B'][-1]])
        assert np.allclose(const_last, tv_last, rtol=1e-8, atol=1e-10)
