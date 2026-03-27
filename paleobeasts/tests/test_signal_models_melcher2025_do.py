"""Tests for paleobeasts.signal_models.melcher2025_do."""

import numpy as np

from paleobeasts.signal_models import Melcher2025DOModel, stommel


class TestSignalModelsMelcher2025DOIntegrate:
    def test_paper_baseline_defaults_t0(self):
        model = Melcher2025DOModel()
        assert model.q0 == -9.0
        assert model.q1 == 12.0
        assert model.b0 == 0.625
        assert model.tau == 0.902

    def test_integrate_euler_maruyama_t0(self):
        model = Melcher2025DOModel(parameter_contract="strict")
        model.integrate(
            t_span=(0, 20),
            y0=[1.0, 0.0],
            method="euler_maruyama",
            kwargs={"dt": 0.01, "random_seed": 123},
        )

        assert model.state_variables.dtype.names == ("delta_b", "B")
        assert "q" in model.diagnostic_variables
        assert np.all(np.isfinite(model.diagnostic_variables["q"]))
        assert np.all(np.isfinite(model.diagnostic_variables["amoc_dim"]))
        assert np.all(np.isfinite(model.diagnostic_variables["aabw_dim"]))

    def test_seed_reproducibility_t0(self):
        kwargs = {"dt": 0.01, "random_seed": 42}
        m1 = Melcher2025DOModel()
        m2 = Melcher2025DOModel()
        m3 = Melcher2025DOModel()

        m1.integrate(t_span=(0, 20), y0=[1.0, 0.0], method="euler_maruyama", kwargs=kwargs)
        m2.integrate(t_span=(0, 20), y0=[1.0, 0.0], method="euler_maruyama", kwargs=kwargs)
        m3.integrate(
            t_span=(0, 20),
            y0=[1.0, 0.0],
            method="euler_maruyama",
            kwargs={"dt": 0.01, "random_seed": 7},
        )

        assert np.allclose(m1.state_variables["delta_b"], m2.state_variables["delta_b"])
        assert np.allclose(m1.state_variables["B"], m2.state_variables["B"])
        assert not np.allclose(m1.state_variables["delta_b"], m3.state_variables["delta_b"])

    def test_transport_sign_reversal_t0(self):
        model = Melcher2025DOModel(q0=-9.0, q1=12.0, b0=1.38)
        q_low = model.transport(0.0, np.array([1.0, 0.0]))
        q_high = model.transport(0.0, np.array([2.5, 0.0]))
        assert q_low < 0.0
        assert q_high > 0.0

    def test_time_varying_alpha_gamma_strict_contract_t0(self):
        alpha_t = lambda t: -0.8 + 0.1 * np.sin(2.0 * np.pi * t / 80.0)
        gamma_t = lambda t: 1.2 + 0.2 * np.cos(2.0 * np.pi * t / 100.0)
        model = Melcher2025DOModel(parameter_contract="strict", alpha=alpha_t, gamma=gamma_t)
        model.integrate(
            t_span=(0, 20),
            y0=[1.0, 0.0],
            method="euler_maruyama",
            kwargs={"dt": 0.01, "random_seed": 3},
        )
        assert np.all(np.isfinite(model.state_variables["delta_b"]))
        assert np.all(np.isfinite(model.state_variables["B"]))

    def test_sigma_vector_form_t0(self):
        model = Melcher2025DOModel(sigma=lambda t: np.array([0.2, 0.1]), parameter_contract="strict")
        model.integrate(
            t_span=(0, 10),
            y0=[1.0, 0.0],
            method="euler_maruyama",
            kwargs={"dt": 0.01, "random_seed": 1},
        )
        assert len(model.time) == len(model.state_variables)


class TestBackwardCompatibilitySanity:
    def test_existing_euler_model_unchanged_t0(self):
        model = stommel.Stommel(forcing=None)
        model.integrate(t_span=(0, 1), y0=[1.0, 0.5], method="euler", kwargs={"dt": 0.01})
        assert np.isfinite(model.state_variables["T"][-1])
        assert np.isfinite(model.state_variables["S"][-1])
