''' Tests for climatecritters.core.ccmodel time axis utilities

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}
'''

import warnings

import numpy as np
import pytest
import climatecritters as cc

from climatecritters.core.ccmodel import CCModel
from climatecritters.model_critters import lorenz


class TestCoreCCModelReframeTimeAxis:
    def test_reframe_time_axis_rk45_t0(self):
        model = lorenz.Lorenz63()
        output = model.integrate(t_span=(0, 5), y0=[1, 1, 1], method='RK45')

        t_eval = np.linspace(0, 5, 51)
        reframed = output.reframe_time_axis(t_eval)

        assert len(reframed) == len(t_eval)
        assert set(reframed.dtype.names) == {'x', 'y', 'z'}

    def test_reframe_time_axis_euler_t0(self):
        model = lorenz.Lorenz63()
        output = model.integrate(t_span=(0, 5), y0=[1, 1, 1], method='euler', dt=0.1)

        t_eval = np.linspace(0, 5, 26)
        output.reframe_time_axis(t_eval)

        assert len(output.time) == len(t_eval)
        assert np.allclose(output.time, t_eval)


class _PostHistoryModel(CCModel):
    def __init__(self):
        super().__init__(variable_name='post_history', state_variables=['x'],
                         diagnostic_variables=['x_squared'])

    uses_post_history = True

    def dydt(self, t, x):
        return [-x[0]]

    def populate_diagnostics_from_history(self, time, history):
        self.diagnostic_variables['x_squared'] = history[:, 0] ** 2


class TestCoreCCModelPostHistoryHooks:
    def test_post_history_model_integrates_t0(self):
        model = _PostHistoryModel()
        model.integrate(t_span=(0, 1), y0=[1.0], method='euler', dt=0.1)

        assert model.state_variables.dtype.names == ('x',)
        assert len(model.time) == len(model.diagnostic_variables['x_squared'])
        assert np.isclose(model.state_variables['x'][0], 1.0)


class _SDEPostHistoryModel(CCModel):
    """uses_post_history=True model with additive noise, for si/forcing tests."""

    uses_post_history = True

    def __init__(self):
        super().__init__(variable_name='sde_post_history', state_variables=['x'])
        self.param_values = {'k': 0.0}
        self.params = ()

    def dydt(self, t, x):
        k = self.get_param_value('k', t, x)
        return [-x[0] + k]

    def sde_noise(self, t, x):
        return np.array([0.1])


class _SDENoPostHistoryModel(CCModel):
    """uses_post_history=False model, to exercise the si guard."""

    uses_post_history = False

    def __init__(self):
        super().__init__(variable_name='sde_no_post_history', state_variables=['x'])
        self.param_values = {}
        self.params = ()

    def dydt(self, t, x):
        return [-x[0]]

    def sde_noise(self, t, x):
        return np.array([0.1])


class _SDENoNoiseOverrideModel(CCModel):
    """uses_post_history=True model that does NOT override sde_noise, to
    confirm the base-class stub recovers deterministic integration."""

    uses_post_history = True

    def __init__(self):
        super().__init__(variable_name='sde_no_noise_override', state_variables=['x'])
        self.param_values = {}
        self.params = ()

    def dydt(self, t, x):
        return [-x[0]]


class TestCoreCCModelSDENoiseStub:
    def test_default_sde_noise_returns_zeros_t0(self):
        model = _SDENoNoiseOverrideModel()
        zeros = model.sde_noise(0.0, [1.0, 2.0])
        np.testing.assert_array_equal(zeros, [0.0, 0.0])

    @pytest.mark.parametrize('method', ['euler_maruyama', 'heun_maruyama', 'milstein'])
    def test_unoverridden_sde_noise_is_seed_independent_t1(self, method):
        """Without an sde_noise override, the diffusion term is always zero,
        so different random seeds must produce identical (deterministic)
        trajectories for euler_maruyama/heun_maruyama/milstein."""
        out_seed0 = _SDENoNoiseOverrideModel().integrate(
            t_span=(0.0, 1.0), y0=[1.0], method=method, dt=0.1,
            kwargs={'random_seed': 0},
        )
        out_seed1 = _SDENoNoiseOverrideModel().integrate(
            t_span=(0.0, 1.0), y0=[1.0], method=method, dt=0.1,
            kwargs={'random_seed': 1},
        )

        np.testing.assert_array_equal(
            out_seed0.state_variables['x'], out_seed1.state_variables['x']
        )

    def test_unoverridden_sde_noise_matches_deterministic_euler_for_euler_maruyama_t2(self):
        """euler_maruyama's drift discretization is forward Euler, so with
        zero diffusion it should match plain euler exactly."""
        sde_out = _SDENoNoiseOverrideModel().integrate(
            t_span=(0.0, 1.0), y0=[1.0], method='euler_maruyama', dt=0.1,
            kwargs={'random_seed': 0},
        )
        euler_out = _SDENoNoiseOverrideModel().integrate(
            t_span=(0.0, 1.0), y0=[1.0], method='euler', dt=0.1,
        )

        np.testing.assert_allclose(
            sde_out.state_variables['x'], euler_out.state_variables['x'], atol=1e-8
        )


class TestCoreCCModelSDESamplingInterval:
    @pytest.mark.parametrize('method', ['euler_maruyama', 'heun_maruyama', 'milstein'])
    def test_si_subsamples_output_t0(self, method):
        model = _SDEPostHistoryModel()
        output = model.integrate(
            t_span=(0.0, 5.0), y0=[1.0], method=method, dt=0.01,
            kwargs={'random_seed': 0, 'si': 0.1},
        )
        assert len(output.time) == 51
        np.testing.assert_allclose(output.time, np.linspace(0.0, 5.0, 51), atol=1e-9)

    @pytest.mark.parametrize('method', ['euler_maruyama', 'heun_maruyama', 'milstein'])
    def test_si_requires_uses_post_history_t1(self, method):
        model = _SDENoPostHistoryModel()
        with pytest.raises(ValueError, match="uses_post_history"):
            model.integrate(
                t_span=(0.0, 1.0), y0=[1.0], method=method, dt=0.01,
                kwargs={'si': 0.1},
            )

    @pytest.mark.parametrize('method', ['euler_maruyama', 'heun_maruyama', 'milstein'])
    def test_reframe_stochastic_coarser_grid_warns_t2(self, method):
        """t_eval spacing (0.5) coarser than the integrated grid (dt=0.1) warns,
        but still resamples (it's a soft warning, not a hard failure)."""
        model = _SDEPostHistoryModel()
        output = model.integrate(
            t_span=(0.0, 5.0), y0=[1.0], method=method, dt=0.1,
            kwargs={'random_seed': 0},
        )
        with pytest.warns(UserWarning, match="coarser grid"):
            reframed = output.reframe_time_axis(np.linspace(0.0, 5.0, 11))
        assert len(reframed) == 11
        np.testing.assert_allclose(output.time, np.linspace(0.0, 5.0, 11))

    @pytest.mark.parametrize('method', ['euler_maruyama', 'heun_maruyama', 'milstein'])
    def test_reframe_stochastic_matching_grid_no_warning_t2b(self, method):
        """t_eval spacing equal to the integrated grid (dt=0.1) is an exact
        subsample, not interpolation, so no warning should fire."""
        model = _SDEPostHistoryModel()
        output = model.integrate(
            t_span=(0.0, 5.0), y0=[1.0], method=method, dt=0.1,
            kwargs={'random_seed': 0},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            output.reframe_time_axis(np.linspace(0.0, 5.0, 51))

    @pytest.mark.parametrize('method', ['heun_maruyama', 'milstein'])
    def test_pre_step_forcing_applied_t3(self, method):
        """heun_maruyama/milstein previously called self.dydt directly,
        silently dropping registered forcings; they must now use the
        forcing-wrapped dydt."""
        unforced = _SDEPostHistoryModel()
        out_unforced = unforced.integrate(
            t_span=(0.0, 1.0), y0=[1.0], method=method, dt=0.1,
            kwargs={'random_seed': 0},
        )

        forced = _SDEPostHistoryModel()
        forced.register_forcing('k', lambda t: 5.0)
        out_forced = forced.integrate(
            t_span=(0.0, 1.0), y0=[1.0], method=method, dt=0.1,
            kwargs={'random_seed': 0},
        )

        assert not np.isclose(
            out_unforced.state_variables['x'][-1], out_forced.state_variables['x'][-1]
        )


class _ParamContractModel(CCModel):
    def __init__(self, coeff=1.0):
        super().__init__(
            variable_name='param_contract',
            state_variables=['x'],
            diagnostic_variables=[],
        )
        self.coeff = coeff
        self.param_values = {'coeff': coeff}

    def dydt(self, t, x):
        coeff = self.get_param_value('coeff', t, x)
        return [coeff * x[0]]


class TestCoreCCModelParameterContract:
    def test_callable_accepts_supported_signatures_t0(self):
        model_t = _ParamContractModel(coeff=lambda t: 2.0)
        model_ts = _ParamContractModel(coeff=lambda t, state: 2.0)
        model_tsm = _ParamContractModel(coeff=lambda t, state, model: 2.0)

        assert model_t.get_param_value('coeff', 0.0, [1.0]) == 2.0
        assert model_ts.get_param_value('coeff', 0.0, [1.0]) == 2.0
        assert model_tsm.get_param_value('coeff', 0.0, [1.0]) == 2.0

    def test_non_compliant_callable_raises_t0(self):
        model = _ParamContractModel(coeff=lambda model, state: 2.0)
        with pytest.raises(TypeError):
            model.get_param_value('coeff', 0.0, [1.0])

    def test_attribute_assignment_syncs_param_values_t0(self):
        model = _ParamContractModel(coeff=1.0)
        model.coeff = lambda t: 3.0
        assert model.param_values['coeff'](0.0) == 3.0


class _FunctionSwapModel(CCModel):
    def __init__(self):
        super().__init__(variable_name='function_swap', state_variables=['x'])

    def calc_term(self, value):
        return value + 1


class TestCoreCCModelSetFunction:
    def test_set_function_plain_callable_t0(self):
        model = _FunctionSwapModel()

        def plain_calc_term(value):
            return value + 5

        model.set_function('calc_term', plain_calc_term)
        assert model.calc_term(2) == 7

    def test_set_function_bound_callable_t1(self):
        model = _FunctionSwapModel()

        def bound_calc_term(self, value):
            return value + 9

        model.set_function('calc_term', bound_calc_term)
        assert model.calc_term(1) == 10

    def test_set_function_force_bind_t2(self):
        model = _FunctionSwapModel()

        def bound_calc_term(self, value):
            return value + 4

        model.set_function('calc_term', bound_calc_term, bind=True)
        assert model.calc_term(3) == 7

    def test_set_function_errors_t3(self):
        model = _FunctionSwapModel()
        with pytest.raises(AttributeError):
            model.set_function('missing_function', lambda x: x)
        with pytest.raises(TypeError):
            model.set_function('calc_term', 123)
