''' Tests for paleobeasts.core.pbmodel time axis utilities

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}
'''

import numpy as np
import paleobeasts as pb

from paleobeasts.core.pbmodel import PBModel
from paleobeasts.signal_models import lorenz


class TestCorePBModelReframeTimeAxis:
    def test_reframe_time_axis_rk45_t0(self):
        def func(x):
            return 0

        forcing = pb.core.Forcing(func)
        model = lorenz.Lorenz63(forcing=forcing)
        model.integrate(t_span=(0, 5), y0=[1, 1, 1], method='RK45')

        t_eval = np.linspace(0, 5, 51)
        reframed = model.reframe_time_axis(t_eval, update_state=False)

        assert len(reframed) == len(t_eval)
        assert set(reframed.dtype.names) == {'x', 'y', 'z'}

    def test_reframe_time_axis_euler_t0(self):
        def func(x):
            return 0

        forcing = pb.core.Forcing(func)
        model = lorenz.Lorenz63(forcing=forcing)
        model.integrate(t_span=(0, 5), y0=[1, 1, 1], method='euler', kwargs={'dt': 0.1})

        t_eval = np.linspace(0, 5, 26)
        reframed = model.reframe_time_axis(t_eval, update_state=True)

        assert len(reframed) == len(t_eval)
        assert np.allclose(model.time, t_eval)


class _PostHistoryModel(PBModel):
    def __init__(self):
        super().__init__(forcing=None, variable_name='post_history', state_variables=['x'],
                         diagnostic_variables=['x_squared'])

    def uses_post_history(self):
        return True

    def dydt(self, t, x):
        return [-x[0]]

    def populate_diagnostics_from_history(self, time, history):
        self.diagnostic_variables['x_squared'] = history[:, 0] ** 2


class TestCorePBModelPostHistoryHooks:
    def test_post_history_model_integrates_t0(self):
        model = _PostHistoryModel()
        model.integrate(t_span=(0, 1), y0=[1.0], method='euler', kwargs={'dt': 0.1})

        assert model.state_variables.dtype.names == ('x',)
        assert len(model.time) == len(model.diagnostic_variables['x_squared'])
        assert np.isclose(model.state_variables['x'][0], 1.0)
