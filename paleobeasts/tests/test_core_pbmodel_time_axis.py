''' Tests for paleobeasts.core.pbmodel time axis utilities

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}
'''

import numpy as np
import paleobeasts as pb

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
