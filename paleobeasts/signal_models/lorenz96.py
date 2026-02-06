import numpy as np

from ..core.pbmodel import PBModel


class Lorenz96(PBModel):
    """Lorenz (1996) system.

    Parameters
    ----------
    forcing : pb.Forcing or None
        Forcing object providing F(t). If None, the constant F parameter is used.

    var_name : str
        Name of the variable being modeled. Default is 'lorenz96'.

    n : int
        Number of state variables. Default is 40.

    F : float or callable or pb.Forcing
        Constant forcing when forcing is None, or a time-varying parameter. Default is 8.

    Notes
    -----
    If ``forcing`` is provided, it takes precedence over ``F``. Otherwise ``F`` can be
    a constant, callable, or ``pb.Forcing`` for time-varying behavior.
    """

    def __init__(self, forcing=None, var_name='lorenz96', n=40, F=8.0,
                 state_variables=None, diagnostic_variables=None, *args, **kwargs):
        if state_variables is None:
            state_variables = [f'x{i}' for i in range(n)]
        if diagnostic_variables is None:
            diagnostic_variables = []

        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables, *args, **kwargs)

        self.n = n
        self.F = F
        self.param_values = {
            'F': F,
        }
        self.params = ()

    def _forcing_value(self, t, x):
        if self.forcing is None:
            return self.get_param('F', t, x)
        return self.forcing.get_forcing(self.time_util(t))

    def dydt(self, t, x):
        x = np.asarray(x, dtype=float)
        F_t = self._forcing_value(t, x)
        n = self.n

        dxdt = np.zeros(n, dtype=float)
        for i in range(n):
            dxdt[i] = (x[(i + 1) % n] - x[i - 2]) * x[i - 1] - x[i] + F_t

        new_row = np.array([tuple(x)], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        if t > 0:
            self.time.append(t)

        return dxdt.tolist()


class Lorenz96TwoScale(PBModel):
    """Lorenz (1996) two-time scale system.

    Parameters
    ----------
    forcing : pb.Forcing or None
        Optional forcing object providing F(t). If None, the constant F parameter is used.

    var_name : str
        Name of the variable being modeled. Default is 'lorenz96_two_scale'.

    K : int
        Number of global-scale variables X. Default is 36.

    J : int
        Number of local-scale variables Y per global-scale variable. Default is 10.

    F : float or callable or pb.Forcing
        Constant forcing when forcing is None, or a time-varying parameter. Default is 10.

    h : float or callable or pb.Forcing
        Coupling coefficient between X and Y. Default is 1.

    b : float or callable or pb.Forcing
        Ratio of amplitudes between X and Y. Default is 10.

    c : float or callable or pb.Forcing
        Time-scale ratio between X and Y. Default is 10.
    """

    def __init__(self, forcing=None, var_name='lorenz96_two_scale', K=36, J=10,
                 F=10.0, h=1.0, b=10.0, c=10.0,
                 state_variables=None, diagnostic_variables=None, *args, **kwargs):
        if state_variables is None:
            x_names = [f'x{k}' for k in range(K)]
            y_names = [f'y{j}' for j in range(K * J)]
            state_variables = x_names + y_names
        if diagnostic_variables is None:
            diagnostic_variables = []

        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables, *args, **kwargs)

        self.K = K
        self.J = J
        self.F = F
        self.h = h
        self.b = b
        self.c = c
        self.param_values = {
            'F': F,
            'h': h,
            'b': b,
            'c': c,
        }
        self.params = ()

    def _forcing_value(self, t, x):
        if self.forcing is None:
            return self.get_param('F', t, x)
        return self.forcing.get_forcing(self.time_util(t))

    def _split_state(self, x):
        x = np.asarray(x, dtype=float)
        X = x[:self.K]
        Y = x[self.K:]
        return X, Y

    def _state_to_arrays(self):
        state = self.state_variables
        X = np.column_stack([state[f'x{i}'] for i in range(self.K)])
        Y = np.column_stack([state[f'y{i}'] for i in range(self.K * self.J)])
        t = self.time
        return X, Y, t

    def dydt(self, t, x):
        X, Y = self._split_state(x)

        F_t = self._forcing_value(t, x)
        h = self.get_param('h', t, x)
        b = self.get_param('b', t, x)
        c = self.get_param('c', t, x)

        K = self.K
        J = self.J

        dX = np.zeros(K, dtype=float)
        dY = np.zeros(K * J, dtype=float)

        Y_reshaped = Y.reshape(K, J)
        coupling = Y_reshaped.sum(axis=1)

        for k in range(K):
            xm1 = X[(k - 1) % K]
            xm2 = X[(k - 2) % K]
            xp1 = X[(k + 1) % K]
            dX[k] = -xm1 * (xm2 - xp1) - X[k] + F_t - (h * c / b) * coupling[k]

        for k in range(K):
            for j in range(J):
                jm1 = (j - 1) % J
                jp1 = (j + 1) % J
                jp2 = (j + 2) % J
                yjk = Y_reshaped[k, j]
                dY[k * J + j] = (-c * b * Y_reshaped[k, jp1] *
                                 (Y_reshaped[k, jp2] - Y_reshaped[k, jm1]) -
                                 c * yjk + (h * c / b) * X[k])

        new_row = np.array([tuple(np.concatenate([X, Y]))], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        if t > 0:
            self.time.append(t)

        return np.concatenate([dX, dY]).tolist()

    def run(self, si, total_time, y0, dt=None, method='euler'):
        """Integrate and return X, Y, t arrays sampled at interval si."""
        if dt is None:
            dt = si

        if method == 'euler':
            self.integrate(t_span=(0, total_time), y0=y0, method=method, kwargs={'dt': dt})
            X, Y, t = self._state_to_arrays()
            if si is not None and dt is not None:
                step = int(round(si / dt))
                if step > 1:
                    X = X[::step]
                    Y = Y[::step]
                    t = t[::step]
        else:
            self.integrate(t_span=(0, total_time), y0=y0, method=method)
            if si is not None:
                t_eval = np.arange(0, total_time + si, si)
                self.reframe_time_axis(t_eval, update_state=True)
            X, Y, t = self._state_to_arrays()

        return X, Y, t
