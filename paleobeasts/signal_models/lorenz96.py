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

    F : float
        Constant forcing when forcing is None. Default is 8.
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
        self.params = (n, F)

    def _forcing_value(self, t):
        if self.forcing is None:
            return self.F
        return self.forcing.get_forcing(self.time_util(t))

    def dydt(self, t, x, n, F):
        x = np.asarray(x, dtype=float)
        F_t = self._forcing_value(t)

        dxdt = np.zeros(n, dtype=float)
        for i in range(n):
            dxdt[i] = (x[(i + 1) % n] - x[i - 2]) * x[i - 1] - x[i] + F_t

        new_row = np.array([tuple(x)], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        if t > 0:
            self.time.append(t)

        return dxdt.tolist()
