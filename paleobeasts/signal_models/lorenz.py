import numpy as np

from ..core.pbmodel import PBModel


class Lorenz63(PBModel):
    """Lorenz (1963) system.

    Parameters
    ----------
    forcing : pb.Forcing
        Forcing object providing f(t). If f(t) is a scalar, it is added to dx/dt.
        If f(t) is array-like with 3 entries, it is added to (dx/dt, dy/dt, dz/dt).

    var_name : str
        Name of the variable being modeled. Default is 'lorenz63'.

    sigma : float
        Prandtl number. Default is 10.

    rho : float
        Rayleigh number. Default is 28.

    beta : float
        Geometric factor. Default is 8/3.
    """

    def __init__(self, forcing, var_name='lorenz63', sigma=10.0, rho=28.0, beta=8 / 3,
                 state_variables=None, diagnostic_variables=None, *args, **kwargs):
        if state_variables is None:
            state_variables = ['x', 'y', 'z']
        if diagnostic_variables is None:
            diagnostic_variables = []

        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables, *args, **kwargs)

        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.params = (sigma, rho, beta)

    def _forcing_vector(self, t):
        if self.forcing is None:
            return np.zeros(3)

        f_val = self.forcing.get_forcing(self.time_util(t))

        if np.isscalar(f_val):
            return np.array([f_val, 0.0, 0.0])

        f_arr = np.asarray(f_val, dtype=float)
        if f_arr.size != 3:
            raise ValueError("Forcing must be a scalar or an array-like with 3 entries.")

        return f_arr.reshape(3,)

    def dydt(self, t, x, sigma, rho, beta):
        x_val, y_val, z_val = x[0], x[1], x[2]
        f_vec = self._forcing_vector(t)

        dxdt = sigma * (y_val - x_val) + f_vec[0]
        dydt = x_val * (rho - z_val) - y_val + f_vec[1]
        dzdt = x_val * y_val - beta * z_val + f_vec[2]

        new_row = np.array([(x_val, y_val, z_val)], dtype=self.dtypes)
        self.state_variables = np.concatenate([self.state_variables, new_row], axis=0)
        if t > 0:
            self.time.append(t)

        return [dxdt, dydt, dzdt]
