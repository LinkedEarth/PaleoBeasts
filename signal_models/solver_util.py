from scipy.integrate import solve_ivp
import numpy as np


class Solver:
    def __init__(self, model, t_span, y0, method='RK45', kwargs=None):
        self.model = model
        self.t_span = t_span
        self.y0 = y0
        self.solution = None
        self.method = method
        self.t_eval = None
        self.kwargs = kwargs if kwargs is not None else {}
        self.diagnostics = {}

    def define_t_eval(self, delta_t=None, num_points=None):
        if num_points is not None:
            self.t_eval = np.linspace(self.t_span[0], self.t_span[1], num_points)
        elif delta_t is not None:
            self.t_eval = np.arange(self.t_span[0], self.t_span[1], delta_t)
        else:
            raise ValueError("Either 'delta_t' or 'num_points' must be provided.")

    def integrate(self, kwargs=None):
        if kwargs is None:
            kwargs = self.kwargs
        else:
            kwargs = {**self.kwargs, **kwargs}
        if self.t_eval is not None:
            kwargs['t_eval'] = self.t_eval
        if 'method' in kwargs:
            self.method = kwargs['method']

        if self.method == 'euler':
            solution = euler_method(self.model.dydt, self.y0, self.t_span[0], self.t_span[1], kwargs['dt'],
                                               args=self.model.params)

            self.model.state_variables = np.array(self.model.state_variables).reshape(len(solution.y),
                                                              len(self.model.state_variables[0]))

            solution.y = np.concatenate((solution.y, self.model.state_variables), axis=1)

        else:
            solution = solve_ivp(self.model.dydt, self.t_span,
                                 self.y0,
                                 dense_output=kwargs['dense_output'] if 'dense_output' in kwargs else True,
                                 method=self.method,
                                 args=self.model.params,
                                 **kwargs)
            solution.y = solution.y.T


        self.solution = solution


class Solution:
    def __init__(self, t, y):
        self.t = t
        self.y = y


def euler_method(f, y0, t0, tf, dt, args=()):
    """
    Solves an ODE using the Euler method with a fixed timestep.

    :param args:
    :param f: callable - The derivative function of the ODE (dy/dt).
    :param y0: float - Initial condition.
    :param t0: float - Initial time.
    :param tf: float - Final time.
    :param dt: float - Timestep for the integration.
    :return: Solution - An object containing the time and value arrays.
    """
    n_steps = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, n_steps)
    y = np.zeros((n_steps, len(y0)))
    y[0] = y0

    for i in range(1, n_steps):
        dy = f(t[i - 1], y[i - 1], *args)
        y[i] = y[i - 1] + np.multiply(dy,dt)

    solution = Solution(t, y)
    return solution



