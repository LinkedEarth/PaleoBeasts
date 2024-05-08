from scipy.integrate import solve_ivp
import numpy as np
from ..utils.solver_util import euler_method
class PBModel:

    def __init__(self, forcing, variable_name, state_variables=None, non_integrated_state_vars=None,
                 diagnostic_variables=None):
        self.variable_name = variable_name
        self.forcing = forcing

        if state_variables is None:
            state_variables = []
        if non_integrated_state_vars is None:
            non_integrated_state_vars = []

        self.state_variables_names = state_variables
        self.non_integrated_state_vars = non_integrated_state_vars
        self.integrated_state_vars = [var for var in state_variables if var not in self.non_integrated_state_vars]
        self.dtypes = None
        self.state_variables = None

        if diagnostic_variables is None:
            diagnostic_variables = ['time']
        if 'time' not in diagnostic_variables:
            diagnostic_variables.append('time')
        self.diagnostic_variables = {var:[] for var in diagnostic_variables}
        self.params = None

        self.t_span = None
        self.y0 = None
        self.solution = None
        self.method = None
        self.t_eval = None
        self.kwargs = None

    def dydt(self):
        pass

    def integrate(self, t_span, y0, method='RK45', kwargs=None):

        self.t_span = t_span
        self.y0 = y0
        self.solution = None
        self.method = method
        self.t_eval = None
        self.kwargs = kwargs if kwargs is not None else {}

        # Define the structured array
        if len(self.state_variables_names) > 0:
            dtype = [(var, float) for i, var in enumerate(self.state_variables_names)]
            self.dtypes = dtype
        else:
            dtype = [type(val) for i, val in enumerate(self.y0)]
            self.dtypes = dtype

        array = np.array(self.y0, dtype=dtype)
        self.state_variables = array

        if kwargs is None:
            kwargs = self.kwargs
        else:
            kwargs = {**self.kwargs, **kwargs}
        if self.t_eval is not None:
            kwargs['t_eval'] = self.t_eval
        if 'method' in kwargs:
            self.method = kwargs['method']

        if self.method == 'euler':
            solution = euler_method(self.dydt, self.t_span,self.y0[:len(self.integrated_state_vars)],  kwargs['dt'],
                                    args=self.params)

        else:
            solution = solve_ivp(self.dydt, self.t_span,
                                 self.y0[:len(self.integrated_state_vars)],
                                 dense_output=kwargs['dense_output'] if 'dense_output' in kwargs else True,
                                 method=self.method,
                                 args=self.params,
                                 **kwargs)
            solution.y = solution.y.T

        self.solution = solution
        self.state_variables = self.state_variables[1:]

        for var in self.diagnostic_variables.keys():
            self.diagnostic_variables[var] = np.array(
                self.diagnostic_variables[var])  # .reshape(len(solution.y)





