from scipy.integrate import solve_ivp
import numpy as np
import pyleoclim as pyleo

from ..utils.solver_util import euler_method
class PBModel:
    '''The overarching model structure for Paleobeasts. 
    
    PBModel serves as the archetype/parent class for models within the signal_models directory.
    This class is not meant to be instantiated, but rather to be inherited by other classes.

    
    '''

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
            diagnostic_variables = []
        # if 'time' not in diagnostic_variables:
        #     diagnostic_variables.append('time')
        self.diagnostic_variables = {var:[] for var in diagnostic_variables}
        self.params = None

        self.t_span = None
        self.y0 = None
        self.solution = None
        self.method = None
        self.time = []
        self.kwargs = None
        self.t_eval= None
        self.run_name = None

    def dydt(self):
        '''The differential equation of the model. 
        
        This method should be implemented and used from the child class.'''
        pass

    def integrate(self, t_span, y0, method='RK45', kwargs=None, run_name=None):
        '''Integrates the model over a given time span.
        
        Parameters
        ----------
        
        t_span : tuple, list
            The time span over which the model will be integrated.
            
        y0 : list
            Initial conditions for the model. The length of this list should be equal to the number of model state variables.
        
        method : str
            The integration method to be use; options include 'RK45' (Runge Kutta) and 'euler'. Default is 'RK45'.

        kwargs : dict
            Additional keyword arguments to be passed to the solver.


        '''

        self.t_span = t_span
        self.y0 = y0
        self.solution = None
        self.method = method
        # self.t_eval = None
        self.kwargs = kwargs if kwargs is not None else {}
        if self.method == 'euler':
            assert 'dt' in kwargs, "Please provide a time step for the Euler method."

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
            self.kwargs['dt'] = 'variable'
            solution.y = solution.y.T

        self.run_name = run_name if run_name is not None else f'{self.method}, dt={self.kwargs["dt"]}'
        self.solution = solution
        self.state_variables = self.state_variables[1:]
        self.time = np.array(self.time)

        for var in self.diagnostic_variables.keys():
            self.diagnostic_variables[var] = np.array(
                self.diagnostic_variables[var])  # .reshape(len(solution.y)

    def to_pyleo(self,state_var_name):
        '''Function to create a pyleoclim Series object from a state variable.
        
        Parameters
        ----------
        
        state_var_name : str
            The name of the state variable to be converted.'''
        
        if state_var_name not in self.state_variables_names:
            raise ValueError(f"{state_var_name} not found. Please check the state variables.")
        
        if self.t_eval is None:
            raise ValueError("Time axis not found. Please integrate the model first.")

        series = pyleo.Series(
            time = self.t_eval, 
            value = self.state_variables[state_var_name],
            value_name = state_var_name,
            verbose=False,
            auto_time_params=True
            )

        return series


