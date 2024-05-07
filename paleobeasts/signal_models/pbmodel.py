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
        self.diagnostic_variables = {var:[] for var in diagnostic_variables}
        self.params = None





