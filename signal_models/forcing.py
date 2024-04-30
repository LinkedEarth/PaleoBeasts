
import numpy as np

class Forcing:
    def __init__(self, data, time=None, derivative=None, params=None):
        """
        Initialize the Forcing class.
        --forcing from vector untested!!!!

        Parameters:
        -----------
        data : callable function or array
            representing the forcing over time.

        time : numeric or array
            time values corresponding to the data points
            if data is an array, time must be provided.

        derivative : callable function representing the derivative of the forcing, or None if `data` is an array.

        """
        self.data = data
        self.time = time
        self.params = params if params is not None else {}
        self.derivative = derivative

    def get_forcing(self, t):
        """
        Get the forcing value at time t.
        """
        if callable(self.data):
            return self.data(t, **self.params)
        elif isinstance(self.data, np.ndarray):
            idx = int(t)  # TODO Assuming t is an index; interpolate if not
            return self.data[idx]

    def get_derivative(self, t):
        """
        Get the derivative of the forcing at time t.
        """
        if self.derivative is not None:
            return self.derivative(t)
        elif isinstance(self.data, np.ndarray):
            # Calculate numerical derivative if `derivative` is not provided and `data` is an array
            if not hasattr(self, 'numeric_derivative'):
                if self.time is not None:
                    self.numeric_derivative = np.gradient(self.data, self.time)
                else:
                    self.numeric_derivative = np.gradient(self.data)  # Assumes uniform spacing of 1
            idx = int(t)
            return self.numeric_derivative[idx]
        else:
            raise ValueError("No method for derivative calculation provided.")
