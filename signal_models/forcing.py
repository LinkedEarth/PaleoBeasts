
import numpy as np

class Forcing:
    def __init__(self, data, time=None, derivative=None):
        """
        Initialize the Forcing class.
        --forcing from vector untested!!!!

        Parameters:
        data: Could be a callable function or an array representing the forcing over time.
        time: An array representing the time values corresponding to the data points or a single value representing uniform spacing.
        derivative: A callable function representing the derivative of the forcing, or None if `data` is an array.
        """
        self.data = data
        self.time = time
        self.derivative = derivative

    def get_forcing(self, t):
        if callable(self.data):
            return self.data(t)
        elif isinstance(self.data, np.ndarray):
            idx = int(t)  # Assuming t is an index; interpolate if not
            return self.data[idx]

    def get_derivative(self, t):
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
