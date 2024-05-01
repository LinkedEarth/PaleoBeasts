
import numpy as np

class Forcing:
    def __init__(self, data, time=None, params=None):
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

    def get_forcing(self, t):
        """
        Get the forcing value at time t.
        """
        if callable(self.data):
            return self.data(t, **self.params)
        elif isinstance(self.data, np.ndarray):
            idx = int(t)  # TODO Assuming t is an index; interpolate if not
            return self.data[idx]

