import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import functools


class Forcing:
    def __init__(self, data, time=None, params=None, interpolation='cubic'):
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
        self.forcing_type = None

        if isinstance(self.data, np.ndarray):
            print('data is an array')
            self.forcing_type = 'interpolated array {}'.format(interpolation)
            if interpolation == 'cubic':
                interp_func = CubicSpline
            elif interpolation == 'linear':
                interp_func = interp1d

            if self.time is not None:
                self.forcing_func = interp_func(time, data, **self.params)
            else:
                self.forcing_func = interp_func(time, data, **self.params)
        elif callable(self.data):
            self.forcing_type = 'function'
            self.forcing_func = functools.partial(self.data, **self.params)

    def get_forcing(self, t):
        """
        Get the forcing value at time t.
        """
        return self.forcing_func(t)
        # if callable(self.data):
        #     return self.data(t, **self.params)
        # elif isinstance(self.data, np.ndarray):
        #     idx = int(t)  # TODO Assuming t is an index; interpolate if not
        #     return self.data[idx]

