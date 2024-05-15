import numpy as np
from scipy.interpolate import CubicSpline

__all__ = [
    'make_derivative_func'
]
def make_derivative_func(method='numpy', derivative=None, data=None, time=None):
    """
    Get the derivative of the forcing at time t.

    ## UNTESTED!!!!
    """
    if derivative is not None:
        if callable(derivative):
            return derivative
    elif isinstance(data, np.ndarray):
        # Calculate numerical derivative if `derivative` is not provided and `data` is an array
        if method == 'numpy':
            if time is None:
                time = np.arange(len(data))

            numeric_derivative = np.gradient(data, time)
            cubic_interp = CubicSpline(time, numeric_derivative)
            return cubic_interp

        if method == 'scipy':
            if time is None:
                time = np.arange(len(data))
            cubic_interp = CubicSpline(time, data)
            d1 = cubic_interp.derivative(nu=1)
            return d1
    else:
        raise ValueError("No method for derivative calculation provided.")
