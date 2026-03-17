import numpy as np

from ..core.forcing import Forcing

__all__ = [
    "create_periodic_forcing_function",
    "create_periodic_forcing",
    "create_constant_forcing",
    "create_sinusoid_forcing",
    "create_piecewise_forcing",
]


def _validate_periods_powers(periods_powers):
    if periods_powers is None:
        raise ValueError("periods_powers must not be None.")

    pairs = list(periods_powers)
    if len(pairs) == 0:
        raise ValueError("periods_powers must contain at least one (period, power) pair.")

    normalized = []
    for item in pairs:
        if len(item) != 2:
            raise ValueError("Each periods_powers entry must be a 2-tuple: (period, power).")
        period, power = float(item[0]), float(item[1])
        if period <= 0.0:
            raise ValueError("Periods must be > 0.")
        normalized.append((period, power))

    total_max_amplitude = float(sum(power for _, power in normalized))
    if np.isclose(total_max_amplitude, 0.0):
        raise ValueError("Sum of powers must be non-zero.")

    return normalized, total_max_amplitude


def create_periodic_forcing_function(periods_powers, desired_amplitude=1, y0=0):
    """
    Create a composite periodic forcing function made of sine components.

    The component amplitudes are rescaled so their summed max amplitude equals
    ``desired_amplitude``.
    """

    pairs, total_max_amplitude = _validate_periods_powers(periods_powers)
    desired_amplitude = float(desired_amplitude)
    y0 = float(y0)

    def forcing_function(t):
        t_arr = np.asarray(t, dtype=float)
        result = np.full(t_arr.shape, y0, dtype=float)
        for period, power in pairs:
            frequency = 1.0 / period
            scaled_power = power / total_max_amplitude * desired_amplitude
            result += scaled_power * np.sin(2.0 * np.pi * frequency * t_arr)
        if t_arr.ndim == 0:
            return float(result)
        return result

    return forcing_function


def create_periodic_forcing(periods_powers, desired_amplitude=1, y0=0):
    """Return a ``Forcing`` built from ``create_periodic_forcing_function``."""
    func = create_periodic_forcing_function(periods_powers, desired_amplitude=desired_amplitude, y0=y0)
    return Forcing(func)


def create_constant_forcing(value):
    """Return a constant forcing object."""

    def _constant(t):
        t_arr = np.asarray(t, dtype=float)
        out = np.full(t_arr.shape, float(value), dtype=float)
        if t_arr.ndim == 0:
            return float(out)
        return out

    return Forcing(_constant)


def create_sinusoid_forcing(A, period, y0=0.0):
    """Return a sinusoidal forcing object."""
    A = float(A)
    period = float(period)
    y0 = float(y0)
    if period <= 0.0:
        raise ValueError("period must be > 0.")

    def _sinusoid(t):
        t_arr = np.asarray(t, dtype=float)
        out = y0 + A * np.sin(2.0 * np.pi * t_arr / period)
        if t_arr.ndim == 0:
            return float(out)
        return out

    return Forcing(_sinusoid)


def create_piecewise_forcing(elements, y0=0.0, label="forcing"):
    """Return a piecewise forcing object from dict specs or forcing elements."""
    return Forcing.from_elements(elements=elements, y0=y0, label=label)
