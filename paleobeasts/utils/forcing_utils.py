import numpy as np

__all__ = [
    'create_periodic_forcing_function'
]

def create_periodic_forcing_function(periods_powers, desired_amplitude=1, y0=0):
    """
    Creates a composite forcing function for a differential equation, where the force is the sum
    of sine waves, each defined by a period and an amplitude (power). The amplitudes of the sine
    waves are rescaled so that the sum of their individual maximum amplitudes is equal to the
    desired amplitude.

    :param periods_powers: List of tuples where each tuple is (period, power)
    :param desired_amplitude: Desired amplitude of the composite forcing function
    :param y0: Initial value of the forcing function
    :return: A function that takes time t and returns the sum of sine wave amplitudes at that time
    """

    # Sum of individual max amplitudes
    total_max_amplitude = sum(power for _, power in periods_powers)

    def forcing_function(t):
        result = y0
        for period, power in periods_powers:
            frequency = 1 / period
            scaled_power = power / total_max_amplitude * desired_amplitude
            result += scaled_power * np.sin(2 * np.pi * frequency * t)
        return result

    return forcing_function
