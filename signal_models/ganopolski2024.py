import numpy as np
from scipy.integrate import solve_ivp

"""
# Example of using this module:

import numpy as np
from scipy.integrate import solve_ivp

k_init = 1
v_init = 0
initial_conditions = [v_init, k_init]  # Initial v and k
tn = 2000
t_span = (0, tn)
A = 25
eps = 0.5
f1 = -16
f2 = 16
vc = 1.4
t1 = 30
t2 = 100
vi = 0
t_eval = np.arange(0, tn, 1)

solution = solve_ivp(
    dVdt,
    t_span,
    initial_conditions,
    dense_output=True,
    t_eval=t_eval, 
    args=(A, eps, f1, f2, t2, t1, vc),
)
"""

def calc_f(t, A, eps, T1=100, T2=23):
    """
    Calculate the orbital forcing value at time t.
    """
    return A * (1 + eps * np.sin(2 * np.pi * t / T1)) * np.cos(2 * np.pi * t / T2)

def calc_df(t, A, eps, T1=100, T2=23):
    """
    Calculate the derivative of the orbital forcing at time t.
    """
    return A * eps * ((2 * np.pi / T1) * np.cos(2 * np.pi * t / T1) * np.cos(2 * np.pi * t / T2) -
                      (2 * np.pi / T2) * np.sin(2 * np.pi * t / T2) * np.sin(2 * np.pi * t / T1))

def calc_vg(f, f1, f2):
    """
    Calculate glacial equilibrium state.
    """
    return 1 + np.sqrt((f2 - f) / (f2 - f1))

def calc_vu(f, f1, f2):
    """
    Calculate unstable equilibrium which separates the glacial and interglacial attraction domains.
    """
    return 1 - np.sqrt((f2 - f) / (f2 - f1))

def calc_ve(v, f, f1, f2):
    """
    Calculate equilibrium state towards which the system is attracted is a function of orbital forcing and,
    for the bi-stable regime, also depends on ice volume
    """
    vg = calc_vg(f, f1, f2)
    vu = calc_vu(f, f1, f2)

    if f < f1:
        return vg
    elif f > f2:
        return 0
    elif f1 < f < f2 and v > vu:
        return vg
    elif f1 < f < f2 and v < vu:
        return 0

def calc_vc(t, vc1, vc2, t1, tau1):
    """
    Calculate the value for critical ice volume
    """
    return 0.5 * (vc1 + vc2) + 0.5 * (vc2 - vc1) * np.tanh((t - t1) / tau1)

def dVdt(t, x, A, eps, f1, f2, t2, t1, vc):

    """
    Differential equation for dv/dt (where v=ice volume) at state k.

    From Ganopolski et al. 2024:
    vc = value for critical ice volume; controls the dominant periodicity and degree of asymmetry of glacial cycles
    f1 = insolation threshold for glacial inception (pinned at -20 to -15 W/m^2)
    t1 = relaxation time scale for glacial inception (in kyr)
    f2 = insolation threshold for deglaciation inception (tunable; positive)
    t2 = relaxation time scale for deglaciation (in kyr)

    A = magnitude of forcing in Wm−2
    eps = nondimensional magnitude of amplitude modulation

    t = time in kyr
    x = [v, k] where v = ice volume and k = state of the system (1 = glacial, 2 = deglaciation)
    Returns dv/dt and dk/dt.

    The transition from a glacial (k=1) to deglaciation regime (k = 2) occurs if three conditions are met:
        v > vc, dfdt > 0, and f > 0.
    The transition from deglaciation to glacial regime occurs if:
        f < glaciation threshold f1.
    The interglacial state formally belongs to the deglaciation regime.

    """

    v, k = x
    k = int(k)
    f = calc_f(t, A, eps)
    dfdt = calc_df(t, A, eps)

    if k == 1 and dfdt > 0 and f > 0 and v > vc:
        k = 2
    elif k == 2 and f < f1:
        k = 1

    if k == 1:
        ve = calc_ve(v, f, f1, f2)
        dvdt = (ve - v) / t1
    elif k == 2:
        dvdt = -vc / t2

    return [dvdt, 0]