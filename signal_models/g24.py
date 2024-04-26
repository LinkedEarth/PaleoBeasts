import numpy as np


class Model3:
    """
    Differential equation for dv/dt (where v=ice volume) at state k.

    From Ganopolski 2024:
    vc = value for critical ice volume; controls the dominant periodicity and degree of asymmetry of glacial cycles
    f1 = insolation threshold for glacial inception (pinned at -20 to -15 W/m^2)
    t1 = relaxation timescale for glacial inception (in kyr)
    f2 = insolation threshold for deglaciation inception (tunable; positive)
    t2 = relaxation timescale for deglaciation (in kyr)

    A = magnitude of forcing in Wm−2
    eps = nondimensional magnitude of amplitude modulation

    t = time in kyr
    x = [v, k] where v = ice volume and k = state of the system (1 = glacial, 2 = deglaciation)
    Returns dv/dt and dk/dt.

    The transition from a glacial (k=1) to deglaciation regime (k = 2) occurs if:
        v > vc, dfdt > 0, and f > 0.
    The transition from deglaciation (k=2) to glacial regime (k=1) occurs if:
        f < glaciation threshold f1.
    The interglacial state formally belongs to the deglaciation regime.
    """

    def __init__(self, forcing, var_name, f1=-16, f2=16, t1=30, t2=100, vc=1.4):
        self.forcing = forcing
        self.f1 = f1
        self.f2 = f2
        self.t1 = t1
        self.t2 = t2
        self.vc = vc
        self.variable_name = var_name
        self.k = 1
        self.k_arr = []
        self.t_arr = []
        self.params = (f1, f2, t1, t2, vc)

    def dVdt(self, t, x, f1, f2, t1, t2, vc):

        v, k = x
        k = int(self.k_arr[-1])
        f = self.forcing.get_forcing(t)
        dfdt = self.forcing.get_derivative(t)

        vc = self.calc_vc(t)

        if k == 1 and dfdt > 0 and f > 0 and v > vc:
            k = 2
        elif k == 2 and f < f1:  # self.f1:
            k = 1

        if k == 1:
            ve = self.calc_ve(v, f)
            dvdt = (ve - v) / t1  # self.t1
        elif k == 2:
            dvdt = -vc / t2  # self.t2


        self.k_arr.append(k)
        self.t_arr.append(t)


        return [dvdt, 0]

    def calc_ve(self, v, f):
        """
        Calculate equilibrium state towards which the system is attracted is a function of orbital forcing and,
        for the bi-stable regime, also depends on ice volume
        """
        vg = self.calc_vg(f)
        vu = self.calc_vu(f)

        if f < self.f1:
            return vg
        elif f > self.f2:
            return 0
        elif self.f1 < f < self.f2 and v > vu:
            return vg
        elif self.f1 < f < self.f2 and v < vu:
            return 0

    def calc_vg(self, f):
        """
        Calculate glacial equilibrium state.
        """
        return 1 + np.sqrt((self.f2 - f) / (self.f2 - self.f1))

    def calc_vu(self, f):
        """
        Calculate unstable equilibrium which separates the glacial and interglacial attraction domains.
        """
        return 1 - np.sqrt((self.f2 - f) / (self.f2 - self.f1))

    def calc_vc(self, t):
        """Evaluate vc which can be a function of time and state or a constant."""
        if callable(self.vc):
            return self.vc(t)  # Assuming vc is a function of time and the state vector x
        else:
            return self.vc  # vc is a constant


def calc_f(t, A=25, eps=0.5, T1=100, T2=30):
    """
    Calculate the orbital forcing value at time t.
    """
    return A * (1 + eps * np.sin(2 * np.pi * t / T1)) * np.cos(2 * np.pi * t / T2)


def calc_df(t, A=25, eps=0.5, T1=100, T2=30):
    """
    Calculate the derivative of the orbital forcing at time t.
    """
    return A * eps * ((2 * np.pi / T1) * np.cos(2 * np.pi * t / T1) * np.cos(2 * np.pi * t / T2) -
                      (2 * np.pi / T2) * np.sin(2 * np.pi * t / T2) * np.sin(2 * np.pi * t / T1))


def vc_func(t, vc1=.65, vc2=1.38, t1_mpt=-1050, tau1=250):
    """
    Calculate the value for critical ice volume as in Ganopolski (2024) for MPT
    """
    return 0.5 * (vc1 + vc2) + 0.5 * (vc2 - vc1) * np.tanh((t - t1_mpt) / tau1)
