import numpy as np
from scipy.interpolate import CubicSpline


class Model3:
    """
    Model 3 from Ganopolski (2024) which describes the evolution of ice volume as a function of orbital forcing.

    Parameters
    ----------
    Parameter defaults taken From Ganopolski 2024:

    forcing : object
        Object that provides the orbital forcing at time t and its derivative.

    var_name : str
        Name of the variable being modeled.
        Default is 'ice volume'

    vc : numeric or function
        value for critical ice volume; controls the dominant periodicity and degree of asymmetry of glacial cycles
        default is 1.4

    f1 : numeric (float, int)
        insolation threshold for glacial inception (pinned at -20 to -15 W/m^2)
        default is -16

    t1 : numeric (float, int)
        relaxation timescale for glacial inception (in kyr)
        default is 30

    f2 : numeric (float, int)
        insolation threshold for deglaciation inception (tunable; positive)
        default is 16

    t2 : numeric (float, int)
        relaxation timescale for deglaciation (in kyr)
        default is 10

    k_arr : list
        List to store the state of the system at each time step.

    t_arr : list
        List to store the time at each time step.

    params : tuple
        Tuple of parameters (f1, f2, t1, t2, vc) for the model.

    """

    def __init__(self, forcing, var_name='ice volume', f1=-16, f2=16, t1=30, t2=10, vc=1.4):
        self.forcing = forcing
        self.f1 = f1
        self.f2 = f2
        self.t1 = t1
        self.t2 = t2
        self.vc = vc
        self.variable_name = var_name
        self.state_variables = []
        self.params = (f1, f2, t1, t2, vc)
        self.dfdt = None

    def dydt(self, t, x, f1, f2, t1, t2, vc):
        """
        Differential equation for dv/dt (where v=ice volume) at state k.

        The transition from a glacial (k=1) to deglaciation regime (k = 2) occurs if:
            - v > vc, dfdt > 0, and f > 0.
        The transition from deglaciation (k=2) to glacial regime (k=1) occurs if:
            - f < glaciation threshold f1.
        The interglacial state formally belongs to the deglaciation regime.

        Inputs:
            t : time in kyr
            x : [v] where v = ice volume
            f1 : insolation threshold for glacial inception (pinned at -20 to -15 W/m^2)
            f2 : insolation threshold for deglaciation inception (tunable; positive)
            t1 : relaxation time scale for glacial inception (in kyr)
            t2 : relaxation time scale for deglaciation (in kyr)

        Returns:
            dv/dt and dk/dt.

        """

        v = x
        k = int(self.state_variables[-1][0])
        # k = int(self.k_arr[-1])
        f = self.forcing.get_forcing(t)
        # dfdt = self.forcing.get_derivative(t)
        dfdt = self.calc_dfdt(t)


        vc = self.calc_vc(t)

        k = self.calc_k(k, dfdt, f, v, vc)

        # if k == 1 and dfdt > 0 and f > 0 and v > vc:
        #     k = 2
        # elif k == 2 and f < f1:  # self.f1:
        #     k = 1

        if k == 1:
            ve = self.calc_ve(v, f)
            dvdt = (ve - v) / t1  # self.t1
        elif k == 2:
            dvdt = -vc / t2  # self.t2

        self.state_variables.append([k])
        # self.k_arr.append(k)
        # self.t_arr.append(t)

        return [dvdt]

    def calc_k(self, k, dfdt, f, v, vc):
        if k == 1 and dfdt > 0 and f > 0 and v > vc:
            k = 2
        elif k == 2 and f < self.f1:
            k = 1
        return k

    def calc_ve(self, v, f, vi=0):
        """
        Calculate equilibrium state towards which the system is attracted is a function of orbital forcing and,
        for the bi-stable regime, also depends on ice volume

        Parameters
        ----------
        v : float
            Ice volume

        f : float
            Orbital forcing

        Returns
        -------
        ve : float
            ice volume of the equilibrium state to which the system is attracted

        """
        vg = self.calc_vg(f)
        vu = self.calc_vu(f)

        if f < self.f1:
            return vg
        elif f > self.f2:
            return vi
        elif self.f1 < f < self.f2 and v > vu:
            return vg
        elif self.f1 < f < self.f2 and v < vu:
            return vi

    def calc_vg(self, f):
        """
        Calculate glacial equilibrium state.

        Parameters
        ----------
        f : float
            Orbital forcing

        Returns
        -------
        vg : float
            ice volume of the glacial equilibrium state

        """
        return 1 + np.sqrt((self.f2 - f) / (self.f2 - self.f1))

    def calc_vu(self, f):
        """
        Calculate unstable equilibrium ice volume which separates the glacial and interglacial attraction domains.

        Parameters
        ----------
        f : float
            Orbital forcing

        Returns
        -------
        vu : float
            ice volume of the unstable equilibrium state

        """
        return 1 - np.sqrt((self.f2 - f) / (self.f2 - self.f1))

    def calc_vc(self, t):
        """
        Evaluate vc which can be a function of time and state or a constant.

        Parameters
        ----------
        t : float
            Time

        Returns
        -------
        vc : float
            Critical ice volume

        """

        if callable(self.vc):
            return self.vc(t)  # Assuming vc is a function of time and the state vector x
        else:
            return self.vc  # vc is a constant

    # def set_dfdt(self, time_range=None):
    #     """
    #     Set the derivative of the orbital forcing for the model.
    #     """
    #     if isinstance(self.forcing.data, np.ndarray):
    #         data = self.forcing.data
    #         time = self.forcing.time
    #     elif callable(self.forcing.data):
    #         data = self.forcing.get_forcing(time_range)
    #         time= time_range
    #
    #     cs = CubicSpline(time, data)
    #     d1 = cs.derivative(nu=1)
    #     self.dfdt = d1

    def calc_dfdt(self, t):
        """
        Calculate the derivative of the orbital forcing at time t.
        """
        if callable(self.dfdt):
            return self.dfdt(t)  # Assuming vc is a function of time and the state vector x
        else:
            return self.dfdt  # vc is a constant

    # def calc_df(self, t):
    #     """
    #     Calculate the derivative of the orbital forcing at time t.
    #
    #     Parameters
    #     ----------
    #     t : float
    #         Time.
    #
    #     Returns
    #     -------
    #     df : float
    #         The value of the derivative of the orbital forcing at time t.
    #
    #     """
    #
    #     if self.dfdt is not None:
    #         return self.derivative(t)
    #     elif isinstance(self.data, np.ndarray):
    #         # Calculate numerical derivative if `derivative` is not provided and `data` is an array
    #         if not hasattr(self, 'numeric_derivative'):
    #             if self.time is not None:
    #                 self.numeric_derivative = np.gradient(self.data, self.time)
    #             else:
    #                 self.numeric_derivative = np.gradient(self.data)  # Assumes uniform spacing of 1
    #         idx = int(t)
    #         return self.numeric_derivative[idx]
    #     else:
    #         raise ValueError("No method for derivative calculation provided.")


def calc_df(t, A=25, eps=0.5, T1=100, T2=30):
    """
    Calculate the derivative of the orbital forcing at time t.

    Parameters
    ----------
    t : float
        Time.

    A : float
        Magnitude of forcing in Wm−2
        Default is 25 w/m^2

    eps : float
        Nondimensional magnitude of eccentricity modulation. eps = 0 corresponds to no eccentricty modulation.
        Default is 0.5

    T1 : float
        Eccentricity timescale
        Default is 100 kyr

    T2 : float
        Precession timescale
        Default is 30 kyr

    Returns
    -------
    df : float
        The value of the derivative of the orbital forcing at time t.

    """
    return A * eps * ((2 * np.pi / T1) * np.cos(2 * np.pi * t / T1) * np.cos(2 * np.pi * t / T2) -
                      (2 * np.pi / T2) * np.sin(2 * np.pi * t / T2) * np.sin(2 * np.pi * t / T1))



def calc_f(t, A=25, eps=0.5, T1=100, T2=30):
    """
    Calculate the orbital forcing value at time t.

    Parameters
    ----------
    t : float
        Time.

    A : float
        Magnitude of forcing in Wm−2
        Default is 25 w/m^2

    eps : float
        Nondimensional magnitude of eccentricity modulation. eps = 0 corresponds to no eccentricty modulation.
        Default is 0.5

    T1 : float
        Eccentricity timescale
        Default is 100 kyr

    T2 : float
        Precession timescale
        Default is 30 kyr


    Returns
    -------
    f : float
        The value of the orbital forcing at time t.

    """

    return A * (1 + eps * np.sin(2 * np.pi * t / T1)) * np.cos(2 * np.pi * t / T2)




def vc_func(t, vc1=.65, vc2=1.38, t1_mpt=-1050, tau1=250):
    """
    Calculate the value for critical ice volume as in Ganopolski (2024) for MPT
    """
    return 0.5 * (vc1 + vc2) + 0.5 * (vc2 - vc1) * np.tanh((t - t1_mpt) / tau1)
