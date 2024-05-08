import numpy as np
from ..utils import constants as phys
from ..core.pbmodel import PBModel


class EBM(PBModel):
    def __init__(self, forcing, state_variables= ['T'],
                 diagnostic_variables = ['albedo', 'absorbed_SW', 'OLR', 'solar_incoming'],
                 var_name='temperature', OLR=None, C=4, merid_diff=0, albedo=.3):
        super().__init__(forcing, var_name, state_variables=state_variables,
                         diagnostic_variables=diagnostic_variables )
        self.forcing = forcing #insolation
        self.variable_name = var_name
        self.C = C
        self.params = None  # (f1, f2, t1, t2, vc)
        self.albedo = albedo
        self.OLR = OLR if OLR is not None else OLR_func
        self.merid_diff = merid_diff
        self.phi = np.linspace(-np.pi / 2, np.pi / 2, 100)

    def dydt(self, t, x):
        T =x
        f_solar_incoming = self.forcing.get_forcing(t)
        albedo = self.calc_albedo(T)
        absorbed_SW = (1 - albedo) * f_solar_incoming
        OLR = self.OLR(T)
        self.diagnostic_variables['time'].append(t)
        self.diagnostic_variables['albedo'].append(albedo)
        self.diagnostic_variables['absorbed_SW'].append(absorbed_SW)
        self.diagnostic_variables['OLR'].append(OLR)
        self.diagnostic_variables['solar_incoming'].append(f_solar_incoming)
         # self.diagnostic_variables.append([albedo, absorbed_SW, OLR])
        dTdt = 1 / self.C * (absorbed_SW - OLR + self.calc_merid_diff(T))
        return [dTdt]

    def calc_OLR(self, T):
        return self.OLR(T)

    def calc_albedo(self, T):
        if callable(self.albedo):
            return self.albedo(self, T)  # Assuming vc is a function of time and the state vector x
        else:
            return self.albedo  # vc is a constant

    def calc_merid_diff(self, T):
        if callable(self.merid_diff):
            return self.merid_diff(self, T)
        else:
            return self.merid_diff

    def calc_C(self, T):

        if callable(self.C):
            return self.C(self, T)
        else:
            return self.C



def albedo_func(ebm_model, Ts, alpha_ice=.6, alpha_0=.3, T1=260., T2=290.):
    if isinstance(Ts, np.ndarray):
        Ts = Ts[-1]
    if Ts < T1:
        a0 = alpha_ice
    elif (Ts >= T1) & (Ts <= T2):
        r = (Ts - T2) ** 2 / (T2 - T1) ** 2
        a0 = alpha_0 + (alpha_ice - alpha_0) * r
    else:
        a0 = alpha_0

    return a0


def albedo_func1D(ebm_model, Ts, a2=.25, alpha_ice=.6, alpha_0=.1, T1=260., T2=290.):
    phi = ebm_model.phi
    a0 = albedo_func(Ts, alpha_ice, alpha_0, T1, T2)
    P2 = .5 * (3 * np.sin(phi) ** 2 - 1)
    return a0 + a2 * P2


def OLR_func(Ts, pRad=650, ps=1000):
    if isinstance(Ts, np.ndarray):
        Ts = Ts[-1]
        # print(Ts)
    Te = (pRad / ps) ** (2. / 7.) * Ts  # emission temperature assuming dry adiabat
    # print('Te', Te)
    return phys.sigma * (Te ** 4.)


def incoming_SW_func(t, S_0=1360.8):
    return np.ones(np.array(t).shape)*S_0 / 4


# def advection_diffusion(ebm_model, T, D):
#     phi = ebm_model.phi
#     return D / np.cos(phi) * (np.gradient(phi, T) * -np.sin(phi) + np.gradient(T, phi) * np.cos(phi))

def calc_f(t, S_0=1360.8, T1=11):
    return S_0/4+ np.sin(t*(2 * np.pi)/T1)
