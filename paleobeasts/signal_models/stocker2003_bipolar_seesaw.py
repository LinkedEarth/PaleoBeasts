"""Stocker & Johnsen (2003) thermal bipolar seesaw model.

This module implements a minimum thermodynamic model in which a southern
temperature anomaly responds to a prescribed northern anomaly with a single
timescale:

    dTs/dt = (beta * Tn(t) - Ts) / tau
"""

from __future__ import annotations

import numpy as np

from ..core.pbmodel import PBModel


class Stocker2003BipolarSeesaw(PBModel):
    """Minimum thermodynamic model for the thermal bipolar seesaw."""

    def __init__(
        self,
        forcing=None,
        var_name="stocker2003_bipolar_seesaw",
        tau=1000.0,
        beta=-1.0,
        Tn=0.0,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["Ts"]
        if diagnostic_variables is None:
            diagnostic_variables = ["Tn"]

        super().__init__(
            forcing,
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.tau = tau
        self.beta = beta
        self.Tn = Tn
        self.param_values = {
            "tau": tau,
            "beta": beta,
            "Tn": Tn,
        }
        self.params = ()

    def uses_post_history(self):
        return True

    def dydt(self, t, x):
        Ts = float(np.asarray(x, dtype=float)[0])
        tau = float(self.get_param("tau", t, x))
        if tau <= 0:
            raise ValueError("tau must be > 0.")
        beta = float(self.get_param("beta", t, x))
        if self.forcing is not None:
            Tn_t = float(self.forcing.get_forcing(self.time_util(t)))
        else:
            Tn_t = float(self.get_param("Tn", t, x))
        dTsdt = (beta * Tn_t - Ts) / tau
        return [dTsdt]

    def populate_diagnostics_from_history(self, time, history):
        time = np.asarray(time, dtype=float)
        history = np.asarray(history, dtype=float)
        Tn_vals = []
        for t, row in zip(time, history):
            if self.forcing is not None:
                Tn_vals.append(float(self.forcing.get_forcing(self.time_util(t))))
            else:
                Tn_vals.append(float(self.get_param("Tn", t, row)))
        Tn_vals = np.asarray(Tn_vals, dtype=float)
        self.diagnostic_variables = {"Tn": Tn_vals}
