"""Melcher et al. (2025) conceptual DO-event model.

This model implements a stochastic, two-equation slow-fast system:

d(delta_b) = [-B - |q| * (delta_b - b0)] dt + sigma dW1
dB        = [(delta_b + alpha * B - gamma) / tau] dt + sigma dW2

with q = q0 + q1 * (delta_b - b0).
"""

from __future__ import annotations

import numpy as np

from ..core.pbmodel import PBModel


class Melcher2025DOModel(PBModel):
    """Minimal stochastic DO-event model (Melcher et al., 2025-style).

    Defaults for ``q0``, ``q1``, ``b0``, and ``tau`` match the figure-code
    baseline in ``reference_papers/DO_events/2023_paper_Melcher_Halkjaer-main``.
    """

    def __init__(
        self,
        forcing=None,
        var_name="melcher2025_do",
        q0=-9.0,
        q1=12.0,
        b0=0.625,
        tau=0.902,
        alpha=-0.6,
        gamma=1.2,
        sigma=0.2,
        psi0=-4.5e6,
        psi1=20.0e6,
        psi_a=5.0e6,
        chi_a=2.5,
        b_c=0.004,
        B_c=3.8e-10,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["delta_b", "B"]
        if diagnostic_variables is None:
            diagnostic_variables = ["q", "amoc_dim", "aabw_dim"]

        super().__init__(
            forcing,
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.q0 = q0
        self.q1 = q1
        self.b0 = b0
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.psi0 = psi0
        self.psi1 = psi1
        self.psi_a = psi_a
        self.chi_a = chi_a
        self.b_c = b_c
        self.B_c = B_c

        self.param_values = {
            "q0": q0,
            "q1": q1,
            "b0": b0,
            "tau": tau,
            "alpha": alpha,
            "gamma": gamma,
            "sigma": sigma,
            "psi0": psi0,
            "psi1": psi1,
            "psi_a": psi_a,
            "chi_a": chi_a,
            "b_c": b_c,
            "B_c": B_c,
        }
        self.params = ()

    def uses_post_history(self):
        return True

    def transport(self, t, x):
        delta_b = np.asarray(x, dtype=float)[0]
        q0 = self.get_param("q0", t, x)
        q1 = self.get_param("q1", t, x)
        b0 = self.get_param("b0", t, x)
        return q0 + q1 * (delta_b - b0)

    def dydt(self, t, x):
        delta_b, B = np.asarray(x, dtype=float)
        q = self.transport(t, x)

        b0 = self.get_param("b0", t, x)
        tau = self.get_param("tau", t, x)
        alpha = self.get_param("alpha", t, x)
        gamma = self.get_param("gamma", t, x)

        d_delta_b = -B - np.abs(q) * (delta_b - b0)
        d_B = (delta_b + alpha * B - gamma) / tau
        return [d_delta_b, d_B]

    def sde_noise(self, t, x):
        sigma = self.get_param("sigma", t, x)
        if np.isscalar(sigma):
            return np.array([float(sigma), float(sigma)], dtype=float)

        arr = np.asarray(sigma, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.array([float(arr[0]), float(arr[0])], dtype=float)
        if arr.size != 2:
            raise ValueError("sigma must be scalar or length-2 for Melcher2025DOModel.")
        return arr

    def _redimensionalized_diagnostics(self, t, x):
        delta_b, B = np.asarray(x, dtype=float)
        psi0 = self.get_param("psi0", t, x)
        psi1 = self.get_param("psi1", t, x)
        psi_a = self.get_param("psi_a", t, x)
        chi_a = self.get_param("chi_a", t, x)
        b_c = self.get_param("b_c", t, x)
        B_c = self.get_param("B_c", t, x)

        amoc_dim = psi0 + psi1 * delta_b
        aabw_dim = psi_a + chi_a * (b_c / B_c) * B
        return float(amoc_dim), float(aabw_dim)

    def populate_diagnostics_from_history(self, time, history):
        time = np.asarray(time, dtype=float)
        history = np.asarray(history, dtype=float)
        diagnostics = {name: [] for name in self.diagnostic_variables}

        for t, row in zip(time, history):
            q = self.transport(t, row)
            amoc_dim, aabw_dim = self._redimensionalized_diagnostics(t, row)
            diagnostics["q"].append(float(q))
            diagnostics["amoc_dim"].append(amoc_dim)
            diagnostics["aabw_dim"].append(aabw_dim)

        self.diagnostic_variables = {k: np.asarray(v) for k, v in diagnostics.items()}
