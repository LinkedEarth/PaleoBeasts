"""Two-box methane/d13C model following Sapart et al. (2012) supplement.

This module implements a minimal two-hemisphere methane box model with explicit
tracking of 12CH4 and 13CH4. The formulation is based on the supplementary
information of Nature 490, 85-89 (2012), doi:10.1038/nature11461.

The model tracks methane in two perfectly mixed boxes representing the Northern
and Southern Hemispheres. Sources are grouped into four categories:
biogenic, pyrogenic, geological, and fossil. Sinks are grouped into OH
oxidation, soil removal, and stratospheric loss.

The public API includes:
- forward integration of NH/SH 12CH4 and 13CH4
- diagnostic conversion to NH/SH total CH4 and d13C
- helper methods for annual smoothing/interpolation
- total-source reconstruction from CH4 and d13C time series
- inversion for biogenic and pyrogenic emissions with geological and fossil
  emissions prescribed
- a lightweight Monte Carlo wrapper for uncertainty exploration
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.pbmodel import PBModel
from ..utils.func import smooth_and_interpolate

R_STD_VPDB = 0.0112372
SOURCE_CATEGORIES = ("biogenic", "pyrogenic", "geological", "fossil")
SINK_CATEGORIES = ("oh", "soil", "stratosphere")


def _as_array(values):
    return np.asarray(values, dtype=float)


@dataclass
class InversionResult:
    years: np.ndarray
    total_source_nh: np.ndarray
    total_source_sh: np.ndarray
    source_delta_nh: np.ndarray
    source_delta_sh: np.ndarray
    biogenic_nh: np.ndarray
    biogenic_sh: np.ndarray
    pyrogenic_nh: np.ndarray
    pyrogenic_sh: np.ndarray


class MethaneD13C(PBModel):
    """Two-box methane/d13C model.

    Parameters
    ----------
    forcing : optional
        Unused placeholder for consistency with the `PBModel` API.

    source_strengths : dict
        Total source strengths in Tg/yr keyed by category name. Each value may
        be a scalar, callable, or `pb.core.Forcing`. Defaults are paper-inspired
        and synthetic rather than observationally fitted.

    source_signatures : dict
        Source d13C signatures in permil vs VPDB keyed by category name.

    source_shares : dict
        NH source fraction keyed by category name. Values may be scalars,
        callables, or dicts with `nh`/`sh` entries.

    sink_inverse_lifetimes : dict
        Inverse lifetimes (1/yr) for the three sink categories.

    sink_kies : dict
        Kinetic isotope effects for the three sink categories. The 13CH4 sink
        coefficient is computed as k_12 / KIE.

    tau_ex : float or callable or Forcing
        Interhemispheric exchange time in years.

    Notes
    -----
    The model is continuous in time, but using `integrate(method='euler',
    kwargs={'dt': 1.0})` reproduces the annual forward-Euler logic of the
    supplement closely and is the intended mode for synthetic round-trip tests.
    """

    def __init__(
        self,
        forcing=None,
        var_name="methane_d13c",
        source_strengths=None,
        source_signatures=None,
        source_shares=None,
        sink_inverse_lifetimes=None,
        sink_kies=None,
        tau_ex=1.0,
        state_variables=None,
        diagnostic_variables=None,
        *args,
        **kwargs,
    ):
        if state_variables is None:
            state_variables = ["ch4_12_nh", "ch4_12_sh", "ch4_13_nh", "ch4_13_sh"]
        if diagnostic_variables is None:
            diagnostic_variables = [
                "ch4_total_nh",
                "ch4_total_sh",
                "delta13c_nh",
                "delta13c_sh",
                "source_total_nh",
                "source_total_sh",
                "source_delta_nh",
                "source_delta_sh",
                "source_biogenic_nh",
                "source_biogenic_sh",
                "source_pyrogenic_nh",
                "source_pyrogenic_sh",
                "source_geological_nh",
                "source_geological_sh",
                "source_fossil_nh",
                "source_fossil_sh",
            ]

        super().__init__(
            forcing,
            var_name,
            state_variables=state_variables,
            diagnostic_variables=diagnostic_variables,
            *args,
            **kwargs,
        )

        self.source_strengths = source_strengths or self.default_source_strengths()
        self.source_signatures = source_signatures or self.default_source_signatures()
        self.source_shares = source_shares or self.default_source_shares()
        self.sink_inverse_lifetimes = (
            sink_inverse_lifetimes or self.default_sink_inverse_lifetimes()
        )
        self.sink_kies = sink_kies or self.default_sink_kies()
        self.tau_ex = tau_ex

        self.param_values = {
            "tau_ex": tau_ex,
        }
        self.params = ()

    @staticmethod
    def default_source_strengths():
        return {
            "biogenic": 220.0,
            "pyrogenic": 28.0,
            "geological": 44.0,
            "fossil": 0.0,
        }

    @staticmethod
    def default_source_signatures():
        return {
            "biogenic": -60.5,
            "pyrogenic": -22.0,
            "geological": -38.0,
            "fossil": -38.0,
        }

    @staticmethod
    def default_source_shares():
        return {
            "biogenic": 0.80,
            "pyrogenic": 0.44,
            "geological": 0.90,
            "fossil": 0.90,
        }

    @staticmethod
    def default_sink_inverse_lifetimes():
        return {
            "oh": 0.133,
            "soil": 0.008,
            "stratosphere": 0.006,
        }

    @staticmethod
    def default_sink_kies():
        return {
            "oh": 1.0054,
            "soil": 1.0180,
            "stratosphere": 1.0120,
        }

    @staticmethod
    def delta_to_ratio(delta):
        return R_STD_VPDB * (np.asarray(delta, dtype=float) / 1000.0 + 1.0)

    @staticmethod
    def ratio_to_delta(ratio):
        ratio = np.asarray(ratio, dtype=float)
        return (ratio / R_STD_VPDB - 1.0) * 1000.0

    @classmethod
    def split_total_and_delta(cls, total_ch4, delta13c):
        ratio = cls.delta_to_ratio(delta13c)
        ch4_12 = np.asarray(total_ch4, dtype=float) / (1.0 + ratio)
        ch4_13 = ch4_12 * ratio
        return ch4_12, ch4_13

    @classmethod
    def combine_isotopologues(cls, ch4_12, ch4_13):
        ch4_12 = np.asarray(ch4_12, dtype=float)
        ch4_13 = np.asarray(ch4_13, dtype=float)
        total = ch4_12 + ch4_13
        ratio = np.divide(
            ch4_13,
            ch4_12,
            out=np.zeros_like(total, dtype=float),
            where=np.abs(ch4_12) > 0,
        )
        return total, cls.ratio_to_delta(ratio)

    @classmethod
    def synthetic_base_scenario(cls):
        return {
            "source_strengths": cls.default_source_strengths(),
            "source_signatures": cls.default_source_signatures(),
            "source_shares": cls.default_source_shares(),
            "sink_inverse_lifetimes": cls.default_sink_inverse_lifetimes(),
            "sink_kies": cls.default_sink_kies(),
            "tau_ex": 1.0,
        }

    def _resolve_spec(self, spec, t, state):
        return float(self.resolve_param(spec, t, state))

    def _resolve_source_strengths(self, t, state):
        return {
            category: self._resolve_spec(self.source_strengths[category], t, state)
            for category in SOURCE_CATEGORIES
        }

    def _resolve_source_signatures(self, t, state):
        return {
            category: self._resolve_spec(self.source_signatures[category], t, state)
            for category in SOURCE_CATEGORIES
        }

    def _resolve_source_shares(self, t, state):
        shares = {}
        for category in SOURCE_CATEGORIES:
            share_spec = self.source_shares[category]
            if isinstance(share_spec, dict):
                nh = self._resolve_spec(share_spec["nh"], t, state)
                sh = self._resolve_spec(share_spec["sh"], t, state)
                total = nh + sh
                if total <= 0:
                    raise ValueError(f"Non-positive hemispheric source share total for {category}.")
                shares[category] = {"nh": nh / total, "sh": sh / total}
            else:
                nh = self._resolve_spec(share_spec, t, state)
                shares[category] = {"nh": nh, "sh": 1.0 - nh}
        return shares

    def _resolve_sink_inverse_lifetimes(self, t, state):
        return {
            sink: self._resolve_spec(self.sink_inverse_lifetimes[sink], t, state)
            for sink in SINK_CATEGORIES
        }

    def _resolve_sink_kies(self, t, state):
        return {
            sink: self._resolve_spec(self.sink_kies[sink], t, state)
            for sink in SINK_CATEGORIES
        }

    def _resolve_tau_ex(self, t, state):
        return self._resolve_spec(self.tau_ex, t, state)

    def _source_terms(self, t, state):
        source_strengths = self._resolve_source_strengths(t, state)
        source_signatures = self._resolve_source_signatures(t, state)
        source_shares = self._resolve_source_shares(t, state)

        total_12 = {"nh": 0.0, "sh": 0.0}
        total_13 = {"nh": 0.0, "sh": 0.0}
        per_category = {}

        for category in SOURCE_CATEGORIES:
            total_source = source_strengths[category]
            ratio = self.delta_to_ratio(source_signatures[category])
            source_12_total = total_source / (1.0 + ratio)
            source_13_total = source_12_total * ratio

            nh_share = source_shares[category]["nh"]
            sh_share = source_shares[category]["sh"]

            cat = {
                "nh_12": source_12_total * nh_share,
                "sh_12": source_12_total * sh_share,
                "nh_13": source_13_total * nh_share,
                "sh_13": source_13_total * sh_share,
            }
            per_category[category] = cat
            total_12["nh"] += cat["nh_12"]
            total_12["sh"] += cat["sh_12"]
            total_13["nh"] += cat["nh_13"]
            total_13["sh"] += cat["sh_13"]

        return total_12, total_13, per_category

    def _state_diagnostics(self, t, state):
        ch4_12_nh, ch4_12_sh, ch4_13_nh, ch4_13_sh = _as_array(state)
        total_12, total_13, per_category = self._source_terms(t, state)
        total_nh, delta_nh = self.combine_isotopologues(ch4_12_nh, ch4_13_nh)
        total_sh, delta_sh = self.combine_isotopologues(ch4_12_sh, ch4_13_sh)

        source_total_nh, source_delta_nh = self.combine_isotopologues(
            total_12["nh"], total_13["nh"]
        )
        source_total_sh, source_delta_sh = self.combine_isotopologues(
            total_12["sh"], total_13["sh"]
        )

        diagnostics = {
            "ch4_total_nh": float(total_nh),
            "ch4_total_sh": float(total_sh),
            "delta13c_nh": float(delta_nh),
            "delta13c_sh": float(delta_sh),
            "source_total_nh": float(source_total_nh),
            "source_total_sh": float(source_total_sh),
            "source_delta_nh": float(source_delta_nh),
            "source_delta_sh": float(source_delta_sh),
        }
        for category in SOURCE_CATEGORIES:
            total_cat_nh = per_category[category]["nh_12"] + per_category[category]["nh_13"]
            total_cat_sh = per_category[category]["sh_12"] + per_category[category]["sh_13"]
            diagnostics[f"source_{category}_nh"] = float(total_cat_nh)
            diagnostics[f"source_{category}_sh"] = float(total_cat_sh)
        return diagnostics

    def dydt(self, t, x):
        ch4_12_nh, ch4_12_sh, ch4_13_nh, ch4_13_sh = _as_array(x)
        state = np.array([ch4_12_nh, ch4_12_sh, ch4_13_nh, ch4_13_sh], dtype=float)

        sources_12, sources_13, _per_category = self._source_terms(t, state)
        sink_12 = self._resolve_sink_inverse_lifetimes(t, state)
        kies = self._resolve_sink_kies(t, state)
        tau_ex = self._resolve_tau_ex(t, state)

        k12_total = sum(sink_12.values())
        k13_total = sum(sink_12[sink] / kies[sink] for sink in SINK_CATEGORIES)

        exchange_12 = 1.0 / (2.0 * tau_ex)
        exchange_13 = 1.0 / (2.0 * tau_ex)

        d_ch4_12_nh = (
            sources_12["nh"]
            - k12_total * ch4_12_nh
            + exchange_12 * (ch4_12_sh - ch4_12_nh)
        )
        d_ch4_12_sh = (
            sources_12["sh"]
            - k12_total * ch4_12_sh
            + exchange_12 * (ch4_12_nh - ch4_12_sh)
        )
        d_ch4_13_nh = (
            sources_13["nh"]
            - k13_total * ch4_13_nh
            + exchange_13 * (ch4_13_sh - ch4_13_nh)
        )
        d_ch4_13_sh = (
            sources_13["sh"]
            - k13_total * ch4_13_sh
            + exchange_13 * (ch4_13_nh - ch4_13_sh)
        )

        return [d_ch4_12_nh, d_ch4_12_sh, d_ch4_13_nh, d_ch4_13_sh]

    def uses_post_history(self):
        return True

    def populate_diagnostics_from_history(self, time, history):
        time = _as_array(time)
        history = _as_array(history)
        diagnostics = {name: [] for name in self.diagnostic_variables}
        for t, row in zip(time, history):
            diag = self._state_diagnostics(t, row)
            for name in diagnostics:
                diagnostics[name].append(diag[name])

        self.diagnostic_variables = {name: np.asarray(values) for name, values in diagnostics.items()}

    def _series_from_spec(self, spec, years):
        years = _as_array(years)
        reference_state = np.zeros(len(self.state_variables_names), dtype=float)
        if np.isscalar(spec):
            return np.full_like(years, float(spec), dtype=float)
        if hasattr(spec, "get_forcing"):
            return np.asarray([spec.get_forcing(t) for t in years], dtype=float)
        if callable(spec):
            return np.asarray([self.resolve_param(spec, t, reference_state) for t in years], dtype=float)
        arr = _as_array(spec)
        if arr.shape == years.shape:
            return arr
        raise ValueError("Unsupported source specification for inversion helper.")

    def _share_series(self, share_spec, years):
        years = _as_array(years)
        if isinstance(share_spec, dict):
            nh = self._series_from_spec(share_spec["nh"], years)
            sh = self._series_from_spec(share_spec["sh"], years)
            total = nh + sh
            if np.any(total <= 0):
                raise ValueError("Non-positive hemispheric source share total encountered.")
            return {"nh": nh / total, "sh": sh / total}

        nh = self._series_from_spec(share_spec, years)
        return {"nh": nh, "sh": 1.0 - nh}

    def _series_by_category(self, mapping, years):
        return {category: self._series_from_spec(mapping[category], years) for category in SOURCE_CATEGORIES}

    def _share_series_by_category(self, years):
        return {
            category: self._share_series(self.source_shares[category], years)
            for category in SOURCE_CATEGORIES
        }

    def reconstruct_total_sources(self, years, ch4_total_nh, ch4_total_sh, delta13c_nh, delta13c_sh):
        years = _as_array(years)
        ch4_total_nh = _as_array(ch4_total_nh)
        ch4_total_sh = _as_array(ch4_total_sh)
        delta13c_nh = _as_array(delta13c_nh)
        delta13c_sh = _as_array(delta13c_sh)

        ch4_12_nh, ch4_13_nh = self.split_total_and_delta(ch4_total_nh, delta13c_nh)
        ch4_12_sh, ch4_13_sh = self.split_total_and_delta(ch4_total_sh, delta13c_sh)

        def forward_difference(values, years_in):
            dt = np.diff(years_in)
            diff = np.diff(values) / dt
            return np.concatenate([diff, diff[-1:]])

        d12_nh = forward_difference(ch4_12_nh, years)
        d12_sh = forward_difference(ch4_12_sh, years)
        d13_nh = forward_difference(ch4_13_nh, years)
        d13_sh = forward_difference(ch4_13_sh, years)

        sink_12 = {sink: self._series_from_spec(self.sink_inverse_lifetimes[sink], years) for sink in SINK_CATEGORIES}
        sink_kies = {sink: self._series_from_spec(self.sink_kies[sink], years) for sink in SINK_CATEGORIES}
        k12 = sum(sink_12.values())
        k13 = sum(sink_12[sink] / sink_kies[sink] for sink in SINK_CATEGORIES)
        tau_ex = self._series_from_spec(self.tau_ex, years)

        exchange = 1.0 / (2.0 * tau_ex)

        source12_nh = d12_nh + k12 * ch4_12_nh + exchange * (ch4_12_nh - ch4_12_sh)
        source12_sh = d12_sh + k12 * ch4_12_sh + exchange * (ch4_12_sh - ch4_12_nh)
        source13_nh = d13_nh + k13 * ch4_13_nh + exchange * (ch4_13_nh - ch4_13_sh)
        source13_sh = d13_sh + k13 * ch4_13_sh + exchange * (ch4_13_sh - ch4_13_nh)

        total_source_nh, source_delta_nh = self.combine_isotopologues(source12_nh, source13_nh)
        total_source_sh, source_delta_sh = self.combine_isotopologues(source12_sh, source13_sh)
        return {
            "years": years,
            "total_source_nh": total_source_nh,
            "total_source_sh": total_source_sh,
            "source_delta_nh": source_delta_nh,
            "source_delta_sh": source_delta_sh,
        }

    def prescribed_hemispheric_sources(self, years):
        years = _as_array(years)
        source_signatures = self._series_by_category(self.source_signatures, years)
        source_shares = self._share_series_by_category(years)
        result = {}
        for category in ("geological", "fossil"):
            total_series = self._series_from_spec(self.source_strengths[category], years)
            nh = total_series * source_shares[category]["nh"]
            sh = total_series * source_shares[category]["sh"]
            result[category] = {
                "nh": nh,
                "sh": sh,
                "delta": source_signatures[category],
            }
        return result

    def invert_biogenic_pyrogenic(self, years, ch4_total_nh, ch4_total_sh, delta13c_nh, delta13c_sh):
        reconstructed = self.reconstruct_total_sources(
            years, ch4_total_nh, ch4_total_sh, delta13c_nh, delta13c_sh
        )
        years = reconstructed["years"]
        prescribed = self.prescribed_hemispheric_sources(years)
        source_signatures = self._series_by_category(self.source_signatures, years)

        def solve_one_hemisphere(total_source, source_delta, hemisphere):
            geological = prescribed["geological"][hemisphere]
            fossil = prescribed["fossil"][hemisphere]
            unknown_total = total_source - geological - fossil
            rhs = (
                source_delta * total_source
                - geological * source_signatures["geological"]
                - fossil * source_signatures["fossil"]
            )
            pyrogenic = (
                rhs - unknown_total * source_signatures["biogenic"]
            ) / (source_signatures["pyrogenic"] - source_signatures["biogenic"])
            biogenic = unknown_total - pyrogenic
            return biogenic, pyrogenic

        biogenic_nh, pyrogenic_nh = solve_one_hemisphere(
            reconstructed["total_source_nh"], reconstructed["source_delta_nh"], "nh"
        )
        biogenic_sh, pyrogenic_sh = solve_one_hemisphere(
            reconstructed["total_source_sh"], reconstructed["source_delta_sh"], "sh"
        )

        return InversionResult(
            years=years,
            total_source_nh=reconstructed["total_source_nh"],
            total_source_sh=reconstructed["total_source_sh"],
            source_delta_nh=reconstructed["source_delta_nh"],
            source_delta_sh=reconstructed["source_delta_sh"],
            biogenic_nh=biogenic_nh,
            biogenic_sh=biogenic_sh,
            pyrogenic_nh=pyrogenic_nh,
            pyrogenic_sh=pyrogenic_sh,
        )

    def monte_carlo_inversion(
        self,
        years,
        ch4_total_nh,
        ch4_total_sh,
        delta13c_nh,
        delta13c_sh,
        n_samples=100,
        random_seed=None,
        ch4_sigma=2.0,
        delta_sigma=0.05,
    ):
        years = _as_array(years)
        rng = np.random.default_rng(random_seed)
        samples = {
            "biogenic_nh": [],
            "biogenic_sh": [],
            "pyrogenic_nh": [],
            "pyrogenic_sh": [],
        }

        for _ in range(int(n_samples)):
            ch4_nh_sample = _as_array(ch4_total_nh) + rng.normal(0.0, ch4_sigma, size=len(years))
            ch4_sh_sample = _as_array(ch4_total_sh) + rng.normal(0.0, ch4_sigma, size=len(years))
            delta_nh_sample = _as_array(delta13c_nh) + rng.normal(0.0, delta_sigma, size=len(years))
            delta_sh_sample = _as_array(delta13c_sh) + rng.normal(0.0, delta_sigma, size=len(years))

            inversion = self.invert_biogenic_pyrogenic(
                years, ch4_nh_sample, ch4_sh_sample, delta_nh_sample, delta_sh_sample
            )
            samples["biogenic_nh"].append(inversion.biogenic_nh)
            samples["biogenic_sh"].append(inversion.biogenic_sh)
            samples["pyrogenic_nh"].append(inversion.pyrogenic_nh)
            samples["pyrogenic_sh"].append(inversion.pyrogenic_sh)

        return {
            "years": years,
            **{key: np.asarray(value) for key, value in samples.items()},
        }
