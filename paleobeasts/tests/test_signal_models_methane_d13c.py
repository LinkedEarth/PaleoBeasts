"""Tests for paleobeasts.signal_models.methane_d13c."""

import numpy as np
import paleobeasts as pb

from paleobeasts.signal_models import methane_d13c


def _default_y0(total=700.0, delta=-47.0):
    ch4_12, ch4_13 = methane_d13c.MethaneD13C.split_total_and_delta(total, delta)
    return [ch4_12, ch4_12, ch4_13, ch4_13]


class TestSignalModelsMethaneD13CIntegrate:
    def test_integrate_t0(self):
        model = methane_d13c.MethaneD13C()
        model.integrate(t_span=(0, 50), y0=_default_y0(), method="euler", kwargs={"dt": 1.0})

        assert model.state_variables.dtype.names == (
            "ch4_12_nh",
            "ch4_12_sh",
            "ch4_13_nh",
            "ch4_13_sh",
        )
        assert np.all(np.isfinite(model.diagnostic_variables["ch4_total_nh"]))
        assert np.all(np.isfinite(model.diagnostic_variables["delta13c_nh"]))
        assert model.uses_post_history() is True

    def test_ratio_conversion_t0(self):
        delta = -47.0
        ratio = methane_d13c.MethaneD13C.delta_to_ratio(delta)
        delta_back = methane_d13c.MethaneD13C.ratio_to_delta(ratio)
        assert np.isclose(delta, delta_back)

    def test_time_varying_param_matches_constant_t0(self):
        y0 = _default_y0()
        model_const = methane_d13c.MethaneD13C(tau_ex=1.0)
        model_tv = methane_d13c.MethaneD13C(tau_ex=lambda t: 1.0)

        model_const.integrate(t_span=(0, 20), y0=y0, method="euler", kwargs={"dt": 1.0})
        model_tv.integrate(t_span=(0, 20), y0=y0, method="euler", kwargs={"dt": 1.0})

        assert np.allclose(
            model_const.diagnostic_variables["ch4_total_nh"],
            model_tv.diagnostic_variables["ch4_total_nh"],
            rtol=1e-10,
            atol=1e-10,
        )


class TestSignalModelsMethaneD13CSyntheticCases:
    def test_steady_state_restart_t0(self):
        model = methane_d13c.MethaneD13C()
        model.integrate(t_span=(0, 400), y0=_default_y0(), method="euler", kwargs={"dt": 1.0})

        final = [
            model.state_variables["ch4_12_nh"][-1],
            model.state_variables["ch4_12_sh"][-1],
            model.state_variables["ch4_13_nh"][-1],
            model.state_variables["ch4_13_sh"][-1],
        ]
        restart = methane_d13c.MethaneD13C()
        restart.integrate(t_span=(0, 20), y0=final, method="euler", kwargs={"dt": 1.0})

        for name in restart.state_variables.dtype.names:
            assert np.max(np.abs(restart.state_variables[name] - restart.state_variables[name][0])) < 1e-2

    def test_symmetric_hemispheres_t0(self):
        equal_shares = {category: 0.5 for category in methane_d13c.SOURCE_CATEGORIES}
        model = methane_d13c.MethaneD13C(source_shares=equal_shares)
        model.integrate(t_span=(0, 100), y0=_default_y0(), method="euler", kwargs={"dt": 1.0})

        assert np.allclose(model.state_variables["ch4_12_nh"], model.state_variables["ch4_12_sh"])
        assert np.allclose(model.state_variables["ch4_13_nh"], model.state_variables["ch4_13_sh"])
        assert np.allclose(model.diagnostic_variables["ch4_total_nh"], model.diagnostic_variables["ch4_total_sh"])
        assert np.allclose(model.diagnostic_variables["delta13c_nh"], model.diagnostic_variables["delta13c_sh"])

    def test_single_source_ordering_t0(self):
        results = {}
        for category in methane_d13c.SOURCE_CATEGORIES:
            strengths = {key: 0.0 for key in methane_d13c.SOURCE_CATEGORIES}
            strengths[category] = 100.0
            model = methane_d13c.MethaneD13C(source_strengths=strengths)
            model.integrate(t_span=(0, 300), y0=_default_y0(total=200.0), method="euler", kwargs={"dt": 1.0})
            results[category] = np.mean(model.diagnostic_variables["delta13c_nh"][-50:])

        assert results["pyrogenic"] > results["geological"]
        assert results["geological"] > results["biogenic"]
        assert results["fossil"] > results["biogenic"]

    def test_synthetic_inversion_roundtrip_t0(self):
        years = np.arange(0, 201, 1.0)
        shares = methane_d13c.MethaneD13C.default_source_shares()

        bio = lambda t: 220.0 + 12.0 * np.sin(2.0 * np.pi * t / 80.0)
        pyro = lambda t: 28.0 + 4.0 * np.cos(2.0 * np.pi * t / 55.0)

        spinup = methane_d13c.MethaneD13C(
            source_strengths={"biogenic": bio(0.0), "pyrogenic": pyro(0.0), "geological": 44.0, "fossil": 0.0}
        )
        spinup.integrate(t_span=(0, 400), y0=_default_y0(), method="euler", kwargs={"dt": 1.0})
        y0 = [
            spinup.state_variables["ch4_12_nh"][-1],
            spinup.state_variables["ch4_12_sh"][-1],
            spinup.state_variables["ch4_13_nh"][-1],
            spinup.state_variables["ch4_13_sh"][-1],
        ]

        model = methane_d13c.MethaneD13C(
            source_strengths={"biogenic": bio, "pyrogenic": pyro, "geological": 44.0, "fossil": 0.0}
        )
        model.integrate(t_span=(0, 200), y0=y0, method="euler", kwargs={"dt": 1.0})

        inversion = model.invert_biogenic_pyrogenic(
            model.time,
            model.diagnostic_variables["ch4_total_nh"],
            model.diagnostic_variables["ch4_total_sh"],
            model.diagnostic_variables["delta13c_nh"],
            model.diagnostic_variables["delta13c_sh"],
        )

        expected_bio_nh = np.asarray([bio(t) for t in years]) * shares["biogenic"]
        expected_bio_sh = np.asarray([bio(t) for t in years]) * (1.0 - shares["biogenic"])
        expected_py_nh = np.asarray([pyro(t) for t in years]) * shares["pyrogenic"]
        expected_py_sh = np.asarray([pyro(t) for t in years]) * (1.0 - shares["pyrogenic"])

        assert np.allclose(inversion.biogenic_nh[1:-1], expected_bio_nh[1:-1], atol=2e-2)
        assert np.allclose(inversion.biogenic_sh[1:-1], expected_bio_sh[1:-1], atol=2e-2)
        assert np.allclose(inversion.pyrogenic_nh[1:-1], expected_py_nh[1:-1], atol=2e-2)
        assert np.allclose(inversion.pyrogenic_sh[1:-1], expected_py_sh[1:-1], atol=2e-2)

    def test_monte_carlo_shapes_t0(self):
        model = methane_d13c.MethaneD13C()
        model.integrate(t_span=(0, 20), y0=_default_y0(), method="euler", kwargs={"dt": 1.0})
        result = model.monte_carlo_inversion(
            model.time,
            model.diagnostic_variables["ch4_total_nh"],
            model.diagnostic_variables["ch4_total_sh"],
            model.diagnostic_variables["delta13c_nh"],
            model.diagnostic_variables["delta13c_sh"],
            n_samples=8,
            random_seed=42,
        )

        assert result["biogenic_nh"].shape == (8, len(model.time))
        assert result["pyrogenic_sh"].shape == (8, len(model.time))

    def test_inversion_uses_non_default_sink_parameters_t0(self):
        years = np.arange(0, 121, 1.0)
        shares = methane_d13c.MethaneD13C.default_source_shares()
        sink_inverse_lifetimes = {"oh": 0.11, "soil": 0.01, "stratosphere": 0.004}
        sink_kies = {"oh": 1.0045, "soil": 1.02, "stratosphere": 1.011}

        bio = lambda t: 210.0 + 8.0 * np.sin(2.0 * np.pi * t / 60.0)
        pyro = lambda t: 24.0 + 2.5 * np.cos(2.0 * np.pi * t / 45.0)

        spinup = methane_d13c.MethaneD13C(
            source_strengths={"biogenic": bio(0.0), "pyrogenic": pyro(0.0), "geological": 44.0, "fossil": 0.0},
            sink_inverse_lifetimes=sink_inverse_lifetimes,
            sink_kies=sink_kies,
        )
        spinup.integrate(t_span=(0, 300), y0=_default_y0(), method="euler", kwargs={"dt": 1.0})
        y0 = [
            spinup.state_variables["ch4_12_nh"][-1],
            spinup.state_variables["ch4_12_sh"][-1],
            spinup.state_variables["ch4_13_nh"][-1],
            spinup.state_variables["ch4_13_sh"][-1],
        ]

        model = methane_d13c.MethaneD13C(
            source_strengths={"biogenic": bio, "pyrogenic": pyro, "geological": 44.0, "fossil": 0.0},
            sink_inverse_lifetimes=sink_inverse_lifetimes,
            sink_kies=sink_kies,
        )
        model.integrate(t_span=(0, 120), y0=y0, method="euler", kwargs={"dt": 1.0})
        inversion = model.invert_biogenic_pyrogenic(
            model.time,
            model.diagnostic_variables["ch4_total_nh"],
            model.diagnostic_variables["ch4_total_sh"],
            model.diagnostic_variables["delta13c_nh"],
            model.diagnostic_variables["delta13c_sh"],
        )

        expected_bio_nh = np.asarray([bio(t) for t in years]) * shares["biogenic"]
        expected_py_nh = np.asarray([pyro(t) for t in years]) * shares["pyrogenic"]
        assert np.allclose(inversion.biogenic_nh[1:-1], expected_bio_nh[1:-1], atol=2e-2)
        assert np.allclose(inversion.pyrogenic_nh[1:-1], expected_py_nh[1:-1], atol=2e-2)

    def test_inversion_respects_time_varying_prescribed_metadata_t0(self):
        years = np.arange(0, 101, 1.0)

        geo_strength = lambda t: 40.0 + 2.0 * np.sin(2.0 * np.pi * t / 70.0)
        geo_share = lambda t: 0.85 + 0.03 * np.sin(2.0 * np.pi * t / 80.0)
        geo_delta = lambda t: -38.0 + 0.4 * np.cos(2.0 * np.pi * t / 90.0)
        fossil_strength = lambda t: 2.0 + 0.5 * np.cos(2.0 * np.pi * t / 50.0)
        fossil_share = lambda t: 0.88 + 0.02 * np.sin(2.0 * np.pi * t / 65.0)
        fossil_delta = lambda t: -37.5 + 0.3 * np.sin(2.0 * np.pi * t / 75.0)

        bio = lambda t: 215.0 + 10.0 * np.sin(2.0 * np.pi * t / 55.0)
        pyro = lambda t: 26.0 + 3.0 * np.cos(2.0 * np.pi * t / 40.0)

        source_strengths = {
            "biogenic": bio,
            "pyrogenic": pyro,
            "geological": geo_strength,
            "fossil": fossil_strength,
        }
        source_signatures = {
            "biogenic": -60.5,
            "pyrogenic": -22.0,
            "geological": geo_delta,
            "fossil": fossil_delta,
        }
        source_shares = {
            "biogenic": 0.80,
            "pyrogenic": 0.44,
            "geological": geo_share,
            "fossil": fossil_share,
        }

        spinup = methane_d13c.MethaneD13C(
            source_strengths={
                "biogenic": bio(0.0),
                "pyrogenic": pyro(0.0),
                "geological": geo_strength(0.0),
                "fossil": fossil_strength(0.0),
            },
            source_signatures={
                "biogenic": -60.5,
                "pyrogenic": -22.0,
                "geological": geo_delta(0.0),
                "fossil": fossil_delta(0.0),
            },
            source_shares={
                "biogenic": 0.80,
                "pyrogenic": 0.44,
                "geological": geo_share(0.0),
                "fossil": fossil_share(0.0),
            },
        )
        spinup.integrate(t_span=(0, 300), y0=_default_y0(), method="euler", kwargs={"dt": 1.0})
        y0 = [
            spinup.state_variables["ch4_12_nh"][-1],
            spinup.state_variables["ch4_12_sh"][-1],
            spinup.state_variables["ch4_13_nh"][-1],
            spinup.state_variables["ch4_13_sh"][-1],
        ]

        model = methane_d13c.MethaneD13C(
            source_strengths=source_strengths,
            source_signatures=source_signatures,
            source_shares=source_shares,
        )
        model.integrate(t_span=(0, 100), y0=y0, method="euler", kwargs={"dt": 1.0})
        inversion = model.invert_biogenic_pyrogenic(
            model.time,
            model.diagnostic_variables["ch4_total_nh"],
            model.diagnostic_variables["ch4_total_sh"],
            model.diagnostic_variables["delta13c_nh"],
            model.diagnostic_variables["delta13c_sh"],
        )

        expected_bio_nh = np.asarray([bio(t) for t in years]) * 0.80
        expected_py_nh = np.asarray([pyro(t) for t in years]) * 0.44
        assert np.allclose(inversion.biogenic_nh[1:-1], expected_bio_nh[1:-1], atol=3e-2)
        assert np.allclose(inversion.pyrogenic_nh[1:-1], expected_py_nh[1:-1], atol=3e-2)
