"""Merger-rate density models and expected event-count utilities.

The rate templates return source-frame merger-rate densities in
Gpc^-3 yr^-1 dex^-1. Expected event counts additionally depend on survey
solid angle, comoving volume, redshift-bin width, mass-bin width, observing
time, and the source-to-observer time-dilation factor 1/(1+z).

"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM


# Menote et al. (2025), Table 1 parameters supplied by the analysis.
P_BBH = {
    "N_R": 7.27,
    "A": 0.09,
    "B": 5.28,
    "C": 4.19,
    "K1": 4.73,
    "K2": 0.99,
    "M1": 1.46e7,
    "M2": 4.30e9,
    "C1": -3.4e-4,
    "C2": -9.53e-5,
    "G": 2.52,
    "Gc": 10.69,
    "Gsig": 0.41,
    "Gd": 0.10,
    "nu": 0.07,
}

P_BHNS = {
    "N_R": 2.78,
    "A": 0.82,
    "B": 4.55,
    "C": 4.55,
    "K1": 5.38,
    "K2": 1.03,
    "M1": 1.17e7,
    "M2": 4.95e9,
    "C1": 1.75e-2,
    "C2": -1.19e-2,
    "G": 1.67,
    "Gc": 10.68,
    "Gsig": 0.34,
    "Gd": 0.10,
    "nu": 0.04,
}

P_BNS = {
    "N_R": 541.7,
    "A": 1.17,
    "B": 4.95,
    "C": 2.73,
    "nu_m": 11.68,
    "K": 7.8e-4,
    "Lc": -0.029,
    "sigma_R": 0.22,
}

# Outer Rim / CosmoDC2 cosmology used by the supplied calculation.
OUTER_RIM_COSMOLOGY = FlatLambdaCDM(H0=71.0, Om0=0.265)


def _return_scalar_if_scalar(value, *inputs):
    """Return float for scalar inputs and ndarray otherwise."""
    array = np.asarray(value)
    if all(np.ndim(item) == 0 for item in inputs):
        return float(array)
    return array


def psi(z, parameters: Mapping[str, float]):
    """Universal source-frame redshift factor."""
    z_array = np.asarray(z, dtype=np.float64)
    value = (
        parameters["N_R"]
        * (1.0 + z_array) ** parameters["A"]
        / (
            1.0
            + ((1.0 + z_array) / parameters["C"]) ** parameters["B"]
        )
    )
    return _return_scalar_if_scalar(value, z)


def heaviside_term(z, log_mass, parameters: Mapping[str, float]):
    """Mass-dependent smooth Heaviside component for BBH/BHNS."""
    z_array = np.asarray(z, dtype=np.float64)
    log_mass_array = np.asarray(log_mass, dtype=np.float64)

    first = np.tanh(
        parameters["K1"]
        * (log_mass_array - np.log10(parameters["M1"]))
    )
    second_argument = parameters["K2"] * (
        log_mass_array
        - np.log10(parameters["M2"])
        - parameters["C1"] * z_array
        - parameters["C2"] * z_array**2
    )
    value = first - np.tanh(second_argument)
    return _return_scalar_if_scalar(value, z, log_mass)


def gaussian_term(z, log_mass, parameters: Mapping[str, float]):
    """Mass-dependent Gaussian component for BBH/BHNS."""
    z_array = np.asarray(z, dtype=np.float64)
    log_mass_array = np.asarray(log_mass, dtype=np.float64)
    center = parameters["Gc"] - parameters["nu"] * z_array**2
    exponent = (
        -0.5 * ((log_mass_array - center) / parameters["Gsig"]) ** 2
        - parameters["Gd"] * z_array**2
    )
    value = parameters["G"] * np.exp(exponent)
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bbh_bhns(z, log_mass, parameters: Mapping[str, float]):
    """BBH/BHNS source-frame rate density [Gpc^-3 yr^-1 dex^-1]."""
    value = psi(z, parameters) * (
        heaviside_term(z, log_mass, parameters)
        + gaussian_term(z, log_mass, parameters)
    )
    value = np.where(np.isfinite(value), np.maximum(value, 0.0), 0.0)
    return _return_scalar_if_scalar(value, z, log_mass)


def rate_bbh(z, log_mass):
    """BBH source-frame rate density [Gpc^-3 yr^-1 dex^-1]."""
    return rate_bbh_bhns(z, log_mass, P_BBH)


def rate_bhns(z, log_mass):
    """BHNS source-frame rate density [Gpc^-3 yr^-1 dex^-1]."""
    return rate_bbh_bhns(z, log_mass, P_BHNS)


def rate_bns(z, log_mass, parameters: Mapping[str, float] = P_BNS):
    """BNS source-frame rate density [Gpc^-3 yr^-1 dex^-1]."""
    z_array = np.asarray(z, dtype=np.float64)
    log_mass_array = np.asarray(log_mass, dtype=np.float64)

    nu_z = parameters["nu_m"] / (1.0 + parameters["K"] * z_array**2)
    ratio = nu_z - log_mass_array

    # Evaluate log10 only where its argument is positive. This vectorizes the
    # scalar guard used in the original notebook without generating warnings.
    log_ratio = np.full(np.broadcast(z_array, log_mass_array).shape, np.nan)
    ratio_broadcast = np.broadcast_to(ratio, log_ratio.shape)
    positive = ratio_broadcast > 0.0
    np.log10(ratio_broadcast, out=log_ratio, where=positive)

    argument = log_ratio - parameters["Lc"]
    exponent = -0.5 * (argument / parameters["sigma_R"]) ** 2
    value = psi(z_array, parameters) * np.exp(exponent)
    value = np.where(positive & np.isfinite(value), np.maximum(value, 0.0), 0.0)
    return _return_scalar_if_scalar(value, z, log_mass)


RATE_FUNCTIONS = {
    "BBH": rate_bbh,
    "BNS": rate_bns,
    "BHNS": rate_bhns,
}


def solid_angle_sr(area_deg2):
    """Convert square degrees to steradians."""
    area = np.asarray(area_deg2, dtype=np.float64)
    if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("area_deg2 must be positive and finite")
    value = area * (np.pi / 180.0) ** 2
    return _return_scalar_if_scalar(value, area_deg2)


def differential_comoving_volume(z, cosmology=OUTER_RIM_COSMOLOGY):
    """Return dV_c/(dz dOmega) in Gpc^3 sr^-1."""
    value_mpc3 = cosmology.differential_comoving_volume(z).to_value(
        u.Mpc**3 / u.sr
    )
    value = np.asarray(value_mpc3) * 1.0e-9
    return _return_scalar_if_scalar(value, z)


def expected_events_in_bin(
    rate_density,
    z_center,
    dz,
    dlog_mass,
    area_deg2,
    observing_years,
    cosmology=OUTER_RIM_COSMOLOGY,
):
    """Expected observer-frame events in one (z, logM) bin.

    ``rate_density`` must be in Gpc^-3 source-frame yr^-1 dex^-1.
    """
    dz_array = np.asarray(dz, dtype=np.float64)
    mass_width_array = np.asarray(dlog_mass, dtype=np.float64)
    time_array = np.asarray(observing_years, dtype=np.float64)
    if (
        np.any(dz_array <= 0.0)
        or np.any(mass_width_array <= 0.0)
        or np.any(time_array < 0.0)
    ):
        raise ValueError("dz and dlog_mass must be positive; time non-negative")

    volume_factor = (
        differential_comoving_volume(z_center, cosmology)
        * solid_angle_sr(area_deg2)
        * dz_array
    )
    expected = (
        np.asarray(rate_density, dtype=np.float64)
        / (1.0 + z_center)
        * volume_factor
        * mass_width_array
        * time_array
    )
    expected = np.where(
        np.isfinite(expected), np.maximum(expected, 0.0), 0.0
    )
    return _return_scalar_if_scalar(
        expected, rate_density, z_center, dz, dlog_mass, observing_years
    )


def expected_events_for_slice(
    rate_function: Callable,
    z_initial,
    z_final,
    log_mass_edges,
    area_deg2,
    observing_years,
    cosmology=OUTER_RIM_COSMOLOGY,
):
    """Midpoint-integrated expected events in one unequal-width z slice."""
    if z_final <= z_initial:
        raise ValueError("z_final must be greater than z_initial")

    mass_edges = np.asarray(log_mass_edges, dtype=np.float64)
    if mass_edges.ndim != 1 or mass_edges.size < 2:
        raise ValueError("log_mass_edges must be a 1D array with at least 2 edges")
    if np.any(np.diff(mass_edges) <= 0.0):
        raise ValueError("log_mass_edges must be strictly increasing")

    z_center = 0.5 * (z_initial + z_final)
    dz = z_final - z_initial
    mass_centers = 0.5 * (mass_edges[:-1] + mass_edges[1:])
    mass_widths = np.diff(mass_edges)
    rates = np.asarray(rate_function(z_center, mass_centers))

    volume_time_factor = (
        differential_comoving_volume(z_center, cosmology)
        * solid_angle_sr(area_deg2)
        * dz
        * observing_years
        / (1.0 + z_center)
    )
    expected = np.sum(rates * mass_widths) * volume_time_factor
    if not np.isfinite(expected) or expected < 0.0:
        return 0.0
    return float(expected)


def expected_events_all_slices(
    zi,
    zf,
    log_mass_edges,
    area_deg2,
    observing_years,
    rate_functions=RATE_FUNCTIONS,
    cosmology=OUTER_RIM_COSMOLOGY,
):
    """Expected BBH/BNS/BHNS event counts for every redshift slice."""
    zi_array = np.atleast_1d(np.asarray(zi, dtype=np.float64))
    zf_array = np.atleast_1d(np.asarray(zf, dtype=np.float64))
    if zi_array.shape != zf_array.shape:
        raise ValueError("zi and zf must have equal shape")

    result = {}
    for name, function in rate_functions.items():
        result[name] = np.array(
            [
                expected_events_for_slice(
                    function,
                    z_initial,
                    z_final,
                    log_mass_edges,
                    area_deg2,
                    observing_years,
                    cosmology,
                )
                for z_initial, z_final in zip(zi_array, zf_array)
            ],
            dtype=np.float64,
        )
    return result


def draw_poisson(expected, rng=None):
    """Draw reproducible Poisson counts from scalar or array expectations."""
    expected_array = np.asarray(expected, dtype=np.float64)
    if np.any(~np.isfinite(expected_array)) or np.any(expected_array < 0.0):
        raise ValueError("expected must contain only finite non-negative values")
    if rng is None:
        rng = np.random.default_rng()
    draw = rng.poisson(expected_array)
    return _return_scalar_if_scalar(draw, expected)


__all__ = [
    "OUTER_RIM_COSMOLOGY",
    "P_BBH",
    "P_BHNS",
    "P_BNS",
    "RATE_FUNCTIONS",
    "differential_comoving_volume",
    "draw_poisson",
    "expected_events_all_slices",
    "expected_events_for_slice",
    "expected_events_in_bin",
    "gaussian_term",
    "heaviside_term",
    "psi",
    "rate_bbh",
    "rate_bbh_bhns",
    "rate_bhns",
    "rate_bns",
    "solid_angle_sr",
]
