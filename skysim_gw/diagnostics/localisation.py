"""Targeted diagnosis of unexpectedly broad GWDALI sky localisations.

For one detected source this script compares parameter marginalisation choices
and finite-difference step sizes. It reports matrix conditioning, Menote et
al.'s inversion residual, and sky areas under both the conventional declination
Jacobian and Menote's published equation (45).
"""

from pathlib import Path
import time

import GWDALI as gw
import numpy as np

from skysim_gw.products import snr_output_path
from skysim_gw.catalogue import (
    catalog_path,
    get_gwdali_source,
    load_gwdali_catalogue,
)
from skysim_gw.detectors import get_lvk_detectors
from skysim_gw.results import find_matrix, unwrap_gwdali_result


from skysim_gw.settings.localisation import (
    BASE_DIR,
    INPUT_DIR,
    SNR_DIR,
    OUTPUT_DIR,
    SOURCE_TYPE,
    ZI,
    ZF,
    SOURCE_INDEX,
    APPROXIMANTS,
    FMIN_HZ,
    FMAX_HZ,
    FSIZE,
    PARAMETER_SETS,
    STEP_SIZES,
)


# The full grid is modest: 3 parameter sets x 4 steps at about 5 s each.


RESULT_DTYPE = np.dtype([
    ("parameter_set", "U32"),
    ("step_size", "f8"),
    ("n_parameters", "i4"),
    ("condition_number", "f8"),
    ("inversion_residual", "f8"),
    ("sigma_ra_deg", "f8"),
    ("sigma_dec_deg", "f8"),
    ("correlation_ra_dec", "f8"),
    ("area90_conventional_deg2", "f8"),
    ("area95_menote_deg2", "f8"),
    ("runtime_s", "f8"),
    ("success", "?"),
    ("error_message", "U512"),
])


def load_snr(path):
    with np.load(path, allow_pickle=False) as data:
        return data["results"], data["completed"].astype(bool)


def invert_and_check(fisher):
    fisher = 0.5 * (fisher + fisher.T)
    covariance = np.linalg.inv(fisher)
    covariance = 0.5 * (covariance + covariance.T)
    identity_error = fisher @ covariance - np.eye(len(fisher))
    residual = float(np.max(np.abs(identity_error)))
    return covariance, residual


def angular_summary(covariance, free_params, declination_deg):
    """Summarise the RA--Dec block returned by GWDALI.

    GWDALI differentiates with respect to RA and Dec in degrees, so this
    covariance block is already in degree squared.  Only the declination used
    in the spherical Jacobian must be converted to radians for sin/cos.
    """
    i_ra = free_params.index("RA")
    i_dec = free_params.index("Dec")
    angular_covariance = covariance[np.ix_([i_ra, i_dec], [i_ra, i_dec])]
    determinant = float(np.linalg.det(angular_covariance))
    if determinant <= 0.0 or not np.isfinite(determinant):
        raise ValueError(f"Invalid angular covariance determinant {determinant}")

    sigma_ra = np.sqrt(angular_covariance[0, 0])
    sigma_dec = np.sqrt(angular_covariance[1, 1])
    correlation = angular_covariance[0, 1] / (sigma_ra * sigma_dec)
    declination_rad = np.radians(declination_deg)

    # For a two-dimensional Gaussian, chi2_2(90%) = -2 ln(0.1).
    area90_conventional = (
        np.pi * (-2.0 * np.log(0.1))
        * abs(np.cos(declination_rad)) * np.sqrt(determinant)
    )

    # Menote equations (45)--(46), reproduced literally: |sin(delta)| and 95%.
    characteristic_area = (
        2.0 * np.pi * abs(np.sin(declination_rad)) * np.sqrt(determinant)
    )
    area95_menote = -characteristic_area * np.log(0.05)
    return (
        float(sigma_ra),
        float(sigma_dec),
        float(correlation),
        float(area90_conventional),
        float(area95_menote),
    )


def main():
    source_type = SOURCE_TYPE.upper()
    injection_path = catalog_path(INPUT_DIR, source_type, ZI, ZF)
    snr_path = snr_output_path(SNR_DIR, source_type, ZI, ZF)
    catalogue, _ = load_gwdali_catalogue(injection_path)
    snr_results, snr_completed = load_snr(snr_path)

    index = int(SOURCE_INDEX)
    if not snr_completed[index] or not snr_results["success"][index]:
        raise ValueError(f"Source {index} has no successful SNR result")
    catalogue_prms = get_gwdali_source(catalogue, index)
    if not np.isclose(catalogue_prms["t_coal"], 0.0, atol=1e-15, rtol=0.0):
        raise ValueError("This diagnostic expects injected t_coal=0")

    # The stored injection catalogue uses radians for all sky angles. GWDALI's
    # AngTransf converts RA and Dec internally with pi/180, so those two values
    # must be supplied in degrees. Its other angular parameters remain radians.
    GwPrms = catalogue_prms.copy()
    GwPrms["RA"] = float(np.degrees(catalogue_prms["RA"]))
    GwPrms["Dec"] = float(np.degrees(catalogue_prms["Dec"]))

    detectors = get_lvk_detectors(include_kagra=True)
    approximant = APPROXIMANTS[source_type]
    detector_snrs, corrected_network_snr = gw.get_SNR(
        detectors,
        GwPrms,
        approximant,
        enable_jax_waveforms=False,
        fmin=FMIN_HZ,
        fmax=FMAX_HZ,
        fsize=FSIZE,
    )
    detector_snrs = np.asarray(detector_snrs, dtype=float)
    corrected_network_snr = float(corrected_network_snr)
    rows = np.zeros(len(PARAMETER_SETS) * len(STEP_SIZES), dtype=RESULT_DTYPE)
    for name in RESULT_DTYPE.names:
        if np.issubdtype(RESULT_DTYPE[name], np.floating):
            rows[name] = np.nan

    print(f"Source index: {index}")
    print(
        "Old checkpoint network SNR (wrong RA/Dec units): "
        f"{snr_results['snr_network'][index]:.8f}"
    )
    print(f"Corrected network SNR: {corrected_network_snr:.8f}")
    print(f"Corrected detector SNRs: {detector_snrs.tolist()}")
    print(
        f"Catalogue: RA={catalogue_prms['RA']:.8f} rad, "
        f"Dec={catalogue_prms['Dec']:.8f} rad"
    )
    print(
        f"GWDALI:   RA={GwPrms['RA']:.8f} deg, "
        f"Dec={GwPrms['Dec']:.8f} deg"
    )
    print(f"Approximant: {approximant}")
    print()
    print(
        f"{'parameters':22s} {'step':>9s} {'cond(F)':>12s} "
        f"{'|FC-I|max':>11s} {'sigRA':>9s} {'sigDec':>9s} "
        f"{'A90':>11s} {'A95Menote':>12s}"
    )

    row_index = 0
    for parameter_name, free_params_tuple in PARAMETER_SETS.items():
        free_params = list(free_params_tuple)
        for step_size in STEP_SIZES:
            row = rows[row_index]
            row["parameter_set"] = parameter_name
            row["step_size"] = step_size
            row["n_parameters"] = len(free_params)
            started = time.perf_counter()
            try:
                raw_result = gw.GWDALI(
                    GwPrms,
                    detectors,
                    free_params,
                    approx=approximant,
                    method="Fisher",
                    diff_method="numdiff",
                    step_size=[step_size],
                    run_sampler=False,
                    hide_info=True,
                    output_name=None,
                    save_bilby_path=False,
                    enable_jax_waveforms=False,
                    fmin=FMIN_HZ,
                    fmax=FMAX_HZ,
                    fsize=FSIZE,
                )
                tensors = unwrap_gwdali_result(raw_result)
                _, fisher = find_matrix(tensors, ("Fisher", "fisher"))
                if fisher is None or fisher.shape != (len(free_params), len(free_params)):
                    raise ValueError("Missing or incorrectly shaped Fisher matrix")
                covariance, residual = invert_and_check(fisher)
                sigma_ra, sigma_dec, correlation, area90, area95 = angular_summary(
                    covariance, free_params, GwPrms["Dec"]
                )
                row["condition_number"] = np.linalg.cond(fisher)
                row["inversion_residual"] = residual
                row["sigma_ra_deg"] = sigma_ra
                row["sigma_dec_deg"] = sigma_dec
                row["correlation_ra_dec"] = correlation
                row["area90_conventional_deg2"] = area90
                row["area95_menote_deg2"] = area95
                row["success"] = True
            except Exception as error:
                row["success"] = False
                row["error_message"] = f"{type(error).__name__}: {error}"[:512]
            row["runtime_s"] = time.perf_counter() - started

            if row["success"]:
                print(
                    f"{parameter_name:22s} {step_size:9.1e} "
                    f"{row['condition_number']:12.4e} "
                    f"{row['inversion_residual']:11.3e} "
                    f"{row['sigma_ra_deg']:9.3f} "
                    f"{row['sigma_dec_deg']:9.3f} "
                    f"{row['area90_conventional_deg2']:11.3f} "
                    f"{row['area95_menote_deg2']:12.3f}",
                    flush=True,
                )
            else:
                print(
                    f"{parameter_name:22s} {step_size:9.1e} FAILED: "
                    f"{row['error_message']}",
                    flush=True,
                )
            row_index += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / (
        f"skysim5000_{source_type}_source{index:06d}_"
        f"localisation_diagnosis.npz"
    )
    np.savez_compressed(
        output_path,
        results=rows,
        source_index=np.asarray(index),
        source_type=np.asarray(source_type),
        old_checkpoint_network_snr=np.asarray(
            snr_results["snr_network"][index]
        ),
        corrected_network_snr=np.asarray(corrected_network_snr),
        corrected_detector_snrs=detector_snrs,
        catalogue_parameters=np.asarray(str(catalogue_prms)),
        gwdali_parameters=np.asarray(str(GwPrms)),
        ra_dec_catalogue_unit=np.asarray("radian"),
        ra_dec_gwdali_unit=np.asarray("degree"),
    )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
