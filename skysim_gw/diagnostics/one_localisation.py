"""Steps 8--10 diagnostic: localize one detected source with full LVK.

This uses 100% duty cycle, joins the injection and SNR catalogues through
``source_index``, selects one detected event, runs a Fisher calculation, and
reports the marginalized 90% RA--Dec ellipse when a covariance is returned.
"""

from pathlib import Path
import time
from skysim_gw.results import find_matrix, unwrap_gwdali_result

import GWDALI as gw
import numpy as np

from skysim_gw.products import snr_output_path
from skysim_gw.catalogue import (
    catalog_path,
    get_gwdali_source,
    load_gwdali_catalogue,
)
from skysim_gw.detectors import LVK_LABELS, get_lvk_detectors


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

from skysim_gw.settings.one_localisation import (
    BASE_DIR,
    INPUT_DIR,
    SNR_DIR,
    SOURCE_TYPE,
    ZI,
    ZF,
    SOURCE_INDEX,
    APPROXIMANTS,
    FMIN_HZ,
    FMAX_HZ,
    FSIZE,
)


# None selects the first successful detected event. Set an integer to test a
# particular original injection-catalogue row.


# Start with the correlated extrinsic parameters. Masses and spins are held
# fixed for this first localization validation.
FREE_PARAMS = [
    "RA",
    "Dec",
    "dL",
    "iota",
    "psi",
    "phi_coal",
]

def load_snr_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        results = data["results"]
        completed = data["completed"].astype(bool)
    if len(results) != len(completed):
        raise ValueError("SNR results and completion mask have different sizes")
    return results, completed


def select_source_index(results, completed, requested_index=None):
    valid = completed & results["success"] & results["detected"]
    if requested_index is None:
        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            raise RuntimeError("The checkpoint contains no completed detections")
        return int(candidates[0])

    index = int(requested_index)
    if index < 0 or index >= len(results):
        raise IndexError(f"Source index {index} outside [0, {len(results)})")
    if not completed[index]:
        raise ValueError(f"Source {index} has not completed its SNR calculation")
    if not results["success"][index]:
        raise ValueError(
            f"Source {index} SNR failed: {results['error_message'][index]}"
        )
    if not results["detected"][index]:
        raise ValueError(
            f"Source {index} is not detected; "
            f"network SNR={results['snr_network'][index]:.4f}"
        )
    return index


def sky_area_90_from_covariance(covariance, free_params, declination):
    """Small-angle marginalized 90% sky ellipse in square degrees."""
    index_ra = free_params.index("RA")
    index_dec = free_params.index("Dec")
    sky_covariance = covariance[np.ix_([index_ra, index_dec], [index_ra, index_dec])]
    determinant = float(np.linalg.det(sky_covariance))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError(f"RA--Dec covariance determinant is {determinant}")

    chi2_90_for_two_dimensions = 4.605170185988092
    area_steradians = (
        np.pi
        * chi2_90_for_two_dimensions
        * abs(np.cos(declination))
        * np.sqrt(determinant)
    )
    return float(area_steradians * (180.0 / np.pi) ** 2), sky_covariance


def main():
    source_type = SOURCE_TYPE.upper()
    injection_path = catalog_path(INPUT_DIR, source_type, ZI, ZF)
    snr_path = snr_output_path(SNR_DIR, source_type, ZI, ZF)

    catalog, _ = load_gwdali_catalogue(injection_path)
    snr_results, completed = load_snr_checkpoint(snr_path)
    if len(catalog) != len(snr_results):
        raise ValueError("Injection and SNR catalogues have different lengths")

    source_index = select_source_index(snr_results, completed, SOURCE_INDEX)
    GwPrms = get_gwdali_source(catalog, source_index)
    detectors = get_lvk_detectors(include_kagra=True)
    approximant = APPROXIMANTS[source_type]

    print(f"Injection catalogue: {injection_path}")
    print(f"SNR checkpoint: {snr_path}")
    print(f"Selected source_index: {source_index}")
    print(f"Network SNR: {snr_results['snr_network'][source_index]:.8f}")
    print("Detector SNRs:")
    snr_fields = (
        "snr_hanford", "snr_livingston", "snr_virgo", "snr_kagra"
    )
    for label, field in zip(LVK_LABELS, snr_fields):
        print(f"  {label:16s}: {snr_results[field][source_index]:.8f}")
    print(f"Approximant: {approximant}")
    print(f"FreeParams: {FREE_PARAMS}")
    print("Running Fisher localization with the full LVK network...")

    start = time.perf_counter()
    result = gw.GWDALI(
        GwPrms,
        detectors,
        FREE_PARAMS,
        approx=approximant,
        method="Fisher",
        # IMRPhenomXPHM/IMRPhenomXHM are generated through LALSuite and
        # therefore cannot be differentiated with JAX autodiff.
        diff_method="numdiff",
        run_sampler=False,
        hide_info=False,
        output_name=None,
        save_bilby_path=False,
        enable_jax_waveforms=False,
        fmin=FMIN_HZ,
        fmax=FMAX_HZ,
        fsize=FSIZE,
    )
    elapsed = time.perf_counter() - start

    print(f"\nElapsed: {elapsed:.3f} s")
    print("Return type:", type(result))
    if isinstance(result, tuple):
        print(f"Samples container type: {type(result[0])}, length={len(result[0])}")
    tensor_result = unwrap_gwdali_result(result)
    print("Tensor keys:", tuple(tensor_result))
    for name, value in tensor_result.items():
        if isinstance(value, (list, tuple)):
            shapes = [np.shape(item) for item in value]
            print(f"  {name!r}: {type(value).__name__}, item shapes={shapes}")
        else:
            print(f"  {name!r}: type={type(value)}, shape={np.shape(value)}")

    covariance_name, covariance = find_matrix(
        tensor_result,
        ("CovFisher", "Covariance", "Cov", "covariance", "cov"),
    )
    fisher_name, fisher = find_matrix(
        tensor_result,
        ("Fisher", "fisher", "FisherMatrix", "fisher_matrix"),
    )

    if fisher is not None:
        print(f"\nFisher key: {fisher_name}")
        print(f"Fisher shape: {fisher.shape}")
        condition_number = float(np.linalg.cond(fisher))
        print(f"Fisher condition number: {condition_number:.6e}")

        # In Fisher-only mode GWDALI v1 returns the information matrix but no
        # covariance. The marginalized covariance is its inverse. Fall back
        # to a Moore-Penrose inverse if numerical inversion fails.
        if covariance is None:
            try:
                covariance = np.linalg.inv(fisher)
                covariance_name = "inverse(Fisher)"
            except np.linalg.LinAlgError:
                covariance = np.linalg.pinv(fisher, rcond=1.0e-12)
                covariance_name = "pinv(Fisher, rcond=1e-12)"
    else:
        print("\nNo Fisher matrix found under the known result keys.")

    if covariance is not None:
        print(f"\nCovariance key: {covariance_name}")
        print("Covariance matrix:")
        print(covariance)
        try:
            area_90, sky_covariance = sky_area_90_from_covariance(
                covariance, FREE_PARAMS, GwPrms["Dec"]
            )
            print("Marginalized RA--Dec covariance [rad^2]:")
            print(sky_covariance)
            print(f"Approximate 90% sky area: {area_90:.8g} deg^2")
        except Exception as error:
            print(f"Could not calculate sky area: {type(error).__name__}: {error}")
    else:
        print("\nNo covariance matrix found under the known result keys.")


if __name__ == "__main__":
    main()
