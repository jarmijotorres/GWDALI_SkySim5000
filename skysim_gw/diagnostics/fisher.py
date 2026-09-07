"""Diagnostic plots for the GWDALI Fisher-localization catalogue.

This is a quality-control stage for map-level peak-siren analyses. It does not
attempt host identification. The plots assess localization scale, numerical
conditioning, redshift/SNR trends, and the range over which a tangent-plane
Gaussian ellipse is becoming too broad for later spherical-map construction.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skysim_gw.products import localization_output_path
from skysim_gw.catalogue import catalog_path, load_gwdali_catalog


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

from skysim_gw.settings.fisher import (
    BASE_DIR,
    INPUT_DIR,
    LOCALIZATION_DIR,
    DIAGNOSTIC_DIR,
    SOURCE_TYPE,
    ZI,
    ZF,
    CONDITION_WARNING,
)


# SkySim5000 covers one octant. This line is a reference scale, not a cut.
FULL_SKY_AREA_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2
FOOTPRINT_AREA_DEG2 = FULL_SKY_AREA_DEG2 / 8.0

# A diagnostic marker only; events are not discarded based on this value.


def finite_positive(values):
    values = np.asarray(values, dtype=float)
    return np.isfinite(values) & (values > 0.0)


def percentile_text(values, percentiles=(5, 16, 50, 84, 95)):
    quantiles = np.percentile(values, percentiles)
    return ", ".join(
        f"p{percentile}={value:.6g}"
        for percentile, value in zip(percentiles, quantiles)
    )


def representative_indices(source_indices, areas):
    order = np.argsort(areas)
    positions = {
        "best": 0,
        "median": len(order) // 2,
        "worst": len(order) - 1,
    }
    return {
        label: int(source_indices[order[position]])
        for label, position in positions.items()
    }


def main():
    source_type = SOURCE_TYPE.upper()
    localization_path = localization_output_path(
        LOCALIZATION_DIR, source_type, ZI, ZF
    )
    injection_path = catalog_path(INPUT_DIR, source_type, ZI, ZF)

    with np.load(localization_path, allow_pickle=False) as data:
        results = data["results"]
        completed = data["completed"].astype(bool)
        free_params = tuple(data["free_params"].tolist())
    injections, _ = load_gwdali_catalog(injection_path)

    if len(results) != len(injections):
        raise ValueError("Localization and injection catalogues have different lengths")

    attempted = completed
    successful = attempted & results["success"]
    valid_area = successful & finite_positive(results["sky_area_90_deg2"])
    if not np.any(valid_area):
        raise RuntimeError("No successful finite positive localization areas found")

    selected = results[valid_area]
    source_indices = selected["source_index"].astype(int)
    areas = selected["sky_area_90_deg2"].astype(float)
    snr = selected["snr_network"].astype(float)
    condition = selected["fisher_condition_number"].astype(float)
    runtime = selected["runtime_s"].astype(float)
    redshift = injections["redshift_true"][source_indices].astype(float)
    used_pseudoinverse = selected["used_pseudoinverse"].astype(bool)

    finite_condition = finite_positive(condition)
    warning_condition = finite_condition & (condition >= CONDITION_WARNING)
    broader_than_footprint = areas >= FOOTPRINT_AREA_DEG2
    broader_than_full_sky = areas >= FULL_SKY_AREA_DEG2

    print(f"Localization file: {localization_path}")
    print(f"Free parameters: {free_params}")
    print(f"Catalogue rows: {len(results):,}")
    print(f"Attempted: {int(attempted.sum()):,}")
    print(f"Successful: {int(successful.sum()):,}")
    print(f"Failed: {int((attempted & ~results['success']).sum()):,}")
    print(f"Valid areas: {len(areas):,}")
    print(f"Pseudoinverses: {int(used_pseudoinverse.sum()):,}")
    print(f"Condition >= {CONDITION_WARNING:.1e}: {int(warning_condition.sum()):,}")
    print(f"Octant reference area: {FOOTPRINT_AREA_DEG2:.3f} deg^2")
    print(
        "Area >= octant: "
        f"{int(broader_than_footprint.sum()):,} "
        f"({100.0 * broader_than_footprint.mean():.2f}%)"
    )
    print(
        "Area >= full sky (invalid tangent-plane result): "
        f"{int(broader_than_full_sky.sum()):,} "
        f"({100.0 * broader_than_full_sky.mean():.2f}%)"
    )
    print("Area quantiles [deg^2]:", percentile_text(areas))
    print("SNR quantiles:", percentile_text(snr))
    print("Condition quantiles:", percentile_text(condition[finite_condition]))
    print("Representative source indices:", representative_indices(source_indices, areas))

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = DIAGNOSTIC_DIR / (
        f"skysim5000_{source_type}_fisher_diagnostics_"
        f"z{ZI:.4f}_{ZF:.4f}.png"
    )

    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    ax_area_hist, ax_snr, ax_redshift, ax_condition, ax_cdf, ax_runtime = axes.ravel()

    log_area = np.log10(areas)
    bins = np.linspace(log_area.min(), log_area.max(), 28) if len(areas) > 1 else 10
    ax_area_hist.hist(log_area, bins=bins, color="tab:blue", alpha=0.8)
    ax_area_hist.axvline(
        np.log10(FOOTPRINT_AREA_DEG2), color="tab:red", linestyle="--",
        label=f"Octant: {FOOTPRINT_AREA_DEG2:.0f} deg²",
    )
    ax_area_hist.axvline(
        np.log10(FULL_SKY_AREA_DEG2), color="black", linestyle=":",
        label="Full sky",
    )
    ax_area_hist.set_xlabel(r"$\log_{10}(\Delta\Omega_{90}/\mathrm{deg}^2)$")
    ax_area_hist.set_ylabel("Number of sirens")
    ax_area_hist.set_title("Fisher localization-area distribution")
    ax_area_hist.legend(fontsize=9)

    color_values = np.log10(np.clip(condition, 1.0, None))
    scatter = ax_snr.scatter(
        snr, areas, c=color_values, s=24, alpha=0.8, cmap="viridis",
        edgecolors="none",
    )
    ax_snr.axhline(FOOTPRINT_AREA_DEG2, color="tab:red", linestyle="--")
    ax_snr.set_yscale("log")
    ax_snr.set_xlabel(r"Network SNR $\rho_\mathrm{net}$")
    ax_snr.set_ylabel(r"$\Delta\Omega_{90}$ [deg$^2$]")
    ax_snr.set_title("Localization versus network SNR")
    colorbar = figure.colorbar(scatter, ax=ax_snr)
    colorbar.set_label(r"$\log_{10}\,\kappa(F)$")

    ax_redshift.scatter(redshift, areas, s=24, alpha=0.75, color="tab:purple")
    ax_redshift.axhline(FOOTPRINT_AREA_DEG2, color="tab:red", linestyle="--")
    ax_redshift.set_yscale("log")
    ax_redshift.set_xlabel("True redshift")
    ax_redshift.set_ylabel(r"$\Delta\Omega_{90}$ [deg$^2$]")
    ax_redshift.set_title("Localization versus redshift")

    ax_condition.scatter(
        condition[finite_condition], areas[finite_condition],
        s=24, alpha=0.75, color="tab:orange",
    )
    ax_condition.axvline(CONDITION_WARNING, color="black", linestyle=":")
    ax_condition.axhline(FOOTPRINT_AREA_DEG2, color="tab:red", linestyle="--")
    ax_condition.set_xscale("log")
    ax_condition.set_yscale("log")
    ax_condition.set_xlabel(r"Fisher condition number $\kappa(F)$")
    ax_condition.set_ylabel(r"$\Delta\Omega_{90}$ [deg$^2$]")
    ax_condition.set_title("Numerical conditioning")

    sorted_areas = np.sort(areas)
    cumulative = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)
    ax_cdf.plot(sorted_areas, cumulative, color="tab:green", linewidth=2)
    ax_cdf.axvline(FOOTPRINT_AREA_DEG2, color="tab:red", linestyle="--")
    ax_cdf.set_xscale("log")
    ax_cdf.set_ylim(0.0, 1.02)
    ax_cdf.set_xlabel(r"Maximum $\Delta\Omega_{90}$ [deg$^2$]")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_title("Fraction localized below an area")
    ax_cdf.grid(alpha=0.25)

    valid_runtime = runtime[np.isfinite(runtime) & (runtime >= 0.0)]
    ax_runtime.hist(valid_runtime, bins=24, color="tab:cyan", alpha=0.85)
    ax_runtime.set_xlabel("Runtime per siren [s]")
    ax_runtime.set_ylabel("Number of sirens")
    ax_runtime.set_title("Localization runtime")

    figure.suptitle(
        f"{source_type} Fisher diagnostics, {ZI:.3f} ≤ z < {ZF:.3f} "
        f"({len(areas)} successful sirens)",
        fontsize=15,
    )
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    print(f"Saved diagnostic figure: {figure_path}")


if __name__ == "__main__":
    main()
