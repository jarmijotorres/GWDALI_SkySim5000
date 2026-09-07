"""Compare Fisher and Doublet GW sky localisations.

The comparison is restricted to sources successfully localized by both
methods, matched using the original source_index.

For Doublet, the stored covariance is the covariance of the sampled Doublet
posterior. Therefore the reported Doublet sky area is a Gaussian-equivalent
area, making it directly comparable to the Fisher covariance ellipse.

This script reports:
    - number of common successful sources
    - Fisher and Doublet 90% sky areas
    - Doublet / Fisher area ratio
    - RA and Dec uncertainty ratios
    - RA-Dec correlation
    - runtime comparison

It also produces diagnostic plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

from skysim_gw.settings.compare import (
    BASE_DIR,
    INPUT_DIR,
    SOURCE_TYPE,
    ZI,
    ZF,
    FISHER_PATH,
    DOUBLET_PATH,
    OUTPUT_DIR,
)


# -------------------------------------------------------------------------
# Adjust FISHER_PATH only if your original Fisher filename differs.
# -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_localization(path):
    """Load localization checkpoint."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Localization file does not exist:\n{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as saved:

        results = saved["results"]
        covariance = saved["covariance"]
        completed = saved["completed"].astype(bool)

        free_params = tuple(
            saved["free_params"].tolist()
        )

    return {
        "results": results,
        "covariance": covariance,
        "completed": completed,
        "free_params": free_params,
    }


def successful_mask(data):
    """Sources completed and successfully localized."""

    return (
        data["completed"]
        & data["results"]["success"]
    )


def percentile_summary(values):
    """Return median and 16th/84th percentiles."""

    values = np.asarray(values)

    p16, median, p84 = np.percentile(
        values,
        [16.0, 50.0, 84.0],
    )

    return median, p16, p84


def print_ratio_summary(name, ratio):

    median, p16, p84 = percentile_summary(
        ratio
    )

    print(
        f"{name:<28s}: "
        f"median={median:.4f} "
        f"[{p16:.4f}, {p84:.4f}]"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Fisher:")
    print(FISHER_PATH)

    fisher = load_localization(
        FISHER_PATH
    )

    print("\nLoading Doublet:")
    print(DOUBLET_PATH)

    doublet = load_localization(
        DOUBLET_PATH
    )

    if len(fisher["results"]) != len(doublet["results"]):
        raise ValueError(
            "Fisher and Doublet catalogues "
            "have different lengths"
        )

    # ---------------------------------------------------------------------
    # Match the same physical sources.
    # ---------------------------------------------------------------------

    fisher_success = successful_mask(
        fisher
    )

    doublet_success = successful_mask(
        doublet
    )

    common = (
        fisher_success
        & doublet_success
    )

    indices = np.flatnonzero(
        common
    )

    print("\n========================================")
    print("Localization comparison")
    print("========================================")

    print(
        f"Fisher successful:  "
        f"{int(fisher_success.sum()):,}"
    )

    print(
        f"Doublet successful: "
        f"{int(doublet_success.sum()):,}"
    )

    print(
        f"Common successful:  "
        f"{len(indices):,}"
    )

    if len(indices) == 0:
        raise RuntimeError(
            "No sources were successfully "
            "localized by both methods"
        )

    # ---------------------------------------------------------------------
    # Extract quantities
    # ---------------------------------------------------------------------

    fr = fisher["results"][indices]
    dr = doublet["results"][indices]

    area_fisher = fr[
        "sky_area_90_deg2"
    ]

    area_doublet = dr[
        "sky_area_90_deg2"
    ]

    sigma_ra_fisher = fr[
        "sigma_ra_deg"
    ]

    sigma_ra_doublet = dr[
        "sigma_ra_deg"
    ]

    sigma_dec_fisher = fr[
        "sigma_dec_deg"
    ]

    sigma_dec_doublet = dr[
        "sigma_dec_deg"
    ]

    corr_fisher = fr[
        "correlation_ra_dec"
    ]

    corr_doublet = dr[
        "correlation_ra_dec"
    ]

    runtime_fisher = fr[
        "runtime_s"
    ]

    runtime_doublet = dr[
        "runtime_s"
    ]

    snr = dr[
        "snr_network"
    ]

    # ---------------------------------------------------------------------
    # Ratios
    # ---------------------------------------------------------------------

    area_ratio = (
        area_doublet
        / area_fisher
    )

    sigma_ra_ratio = (
        sigma_ra_doublet
        / sigma_ra_fisher
    )

    sigma_dec_ratio = (
        sigma_dec_doublet
        / sigma_dec_fisher
    )

    runtime_ratio = (
        runtime_doublet
        / runtime_fisher
    )

    # ---------------------------------------------------------------------
    # Numerical summary
    # ---------------------------------------------------------------------

    print("\n90% sky area [deg^2]")
    print("----------------------------------------")

    median, p16, p84 = percentile_summary(
        area_fisher
    )

    print(
        f"Fisher : median={median:.4f} "
        f"[{p16:.4f}, {p84:.4f}]"
    )

    median, p16, p84 = percentile_summary(
        area_doublet
    )

    print(
        f"Doublet: median={median:.4f} "
        f"[{p16:.4f}, {p84:.4f}]"
    )

    print("\nDoublet / Fisher")
    print("----------------------------------------")

    print_ratio_summary(
        "sky area",
        area_ratio,
    )

    print_ratio_summary(
        "sigma RA",
        sigma_ra_ratio,
    )

    print_ratio_summary(
        "sigma Dec",
        sigma_dec_ratio,
    )

    print_ratio_summary(
        "runtime",
        runtime_ratio,
    )

    print(
        "\nFraction Doublet area < Fisher area: "
        f"{np.mean(area_ratio < 1.0):.3f}"
    )

    print(
        "Fraction Doublet area > Fisher area: "
        f"{np.mean(area_ratio > 1.0):.3f}"
    )

    # ---------------------------------------------------------------------
    # Per-source table
    # ---------------------------------------------------------------------

    print("\nPer-source comparison")
    print(
        " index      SNR      "
        "Fisher area    Doublet area    "
        "D/F"
    )

    for (
        index,
        source_snr,
        af,
        ad,
        ratio,
    ) in zip(
        indices,
        snr,
        area_fisher,
        area_doublet,
        area_ratio,
    ):

        print(
            f"{index:6d} "
            f"{source_snr:8.2f} "
            f"{af:13.5f} "
            f"{ad:15.5f} "
            f"{ratio:8.3f}"
        )

    # ---------------------------------------------------------------------
    # Save numerical comparison
    # ---------------------------------------------------------------------

    comparison_dtype = np.dtype([
        ("source_index", "i8"),
        ("snr_network", "f8"),
        ("fisher_area90_deg2", "f8"),
        ("doublet_area90_deg2", "f8"),
        ("area_ratio_doublet_fisher", "f8"),
        ("fisher_sigma_ra_deg", "f8"),
        ("doublet_sigma_ra_deg", "f8"),
        ("sigma_ra_ratio", "f8"),
        ("fisher_sigma_dec_deg", "f8"),
        ("doublet_sigma_dec_deg", "f8"),
        ("sigma_dec_ratio", "f8"),
        ("fisher_corr_ra_dec", "f8"),
        ("doublet_corr_ra_dec", "f8"),
        ("fisher_runtime_s", "f8"),
        ("doublet_runtime_s", "f8"),
    ])

    comparison = np.zeros(
        len(indices),
        dtype=comparison_dtype,
    )

    comparison[
        "source_index"
    ] = indices

    comparison[
        "snr_network"
    ] = snr

    comparison[
        "fisher_area90_deg2"
    ] = area_fisher

    comparison[
        "doublet_area90_deg2"
    ] = area_doublet

    comparison[
        "area_ratio_doublet_fisher"
    ] = area_ratio

    comparison[
        "fisher_sigma_ra_deg"
    ] = sigma_ra_fisher

    comparison[
        "doublet_sigma_ra_deg"
    ] = sigma_ra_doublet

    comparison[
        "sigma_ra_ratio"
    ] = sigma_ra_ratio

    comparison[
        "fisher_sigma_dec_deg"
    ] = sigma_dec_fisher

    comparison[
        "doublet_sigma_dec_deg"
    ] = sigma_dec_doublet

    comparison[
        "sigma_dec_ratio"
    ] = sigma_dec_ratio

    comparison[
        "fisher_corr_ra_dec"
    ] = corr_fisher

    comparison[
        "doublet_corr_ra_dec"
    ] = corr_doublet

    comparison[
        "fisher_runtime_s"
    ] = runtime_fisher

    comparison[
        "doublet_runtime_s"
    ] = runtime_doublet

    comparison_path = (
        OUTPUT_DIR
        / (
            f"fisher_doublet_comparison_"
            f"{SOURCE_TYPE}_"
            f"z{ZI:.4f}_{ZF:.4f}.npz"
        )
    )

    np.savez_compressed(
        comparison_path,
        comparison=comparison,
    )

    # ---------------------------------------------------------------------
    # Plot 1: Fisher vs Doublet area
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6.5, 6.0)
    )

    ax.scatter(
        area_fisher,
        area_doublet,
        s=45,
        alpha=0.8,
    )

    positive = np.concatenate([
        area_fisher[
            area_fisher > 0
        ],
        area_doublet[
            area_doublet > 0
        ],
    ])

    minimum = positive.min()
    maximum = positive.max()

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--",
        linewidth=1.5,
        label="Fisher = Doublet",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"Fisher $A_{90}$ [deg$^2$]"
    )

    ax.set_ylabel(
        r"Doublet $A_{90}$ [deg$^2$]"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "fisher_vs_doublet_area90.png",
        dpi=200,
    )

    plt.close(fig)

    # ---------------------------------------------------------------------
    # Plot 2: area ratio versus SNR
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    ax.scatter(
        snr,
        area_ratio,
        s=45,
        alpha=0.8,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Network SNR"
    )

    ax.set_ylabel(
        r"$A_{90}^{\rm Doublet} / "
        r"A_{90}^{\rm Fisher}$"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "doublet_fisher_area_ratio_vs_snr.png",
        dpi=200,
    )

    plt.close(fig)

    # ---------------------------------------------------------------------
    # Plot 3: uncertainty ratios
    # ---------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    ax.scatter(
        indices,
        sigma_ra_ratio,
        s=40,
        label=r"$\sigma_{\rm RA}$",
    )

    ax.scatter(
        indices,
        sigma_dec_ratio,
        s=40,
        label=r"$\sigma_{\rm Dec}$",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Source index"
    )

    ax.set_ylabel(
        "Doublet / Fisher uncertainty"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "doublet_fisher_sigma_ratios.png",
        dpi=200,
    )

    plt.close(fig)

    print("\nSaved comparison:")
    print(comparison_path)

    print("\nPlots saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()