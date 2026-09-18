#!/usr/bin/env python3
"""Plot summed BBH--LSST correlation matrices by angular separation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT = Path("CrossCorrelation/ConfigurationSpace/results/bbh_lsst_y1_nk_fwhm15arcmin.npz")
OUTPUT_MATRIX = Path("CrossCorrelation/ConfigurationSpace/results/bbh_lsst_y1_nk_matrix_4x31x31.npz")
OUTPUT_PLOT = Path("CrossCorrelation/ConfigurationSpace/results/bbh_lsst_y1_nk_matrix_4x31x31.png")
OUTPUT_PLOT_Z = Path("CrossCorrelation/ConfigurationSpace/results/bbh_lsst_y1_nk_matrix_4x31x31_zmeans.png")
FIRST_ZBIN = 4
LAST_ZBIN = 35  # exclusive; retain original bins 4..34


def main():
    with np.load(INPUT, allow_pickle=False) as data:
        xi = np.asarray(data["xi"][FIRST_ZBIN:LAST_ZBIN, FIRST_ZBIN:LAST_ZBIN], dtype=float)
        theta_edges = np.asarray(data["theta_edges"], dtype=float)
        zi = np.asarray(data["zi"][FIRST_ZBIN:LAST_ZBIN], dtype=float)
        zf = np.asarray(data["zf"][FIRST_ZBIN:LAST_ZBIN], dtype=float)

    n_zbins = LAST_ZBIN - FIRST_ZBIN

    # Four TreeCorr bins per one-degree interval; the first interval starts
    # at the estimator's minimum separation of 0.25 degrees.
    groups = [
        (0.25, 1.00, np.arange(0, 3)),
        (1.00, 2.00, np.arange(3, 7)),
        (2.00, 3.00, np.arange(7, 11)),
        (3.00, 4.00, np.arange(11, 15)),
    ]
    matrices = np.full((4, n_zbins, n_zbins), np.nan)
    for index, (_, _, angular_bins) in enumerate(groups):
        values = xi[:, :, angular_bins]
        valid = np.any(np.isfinite(values), axis=2)
        summed = np.nansum(values, axis=2)
        matrices[index][valid] = summed[valid]

    labels = np.asarray([f"{lo:g}–{hi:g} deg" for lo, hi, _ in groups])
    np.savez_compressed(
        OUTPUT_MATRIX,
        matrices=matrices,
        separation_labels=labels,
        separation_ranges=np.asarray([[lo, hi] for lo, hi, _ in groups]),
        zi=zi,
        zf=zf,
        theta_edges=theta_edges,
        definition=np.asarray("sum of TreeCorr xi over angular bins"),
    )

    zcenters = 0.5 * (zi + zf)
    xedges = np.r_[zi[0], zf]
    yedges = xedges.copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for ax, matrix, label in zip(axes.flat, matrices, labels):
        finite = matrix[np.isfinite(matrix)]
        limit = max(abs(np.percentile(finite, 1)), abs(np.percentile(finite, 99)))
        image = ax.imshow(matrix, origin="lower", cmap="bwr", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(f"{label} — summed xi")
        ax.set_xlabel("LSST photo-z bin")
        ax.set_ylabel("BBH true-z bin")
        ax.set_xticks([0, 10, 20, 30])
        ax.set_yticks([0, 10, 20, 30])
        ax.set_xlabel("LSST photo-z bin (original +4)")
        ax.set_ylabel("BBH true-z bin (original +4)")
        fig.colorbar(image, ax=ax, shrink=0.82, label="summed TreeCorr xi")
    fig.suptitle("BBH–LSST Y1 cross-correlation matrix", fontsize=15)
    fig.savefig(OUTPUT_PLOT, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for ax, matrix, label in zip(axes.flat, matrices, labels):
        finite = matrix[np.isfinite(matrix)]
        limit = max(abs(np.percentile(finite, 1)), abs(np.percentile(finite, 99)))
        image = ax.pcolormesh(
            xedges,
            yedges,
            matrix,
            cmap="bwr",
            vmin=-limit,
            vmax=limit,
            shading="flat",
        )
        ax.set_title(f"{label} — summed xi")
        ax.set_xlabel("LSST photo-z bin centre")
        ax.set_ylabel("BBH true-z bin centre")
        ticks = zcenters[::5]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{x:.2f}" for x in ticks])
        ax.set_yticklabels([f"{x:.2f}" for x in ticks])
        fig.colorbar(image, ax=ax, shrink=0.82, label="summed TreeCorr xi")
    fig.suptitle("BBH–LSST Y1 cross-correlation matrix (redshift coordinates)", fontsize=15)
    fig.savefig(OUTPUT_PLOT_Z, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_MATRIX}")
    print(f"wrote {OUTPUT_PLOT}")
    print(f"wrote {OUTPUT_PLOT_Z}")
    print("shape", matrices.shape, "robust colour limit", limit)


if __name__ == "__main__":
    main()
