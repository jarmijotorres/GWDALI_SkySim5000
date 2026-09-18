#!/usr/bin/env python3
"""Plot LSST galaxy overdensity with BBH host pixels overlaid."""

from pathlib import Path

import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


HOST_MAP = Path("/tmp/gw_seed_maps_BBH_true_z.hdf5")
GALAXY_MAP = Path("/home/jarmijo/GW-sirens-data/SkySim5000_photoz_number_count_maps_nside512.hdf5")
OUTPUT_DIR = Path("plots/bbh_vs_lsst_bins25_35_overlay_rot35m25")
BIN_START, BIN_STOP = 25, 35


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(HOST_MAP, "r") as data:
        zi = data["redshift_bins/zi"][:]
        zf = data["redshift_bins/zf"][:]
        host_maps = data["maps/gw_seed_counts"][:]
    with h5py.File(GALAXY_MAP, "r") as data:
        galaxy_maps = data["maps/lsst_y5"][:]
        nside = int(data.attrs["nside"])
        nest = bool(data.attrs["nest"])

    for index in range(BIN_START, BIN_STOP + 1):
        galaxy = np.asarray(galaxy_maps[index], dtype=float)
        positive = galaxy > 0
        mean = float(galaxy[positive].mean())
        log_delta = np.full(galaxy.shape, np.nan)
        log_delta[positive] = np.log10(galaxy[positive] / mean)
        values = log_delta[np.isfinite(log_delta)]
        p1, p99 = np.percentile(values, [1, 99])
        limit = max(abs(float(p1)), abs(float(p99)))

        occupied = np.flatnonzero(host_maps[index] > 0)
        lon, lat = hp.pix2ang(nside, occupied, nest=nest, lonlat=True)
        display = np.ma.masked_invalid(log_delta)

        fig = plt.figure(figsize=(16, 7), facecolor="white")
        titles = (
            f"LSST Y5 log galaxy overdensity: slice {index}\n"
            f"[{zi[index]:.4f}, {zf[index]:.4f})  mean={mean:.2f}",
            f"log galaxy overdensity + BBH hosts\n"
            f"{len(occupied)} occupied pixels, {int(host_maps[index].sum())} BBH hosts",
        )
        for subplot, title in enumerate(titles, 1):
            hp.orthview(
                display,
                fig=fig.number,
                sub=(1, 2, subplot),
                coord="C",
                half_sky=True,
                rot=(35, -25),
                title=title,
                norm="linear",
                min=-limit,
                max=limit,
                cmap="bwr",
                cbar=True,
            )
        if occupied.size:
            hp.projscatter(
                lon,
                lat,
                lonlat=True,
                coord="C",
                s=18,
                c="red",
                edgecolors="white",
                linewidths=0.25,
                alpha=0.95,
            )
        fig.suptitle(
            f"LSST Y5 log galaxy overdensity with BBH host occupation — bin {index}\n"
            f"log10(Ngal/mean Ngal), robust limits 1–99% "
            f"[{-limit:.3f}, {limit:.3f}], rotation=(35,-25)",
            fontsize=14,
        )
        fig.subplots_adjust(top=0.87, wspace=0.02)
        output = OUTPUT_DIR / (
            f"bbh_vs_lsst_y5_bin{index:02d}_overlay_"
            "orthview_rot35m25_halfsky.png"
        )
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
