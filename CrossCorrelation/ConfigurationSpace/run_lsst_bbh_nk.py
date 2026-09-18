#!/usr/bin/env python3
"""Measure BBH-host / LSST-map angular cross-correlations with TreeCorr.

The fixed test configuration is LSST Y5, 15 arcmin Gaussian smoothing, the
stored footprint mask, and angular bins 0.25--4 degrees in 0.25 degree steps.
Hosts are TreeCorr N catalogues; smoothed LSST log-overdensity pixels are K
catalogues. The output contains every BBH true-z slice by LSST photo-z slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import treecorr


SMOOTHING_FWHM_ARCMIN = 15.0
MIN_SEP_DEG = 0.25
MAX_SEP_DEG = 4.0
BIN_SIZE_DEG = 0.25
DEFAULT_MAP_NAME = "lsst_y1"


def read_inputs(host_file: Path, galaxy_map: Path, population: str, map_name: str):
    with h5py.File(galaxy_map, "r") as data:
        zi = np.asarray(data["redshift_bins/zi"][:], dtype=float)
        zf = np.asarray(data["redshift_bins/zf"][:], dtype=float)
        galaxy_maps = data[f"maps/{map_name}"].shape
        footprint = np.asarray(data["masks/footprint"][:], dtype=bool)
        nside = int(np.asarray(data.attrs["nside"]).item())
        nest = bool(np.asarray(data.attrs["nest"]).item())
        if galaxy_maps != (len(zi), hp.nside2npix(nside)):
            raise ValueError("Galaxy map shape does not match redshift bins and NSIDE")

    with h5py.File(host_file, "r") as data:
        prefix = f"sources/{population}"
        ra = np.asarray(data[f"{prefix}/ra"][:], dtype=float) % 360.0
        dec = np.asarray(data[f"{prefix}/dec"][:], dtype=float)
        redshift = np.asarray(data[f"{prefix}/redshift_true"][:], dtype=float)

    if not (len(ra) == len(dec) == len(redshift)):
        raise ValueError("Host RA, Dec, and redshift lengths differ")
    if footprint.shape != (hp.nside2npix(nside),):
        raise ValueError("Footprint length does not match NSIDE")
    return zi, zf, footprint, nside, nest, ra, dec, redshift


def make_galaxy_catalog(count_map, footprint, nside, nest):
    """Smooth one count map and return pixel-centre coordinates plus log delta."""
    smoothed = hp.smoothing(
        np.asarray(count_map, dtype=float),
        fwhm=np.deg2rad(SMOOTHING_FWHM_ARCMIN / 60.0),
        verbose=False,
    )
    valid = footprint & np.isfinite(smoothed) & (smoothed > 0)
    if not np.any(valid):
        return None, np.nan, 0
    mean = float(smoothed[valid].mean())
    k = np.log10(smoothed[valid] / mean)
    pixels = np.flatnonzero(valid)
    ra, dec = hp.pix2ang(nside, pixels, nest=nest, lonlat=True)
    catalog = treecorr.Catalog(
        ra=ra,
        dec=dec,
        k=k,
        ra_units="deg",
        dec_units="deg",
    )
    return catalog, mean, int(valid.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-file", type=Path, required=True)
    parser.add_argument("--galaxy-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", default="BBH")
    parser.add_argument("--map-name", default=DEFAULT_MAP_NAME, choices=("lsst_y1", "lsst_y5"))
    args = parser.parse_args()

    zi, zf, footprint, nside, nest, ra, dec, redshift = read_inputs(
        args.host_file, args.galaxy_map, args.population, args.map_name
    )
    nbin = len(zi)
    theta_edges = np.arange(MIN_SEP_DEG, MAX_SEP_DEG + BIN_SIZE_DEG, BIN_SIZE_DEG)
    ntheta = len(theta_edges) - 1
    host_catalogues = []
    for i in range(nbin):
        selected = (redshift >= zi[i]) & (redshift < zf[i])
        host_catalogues.append(
            treecorr.Catalog(
                ra=ra[selected], dec=dec[selected], ra_units="deg", dec_units="deg"
            ) if np.any(selected) else None
        )

    galaxy_catalogues = []
    galaxy_means = np.zeros(nbin)
    galaxy_pixels = np.zeros(nbin, dtype=int)
    with h5py.File(args.galaxy_map, "r") as data:
        galaxy_maps = data[f"maps/{args.map_name}"]
        for j in range(nbin):
            cat, mean, npix = make_galaxy_catalog(galaxy_maps[j], footprint, nside, nest)
            galaxy_catalogues.append(cat)
            galaxy_means[j] = mean
            galaxy_pixels[j] = npix
            print(f"prepared LSST slice {j + 1}/{nbin}")

    theta = np.full((nbin, nbin, ntheta), np.nan)
    xi = np.full_like(theta, np.nan)
    weight = np.zeros_like(theta)
    npairs = np.zeros((nbin, nbin, ntheta), dtype=np.int64)
    host_counts = np.zeros(nbin, dtype=np.int64)

    config = {
        "min_sep": MIN_SEP_DEG,
        "max_sep": MAX_SEP_DEG,
        "bin_size": BIN_SIZE_DEG,
        "sep_units": "deg",
        "bin_type": "Linear",
    }
    for i, host_cat in enumerate(host_catalogues):
        host_counts[i] = 0 if host_cat is None else host_cat.nobj
        if host_cat is None:
            continue
        for j, galaxy_cat in enumerate(galaxy_catalogues):
            if galaxy_cat is None:
                continue
            corr = treecorr.NKCorrelation(config)
            corr.process(host_cat, galaxy_cat)
            theta[i, j] = corr.meanr
            xi[i, j] = corr.xi
            weight[i, j] = corr.weight
            npairs[i, j] = corr.npairs
        print(f"correlated host slice {i + 1}/{nbin} ({host_counts[i]} hosts)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "estimator": "TreeCorr NKCorrelation",
        "population": args.population,
        "galaxy_map": str(args.galaxy_map),
        "host_file": str(args.host_file),
        "galaxy_map_name": args.map_name,
        "host_redshift_field": "redshift_true",
        "galaxy_redshift_definition": "observed/photo-z",
        "smoothing_fwhm_arcmin": SMOOTHING_FWHM_ARCMIN,
        "galaxy_field": "log10(smoothed_galaxy_count / footprint_mean)",
        "nside": nside,
        "ordering": "NESTED" if nest else "RING",
        "min_sep_deg": MIN_SEP_DEG,
        "max_sep_deg": MAX_SEP_DEG,
        "bin_size_deg": BIN_SIZE_DEG,
        "null_tests": "not included in this initial estimator",
    }
    np.savez_compressed(
        args.output,
        theta=theta,
        xi=xi,
        weight=weight,
        npairs=npairs,
        theta_edges=theta_edges,
        host_counts=host_counts,
        galaxy_means=galaxy_means,
        galaxy_valid_pixels=galaxy_pixels,
        zi=zi,
        zf=zf,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
