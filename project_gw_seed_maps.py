#!/usr/bin/env python3
"""Project host-catalogue objects into galaxy redshift-bin HEALPix maps.

The host input must contain ``/sources/<population>/ra``, ``dec`` and a
redshift field.  In the SkySim5000 host catalogue, ``ra``/``dec`` are degrees,
``redshift_true`` is the true redshift, and ``redshift`` is the observed
catalogue/photo-z value.  An NPZ injection catalogue is also accepted for
backwards compatibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import healpy as hp
import numpy as np


def _scalar(value):
    return np.asarray(value).item()


def read_map_metadata(path: Path, map_name: str):
    with h5py.File(path, "r") as data:
        zi = np.asarray(data["redshift_bins/zi"][:], dtype=float)
        zf = np.asarray(data["redshift_bins/zf"][:], dtype=float)
        footprint = np.asarray(data["masks/footprint"][:], dtype=bool)
        map_shape = data[f"maps/{map_name}"].shape
        nside = int(_scalar(data.attrs["nside"]))
        ordering = str(_scalar(data.attrs.get("ordering", "RING"))).upper()
        nest = bool(_scalar(data.attrs["nest"])) if "nest" in data.attrs else ordering in {"NEST", "NESTED"}

    if zi.ndim != 1 or zf.shape != zi.shape or not len(zi):
        raise ValueError("Invalid galaxy redshift-bin arrays")
    if np.any(~np.isfinite(zi)) or np.any(~np.isfinite(zf)) or np.any(zf <= zi):
        raise ValueError("Galaxy redshift bins must be finite and satisfy zf > zi")
    expected = (len(zi), hp.nside2npix(nside))
    if map_shape != expected:
        raise ValueError(f"Galaxy map shape {map_shape} disagrees with {expected}")
    if footprint.shape != (expected[1],) or not footprint.any():
        raise ValueError("Galaxy footprint has the wrong shape or is empty")
    return zi, zf, footprint, nside, nest


def read_host_catalogue(path: Path, population: str, redshift_field: str):
    prefix = f"sources/{population}"
    with h5py.File(path, "r") as data:
        required = [f"{prefix}/{name}" for name in ("ra", "dec", redshift_field)]
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"Missing host datasets: {missing}")
        ra = np.asarray(data[f"{prefix}/ra"][:], dtype=float) % 360.0
        dec = np.asarray(data[f"{prefix}/dec"][:], dtype=float)
        redshift = np.asarray(data[f"{prefix}/{redshift_field}"][:], dtype=float)
        galaxy_id = np.asarray(data[f"{prefix}/galaxy_id"][:]) if f"{prefix}/galaxy_id" in data else None
    return ra, dec, redshift, galaxy_id


def read_injection_npz(path: Path):
    with np.load(path, allow_pickle=False) as data:
        catalog = data["catalog"]
    required = ("ra_deg", "dec_deg", "redshift_true")
    missing = [name for name in required if name not in catalog.dtype.names]
    if missing:
        raise KeyError(f"Missing injection fields: {missing}")
    return (np.asarray(catalog["ra_deg"], dtype=float) % 360.0,
            np.asarray(catalog["dec_deg"], dtype=float),
            np.asarray(catalog["redshift_true"], dtype=float), None)


def validate_sources(ra, dec, redshift):
    if not (len(ra) == len(dec) == len(redshift)):
        raise ValueError("RA, Dec, and redshift arrays have inconsistent lengths")
    if (np.any(~np.isfinite(ra)) or np.any(~np.isfinite(dec)) or
            np.any(~np.isfinite(redshift))):
        raise ValueError("Source coordinates/redshifts contain non-finite values")
    if np.any((dec < -90) | (dec > 90)):
        raise ValueError("Declinations must be in [-90, 90] degrees")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-file", type=Path, help="Host HDF5 input")
    parser.add_argument("--injection-file", type=Path, help="Legacy NPZ input")
    parser.add_argument("--population", default="BBH", help="Host population group")
    parser.add_argument("--redshift-field", choices=("redshift_true", "redshift"), default="redshift_true")
    parser.add_argument("--galaxy-map", type=Path, required=True)
    parser.add_argument("--map-name", default="lsst_y5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.host_file is None) == (args.injection_file is None):
        parser.error("provide exactly one of --host-file or --injection-file")

    if args.host_file:
        ra, dec, redshift, galaxy_id = read_host_catalogue(args.host_file, args.population, args.redshift_field)
        input_description = f"{args.host_file} ({args.population}/{args.redshift_field})"
    else:
        ra, dec, redshift, galaxy_id = read_injection_npz(args.injection_file)
        input_description = str(args.injection_file)
    validate_sources(ra, dec, redshift)
    zi, zf, footprint, nside, nest = read_map_metadata(args.galaxy_map, args.map_name)
    pixels = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
    maps = np.zeros((len(zi), hp.nside2npix(nside)), dtype=np.float64)
    grouped = [[] for _ in zi]
    for source_index, z in enumerate(redshift):
        matches = np.flatnonzero((z >= zi) & (z < zf))
        if len(matches) > 1:
            raise ValueError(f"Redshift z={z} overlaps multiple bins")
        if len(matches) == 1:
            i = int(matches[0])
            maps[i, pixels[source_index]] += 1.0
            grouped[i].append(source_index)

    seed_counts = np.asarray([len(x) for x in grouped], dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as out:
        out.attrs.update(nside=nside, nest=nest, ordering="NESTED" if nest else "RING",
                         map_type="GW injection/host counts", input_file=str(input_description),
                         population=args.population, redshift_field=args.redshift_field,
                         coordinate_units="degrees", total_seeds=len(redshift),
                         projected_seeds=int(seed_counts.sum()))
        out.create_dataset("redshift_bins/zi", data=zi)
        out.create_dataset("redshift_bins/zf", data=zf)
        out.create_dataset("masks/footprint", data=footprint)
        out.create_dataset("maps/gw_seed_counts", data=maps, compression="gzip", shuffle=True)
        out.create_dataset("seed_counts", data=seed_counts)
        vlen = h5py.vlen_dtype(np.dtype("int64"))
        indices = out.create_dataset("seeds/source_indices", shape=(len(zi),), dtype=vlen)
        for i, group in enumerate(grouped):
            indices[i] = np.asarray(group, dtype=np.int64)
        out["seeds"].attrs["index_definition"] = "Index N maps to input catalogue row N"
        if galaxy_id is not None:
            gids = out.create_dataset("seeds/galaxy_id", shape=(len(zi),), dtype=vlen)
            for i, group in enumerate(grouped):
                gids[i] = np.asarray(galaxy_id[group], dtype=np.int64)
            out["seeds"].attrs["galaxy_id_definition"] = "Galaxy IDs corresponding to source_indices"
    print(f"Wrote {args.output}: {len(redshift)} input objects, {seed_counts.sum()} projected")


if __name__ == "__main__":
    main()
