#!/usr/bin/env python3
"""Make HEALPix maps directly from saved GWDALI Doublet samples.

Each posterior sample is assigned to a HEALPix pixel. Per-event pixel weights
are normalized to one, so summed maps integrate to the number of mapped events.
The 90-percent area is the exact discrete HPD area of the sampled map.
"""
from __future__ import annotations
import argparse
import os
import re
import tempfile
from pathlib import Path
import healpy as hp
import numpy as np

FREE_PARAMS = ("RA", "Dec", "inv_dL", "iota", "psi", "phi_coal")

def atomic_save(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}_", suffix=".npz")
    os.close(fd); tmp = Path(name)
    try:
        np.savez_compressed(tmp, **arrays); os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

def slices(path):
    a = np.loadtxt(path, dtype=float, ndmin=2)
    if a.shape[1] < 2 or np.any(a[:, 1] <= a[:, 0]):
        raise ValueError("redshift slice file must contain zi zf with zf > zi")
    return a[:, 0], a[:, 1]

def event_pixels(samples, nside, nest):
    # GWDALI samples store RA/Dec in degrees; healpy expects colatitude, longitude in radians.
    ra = np.radians(np.asarray(samples[:, 0], dtype=float)) % (2*np.pi)
    dec = np.clip(np.radians(np.asarray(samples[:, 1], dtype=float)), -np.pi/2, np.pi/2)
    pix = hp.ang2pix(nside, 0.5*np.pi-dec, ra, nest=nest)
    unique, counts = np.unique(pix, return_counts=True)
    weights = counts.astype(float) / float(len(pix))
    order = np.argsort(weights)[::-1]
    cumulative = np.cumsum(weights[order])
    n90 = int(np.searchsorted(cumulative, 0.9, side="left") + 1)
    area90 = n90 * hp.nside2pixarea(nside, degrees=True)
    return unique.astype(np.int64), weights, float(area90), n90

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True)
    ap.add_argument("--injection-file", required=True)
    ap.add_argument("--slice-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--slice-output-dir", required=True)
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--nest", action="store_true")
    args = ap.parse_args()
    if not hp.isnsideok(args.nside):
        ap.error("invalid NSIDE")
    with np.load(args.injection_file, allow_pickle=False) as f:
        catalogue = f["catalog"]
    zname = next((x for x in ("redshift_true", "redshift", "z") if x in catalogue.dtype.names), None)
    if zname is None:
        raise KeyError("catalogue has no redshift_true/redshift/z column")
    zi, zf = slices(args.slice_file)
    paths = sorted(Path(args.samples_dir).glob("source_*.npz"))
    npix = hp.nside2npix(args.nside)
    event_ids=[]; event_z=[]; event_areas=[]; sparse_p=[]; sparse_w=[]; offsets=[0]
    summed = np.zeros(npix); failed=[]
    for path in paths:
        match = re.fullmatch(r"source_(\d+)\.npz", path.name)
        if not match: continue
        index = int(match.group(1))
        try:
            with np.load(path, allow_pickle=False) as f:
                if tuple(f["free_params"].tolist()) != FREE_PARAMS: raise ValueError("FreeParams mismatch")
                samples = f["samples"]
            if samples.ndim != 2 or samples.shape[1] != 6 or len(samples) == 0: raise ValueError("invalid samples")
            pix, weights, area, _ = event_pixels(samples, args.nside, args.nest)
            summed[pix] += weights
            event_ids.append(index); event_z.append(float(catalogue[zname][index])); event_areas.append(area)
            sparse_p.append(pix); sparse_w.append(weights); offsets.append(offsets[-1] + len(pix))
        except Exception as exc:
            failed.append((index, f"{type(exc).__name__}: {exc}"))
    if not event_ids: raise ValueError("no valid Doublet sample files found")
    event_ids=np.asarray(event_ids, dtype=np.int64); event_z=np.asarray(event_z); event_areas=np.asarray(event_areas)
    sparse_p=np.concatenate(sparse_p); sparse_w=np.concatenate(sparse_w); offsets=np.asarray(offsets, dtype=np.int64)
    event_slice=np.full(len(event_ids), -1, dtype=np.int32)
    for k, (lo, hi) in enumerate(zip(zi, zf)):
        selected=(event_z >= lo) & (event_z < hi)
        if np.any(event_slice[selected] >= 0): raise ValueError("overlapping redshift slices")
        event_slice[selected]=k
        dense=np.zeros(npix)
        for j in np.flatnonzero(selected): dense[sparse_p[offsets[j]:offsets[j+1]]] += sparse_w[offsets[j]:offsets[j+1]]
        if selected.any():
            out=Path(args.slice_output_dir)/f"doublet_slice_{k:03d}_{lo:.8f}_{hi:.8f}_nside{args.nside}.npz"
            atomic_save(out, probability_map=dense, event_source_index=event_ids[selected], event_redshift_true=event_z[selected], number_of_events=np.asarray(selected.sum()), map_probability_sum=np.asarray(dense.sum()), nside=np.asarray(args.nside), nest=np.asarray(args.nest))
    atomic_save(Path(args.output), summed_probability_map=summed, event_source_index=event_ids, event_redshift_true=event_z, event_slice_index=event_slice, event_area_90_deg2=event_areas, sparse_pixel_index=sparse_p, sparse_probability=sparse_w, sparse_event_offset=offsets, failed_source_index=np.asarray([x[0] for x in failed], dtype=np.int64), failure_message=np.asarray([x[1] for x in failed], dtype="U512"), nside=np.asarray(args.nside), nest=np.asarray(args.nest), number_of_events=np.asarray(len(event_ids)), map_probability_sum=np.asarray(summed.sum()), pixel_area_deg2=np.asarray(hp.nside2pixarea(args.nside, degrees=True)), free_params=np.asarray(FREE_PARAMS), probability_normalisation=np.asarray("unit sum per event"), credible_area_definition=np.asarray("discrete HPD pixel area from Doublet samples"))
    print(f"Mapped {len(event_ids)} events; failed={len(failed)}; map sum={summed.sum():.8g}")
    print(f"Saved: {args.output}")

if __name__ == "__main__": main()
