# Metadata inventory: `~/GW-sirens-data/`

Inspected 2026-09-18. Source files were read only; no source file was changed.

## Pipeline interpretation

The files correspond to this chain:

```text
gw_source_galaxy_targets.hdf5
        -> SkySim5000_gw_host_catalogue1.hdf5
        -> skysim5000_BBH_gwdali_z0.0000_0.8000.npz
        -> skysim5000_BBH_lvk_snr*.npz
```

`SkySim5000_photoz_number_count_maps_nside512.hdf5` is a galaxy-map input for
photometric-redshift slices and footprint construction, not a GW-event product.

## Files

### `gw_source_galaxy_targets.hdf5`

Rate-target product, created `2026-08-11T12:20:48.972083+00:00`. It records
the merger-rate density as a function of redshift and stellar mass, then uses
that density to generate expected and Poisson-drawn event counts.

- 69 redshift bins (`zi <= redshift_true < zf`) × 25 log-stellar-mass bins.
- Contains expected counts and independent Poisson draws for BBH, BHNS, and BNS.
- Drawn totals: BBH 20,701; BHNS 14,256; BNS 593,957.
- Cosmology: `FlatLambdaCDM(H0=71.0, Om0=0.265)`; observing time 10 years.
- This is an upstream target/count file, not a source-event catalogue.

### `SkySim5000_gw_host_catalogue1.hdf5`

Assigned host catalogue, created `2026-08-11T13:09:16.011596+00:00`.

- Populations: `sources/BBH`, `sources/BHNS`, `sources/BNS`.
- Selected rows: BBH 19,819; BHNS 13,143; BNS 572,928.
- Each population includes `galaxy_id`, `ra`, `dec`, `redshift`,
  `redshift_true`, stellar/halo mass, bin indices, and selection priority.
- Host `ra`/`dec` values are in degrees (consistent with the source code and
  the observed numeric ranges); `redshift_true` is dimensionless.
- No replacement sampling was recorded.
- This is the portable seed-host input to the preparation stage.

### `SkySim5000_photoz_number_count_maps_nside512.hdf5`

Photometric-redshift galaxy count maps, created `2026-08-11T13:34:21.991498+00:00`.

- HEALPix NSIDE 512, RING ordering, 3,145,728 pixels, pixel area
  `0.0131139632 deg²`.
- `maps/lsst_y1` and `maps/lsst_y5`: 69 × NSIDE-512 integer count maps.
- Slice redshift is observed/photometric redshift; bins use
  `zi <= z < zf`. This is distinct from GW membership, which uses injected
  `redshift_true`.
- `masks/footprint` is the accumulated LSST Y5 occupation footprint: 395,007
  pixels / 5,180.1073 deg².
- Coordinates are equatorial RA/Dec in degrees.

### `skysim5000_BBH_gwdali_z0.0000_0.8000.npz`

Prepared/final BBH event catalogue according to the legacy workflow. It
contains 3,298 rows and the structured `catalog`
fields:

`m1`, `m2`, `RA`, `Dec`, `psi`, `t_coal`, `phi_coal`, `dL`, `iota`, spin
components, `redshift_true`, `ra_deg`, `dec_deg`, and source-frame masses.

- `RA`, `Dec`, `psi`, `iota`, and other angles are stored in radians
  (`angle_unit=radian`).
- `ra_deg` and `dec_deg` are retained degree copies. They agree with
  `degrees(RA)` and `degrees(Dec)` to floating-point precision.
- `dL` is in Gpc; `m1`/`m2` are detector-frame solar masses.
- `redshift_true` spans 0.0541–0.79995; this is the correct field for assigning
  the event to a true-redshift slice, not a posterior distance sample.
- The file itself has no SNR fields; the associated SNR calculation is recorded
  in the separate SNR product. In the legacy workflow this NPZ is the retained
  event list after applying the valid SNR selection.

### `skysim5000_BBH_lvk_snr_radec_deg_z0.0000_0.8000.npz`

Completed LVK SNR catalogue for all 3,298 prepared rows.

- Contains per-detector SNRs, network SNR, detector-count threshold, `detected`,
  `success`, and `completed`.
- Configuration: IMRPhenomXPHM, 10–2048 Hz, `fsize=3000`, KAGRA included;
  network threshold 12, single-detector threshold 4, minimum 2 detectors.
- Explicit provenance: input RA/Dec stored in radians, passed to GWDALI in
  degrees (`coordinate_convention=gwdali_ra_dec_degrees_v1`). This is the
  mixed-unit run to discard, per the legacy provenance clarification.
- All 3,298 rows completed successfully; 303 satisfy the detection mask.

### `skysim5000_BBH_lvk_snr_z0.0000_0.8000.npz`

Also a completed LVK SNR catalogue for the same 3,298 rows and same nominal
configuration, but without unit/convention metadata.

- It is not a duplicate of the `_radec_deg` file: source indices, completion,
  and success flags match, but all SNR columns differ and the detection count is
  199 rather than 303.
- The difference is therefore consistent with a changed calculation, with the
  documented RA/Dec conversion being the main provenance distinction. Treat the
  `_radec_deg` file as the better-documented candidate, but retain both until a
  numerical rerun confirms the intended legacy convention.

## ID and angle conclusions

- The prepared injection `catalog` has 3,298 rows indexed `0..3297`.
- Both SNR products carry `results/source_index`, and it is exactly `0..3297`.
- Per the README/docs, these are prepared-catalogue row indices—not original
  HDF5 row numbers and not galaxy IDs. The prepared NPZ does not retain the
  original `galaxy_id`, so an exact host-ID join cannot be recovered from these
  files alone.
- The strongest visible unit check passes: stored injection RA/Dec are radians,
  degree copies are consistent, and the `_radec_deg` SNR product records the
  radian-to-degree conversion explicitly.

## BBH host/injection/SNR cross-check

The three BBH products can be joined reliably, with one important limitation:
the GWDALI NPZ does not retain `galaxy_id`, so the join is established through
the preserved row order and positions.

- The host HDF5 contains exactly 3,298 BBH rows satisfying
  `0 <= redshift_true < 0.8`.
- The GWDALI NPZ contains exactly 3,298 rows.
- In the same order, every GWDALI `ra_deg`, `dec_deg`, and `redshift_true`
  value is identical to the corresponding host-HDF5 value (maximum absolute
  difference: zero in the stored arrays).
- The SNR NPZ also contains 3,298 rows with `source_index=0..3297`, so
  `SNR results[i]` maps to GWDALI `catalog[i]` and the matching host-HDF5 row.
- The SNR file has 303 rows marked detected and all 3,298 rows completed
  successfully. Those 303 can therefore be mapped back to the host
  `galaxy_id` using the host BBH row selected by the same index.

Conceptually, for this dataset:

```text
host BBH row i
    == GWDALI catalog[i] (same RA, Dec, true redshift)
    == SNR results[i] (source_index=i)
```

This mapping is valid for the inspected files, but it depends on the prepared
catalogue preserving order. It should be rechecked if the NPZ is regenerated
or filtered. The 303 detection flags above come from the mixed-unit SNR run
that is now being discarded, so they should not be treated as valid science
results.
