# BBH–LSST angular cross-correlation estimator

This document defines the first test estimator for the relation between the
SkySim5000 BBH host positions and LSST galaxy-count maps. It supports both
same-redshift and cross-redshift measurements.

## Goal

For BBH host redshift slice `i` and LSST galaxy/photo-z slice `j`, measure
whether BBH hosts preferentially occupy angular regions with above- or
below-average galaxy density.

The estimator is evaluated for all pairs `(i, j)`:

- same-redshift: `i = j`;
- cross-redshift: `i != j`.

The cross-redshift matrix is scientifically useful because it shows the effect
of the photo-z selection, redshift overlap, survey footprint, and any broad
angular structure shared by the slices.

## Input products

### BBH hosts

`SkySim5000_gw_host_catalogue1.hdf5`:

```text
/sources/BBH/ra
/sources/BBH/dec
/sources/BBH/redshift_true
```

The host catalogue is used as a point catalogue. It is not converted into a
dense NSIDE-512 host map for the correlation estimator. This avoids a map whose
occupation is almost entirely zero.

### LSST galaxy maps

`SkySim5000_photoz_number_count_maps_nside512.hdf5`:

```text
/maps/lsst_y5
/redshift_bins/zi
/redshift_bins/zf
/masks/footprint
```

The galaxy map and host coordinates use equatorial RA/Dec in degrees. The
galaxy slices are selected using observed/photometric redshift, while the BBH
host slices are selected using injected `redshift_true`.

## Galaxy field

For each LSST slice `j`, smooth the galaxy count map with a Gaussian kernel of
FWHM 15 arcmin (`0.25 deg`). The smoothing is a test choice and should be
applied consistently to every galaxy slice.

Let `N_j(p)` be the smoothed galaxy count in pixel `p`, and let `M(p)` be the
common footprint mask. Define the mean only over valid footprint pixels:

```text
mean_j = mean[N_j(p) | M(p) and finite N_j(p)]
```

The scalar field supplied to TreeCorr is:

```text
k_j(p) = log10[N_j(p) / mean_j]
```

This is equivalent to `log10(1 + delta_g)`. Pixels with zero or invalid values
are excluded from the galaxy `K` catalogue rather than assigned an artificial
large negative value.

The same galaxy field can also be tested without the logarithm as a diagnostic.
The logarithmic field is preferred for the first run because the galaxy counts
have a broad dynamic range.

## BBH point catalogue

For each BBH slice `i`, select:

```text
zi[i] <= redshift_true < zf[i]
```

Create a TreeCorr `Catalog` using the selected RA/Dec positions in degrees.
The BBH hosts remain individual points. With the current catalogue, bins
25–35 contain only approximately 86–223 BBH hosts each, so the estimator will
have substantial counting noise.

## TreeCorr estimator

Use TreeCorr `NKCorrelation`:

```text
N = BBH host point catalogue in slice i
K = smoothed LSST scalar field k_j at valid pixel centres
```

For angular separation `theta`, the basic measured quantity is the weighted
mean galaxy overdensity around BBH hosts:

```text
w_NK(i,j,theta) = < k_j(theta) around BBH hosts in slice i >
```

This is a host–galaxy cross-correlation with the galaxy field already
mean-subtracted in logarithmic form. It is not the same normalization as a
traditional pair-count `w(theta)` with two point catalogues, so the output
should be labelled explicitly as `host_to_loggalaxy_overdensity`.

Recommended initial angular bins:

```text
min_sep = 0.25 deg
max_sep = 30 deg
bin_size = 0.2 in log(theta)
sep_units = deg
```

The minimum scale matches the 15 arcmin smoothing scale. Smaller bins would
not add reliable information in this first sparse-host test.

## Same- and cross-redshift products

For every BBH slice `i` and LSST slice `j`, save:

```text
theta[i,j,:]
w_host_loggalaxy[i,j,:]
weight[i,j,:]
npairs[i,j,:]
```

Also save the slice edges, smoothing scale, map name, NSIDE, footprint
definition, host redshift field, galaxy redshift interpretation, and estimator
version as metadata.

The diagonal `i=j` is the same-redshift result. Off-diagonal entries are the
cross-redshift results. Do not force the two redshift definitions to be equal:
the host field uses true injected redshift and the LSST field uses photo-z.

## Null tests and uncertainties

The estimator should not be interpreted from one measurement alone. At minimum
run:

1. **Host shuffle:** randomly permute BBH RA/Dec while preserving the BBH
   redshift-slice counts.
2. **Random rotations:** rotate all BBH positions by random sky rotations and
   recompute the full `(i,j)` matrix.
3. **Footprint jackknife:** divide the footprint into spatial regions and
   recompute while omitting one region at a time.
4. **Smoothing test:** compare 15 arcmin with unsmoothed and, if useful,
   30 arcmin fields.
5. **Field definition test:** compare `log10(N/mean(N))` with the linear
   overdensity `N/mean(N)-1`.

The random/shuffle measurements provide the null mean and covariance. The
jackknife provides a spatial-structure uncertainty estimate, although with
fewer than a few hundred hosts per slice it will itself be noisy.

## Interpretation cautions

- The host catalogue contains selected possible GW hosts, not an unbiased
  galaxy sample. The correlation tests the selection against the galaxy field;
  it does not directly measure the underlying cosmological galaxy bias.
- The galaxy maps are photo-z selected, while hosts are true-z selected.
  Cross-redshift signal is therefore expected from photo-z leakage and broad
  angular footprint structure.
- The 15 arcmin smoothing scale is a starting test, not a validated physical
  scale.
- The mixed-unit SNR catalogue is irrelevant to this estimator and should not
  be used to define the host positions. Use the host HDF5 positions directly.

## First implementation sequence

1. Load the map bins, LSST Y5 map, and footprint.
2. Smooth every LSST slice with 15 arcmin Gaussian smoothing.
3. Build one LSST `K` catalogue per slice from valid pixel centres.
4. Load BBH RA/Dec and `redshift_true` from the host HDF5.
5. Build one BBH `N` catalogue per true-redshift slice.
6. Run TreeCorr `NKCorrelation` for every `(i,j)` pair.
7. Save the full cross-redshift result matrix and metadata.
8. Run shuffled-host null tests before interpreting the diagonal.
