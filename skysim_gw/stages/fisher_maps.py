"""Build sparse per-event and summed HEALPix GW localisation maps.

Each successful Fisher localisation is represented by a Gaussian on the local
east--north tangent plane. Every event posterior is normalised to unit sum.
The summed map therefore integrates to the number of included detections.

The injection catalogue stores RA and Dec in radians. GWDALI's marginalised
RA--Dec covariance is in degree squared because it differentiates those two
parameters in degrees.
"""

from pathlib import Path
import os
import tempfile
import time

import healpy as hp
import numpy as np


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

from skysim_gw.settings.fisher_maps import (
    BASE_DIR,
    INPUT_DIR,
    LOCALISATION_DIR,
    OUTPUT_DIR,
    SLICE_OUTPUT_DIR,
    REDSHIFT_SLICE_FILE,
    SOURCE_TYPE,
    ZI,
    ZF,
    NSIDE,
    NEST,
    NSIGMA,
    MAX_EVENTS,
    PRINT_EVERY,
)


# Retain posterior support through Mahalanobis radius^2 <= NSIGMA^2.
# Six sigma loses a negligible fraction of a two-dimensional Gaussian.

# Set to an integer for a small test, or None for every successful event.


def injection_path():
    return INPUT_DIR / (
        f"skysim5000_{SOURCE_TYPE}_gwdali_z{ZI:.4f}_{ZF:.4f}.npz"
    )


def localisation_path():
    return LOCALISATION_DIR / (
        f"skysim5000_{SOURCE_TYPE}_lvk_fisher_tc_fixed_radec_deg_"
        f"localisation_z{ZI:.4f}_{ZF:.4f}.npz"
    )


def output_path():
    ordering = "nested" if NEST else "ring"
    return OUTPUT_DIR / (
        f"skysim5000_{SOURCE_TYPE}_lvk_gw_probability_maps_"
        f"nside{NSIDE}_{ordering}_z{ZI:.4f}_{ZF:.4f}.npz"
    )


def slice_output_path(slice_index, zi, zf):
    ordering = "nested" if NEST else "ring"
    return SLICE_OUTPUT_DIR / (
        f"skysim5000_{SOURCE_TYPE}_lvk_gw_probability_"
        f"slice_{slice_index:03d}_{zi:.4f}_{zf:.4f}_"
        f"nside{NSIDE}_{ordering}.npz"
    )


def load_redshift_slices(path):
    """Read a two-column zi, zf table without converting it to edge vectors."""
    table = np.loadtxt(path, dtype=float, ndmin=2)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"Expected at least two columns (zi, zf) in {path}")
    zi = np.asarray(table[:, 0], dtype=float)
    zf = np.asarray(table[:, 1], dtype=float)
    if not np.all(np.isfinite(zi)) or not np.all(np.isfinite(zf)):
        raise ValueError("Redshift slice table contains non-finite values")
    if np.any(zf <= zi):
        raise ValueError("Every redshift slice must satisfy zf > zi")
    if np.any(zi[1:] < zi[:-1]) or np.any(zf[1:] < zf[:-1]):
        raise ValueError("Redshift slices must be ordered")
    if np.any(zi[1:] < zf[:-1] - 1.0e-12):
        raise ValueError("Redshift slices overlap")
    return zi, zf


def load_inputs(injection_file, localisation_file):
    with np.load(injection_file, allow_pickle=False) as data:
        if "catalog" not in data.files:
            raise KeyError(f"Missing 'catalog' in {injection_file}")
        catalogue = data["catalog"]

    with np.load(localisation_file, allow_pickle=False) as data:
        required = {"results", "covariance", "completed", "free_params"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(
                f"Missing localisation fields {sorted(missing)} in "
                f"{localisation_file}"
            )
        results = data["results"]
        covariance = data["covariance"]
        completed = data["completed"].astype(bool)
        free_params = tuple(data["free_params"].tolist())

    if len(catalogue) != len(results) or len(results) != len(covariance):
        raise ValueError("Injection and localisation row counts do not match")
    if "RA" not in catalogue.dtype.names or "Dec" not in catalogue.dtype.names:
        raise KeyError("Injection catalogue must contain RA and Dec")
    if "RA" not in free_params or "Dec" not in free_params:
        raise ValueError("Localisation FreeParams must contain RA and Dec")
    return catalogue, results, covariance, completed, free_params


def redshift_column(catalogue):
    for name in ("redshift_true", "redshift", "z"):
        if name in catalogue.dtype.names:
            return name
    return None


def sky_covariance_east_north(covariance, free_params, dec_rad):
    """Transform the coordinate covariance to local east--north degree^2."""
    i_ra = free_params.index("RA")
    i_dec = free_params.index("Dec")
    coordinate_covariance = np.asarray(
        covariance[np.ix_([i_ra, i_dec], [i_ra, i_dec])], dtype=float
    )
    coordinate_covariance = 0.5 * (
        coordinate_covariance + coordinate_covariance.T
    )
    jacobian = np.diag([np.cos(dec_rad), 1.0])
    tangent_covariance = jacobian @ coordinate_covariance @ jacobian.T
    eigenvalues = np.linalg.eigvalsh(tangent_covariance)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
        raise ValueError(
            f"Invalid tangent-plane covariance eigenvalues {eigenvalues}"
        )
    return tangent_covariance, eigenvalues


def local_gnomonic_offsets_deg(pixel_vectors, ra_rad, dec_rad):
    """Return exact gnomonic east/north offsets around one sky position."""
    cos_ra = np.cos(ra_rad)
    sin_ra = np.sin(ra_rad)
    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)

    centre = np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])
    east = np.array([-sin_ra, cos_ra, 0.0])
    north = np.array([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec])

    denominator = centre @ pixel_vectors
    valid = denominator > 0.0
    east_offset = np.full(pixel_vectors.shape[1], np.nan)
    north_offset = np.full(pixel_vectors.shape[1], np.nan)
    radians_to_degrees = 180.0 / np.pi
    east_offset[valid] = (
        (east @ pixel_vectors[:, valid]) / denominator[valid]
        * radians_to_degrees
    )
    north_offset[valid] = (
        (north @ pixel_vectors[:, valid]) / denominator[valid]
        * radians_to_degrees
    )
    return east_offset, north_offset, valid


def event_posterior_pixels(ra_rad, dec_rad, covariance, free_params):
    """Return retained pixel indices and unit-normalised posterior masses."""
    tangent_covariance, eigenvalues = sky_covariance_east_north(
        covariance, free_params, dec_rad
    )
    inverse_covariance = np.linalg.inv(tangent_covariance)

    # Convert the largest tangent-plane radius to an angular query radius.
    maximum_tangent_radius_deg = NSIGMA * np.sqrt(eigenvalues[-1])
    query_radius = np.arctan(np.radians(maximum_tangent_radius_deg))
    query_radius += hp.max_pixrad(NSIDE)
    if query_radius >= 0.5 * np.pi:
        raise ValueError(
            "Localisation is too broad for the tangent-plane approximation"
        )

    centre_vector = hp.ang2vec(0.5 * np.pi - dec_rad, ra_rad)
    pixels = hp.query_disc(
        NSIDE,
        centre_vector,
        query_radius,
        inclusive=True,
        nest=NEST,
    )
    pixel_vectors = np.asarray(hp.pix2vec(NSIDE, pixels, nest=NEST))
    east, north, valid = local_gnomonic_offsets_deg(
        pixel_vectors, ra_rad, dec_rad
    )
    offsets = np.column_stack((east, north))
    mahalanobis_squared = np.full(len(pixels), np.inf)
    mahalanobis_squared[valid] = np.einsum(
        "ni,ij,nj->n",
        offsets[valid],
        inverse_covariance,
        offsets[valid],
        optimize=True,
    )
    retained = np.isfinite(mahalanobis_squared) & (
        mahalanobis_squared <= NSIGMA**2
    )
    pixels = pixels[retained]
    weights = np.exp(-0.5 * mahalanobis_squared[retained])
    normalisation = float(weights.sum())
    if not np.isfinite(normalisation) or normalisation <= 0.0:
        raise ValueError("Event posterior has zero or invalid normalisation")
    weights /= normalisation
    return pixels.astype(np.int64, copy=False), weights


def save_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}_", suffix=".npz"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def assign_complete_redshift_slices(event_redshift, slice_zi, slice_zf):
    """Assign events only to slices fully covered by the input z interval."""
    tolerance = 1.0e-12
    fully_covered = (
        (slice_zi >= ZI - tolerance)
        & (slice_zf <= ZF + tolerance)
    )
    event_slice = np.full(len(event_redshift), -1, dtype=np.int32)
    for slice_index in np.flatnonzero(fully_covered):
        selected = (
            np.isfinite(event_redshift)
            & (event_redshift >= slice_zi[slice_index])
            & (event_redshift < slice_zf[slice_index])
        )
        if np.any(event_slice[selected] >= 0):
            raise ValueError("An event was assigned to overlapping slices")
        event_slice[selected] = slice_index
    return event_slice, fully_covered


def save_redshift_slice_maps(
    event_indices,
    event_redshift,
    event_slice,
    slice_zi,
    slice_zf,
    fully_covered,
    sparse_pixels,
    sparse_probabilities,
    offsets,
    injection_file,
    localisation_file,
):
    """Write one dense summed probability map for every fully covered slice."""
    number_of_pixels = hp.nside2npix(NSIDE)
    saved_paths = []
    for slice_index in np.flatnonzero(fully_covered):
        positions = np.flatnonzero(event_slice == slice_index)
        slice_map = np.zeros(number_of_pixels, dtype=np.float64)
        for event_position in positions:
            start = offsets[event_position]
            stop = offsets[event_position + 1]
            np.add.at(
                slice_map,
                sparse_pixels[start:stop],
                sparse_probabilities[start:stop],
            )

        expected_sum = float(len(positions))
        actual_sum = float(slice_map.sum())
        if not np.isclose(actual_sum, expected_sum, rtol=0.0, atol=1.0e-9):
            raise ValueError(
                f"Slice {slice_index} map sums to {actual_sum}, "
                f"expected {expected_sum}"
            )

        path = slice_output_path(
            slice_index, slice_zi[slice_index], slice_zf[slice_index]
        )
        save_atomic(
            path,
            probability_map=slice_map,
            support_mask=(slice_map > 0.0),
            event_source_index=event_indices[positions],
            event_redshift_true=event_redshift[positions],
            slice_index=np.asarray(slice_index),
            zi=np.asarray(slice_zi[slice_index]),
            zf=np.asarray(slice_zf[slice_index]),
            number_of_events=np.asarray(len(positions)),
            map_probability_sum=np.asarray(actual_sum),
            nside=np.asarray(NSIDE),
            nest=np.asarray(NEST),
            nsigma=np.asarray(NSIGMA),
            pixel_area_deg2=np.asarray(
                hp.nside2pixarea(NSIDE, degrees=True)
            ),
            source_type=np.asarray(SOURCE_TYPE),
            redshift_coordinate=np.asarray("redshift_true"),
            probability_normalisation=np.asarray("unit sum per event"),
            complete_slice_coverage=np.asarray(True),
            redshift_slice_file=np.asarray(str(REDSHIFT_SLICE_FILE)),
            injection_catalogue=np.asarray(str(injection_file)),
            localisation_catalogue=np.asarray(str(localisation_file)),
        )
        saved_paths.append(path)
        print(
            f"slice={slice_index:03d} "
            f"z=[{slice_zi[slice_index]:.6f}, {slice_zf[slice_index]:.6f}) "
            f"events={len(positions):,} sum={actual_sum:.8g}",
            flush=True,
        )
    return saved_paths


def main():
    injection_file = injection_path()
    localisation_file = localisation_path()
    save_file = output_path()
    catalogue, results, covariance, completed, free_params = load_inputs(
        injection_file, localisation_file
    )
    slice_zi, slice_zf = load_redshift_slices(REDSHIFT_SLICE_FILE)

    selected = completed & results["success"]
    event_indices = np.flatnonzero(selected)
    if MAX_EVENTS is not None:
        event_indices = event_indices[:int(MAX_EVENTS)]
    if len(event_indices) == 0:
        raise ValueError("No successful localisations were selected")

    number_of_pixels = hp.nside2npix(NSIDE)
    summed_map = np.zeros(number_of_pixels, dtype=np.float64)
    support_mask = np.zeros(number_of_pixels, dtype=bool)

    sparse_pixels = []
    sparse_probabilities = []
    offsets = np.zeros(len(event_indices) + 1, dtype=np.int64)
    retained_indices = []
    failed_indices = []
    failure_messages = []

    started = time.perf_counter()
    for position, source_index in enumerate(event_indices):
        try:
            ra_rad = float(catalogue["RA"][source_index])
            dec_rad = float(catalogue["Dec"][source_index])
            pixels, probabilities = event_posterior_pixels(
                ra_rad,
                dec_rad,
                covariance[source_index],
                free_params,
            )
            if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
                raise ValueError("Discrete event posterior does not sum to one")

            np.add.at(summed_map, pixels, probabilities)
            support_mask[pixels] = True
            sparse_pixels.append(pixels)
            sparse_probabilities.append(probabilities)
            retained_indices.append(source_index)
            offsets[len(retained_indices)] = (
                offsets[len(retained_indices) - 1] + len(pixels)
            )
        except Exception as error:
            failed_indices.append(source_index)
            failure_messages.append(f"{type(error).__name__}: {error}")

        processed = position + 1
        if processed % PRINT_EVERY == 0 or processed == len(event_indices):
            elapsed = time.perf_counter() - started
            print(
                f"processed={processed:,}/{len(event_indices):,} "
                f"mapped={len(retained_indices):,} failed={len(failed_indices):,} "
                f"elapsed={elapsed/60:.2f} min",
                flush=True,
            )

    retained_indices = np.asarray(retained_indices, dtype=np.int64)
    failed_indices = np.asarray(failed_indices, dtype=np.int64)
    offsets = offsets[:len(retained_indices) + 1]
    if sparse_pixels:
        sparse_pixels = np.concatenate(sparse_pixels)
        sparse_probabilities = np.concatenate(sparse_probabilities)
    else:
        sparse_pixels = np.empty(0, dtype=np.int64)
        sparse_probabilities = np.empty(0, dtype=np.float64)

    expected_total = float(len(retained_indices))
    actual_total = float(summed_map.sum())
    if not np.isclose(actual_total, expected_total, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Summed map integrates to {actual_total}, expected {expected_total}"
        )

    redshift_name = redshift_column(catalogue)
    if redshift_name is None:
        raise KeyError(
            "Injection catalogue has no redshift_true, redshift, or z column"
        )
    event_redshift = np.asarray(
        catalogue[redshift_name][retained_indices], dtype=float
    )

    event_slice, fully_covered_slices = assign_complete_redshift_slices(
        event_redshift, slice_zi, slice_zf
    )

    print("\nWriting true-redshift slice maps")
    saved_slice_paths = save_redshift_slice_maps(
        retained_indices,
        event_redshift,
        event_slice,
        slice_zi,
        slice_zf,
        fully_covered_slices,
        sparse_pixels,
        sparse_probabilities,
        offsets,
        injection_file,
        localisation_file,
    )

    save_atomic(
        save_file,
        summed_probability_map=summed_map,
        gw_support_mask=support_mask,
        event_source_index=retained_indices,
        event_redshift_true=event_redshift,
        event_slice_index=event_slice,
        event_snr_network=np.asarray(
            results["snr_network"][retained_indices], dtype=float
        ),
        event_ra_rad=np.asarray(catalogue["RA"][retained_indices], dtype=float),
        event_dec_rad=np.asarray(catalogue["Dec"][retained_indices], dtype=float),
        sparse_pixel_index=sparse_pixels,
        sparse_probability=sparse_probabilities,
        sparse_event_offset=offsets,
        failed_source_index=failed_indices,
        failure_message=np.asarray(failure_messages, dtype="U512"),
        nside=np.asarray(NSIDE),
        nest=np.asarray(NEST),
        nsigma=np.asarray(NSIGMA),
        number_of_events=np.asarray(len(retained_indices)),
        map_probability_sum=np.asarray(actual_total),
        pixel_area_deg2=np.asarray(hp.nside2pixarea(NSIDE, degrees=True)),
        source_type=np.asarray(SOURCE_TYPE),
        zi=np.asarray(ZI),
        zf=np.asarray(ZF),
        injection_catalogue=np.asarray(str(injection_file)),
        localisation_catalogue=np.asarray(str(localisation_file)),
        stored_ra_dec_unit=np.asarray("radian"),
        fisher_ra_dec_covariance_unit=np.asarray("degree_squared"),
        probability_normalisation=np.asarray("unit sum per event"),
        redshift_slice_file=np.asarray(str(REDSHIFT_SLICE_FILE)),
        redshift_slice_zi=slice_zi,
        redshift_slice_zf=slice_zf,
        fully_covered_redshift_slice=fully_covered_slices,
        support_mask_definition=np.asarray(
            f"union of event posterior pixels with Mahalanobis radius <= {NSIGMA}"
        ),
    )

    print("\nFinished")
    print(f"Mapped events: {len(retained_indices):,}")
    print(f"Failed events: {len(failed_indices):,}")
    print(f"Sparse entries: {len(sparse_pixels):,}")
    print(f"Summed-map probability: {actual_total:.12g}")
    print(f"Saved redshift slice maps: {len(saved_slice_paths):,}")
    print(
        "Events outside fully covered slices: "
        f"{int(np.count_nonzero(event_slice < 0)):,}"
    )
    print(f"Pixel area: {hp.nside2pixarea(NSIDE, degrees=True):.8f} deg2")
    print(f"Saved: {save_file}")


if __name__ == "__main__":
    main()
