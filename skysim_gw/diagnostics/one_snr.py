"""Step 7a: calculate and inspect the SNR of one real catalogue source.

Run this diagnostic before the full serial SNR catalogue.  GWDALI documents
detector and network SNRs but not the exact return container, which has changed
between releases.  This script therefore prints the untouched return value.
"""

from pathlib import Path

import numpy as np
import GWDALI as gw

from skysim_gw.catalogue import (
    catalog_path,
    get_gwdali_source,
    load_gwdali_catalog,
)
from skysim_gw.detectors import LVK_LABELS, get_lvk_detectors


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

from skysim_gw.settings.one_snr import (
    BASE_DIR,
    INPUT_DIR,
    SOURCE_TYPE,
    ZI,
    ZF,
    SOURCE_INDEX,
    INCLUDE_KAGRA,
    APPROXIMANTS,
    FMIN_HZ,
    FMAX_HZ,
    FSIZE,
)


def main():
    source_type = SOURCE_TYPE.upper()
    path = catalog_path(INPUT_DIR, source_type, ZI, ZF)
    catalog, metadata = load_gwdali_catalog(path)
    GwPrms = get_gwdali_source(catalog, SOURCE_INDEX)
    detectors = get_lvk_detectors(include_kagra=INCLUDE_KAGRA)
    labels = LVK_LABELS[:len(detectors)]
    approx = APPROXIMANTS[source_type]

    print(f"Catalogue: {path}")
    print(f"Sources: {len(catalog):,}")
    print(f"Testing source index: {SOURCE_INDEX}")
    print(f"Source type: {source_type}")
    print(f"Approximant: {approx}")
    print(f"Frequency range: {FMIN_HZ:g}--{FMAX_HZ:g} Hz; fsize={FSIZE}")
    print("Detectors:")
    for label, detector in zip(labels, detectors):
        print(f"  {label}: {detector}")

    print("\nGwPrms:")
    for name, value in GwPrms.items():
        print(f"  {name:8s} = {value:.12g}")

    result = gw.get_SNR(
        detectors,
        GwPrms,
        approx,
        enable_jax_waveforms=False,
        fmin=FMIN_HZ,
        fmax=FMAX_HZ,
        fsize=FSIZE,
    )

    print("\nRaw get_SNR result:")
    print(result)
    print("Return type:", type(result))
    if isinstance(result, dict):
        print("Dictionary keys:", tuple(result))
        for key, value in result.items():
            print(f"  {key!r}: type={type(value)}, shape={np.shape(value)}, value={value}")
    elif isinstance(result, (tuple, list)):
        print("Number of entries:", len(result))
        for index, value in enumerate(result):
            print(
                f"  result[{index}]: type={type(value)}, "
                f"shape={np.shape(value)}, value={value}"
            )
    else:
        print("Result shape:", np.shape(result))

    print("\nCatalogue metadata:")
    for name in ("distance_unit", "mass_unit", "angle_unit", "cosmology"):
        if name in metadata:
            print(f"  {name}: {metadata[name]}")


if __name__ == "__main__":
    main()
