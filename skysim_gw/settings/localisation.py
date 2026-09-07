"""Preserved settings for skysim_gw.diagnostics.localisation. Edit before a new production run."""
from pathlib import Path
from .common import (BASE_DIR, SOURCE_TYPE, ZI, ZF, APPROXIMANTS, FMIN_HZ, FMAX_HZ, FSIZE)

INPUT_DIR = BASE_DIR / "GWInputCatalogs"

SNR_DIR = INPUT_DIR / "SNR"

OUTPUT_DIR = INPUT_DIR / "Localization/Diagnostics"

SOURCE_INDEX = 0

PARAMETER_SETS = {
    "sky_only": ("RA", "Dec"),
    "extrinsic_tc_fixed": ("RA", "Dec", "dL", "iota", "psi", "phi_coal"),
    "extrinsic_tc_free": (
        "RA", "Dec", "dL", "iota", "psi", "t_coal", "phi_coal"
    ),
}

STEP_SIZES = (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)
