"""Preserved settings for skysim_gw.diagnostics.compare. Edit before a new production run."""
from pathlib import Path
from .common import (BASE_DIR, SOURCE_TYPE, ZI, ZF)

INPUT_DIR = BASE_DIR / "GWInputCatalogs"

FISHER_PATH = (
    INPUT_DIR
    / "Localization"
    / (
        f"skysim5000_{SOURCE_TYPE}_lvk_fisher_tc_fixed_radec_deg_localisation_"
        f"z{ZI:.4f}_{ZF:.4f}.npz"
    )
)

DOUBLET_PATH = (
    INPUT_DIR
    / "LocalizationDoublet"
    / (
        f"skysim5000_{SOURCE_TYPE}_lvk_doublet_inv_dL_"
        f"tc_fixed_radec_deg_localisation_"
        f"z{ZI:.4f}_{ZF:.4f}.npz"
    )
)

OUTPUT_DIR = INPUT_DIR / "LocalizationComparison"
