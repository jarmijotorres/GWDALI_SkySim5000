"""Preserved settings for skysim_gw.diagnostics.one_localisation. Edit before a new production run."""
from pathlib import Path
from .common import (BASE_DIR, SOURCE_TYPE, ZI, ZF, APPROXIMANTS, FMIN_HZ, FMAX_HZ, FSIZE)

INPUT_DIR = BASE_DIR / "GWInputCatalogs"

SNR_DIR = INPUT_DIR / "SNR"

SOURCE_INDEX = None

