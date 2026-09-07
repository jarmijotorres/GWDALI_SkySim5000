"""Preserved settings for skysim_gw.stages.snr. Edit before a new production run."""
from pathlib import Path
from .common import (BASE_DIR, SOURCE_TYPE, ZI, ZF, APPROXIMANTS, FMIN_HZ, FMAX_HZ, FSIZE)

INPUT_DIR = BASE_DIR / "GWInputCatalogs"

OUTPUT_DIR = INPUT_DIR / "SNR"

INCLUDE_KAGRA = True

NETWORK_SNR_THRESHOLD = 12.0

SINGLE_DETECTOR_SNR_THRESHOLD = 4.0

MIN_DETECTORS_ABOVE_THRESHOLD = 2

START_INDEX = 0

STOP_INDEX = None

CHECKPOINT_EVERY = 25

PRINT_EVERY = 10

RESUME = True
