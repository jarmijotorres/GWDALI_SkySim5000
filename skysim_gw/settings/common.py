"""Shared production defaults; existing NERSC paths remain placeholders.

Change these before starting a new process. Existing checkpoints must match the
chosen scientific settings; use separate output directories for different runs.
"""
from pathlib import Path

BASE_DIR = Path("/pscratch/sd/j/jatorres/data/lsst/SkySim5000/NumberCountMaps")
SOURCE_TYPE = "BBH"
ZI = 0.0
ZF = 0.8
APPROXIMANTS = {
    "BBH": "IMRPhenomXPHM",
    "BHNS": "IMRPhenomXPHM",
    "BNS": "IMRPhenomXHM",
}
FMIN_HZ = 10.0
FMAX_HZ = 2048.0
FSIZE = 3000
REDSHIFT_SLICE_FILE = Path(
    "/global/homes/j/jatorres/SkySim5000Galaxies/Notebooks/redshift_slices.info"
)
