"""LVK detector configuration for GWDALI.

Geometry follows Table 3 of Menote et al. (2026).  In the installed GWDALI
version, for a 90-degree detector ``draw_detectors`` places the two arm axes
at ``rot - 90 deg`` and ``rot`` measured counter-clockwise from east.  An
interferometer arm is an axis, so angles differing by 180 degrees are
equivalent.

The case-sensitive ``name`` values select GWDALI's built-in sensitivity
curves. Hanford and Livingston both use ``LIGO`` but retain distinct geometry.
"""

from copy import deepcopy


LVK_LABELS = ("LIGO Hanford", "LIGO Livingston", "Virgo", "KAGRA")

LVK_DETECTORS = (
    {
        "name": "LIGO",
        "lon": -119.40,
        "lat": 46.45,
        # X: N36 W; Y: W36 S
        "rot": 36.0,
        "shape": 90.0,
    },
    {
        "name": "LIGO",
        "lon": -90.77,
        "lat": 30.56,
        # X: W18 S; Y: S18 E
        "rot": 108.0,
        "shape": 90.0,
    },
    {
        "name": "Virgo",
        "lon": 10.43,
        "lat": 43.51,
        # X: N19 E; Y: W19 N
        "rot": 161.0,
        "shape": 90.0,
    },
    {
        "name": "Kagra",
        "lon": 137.31,
        "lat": 36.41,
        # X: E28.3 N; Y: N28.3 W
        "rot": 118.3,
        "shape": 90.0,
    },
)


def validate_detector_network(detectors):
    """Raise a useful error for malformed GWDALI detector dictionaries."""
    required = ("name", "lon", "lat", "rot", "shape")
    if not detectors:
        raise ValueError("The detector network cannot be empty")

    for index, detector in enumerate(detectors):
        missing = [key for key in required if key not in detector]
        if missing:
            raise KeyError(f"Detector {index} is missing keys {missing}")
        if not -180.0 <= float(detector["lon"]) <= 180.0:
            raise ValueError(f"Detector {index} has invalid longitude")
        if not -90.0 <= float(detector["lat"]) <= 90.0:
            raise ValueError(f"Detector {index} has invalid latitude")
        if not 0.0 < float(detector["shape"]) <= 180.0:
            raise ValueError(f"Detector {index} has invalid arm opening angle")


def get_lvk_detectors(include_kagra=True):
    """Return fresh mutable dictionaries for LVK or the HLV subnetwork."""
    count = 4 if include_kagra else 3
    detectors = deepcopy(list(LVK_DETECTORS[:count]))
    validate_detector_network(detectors)
    return detectors


if __name__ == "__main__":
    detectors = get_lvk_detectors(include_kagra=True)
    for label, detector in zip(LVK_LABELS, detectors):
        print(
            f"{label:16s} PSD={detector['name']:7s} "
            f"lat={detector['lat']:7.2f} lon={detector['lon']:8.2f} "
            f"rot={detector['rot']:7.2f} shape={detector['shape']:5.1f}"
        )
