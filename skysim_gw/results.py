"""Reusable GWDALI return-structure adapters, without running diagnostics."""
from collections.abc import Mapping
import numpy as np

def find_matrix(result, names):
    """Find the first square numeric matrix under any candidate result key."""
    if not isinstance(result, Mapping) and not (
        hasattr(result, "keys") and hasattr(result, "__getitem__")
    ):
        return None, None
    for name in names:
        if name in result:
            matrix = np.asarray(result[name], dtype=float)
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                return name, matrix
    return None, None

def unwrap_gwdali_result(result):
    """Normalize the GWDALI v1 ``(samples, tensors)`` return structure."""
    is_mapping = lambda value: (
        isinstance(value, Mapping)
        or (hasattr(value, "keys") and hasattr(value, "__getitem__"))
    )
    if is_mapping(result):
        return result
    if isinstance(result, tuple):
        # Return tuple length varies across GWDALI v1 builds. Locate the tensor
        # mapping by content instead of assuming it is exactly element 1 of a
        # two-element tuple.
        for item in result:
            if is_mapping(item) and "Fisher" in item:
                return item
            if isinstance(item, (list, tuple)):
                for nested_item in item:
                    if is_mapping(nested_item) and "Fisher" in nested_item:
                        return nested_item
    second_type = (
        repr(type(result[1]))
        if isinstance(result, tuple) and len(result) > 1
        else "n/a"
    )
    raise TypeError(
        "Expected a mapping or (samples, tensor_mapping); received "
        f"outer type={type(result)!r}, second type={second_type}"
    )
