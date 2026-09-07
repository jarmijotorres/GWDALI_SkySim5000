"""Compatibility entry point; implementation: skysim_gw.stages.doublet_maps."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    implementation = importlib.import_module("skysim_gw.stages.doublet_maps")
    if hasattr(implementation, "main"):
        raise SystemExit(implementation.main())
    runpy.run_module("skysim_gw.stages.doublet_maps", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("skysim_gw.stages.doublet_maps")
