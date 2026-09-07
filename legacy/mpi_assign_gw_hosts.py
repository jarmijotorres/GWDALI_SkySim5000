"""Compatibility entry point; implementation: skysim_gw.stages.assign_hosts."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    implementation = importlib.import_module("skysim_gw.stages.assign_hosts")
    if hasattr(implementation, "main"):
        raise SystemExit(implementation.main())
    runpy.run_module("skysim_gw.stages.assign_hosts", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("skysim_gw.stages.assign_hosts")
