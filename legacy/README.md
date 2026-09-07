# Archived entry points

These 18 compatibility wrappers and the empty merge placeholder were moved out of the repository root after the package refactor. They contain no separate scientific implementation. Active calculations and all diagnostics live in `skysim_gw/`.

Use `python -m skysim_gw --list` from the repository root to find the current commands. The filename-to-module mapping is in `skysim_gw/module_map.json`.

For compatibility, wrappers can be invoked from the repository root using module syntax, for example `python -m legacy.calculate_merger_rate_density_targets --config /path/to/gw_rate_config.json`. Prefer the current launcher for new runs. Direct file execution from this folder requires the package to be installed or otherwise on Python's import path.

`merge_gwdali_doublet_samples.py` remains empty and unimplemented. No active diagnosers or scientific modules were archived.
