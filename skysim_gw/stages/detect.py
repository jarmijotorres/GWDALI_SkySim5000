#!/usr/bin/env python3
"""Write catalogue row indices satisfying the validated SNR selection."""
from pathlib import Path
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snr_file")
    ap.add_argument("output")
    args = ap.parse_args()
    with np.load(args.snr_file, allow_pickle=False) as f:
        results = f["results"]
        completed = f["completed"].astype(bool)
    required = {"success", "detected"}
    if not required.issubset(results.dtype.names):
        raise ValueError(f"SNR results must contain {sorted(required)}")
    indices = np.flatnonzero(completed & results["success"] & results["detected"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(map(str, indices)) + "\n")
    print(f"Wrote {len(indices)} detected indices to {args.output}")

if __name__ == "__main__":
    main()
