"""STEP 3 (local) — ingest the Kaggle predictions and refresh the project feed.

Chains:
    merge_predictions <predictions.csv>   (append+dedup into the master)
    -> option-c-value-bidding/run_pipeline.py   (rebuild the value feed)

Usage:
  python -m src.refresh_ingest predictions_kaggle_latest.csv
  python -m src.refresh_ingest <csv> --dry-run
  python -m src.refresh_ingest <csv> --no-optionc   # just merge, skip the feed rebuild
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from .config import ROOT
from . import merge_predictions as MP

OPTION_C = ROOT.parent / "option-c-value-bidding" / "run_pipeline.py"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="predictions.csv downloaded from Kaggle")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-optionc", action="store_true", help="skip the Option C feed rebuild")
    a = ap.parse_args(argv)

    print("== refresh_ingest ==")
    print("[1] merge predictions")
    mp_args = [a.predictions] + (["--dry-run"] if a.dry_run else [])
    rc = MP.main(mp_args)
    if rc != 0:
        return rc

    if a.no_optionc:
        return 0
    print("\n[2] refresh Option C value feed")
    if not OPTION_C.exists():
        print(f"   (skipped — {OPTION_C} not found)")
        return 0
    print(f"   $ python {OPTION_C.name}")
    if a.dry_run:
        return 0
    return subprocess.run([sys.executable, str(OPTION_C)], cwd=str(OPTION_C.parent)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
