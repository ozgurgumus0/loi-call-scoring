r"""PART 1 (manual, one command) — score new LOIs and emit the leadid Excel.

Chains, with NO Option C / ads step (that's Part 2):
    refresh_pull   (Zoom -> calls.csv -> score queue -> jsonl)
    kaggle_run     (push + run on Kaggle GPU + download predictions)   [--go]
    refresh_ingest (merge into the master, --no-optionc)
    export_loi_excel (outputs/loi_scores_<today>.xlsx, leadid-tagged)

Empty-day guard: if no new calls, the Kaggle run is skipped (but the Excel is
still refreshed from the existing master so you always get a current file).

Usage:
  python -m src.part1_score --dry-run     # show the plan
  python -m src.part1_score --go          # run it (needs Kaggle token + Zoom creds)
  python -m src.part1_score --go --skip-ingest   # Zoom already pulled; just score+excel
  python -m src.part1_score --go --since 2026-06-12

Manual fallback if a step fails mid-way (run them one by one):
  python -m src.refresh_pull
  python -m src.kaggle_run --go
  python -m src.refresh_ingest predictions_kaggle_latest.csv --no-optionc
  python -m src.export_loi_excel
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import refresh_pull, kaggle_run, refresh_ingest, export_loi_excel
from .config import ROOT, DATA

PRED = ROOT / "predictions_kaggle_latest.csv"
JSONL = DATA / "train" / "test_prompting_sales.jsonl"
LOG = ROOT / "outputs" / "part1_log.txt"


def _log(msg: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}\n")
    print(f"   [log] {msg}")


def _jsonl_rows() -> int:
    return sum(1 for _ in JSONL.open(encoding="utf-8")) if JSONL.exists() else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="execute (default dry-run)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--since", help="only queue/score sales calls on/after this date")
    ap.add_argument("--skip-ingest", action="store_true", help="don't pull Zoom; just stage+score")
    a = ap.parse_args(argv)
    go = a.go
    dry = [] if go else ["--dry-run"]

    print("################  PART 1 — LOI SCORING  ################\n")

    print(">>> STEP 1/4  refresh_pull")
    pull_args = (["--since", a.since] if a.since else []) + \
                (["--skip-ingest"] if a.skip_ingest else []) + dry
    if refresh_pull.main(pull_args) != 0:
        if go:
            _log("FAIL at refresh_pull")
        return 1

    new = _jsonl_rows() if go else None
    if go and new == 0:
        print("\nNo new calls to score — refreshing the Excel from the existing master.")
        rc = export_loi_excel.main([])
        _log("OK no new calls; excel refreshed")
        return rc
    if go:
        print(f"   ({new} calls queued)")

    print("\n>>> STEP 2/4  kaggle_run")
    if kaggle_run.main(["--go"] if go else []) != 0:
        if go:
            _log("FAIL at kaggle_run (rerun: kaggle_run --go, then refresh_ingest --no-optionc, then export_loi_excel)")
        return 1

    print("\n>>> STEP 3/4  refresh_ingest (--no-optionc)")
    if not go:
        print(f"   $ python -m src.refresh_ingest {PRED.name} --no-optionc")
    else:
        if not PRED.exists():
            _log("FAIL no predictions downloaded")
            return 1
        if refresh_ingest.main([str(PRED), "--no-optionc"]) != 0:
            _log("FAIL at refresh_ingest")
            return 1

    print("\n>>> STEP 4/4  export_loi_excel")
    if not go:
        print("   $ python -m src.export_loi_excel")
        print("\nDRY-RUN complete.")
        return 0
    rc = export_loi_excel.main([])
    _log(f"OK scored {new} new calls; excel written" if rc == 0 else "FAIL at export_loi_excel")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
