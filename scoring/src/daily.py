r"""ONE-COMMAND daily LOI refresh — chains all three steps end to end.

    refresh_pull   (Zoom -> calls.csv -> score queue -> jsonl)
    kaggle_run     (push + run on Kaggle GPU + download predictions)   [--go]
    refresh_ingest (merge predictions -> Option C feed)

Empty-day guard: if there are no new calls to score, the Kaggle run is skipped
(no wasted GPU run on weekends/quiet days). Every run appends a line to
outputs/daily_log.txt so unattended failures leave a trail.

Free Kaggle GPU = ~28 min for a daily incremental, inside the 30h/week quota, so
a daily schedule is safe. Kaggle queue/run latency is ~30-60 min, so step 2 polls.

Usage:
  python -m src.daily --dry-run        # show the whole plan
  python -m src.daily --go             # run it for real (needs Kaggle token + Zoom creds)
  python -m src.daily --go --since 2026-06-12

Schedule (Windows Task Scheduler), once a day:
  schtasks /create /tn "LOI daily" /sc daily /st 07:00 ^
    /tr "cmd /c cd /d \"<repo>\loi-audio-mvp\" && python -m src.daily --go"
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from . import refresh_pull, kaggle_run, refresh_ingest
from .config import ROOT, DATA

PRED = ROOT / "predictions_kaggle_latest.csv"
JSONL = DATA / "train" / "test_prompting_sales.jsonl"
LOG = ROOT / "outputs" / "daily_log.txt"


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
    ap.add_argument("--dry-run", action="store_true", help="show the plan (default if --go absent)")
    ap.add_argument("--since", help="only queue/score sales calls on/after this date")
    a = ap.parse_args(argv)
    go = a.go
    dry = [] if go else ["--dry-run"]

    print("################  DAILY LOI REFRESH  ################\n")

    print(">>> STEP 1/3  refresh_pull")
    if refresh_pull.main((["--since", a.since] if a.since else []) + dry) != 0:
        if go:
            _log("FAIL at refresh_pull")
        return 1

    # empty-day guard
    if go:
        n = _jsonl_rows()
        if n == 0:
            _log("OK no new calls — Kaggle run skipped")
            print("\nNo new calls to score today. Done.")
            return 0
        print(f"   ({n} calls queued for scoring)")

    print("\n>>> STEP 2/3  kaggle_run")
    if kaggle_run.main(["--go"] if go else []) != 0:
        if go:
            _log("FAIL at kaggle_run (fix setup, then rerun kaggle_run + refresh_ingest)")
        else:
            print("   (kaggle step needs token/setup — see kaggle_run output)")
        return 1

    print("\n>>> STEP 3/3  refresh_ingest")
    if not go:
        print(f"   $ python -m src.refresh_ingest {PRED.name} (after Kaggle produces it)")
        print("\nDRY-RUN complete.")
        return 0
    if not PRED.exists():
        _log("FAIL no predictions produced by Kaggle step")
        return 1
    rc = refresh_ingest.main([str(PRED)])
    _log(f"OK scored {_jsonl_rows()} calls" if rc == 0 else "FAIL at refresh_ingest")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
