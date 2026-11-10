"""STEP 1 (local, daily) — pull fresh Zoom transcripts and stage the Kaggle input.

Auto-computes the Zoom window (earlier of: last call in calls.csv, or today
minus LOOKBACK_DAYS -> today; the trailing re-scan retries recordings that
appeared late or were skipped by a mid-pull network drop, before Zoom's
~30-day retention purges them), then chains the local pipeline:
    ingest_zoom  ->  build_calls_csv  ->  build_score_queue  ->  emit_test_prompting

Output: data/train/test_prompting_sales.jsonl  (the file kaggle_run.py uploads).
Idempotent — re-pulling an overlapping window is safe (build_calls_csv dedups).

Usage:
  python -m src.refresh_pull                 # auto window, run it
  python -m src.refresh_pull --dry-run       # show the plan, run nothing
  python -m src.refresh_pull --start 2026-06-15   # override window start
  python -m src.refresh_pull --skip-ingest   # transcripts already pulled; just re-stage
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from datetime import date, timedelta

import pandas as pd

from .config import DATA

JSONL = DATA / "train" / "test_prompting_sales.jsonl"
IDS = DATA / "train" / "ids_sales_new.json"

# re-list at least this far back on every pull; found 116 lost recordings on
# 2026-07-21 (07-06 network drops + forward-only window). All-skip re-lists
# are cheap; must stay >= Zoom's ~30-day retention to catch stragglers in time.
LOOKBACK_DAYS = 35


def _last_zoom_date() -> str:
    calls = pd.read_csv(DATA / "calls.csv", low_memory=False)
    z = calls[calls["source"] == "zoom"]
    d = pd.to_datetime(z["call_date"], errors="coerce").max()
    return (d.date().isoformat() if pd.notna(d) else "2026-04-21")


def _default_start() -> str:
    floor = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    return min(_last_zoom_date(), floor)


def _run(cmd, dry):
    print("   $ " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.run([sys.executable, "-m", *cmd], cwd=str(DATA.parent)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="window start YYYY-MM-DD (default = last zoom call in calls.csv)")
    ap.add_argument("--end", help="window end YYYY-MM-DD (default = today)")
    ap.add_argument("--since", help="only queue/score sales calls on/after this date")
    ap.add_argument("--skip-ingest", action="store_true", help="don't pull Zoom; just rebuild+queue+emit")
    ap.add_argument("--transcripts-only", action="store_true",
                    help="forwarded to ingest_zoom: skip audio downloads")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    start = a.start or _default_start()
    end = a.end or date.today().isoformat()
    print(f"== refresh_pull ==  window {start} -> {end}  (dry-run={a.dry_run})")

    steps = []
    if not a.skip_ingest:
        ing = ["src.ingest_zoom", "--start", start, "--end", end]
        if a.transcripts_only:
            ing.append("--transcripts-only")
        steps.append(ing)
    steps.append(["src.build_calls_csv"])
    if not a.skip_ingest:
        # re-fetch transcripts that landed AFTER their first pull (Zoom is async)
        steps.append(["src.backfill_transcripts", "--days", "10"])
    q = ["src.build_score_queue", "--out", str(IDS)]
    if a.since:
        q += ["--since", a.since]
    steps.append(q)
    steps.append(["src.emit_test_prompting", "--ids", str(IDS), "--out", str(JSONL)])

    for cmd in steps:
        rc = _run(cmd, a.dry_run)
        if rc != 0:
            print(f"!! step failed (rc={rc}): {' '.join(cmd)}")
            return rc

    if not a.dry_run:
        print(f"\nstaged Kaggle input -> {JSONL}")
        print("NEXT: python -m src.kaggle_run   (push + run on Kaggle + fetch predictions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
