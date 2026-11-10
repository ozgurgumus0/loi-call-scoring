"""Re-fetch Zoom transcripts that weren't ready at the first pull.

Zoom generates the VTT transcript ASYNCHRONOUSLY (often hours after the call),
but ingest_zoom skips any meeting it has already seen, and build_calls_csv never
updates transcript fields on existing rows. So a transcript that lands AFTER the
first pull is otherwise lost and that call never gets scored.

This backfill re-probes recent zoom calls that have no/empty transcript; if Zoom
now has a real one, it downloads it and updates calls.csv + zoom_metadata.csv
(both, so a later rebuild keeps it). refresh_pull runs this every time, so late
transcripts self-heal within the --days window.

Usage:
  python -m src.backfill_transcripts                 # last 10 days
  python -m src.backfill_transcripts --days 14
  python -m src.backfill_transcripts --dry-run
"""
from __future__ import annotations
import argparse
import csv
import shutil
import sys
from datetime import date, timedelta

import pandas as pd

from . import config
from .zoom_client import ZoomClient, safe_uuid

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CALLS = config.CALLS_CSV
ZMETA = config.ZOOM_METADATA_CSV
TDIR = config.DATA / "transcripts" / "zoom"
MIN_BYTES = 500   # empty Zoom VTT headers are ~70-95 B; a real call is many KB


def _transcriptless(days: int) -> pd.DataFrame:
    df = pd.read_csv(CALLS, low_memory=False, dtype=str)
    z = df[df["source"] == "zoom"].copy()
    d = pd.to_datetime(z["call_date"], errors="coerce")
    cutoff = pd.Timestamp(date.today() - timedelta(days=days))
    z = z[d >= cutoff]
    tb = pd.to_numeric(z["transcript_bytes"], errors="coerce").fillna(0)
    return z[tb < MIN_BYTES]


def _patch_csv(path, keycol: str, recovered: dict, only_zoom: bool) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames or []
    n = 0
    for row in rows:
        if only_zoom and row.get("source") != "zoom":
            continue
        hit = recovered.get(row.get(keycol, ""))
        if hit:
            row["transcript_file"], row["transcript_bytes"] = hit[0], str(hit[1])
            n += 1
    if n:
        shutil.copy(path, str(path) + ".bak_backfill")
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10, help="re-check window (default 10)")
    ap.add_argument("--min-bytes", type=int, default=MIN_BYTES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if not CALLS.exists():
        print("no calls.csv — nothing to backfill")
        return 0
    miss = _transcriptless(a.days)
    if miss.empty:
        print(f"backfill_transcripts: no transcript-less zoom calls in the last {a.days} days")
        return 0

    dmin = pd.to_datetime(miss["call_date"]).min().date().isoformat()
    dmax = pd.to_datetime(miss["call_date"]).max().date().isoformat()
    targets = set(miss["call_id"].astype(str))
    print(f"backfill_transcripts: {len(targets)} transcript-less zoom calls "
          f"({dmin}..{dmax}) — probing Zoom...")

    # Zoom probe is best-effort: a transient API error must NOT break the daily
    # pull (calls that already have transcripts still score).
    try:
        zc = ZoomClient()
        by_uuid = {m.get("uuid", ""): m for m in zc.list_all_recordings(dmin, dmax)}
    except Exception as e:  # noqa: BLE001
        print(f"  (Zoom probe failed, skipping backfill: {type(e).__name__}: {e})")
        return 0

    TDIR.mkdir(parents=True, exist_ok=True)
    recovered: dict[str, tuple[str, int]] = {}
    for cid in targets:
        m = by_uuid.get(cid)
        if not m:
            continue
        vtt = next((f for f in (m.get("recording_files") or [])
                    if f.get("recording_type") == "audio_transcript"
                    and f.get("file_size", 0) >= a.min_bytes), None)
        if not vtt:
            continue
        su = safe_uuid(cid)
        rel = f"transcripts/zoom/{su}.vtt"
        if a.dry_run:
            recovered[cid] = (rel, vtt["file_size"])
            continue
        try:
            n = zc.download(vtt["download_url"], TDIR / f"{su}.vtt")
        except Exception as e:  # noqa: BLE001
            print(f"  ! {cid[:14]} download failed: {type(e).__name__}")
            continue
        recovered[cid] = (rel, n)
        print(f"  + {cid[:14]}  {n}B")

    print(f"now-available transcripts: {len(recovered)} of {len(targets)} checked")
    if a.dry_run:
        print("(dry-run: nothing written)")
        return 0
    if recovered:
        c = _patch_csv(CALLS, "call_id", recovered, only_zoom=True)
        z = _patch_csv(ZMETA, "meeting_uuid", recovered, only_zoom=False)
        print(f"updated calls.csv ({c} rows) + zoom_metadata.csv ({z} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
