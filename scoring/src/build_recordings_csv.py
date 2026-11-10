"""Build data/recordings.csv: one row per recording with the always-present
fields from /api/v2/conversations/{cid}/recordings.

These fields aren't in genesys_metadata.csv (which captures call-level metadata).
This adds a richer recording-level table.

Skipped on purpose:
  - mediaUris.mediaUri: signed time-limited URL, not useful to persist
  - mediaUris.waveformData: ~200 floats per track — bulky in CSV; see follow-up
  - annotations / externalContacts / users: empty for nearly all calls

Usage:
  python -m src.build_recordings_csv
"""
from __future__ import annotations

import csv
import datetime as dt
import sys
import time

from . import config
from .genesys_client import GenesysClient, GenesysError

FIELDS = [
    "recording_id", "conversation_id", "session_id",
    "recording_start", "recording_end",
    "original_recording_start_time", "creation_time",
    "media", "media_subtype", "media_subject",
    "file_state", "recording_file_role",
    "actual_transcode_time_ms", "max_allowed_restorations_for_org",
]


def _flatten(rec: dict, cid: str) -> dict:
    return {
        "recording_id": rec.get("id", ""),
        "conversation_id": rec.get("conversationId", cid),
        "session_id": rec.get("sessionId", ""),
        "recording_start": rec.get("startTime", ""),
        "recording_end": rec.get("endTime", ""),
        "original_recording_start_time": rec.get("originalRecordingStartTime", ""),
        "creation_time": rec.get("creationTime", ""),
        "media": rec.get("media", ""),
        "media_subtype": rec.get("mediaSubtype", ""),
        "media_subject": rec.get("mediaSubject", ""),
        "file_state": rec.get("fileState", ""),
        "recording_file_role": rec.get("recordingFileRole", ""),
        "actual_transcode_time_ms": rec.get("actualTranscodeTimeMs", ""),
        "max_allowed_restorations_for_org": rec.get("maxAllowedRestorationsForOrg", ""),
    }


def main() -> int:
    src = config.GENESYS_METADATA_CSV
    if not src.exists():
        print(f"missing {src}", file=sys.stderr)
        return 1
    out = config.DATA / "recordings.csv"

    with src.open(newline="", encoding="utf-8") as f:
        meta = [(r["conversation_id"], r["recording_id"])
                for r in csv.DictReader(f)
                if r["conversation_id"] and r["recording_id"]]

    # Resume support: skip recording_ids already present in recordings.csv
    # so an interrupted run can be continued without re-fetching.
    done: set[str] = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            done = {r["recording_id"] for r in csv.DictReader(f) if r.get("recording_id")}
    todo = [(cid, rid) for (cid, rid) in meta if rid not in done]
    print(f"recordings.csv already has {len(done)} rows; "
          f"fetching {len(todo)} of {len(meta)} -> {out}")

    gc = GenesysClient()
    ok = miss = fail = 0
    t0 = dt.datetime.now(dt.timezone.utc)
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, (cid, rid) in enumerate(todo, 1):
            rec = None
            # Retry-by-ID with backoff: Genesys throttles under load and
            # returns 404 on the list endpoint, but the by-ID endpoint is
            # more reliable. Backoff covers transient 404s too.
            for attempt in range(4):
                try:
                    rec = gc._api_request(
                        "GET", f"/api/v2/conversations/{cid}/recordings/{rid}")
                    break
                except GenesysError as e:
                    msg = str(e)
                    transient = "HTTP 404" in msg or "HTTP 429" in msg \
                        or "network error" in msg
                    if transient and attempt < 3:
                        time.sleep(2 ** attempt)  # 1s, 2s, 4s
                        continue
                    fail += 1
                    if fail <= 5 or fail % 100 == 0:
                        print(f"  FAIL {cid[:8]}/{rid[:8]} {msg[:80]}")
                    break
            if not isinstance(rec, dict):
                if isinstance(rec, type(None)) and (rec is None) and \
                        fail == 0:  # ok=False, no error logged: empty body
                    miss += 1
                continue
            if rec.get("media") != "audio":
                miss += 1
                continue
            w.writerow(_flatten(rec, cid))
            ok += 1
            if i % 100 == 0:
                elapsed = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()
                rate = i / elapsed if elapsed else 0
                eta = (len(todo) - i) / rate / 60 if rate else 0
                print(f"  {i}/{len(todo)}  ok={ok}  miss={miss}  fail={fail}  "
                      f"rate={rate:.1f}/s  eta={eta:.1f}min")
            # 0.25s = 4/sec = 240/min, well under Genesys's 300/min cap.
            time.sleep(0.25)

    print(f"DONE  wrote={ok} miss={miss} fail={fail}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
