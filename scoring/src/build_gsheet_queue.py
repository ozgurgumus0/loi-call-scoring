"""Build the COMBINED daily score queue: sales queue + dispatch-sheet calls.

`build_score_queue` only queues call_type=='sales', but the user decided
(2026-07-07) that dispatch-sheet membership OVERRIDES the booking exclusion:
every sheet-matched call gets scored, booker@example.com-hosted ones included.
This script reads the 'Zoom SC1' sheet LIVE (same gspread SA as
fill_gsheet_loi), resolves rows to calls.csv via the validated join
(scheduled meetingUuid == call_id, numeric meetingId -> instance-uuid
fallback), and appends the unscored non-sales matches with a usable VTT
(>=500B) to the sales queue from ids_sales_new.json.

Booking-typed predictions must NOT be merged into the sales master — split by
call_type at ingest time (sales -> merge_predictions, booking -> append to
predictions_qwen3_8b_gsheet_booking.csv).

Usage:
  python -m src.build_gsheet_queue                 # write data/train/ids_daily_<date>.json
  python -m src.build_gsheet_queue --dry-run       # counts only, write nothing
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

from .config import DATA, ROOT

SHEET_ID = os.environ.get("LOI_SHEET_ID", "")
SA_FILE = (ROOT.parent / "ai-traffic-tracker" / "data"
           / "sheets_service_account.json")
MASTER_CSV = ROOT / "predictions_qwen3_8b_sales_full.csv"
BOOKING_CSV = ROOT / "predictions_qwen3_8b_gsheet_booking.csv"
SALES_QUEUE = DATA / "train" / "ids_sales_new.json"
MIN_VTT = 500


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="ids json output path "
                    "(default data/train/ids_daily_<date>.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print counts, write nothing")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else (
        DATA / "train" / f"ids_daily_{date.today():%m%d}.json")

    import gspread
    gc = gspread.service_account(filename=str(SA_FILE))
    ws = gc.open_by_key(SHEET_ID).sheet1
    vals = ws.get_all_values()
    sheet = pd.DataFrame(vals[1:], columns=vals[0])
    sheet = sheet[sheet["meetingUuid"].str.strip() != ""].copy()
    sheet["meetingId"] = sheet["meetingId"].astype(str)
    print(f"sheet rows (live): {len(sheet)}")

    calls = pd.read_csv(DATA / "calls.csv", low_memory=False)
    z = calls[calls["source"] == "zoom"].copy()
    z["call_id"] = z["call_id"].astype(str)
    zinfo = z.set_index("call_id")[["call_type", "transcript_bytes"]]
    corpus_ids = set(zinfo.index)

    zm = pd.read_csv(DATA / "zoom_metadata.csv", low_memory=False)
    zm["meeting_id"] = (zm["meeting_id"].astype(str)
                        .str.replace(r"\.0$", "", regex=True))
    mid2uuid = zm.groupby("meeting_id")["meeting_uuid"].apply(list).to_dict()

    scored = set(pd.read_csv(MASTER_CSV, usecols=["call_id"])
                 ["call_id"].astype(str))
    if BOOKING_CSV.exists():
        scored |= set(pd.read_csv(BOOKING_CSV, usecols=["call_id"])
                      ["call_id"].astype(str))
    silent_file = DATA / "whisper_silent_calls.json"
    silent = (set(json.load(open(silent_file))["call_ids"])
              if silent_file.exists() else set())

    def resolve(row):
        uu = str(row["meetingUuid"])
        if uu in corpus_ids:
            return uu
        for iu in mid2uuid.get(row["meetingId"], []):
            if iu in corpus_ids:
                return iu
        return ""

    matched = {cid for cid in sheet.apply(resolve, axis=1) if cid}
    print(f"sheet-matched calls           : {len(matched)}")

    booking_add = []
    for cid in sorted(matched):
        if cid in scored or cid in silent:
            continue
        if zinfo.at[cid, "call_type"] == "sales":
            continue  # already covered by build_score_queue
        tb = zinfo.at[cid, "transcript_bytes"]
        if pd.isna(tb) or float(tb) < MIN_VTT:
            continue
        booking_add.append(cid)

    # rebuild the sales diff fresh — a stale ids_sales_new.json silently
    # drops every sales call pulled since its last write (seen 2026-07-20)
    from . import build_score_queue
    build_score_queue.main([])
    sales_q = []
    if SALES_QUEUE.exists():
        sales_q = json.load(open(SALES_QUEUE))["call_ids"]
    combined = sales_q + [c for c in booking_add if c not in set(sales_q)]

    print(f"sales queue (ids_sales_new)   : {len(sales_q)}")
    print(f"booking additions (sheet)     : {len(booking_add)}")
    print(f"=> COMBINED QUEUE             : {len(combined)}")

    if a.dry_run:
        print("(dry-run: nothing written)")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"call_ids": combined, "source": "zoom",
         "note": f"daily combined queue {date.today()}: "
                 f"{len(sales_q)} sales + {len(booking_add)} sheet-booking"},
        ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[saved] {out}  ({len(combined)} ids)")
    print(f"NEXT: python -m src.emit_test_prompting --ids {out} "
          f"--out data/train/test_prompting_sales.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
