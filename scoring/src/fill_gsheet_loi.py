"""Fill the 'Zoom SC1' dispatch-meetings Google Sheet with our LOI columns.

Reads the sheet LIVE via gspread (service account), matches every row to our
Zoom corpus (scheduled meetingUuid == calls.csv call_id, with a numeric
meetingId -> zoom_metadata instance-uuid fallback), joins the scored master
(the usual loi_scores Excel columns) plus any extra predictions file, and
writes the result into columns N.. of the sheet in one batch update.

Rows are aligned by re-reading the sheet at write time, so it is safe to run
while new rows are being appended (new/unknown rows just get their computed
status too).

Usage:
  python -m src.fill_gsheet_loi            # dry-run: print counts, write nothing
  python -m src.fill_gsheet_loi --go       # write to the sheet
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .config import DATA, ROOT

SHEET_ID = os.environ.get("LOI_SHEET_ID", "")
SA_FILE = (ROOT.parent / "ai-traffic-tracker" / "data"
           / "sheets_service_account.json")

# newest loi_scores_*.xlsx = the usual output columns
SCORE_COLS = ["leadid", "call_date", "channel", "region", "loi_band",
              "loi_score", "closing_readiness", "confidence",
              "contract_signed"]
OUT_HEADER = (["loi_match_status", "matched_call_id", "matched_via"]
              + SCORE_COLS)
FIRST_COL = "N"  # sheet data ends at column M (calendarEventId)

STAGED_IDS = DATA / "train" / "ids_gsheet_batch.json"
EXTRA_PRED = ROOT / "predictions_qwen3_8b_gsheet_booking.csv"
SALES_MASTER = ROOT / "predictions_qwen3_8b_sales_full.csv"
LATEST_BATCH = ROOT / "predictions_kaggle_latest.csv"
CORPUS_START = pd.Timestamp("2026-04-21", tz="UTC")


def _newest_scores_xlsx() -> Path:
    # strict date pattern: loi_scores_2*.xlsx — a stray suffixed file like
    # loi_scores_xcally_*.xlsx sorts after every date and would shadow the
    # real master (bit us 2026-07-13)
    cands = sorted((ROOT / "outputs").glob("loi_scores_2*.xlsx"))
    if not cands:
        raise SystemExit("no outputs/loi_scores_*.xlsx found")
    return cands[-1]


def load_scores() -> pd.DataFrame:
    """Scored master plus booking scores and any not-yet-exported latest batch."""
    xl = pd.read_excel(_newest_scores_xlsx(), sheet_name=0)
    xl["call_id"] = xl["call_id"].astype(str)
    frames = [xl]
    # The dated xlsx is a hand-off snapshot and can lag behind the live sales
    # master. Preserve its richer CRM fields, but overlay nonblank scoring
    # fields from the live master for established rows as well as adding newer
    # call_ids. Otherwise a stale snapshot can hide later readiness re-scores.
    if SALES_MASTER.exists():
        sales = pd.read_csv(SALES_MASTER)
        sales["call_id"] = sales["call_id"].astype(str)
        if "pred" in sales.columns and "loi_band" not in sales.columns:
            sales = sales.rename(columns={"pred": "loi_band",
                                          "pred_score": "loi_score"})
        scoring_cols = [
            c for c in ("loi_band", "loi_score", "closing_readiness", "confidence")
            if c in sales.columns
        ]
        xl_idx = xl.set_index("call_id")
        sales_idx = sales.set_index("call_id")
        common = xl_idx.index.intersection(sales_idx.index)
        if len(common) and scoring_cols:
            xl_idx.update(sales_idx.loc[common, scoring_cols])
            xl = xl_idx.reset_index()
            frames[0] = xl
        known_ids = set(xl["call_id"])
        sales = sales[~sales["call_id"].isin(known_ids)]
        if not sales.empty:
            frames.append(sales)
    if EXTRA_PRED.exists():
        ep = pd.read_csv(EXTRA_PRED)
        ep["call_id"] = ep["call_id"].astype(str)
        # map prediction-file columns onto the Excel shape where needed
        if "pred" in ep.columns and "loi_band" not in ep.columns:
            ep = ep.rename(columns={"pred": "loi_band",
                                    "pred_score": "loi_score"})
        frames.append(ep)
    # export_loi_excel is intentionally a separate hand-off step, so a freshly
    # downloaded Kaggle batch can be newer than the dated xlsx. Include only
    # call_ids absent from the established sales/booking sources; this avoids a
    # stale latest-batch file overriding their richer CRM metadata later.
    if LATEST_BATCH.exists():
        latest = pd.read_csv(LATEST_BATCH)
        latest["call_id"] = latest["call_id"].astype(str)
        if "pred" in latest.columns and "loi_band" not in latest.columns:
            latest = latest.rename(columns={"pred": "loi_band",
                                            "pred_score": "loi_score"})
        known_ids = set(pd.concat(frames, ignore_index=True, sort=False)["call_id"])
        latest = latest[~latest["call_id"].isin(known_ids)]
        if not latest.empty:
            frames.append(latest)
    allsc = pd.concat(frames, ignore_index=True, sort=False)
    allsc = allsc.drop_duplicates("call_id", keep="last").set_index("call_id")
    return allsc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="write to the sheet")
    a = ap.parse_args(argv)

    import gspread
    gc = gspread.service_account(filename=str(SA_FILE))
    ws = gc.open_by_key(SHEET_ID).sheet1
    vals = ws.get_all_values()
    hdr, rows = vals[0], vals[1:]
    sheet = pd.DataFrame(rows, columns=hdr)
    # drop columns from a previous fill run so re-runs stay idempotent
    sheet = sheet.drop(columns=[c for c in OUT_HEADER if c in sheet.columns])
    sheet = sheet[sheet["meetingUuid"].str.strip() != ""].copy()
    n_rows = len(sheet)
    print(f"sheet rows (live): {n_rows}")
    # the sheet mixes '2026-07-08 12:00:00+00' and '2026-07-08T12:00:00.000Z'
    sheet["start"] = pd.to_datetime(sheet["startTime"], errors="coerce",
                                    utc=True, format="mixed")
    sheet["meetingId"] = sheet["meetingId"].astype(str)

    calls = pd.read_csv(DATA / "calls.csv", low_memory=False)
    z = calls[calls["source"] == "zoom"].copy()
    z["call_id"] = z["call_id"].astype(str)
    zinfo = z.set_index("call_id")[["call_type", "transcript_bytes"]]
    corpus_ids = set(zinfo.index)

    zm = pd.read_csv(DATA / "zoom_metadata.csv", low_memory=False)
    zm["meeting_id"] = (zm["meeting_id"].astype(str)
                        .str.replace(r"\.0$", "", regex=True))
    mid2uuid = zm.groupby("meeting_id")["meeting_uuid"].apply(list).to_dict()

    scores = load_scores()
    scored_ids = set(scores.index)
    staged = set()
    if STAGED_IDS.exists():
        staged = set(json.load(open(STAGED_IDS))["call_ids"])
    silent = set()
    silent_file = DATA / "whisper_silent_calls.json"
    if silent_file.exists():
        silent = set(json.load(open(silent_file))["call_ids"])

    now = pd.Timestamp.now(tz="UTC")

    def resolve(row):
        uu = str(row["meetingUuid"])
        if uu in corpus_ids:
            return uu, "scheduled_uuid"
        for iu in mid2uuid.get(row["meetingId"], []):
            if iu in corpus_ids:
                return iu, "instance_uuid"
        return "", ""

    res = sheet.apply(resolve, axis=1, result_type="expand")
    sheet["matched_call_id"], sheet["matched_via"] = res[0], res[1]

    def status(r):
        cid = r["matched_call_id"]
        if cid:
            if cid in scored_ids:
                return "scored"
            if cid in silent:
                return "verified_silent_no_speech"
            if cid in staged:
                return "scoring_in_progress"
            tb = zinfo.at[cid, "transcript_bytes"]
            if pd.isna(tb) or float(tb) < 500:
                return "recorded_but_empty_or_no_transcript"
            return "recorded_awaiting_score"
        if str(r.get("deletedAt", "")).strip() not in ("", "NULL"):
            return "cancelled"
        if pd.isna(r["start"]):
            return "no_start_time"
        if r["start"] > now:
            return "scheduled_future"
        if r["start"] < CORPUS_START:
            return "pre_2026-04-21_recording_purged"
        return "no_recording_found"

    sheet["loi_match_status"] = sheet.apply(status, axis=1)
    print(sheet["loi_match_status"].value_counts().to_string())

    enr = sheet.merge(scores[SCORE_COLS], how="left",
                      left_on="matched_call_id", right_index=True)
    block = enr[OUT_HEADER].astype(object)
    block = block.where(pd.notna(block), "")
    block = block.map(lambda v: str(v) if v != "" else "")

    print(f"rows with loi_band attached: {(block['loi_band'] != '').sum()}")

    if not a.go:
        print("[dry-run] nothing written; use --go")
        return 0

    last_col = chr(ord(FIRST_COL) + len(OUT_HEADER) - 1)
    payload = [OUT_HEADER] + block.values.tolist()
    ws.update(values=payload,
              range_name=f"{FIRST_COL}1:{last_col}{n_rows + 1}",
              value_input_option="RAW")
    print(f"[written] {FIRST_COL}1:{last_col}{n_rows + 1} "
          f"({len(OUT_HEADER)} cols x {n_rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
