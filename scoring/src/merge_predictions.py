"""Merge a Kaggle predictions.csv into predictions_qwen3_8b_sales_full.csv.

Append the freshly downloaded predictions, dedup on call_id (newest wins), and
write back with a timestamped backup. This is the "ingest" step that closes the
periodic loop: ingest_zoom -> build_calls_csv -> build_score_queue ->
emit_test_prompting -> [Kaggle] -> merge_predictions.

Usage:
  python -m src.merge_predictions ~/Downloads/predictions.csv
  python -m src.merge_predictions ~/Downloads/predictions.csv --dry-run
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import pandas as pd

from .config import ROOT

MASTER = ROOT / "predictions_qwen3_8b_sales_full.csv"

# --- leadid enrichment ------------------------------------------------------
# The Kaggle kernel only knows call_id (the CRM is never uploaded). leadid is
# joined back LOCALLY from the CRM merge so the output is joinable to the lead
# universe (and downstream to clickid_map / Option C). Source = the newest
# calls2_zoom_merged*.xlsx (its `merged` sheet has both call_id + LEADID; the
# zoom_calendar_event_merged file has LEADID but no call_id, so it can't join).
CRM_DIR = ROOT / "data" / "crm"
CALLS2_GLOB = "calls2_zoom_merged*.xlsx"             # has call_id + LEADID (direct)
CALENDAR_GLOB = "zoom_calendar_event_merged*.xlsx"  # has meetingId + LEADID (wider/newer)
CALLS_CSV = ROOT / "data" / "calls.csv"             # call_id <-> meeting_id bridge


def _clean_leadid(v) -> str:
    """'213375.0' / 213375.0 / NaN -> '213375' / ''."""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return s


def _newest(glob: str) -> Path | None:
    cands = sorted(CRM_DIR.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_leadid_map():
    """call_id -> leadid, combined from two CRM sources for max coverage:
      (1) DIRECT call_id->LEADID from the newest calls2_zoom_merged*.xlsx;
      (2) call_id->meeting_id (calls.csv) -> LEADID from the newest calendar
          export — covers recent calls absent from the calls2 snapshot.
    Direct wins on conflict. Returns (map, [source descriptions])."""
    m: dict[str, str] = {}
    sources: list[str] = []

    f1 = _newest(CALLS2_GLOB)
    if f1:
        df = pd.read_excel(f1, sheet_name="merged")
        if {"call_id", "LEADID"} <= set(df.columns):
            for cid, lid in zip(df["call_id"].astype(str), df["LEADID"]):
                lid = _clean_leadid(lid)
                if cid and cid != "nan" and lid:
                    m.setdefault(cid, lid)
            sources.append(f1.name)

    f2 = _newest(CALENDAR_GLOB)
    if f2 and CALLS_CSV.exists():
        cal = pd.read_excel(f2, sheet_name="merged")
        if {"meetingId", "LEADID"} <= set(cal.columns):
            mid2lead = {}
            for mid, lid in zip(pd.to_numeric(cal["meetingId"], errors="coerce"), cal["LEADID"]):
                lid = _clean_leadid(lid)
                if pd.notna(mid) and lid:
                    mid2lead[int(mid)] = lid
            calls = pd.read_csv(CALLS_CSV, low_memory=False)
            calls = calls[calls["source"] == "zoom"]
            for cid, mid in zip(calls["call_id"].astype(str),
                                pd.to_numeric(calls["meeting_id"], errors="coerce")):
                if cid in m or pd.isna(mid):  # prefer the direct map
                    continue
                lid = mid2lead.get(int(mid))
                if lid:
                    m[cid] = lid
            sources.append(f"{f2.name} (via meeting_id)")
    return m, sources


def attach_leadid(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add/refresh a `leadid` column (right after call_id) by joining
    call_id -> LEADID from the local CRM (direct + meeting_id cross-walk).
    Unmatched rows get ''."""
    m, sources = load_leadid_map()
    df = df.copy()
    df["call_id"] = df["call_id"].astype(str)
    df["leadid"] = df["call_id"].map(m).fillna("")
    if verbose:
        matched = int((df["leadid"].astype(str) != "").sum())
        src = " + ".join(sources) if sources else "NO CRM FILE FOUND"
        print(f"[leadid] matched {matched}/{len(df)} call_ids -> leadid  (source: {src})")
    cols = list(df.columns)
    cols.remove("leadid")
    cols.insert(cols.index("call_id") + 1, "leadid")
    return df[cols]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new_csv", help="path to the predictions.csv downloaded from Kaggle")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    a = ap.parse_args(argv)

    new = pd.read_csv(a.new_csv)
    new["call_id"] = new["call_id"].astype(str)
    if MASTER.exists():
        master = pd.read_csv(MASTER)
        master["call_id"] = master["call_id"].astype(str)
    else:
        master = pd.DataFrame(columns=new.columns)

    before = len(master)
    overlap = len(set(master["call_id"]) & set(new["call_id"]))
    # new rows last so dedup(keep='last') lets re-scores overwrite old scores
    combined = (pd.concat([master, new], ignore_index=True)
                .drop_duplicates(subset="call_id", keep="last")
                .reset_index(drop=True))
    net_new = len(combined) - before

    # attach/refresh leadid for the WHOLE master (back-fills old rows too)
    combined = attach_leadid(combined)

    # parse-fail accounting (pred blank / not a band)
    bands = {"Freddo", "Tiepido", "Calduccio", "Caldo", "Non interessato", "Non idoneo"}
    pf = int((~new["pred"].astype(str).isin(bands)).sum()) if "pred" in new.columns else 0

    print(f"master rows         : {before}")
    print(f"incoming rows       : {len(new)}  (overlap w/ master {overlap}, parse-fails {pf})")
    print(f"=> merged total     : {len(combined)}  (net new {net_new}, {overlap} updated)")

    if a.dry_run:
        print("(dry-run: nothing written)")
        return 0

    if MASTER.exists():
        bak = MASTER.with_suffix(".csv.bak")
        shutil.copy(MASTER, bak)
        print(f"[backup] {bak}")
    combined.to_csv(MASTER, index=False)
    print(f"[saved] {MASTER}  ({len(combined)} rows)")
    print("NEXT: refresh the Option C feed -> "
          "cd ../option-c-value-bidding && python run_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
