"""Populate calls.csv (and labels.csv) with human rep LOI + real contract
outcomes from the CRM export, so the app surfaces them.

Source: data/crm/crm_merged_with_loi.xlsx, sheet `merged`
  - LOI              human sales-rep LOI (1/4/6/7) -> mapped to band name
  - CONTRACT_SIGNED  1/0 real signed-contract outcome -> "signed"/"not_signed"
  - LOI_SURVEY_STATUS  keep 'ok' only

Join: xlsx.meetingId (numeric) -> calls.csv.meeting_id -> (source=zoom, call_id).

GUARDRAIL: the human LOI lands in calls.csv's `rep_loi` column, which the
training pipeline (src.build_training_jsonl) NEVER reads — it trains only on
Sonnet scores from outputs/scores/. So this is display/analysis only; it can
not leak into training.

Writes:
  - data/labels.csv        upsert rep_loi + conversion_outcome (re-ingest safe;
                           build_calls_csv.merge backfills from here)
  - data/calls.csv         direct update so the /data page reflects it now
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from .build_calls_csv import CALLS_COLS, _read_csv
from .config import CALLS_CSV, DATA, LABELS_CSV

DEFAULT_XLSX = DATA / "crm" / "crm_merged_with_loi.xlsx"
LOI_TO_BAND = {1: "Freddo", 4: "Tiepido", 6: "Calduccio", 7: "Caldo"}


def _load_xlsx(xlsx_path: Path) -> dict[int, dict]:
    """{meetingId(int) -> {rep_loi, conversion_outcome}} from the CRM export."""
    d = pd.read_excel(xlsx_path, sheet_name="merged")
    d["meetingId"] = pd.to_numeric(d["meetingId"], errors="coerce")
    d = d[d["LOI_SURVEY_STATUS"].isna() | (d["LOI_SURVEY_STATUS"] == "ok")]
    out: dict[int, dict] = {}
    for _, r in d.iterrows():
        mid = r["meetingId"]
        if pd.isna(mid):
            continue
        loi = r.get("LOI")
        rep_loi = LOI_TO_BAND.get(int(loi)) if pd.notna(loi) else ""
        cs = r.get("CONTRACT_SIGNED")
        if pd.isna(cs):
            conv = ""
        else:
            conv = "signed" if int(cs) == 1 else "not_signed"
        out[int(mid)] = {"rep_loi": rep_loi or "", "conversion_outcome": conv}
    return out


def run(xlsx_path: Path = DEFAULT_XLSX) -> None:
    by_meeting = _load_xlsx(xlsx_path)
    print(f"CRM rows with usable meetingId: {len(by_meeting)}")

    calls = _read_csv(CALLS_CSV)
    # meeting_id(str) -> call_id for zoom rows
    mid_to_cid: dict[str, str] = {}
    for r in calls:
        if r.get("source") == "zoom":
            mid = (r.get("meeting_id") or "").strip()
            if mid:
                mid_to_cid[mid] = r["call_id"]

    # Resolve per-call values keyed by (source, call_id)
    per_call: dict[str, dict] = {}  # call_id -> {rep_loi, conversion_outcome}
    n_loi = n_conv = 0
    for mid_int, vals in by_meeting.items():
        cid = mid_to_cid.get(str(mid_int))
        if not cid:
            continue
        per_call[cid] = vals
        if vals["rep_loi"]:
            n_loi += 1
        if vals["conversion_outcome"]:
            n_conv += 1
    print(f"Matched to calls.csv zoom rows: {len(per_call)} "
          f"(rep_loi={n_loi}, conversion={n_conv})")

    # --- 1. Update calls.csv directly (force-set the two columns) ----------
    updated = 0
    for r in calls:
        if r.get("source") != "zoom":
            continue
        v = per_call.get(r["call_id"])
        if not v:
            continue
        before = (r.get("rep_loi", ""), r.get("conversion_outcome", ""))
        if v["rep_loi"]:
            r["rep_loi"] = v["rep_loi"]
        if v["conversion_outcome"]:
            r["conversion_outcome"] = v["conversion_outcome"]
        if (r.get("rep_loi", ""), r.get("conversion_outcome", "")) != before:
            updated += 1
    with CALLS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CALLS_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(calls)
    print(f"[saved] {CALLS_CSV}  ({updated} rows updated)")

    # --- 2. Upsert labels.csv (so a future re-ingest merge preserves them) -
    lab_rows = _read_csv(LABELS_CSV) if LABELS_CSV.exists() else []
    lab_cols = (list(lab_rows[0].keys()) if lab_rows else
                ["call_id", "rep_loi", "rep_reasoning",
                 "conversion_outcome", "rep_id", "call_date", "source"])
    lab_idx = {(r.get("source") or "zoom", r.get("call_id")): r
               for r in lab_rows}
    cid_to_call = {r["call_id"]: r for r in calls}
    added = 0
    for cid, v in per_call.items():
        key = ("zoom", cid)
        row = lab_idx.get(key)
        if row is None:
            src_row = cid_to_call.get(cid, {})
            row = {c: "" for c in lab_cols}
            row.update({"call_id": cid, "source": "zoom",
                        "rep_id": src_row.get("rep_id", ""),
                        "call_date": src_row.get("call_date", "")})
            lab_rows.append(row)
            lab_idx[key] = row
            added += 1
        if v["rep_loi"]:
            row["rep_loi"] = v["rep_loi"]
        if v["conversion_outcome"]:
            row["conversion_outcome"] = v["conversion_outcome"]
    with LABELS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lab_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(lab_rows)
    print(f"[saved] {LABELS_CSV}  ({len(lab_rows)} rows, {added} new)")


def _cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    a = ap.parse_args()
    if not a.xlsx.exists():
        raise SystemExit(f"xlsx not found: {a.xlsx}")
    run(a.xlsx)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
