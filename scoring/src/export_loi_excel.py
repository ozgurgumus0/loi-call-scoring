"""Part 1 OUTPUT — write the leadid-tagged LOI scores to an Excel file.

Takes the production scores (predictions_qwen3_8b_sales_full.csv, already
leadid-tagged by merge_predictions) and joins:
  * call_date / meeting_id / call_type   from data/calls.csv  (by call_id)
  * channel / region / contract_signed   from the newest CRM calendar export
                                          (by meetingId)
-> outputs/loi_scores_<today>.xlsx  with two sheets:
     "loi_scores"  one row per scored call, newest first
     "summary"     band counts, date range, leadid coverage

This is the hand-off artifact until the Snowflake sink is wired (then the same
frame just gets written to a table instead of an xlsx).

Usage:
  python -m src.export_loi_excel                       # full master -> dated xlsx
  python -m src.export_loi_excel --since 2026-06-12    # only recent calls
  python -m src.export_loi_excel --out outputs/loi_today.xlsx
"""
from __future__ import annotations
import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from .config import ROOT, DATA
from .merge_predictions import _newest, CALENDAR_GLOB, _clean_leadid

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MASTER = ROOT / "predictions_qwen3_8b_sales_full.csv"
OUT_DIR = ROOT / "outputs"
COLS = ["leadid", "call_id", "meeting_id", "call_date", "call_type",
        "channel", "region", "loi_band", "loi_score", "closing_readiness",
        "confidence", "contract_signed"]

# Rubric LOI scale (src/rubric.py): Freddo=1, Tiepido=4, Calduccio=6, Caldo=7;
# the exclusions "Non idoneo" / "Non interessato" (and parse-fails) score null.
BAND_TO_SCORE = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7}


def _crm_context() -> pd.DataFrame:
    """meetingId -> channel/region/contract_signed from the newest calendar export."""
    f = _newest(CALENDAR_GLOB)
    if not f:
        return pd.DataFrame(columns=["meetingId", "channel", "region", "contract_signed"])
    df = pd.read_excel(f, sheet_name="merged")
    df["meetingId"] = pd.to_numeric(df["meetingId"], errors="coerce").astype("Int64")
    reg = df.get("LEAD_ZIPREGION")
    if reg is None:
        reg = df.get("LEAD_REGION")
    out = pd.DataFrame({
        "meetingId": df["meetingId"],
        "channel": df.get("LEAD_CHANNEL"),
        "region": reg,
        "contract_signed": pd.to_numeric(df.get("CONTRACT_SIGNED"), errors="coerce"),
    }).dropna(subset=["meetingId"]).drop_duplicates(subset="meetingId")
    return out


def build(since: str | None = None) -> pd.DataFrame:
    pred = pd.read_csv(MASTER)
    pred["call_id"] = pred["call_id"].astype(str)
    pred = pred.rename(columns={"pred": "loi_band"})

    calls = pd.read_csv(DATA / "calls.csv", low_memory=False)
    calls["call_id"] = calls["call_id"].astype(str)
    calls["meetingId"] = pd.to_numeric(calls["meeting_id"], errors="coerce").astype("Int64")
    keep = calls[["call_id", "meeting_id", "meetingId", "call_date", "call_type"]]
    df = pred.merge(keep, on="call_id", how="left")
    df = df.merge(_crm_context(), on="meetingId", how="left")

    if since:
        df = df[pd.to_datetime(df["call_date"], errors="coerce") >= pd.Timestamp(since)]

    # numeric LOI on the rubric scale (1/4/6/7); blank for exclusions / parse-fails
    df["loi_score"] = df["loi_band"].map(BAND_TO_SCORE).astype("Int64")

    # tidy leadid (in case the master holds it as a float-ish string)
    df["leadid"] = df["leadid"].map(_clean_leadid) if "leadid" in df.columns else ""
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLS].sort_values("call_date", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    leadid_cov = int((df["leadid"].astype(str).str.strip() != "").sum())
    bands = df["loi_band"].value_counts()
    dates = pd.to_datetime(df["call_date"], errors="coerce")
    rows = [
        ("scored calls", n),
        ("with leadid", f"{leadid_cov}/{n}"),
        ("date range", f"{dates.min():%Y-%m-%d} -> {dates.max():%Y-%m-%d}" if n else "-"),
        ("", ""),
        ("--- band mix ---", ""),
    ] + [(str(k), int(v)) for k, v in bands.items()]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="only calls on/after YYYY-MM-DD")
    ap.add_argument("--out", help="output xlsx (default outputs/loi_scores_<today>.xlsx)")
    a = ap.parse_args(argv)

    df = build(a.since)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(a.out) if a.out else OUT_DIR / f"loi_scores_{date.today().isoformat()}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="loi_scores", index=False)
        _summary(df).to_excel(xl, sheet_name="summary", index=False)

    cov = int((df["leadid"].astype(str).str.strip() != "").sum())
    print(f"[saved] {out}")
    print(f"  {len(df)} scored calls, {cov} with leadid")
    print(f"  band mix: {df['loi_band'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
