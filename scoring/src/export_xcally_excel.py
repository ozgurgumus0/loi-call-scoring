"""Merge xCally Kaggle prediction batches + metadata into the deliverable Excel.

Joins prediction CSVs (rows with source='xcally') against the emit_xcally meta
sidecar and writes:
  predictions_qwen3_8b_xcally.csv     — enriched master (repo root, append+dedup
                                        newest-wins like merge_predictions)
  outputs/loi_scores_xcally_<date>.xlsx
      scores               one row per scored call + metadata
      summary              band counts overall and by queue group
      disposition_x_band   agent disposition vs Qwen band cross-tab

No Google Sheet writes — xCally output is Excel-only (user decision 2026-07-10).

Usage:  python -m src.export_xcally_excel <predictions1.csv> [predictions2.csv ...]
        (no args = just rebuild the Excel from the existing xCally master)
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from .config import DATA

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "predictions_qwen3_8b_xcally.csv"
BAND_ORDER = ["Caldo", "Calduccio", "Tiepido", "Freddo",
              "Non interessato", "Non idoneo"]


def queue_group(q: str) -> str:
    q = q or ""
    if q.startswith("Booker") or q in ("BOOKING_INBOUND", "BOOKING_NOT_INTERESTED"):
        return "booking"
    if q == "":
        return "outbound_no_queue"
    if q in ("CUSTOMER_CARE_BL", "ACCOUNT_MANAGEMENT_BL"):
        return "care_account"
    return "other"


def _newest_meta() -> Path:
    cands = sorted((DATA / "xcally").glob("omnibus_meta_*.csv"))
    if not cands:
        raise SystemExit("no omnibus_meta_*.csv under data/xcally — run emit_xcally first")
    return cands[-1]


def main(pred_paths: list[Path]) -> int:
    frames = []
    if MASTER.exists():
        frames.append(pd.read_csv(MASTER, dtype={"call_id": str}))
        print(f"existing master: {len(frames[0])} rows")
    for p in pred_paths:
        df = pd.read_csv(p, dtype={"call_id": str})
        df = df[df["source"] == "xcally"].copy()
        print(f"{p.name}: {len(df)} xcally rows")
        frames.append(df)
    if not frames:
        raise SystemExit("nothing to merge and no existing master")

    pred_cols = ["call_id", "source", "pred", "pred_score",
                 "confidence", "closing_readiness"]
    preds = pd.concat(frames, ignore_index=True)
    preds = preds[[c for c in pred_cols if c in preds.columns]]
    preds = preds.drop_duplicates("call_id", keep="last")
    n_fail = preds["pred"].isna().sum()
    print(f"merged: {len(preds)} unique calls ({n_fail} parse-fails)")

    meta_path = _newest_meta()
    meta = pd.read_csv(meta_path, dtype={"call_id": str})
    out = preds.merge(meta.drop(columns=["in_scope", "skip_reason"]),
                      on="call_id", how="left")
    unmatched = out["created_at"].isna().sum()
    if unmatched:
        print(f"  WARNING: {unmatched} predictions missing from {meta_path.name}")
    out["queue_group"] = out["queue"].fillna("").map(queue_group)
    out["call_date"] = pd.to_datetime(out["created_at"], errors="coerce").dt.date
    out = out.sort_values("created_at").reset_index(drop=True)

    out.to_csv(MASTER, index=False)
    print(f"[saved] {MASTER}  ({len(out)} rows)")

    # coverage vs scope
    in_scope = meta[meta["in_scope"]]
    missing = set(in_scope["call_id"]) - set(out["call_id"])
    print(f"scope coverage: {len(out)}/{len(in_scope)} scored"
          + (f" — {len(missing)} still pending" if missing else ""))

    tag = dt.date.today().isoformat()
    # "xcally_" PREFIX, not suffix: fill_gsheet_loi and the frontend pick the
    # newest outputs/loi_scores_*.xlsx by string sort, and a loi_scores_xcally_*
    # name sorts after every date -> it would shadow the real Zoom master.
    xlsx = ROOT / "outputs" / f"xcally_loi_scores_{tag}.xlsx"

    scores = out[["call_id", "call_date", "agent", "queue", "queue_group",
                  "type", "direction", "caller_num", "user_disposition",
                  "duration_min", "customer_words", "pred", "pred_score",
                  "confidence", "closing_readiness"]].rename(
        columns={"pred": "loi_band", "pred_score": "loi_score"})

    band = out["pred"].fillna("(parse-fail)")
    summary = (band.value_counts().rename_axis("loi_band")
               .reset_index(name="calls"))
    summary["share"] = (summary["calls"] / len(out)).round(3)
    by_group = (out.assign(loi_band=band)
                .pivot_table(index="loi_band", columns="queue_group",
                             values="call_id", aggfunc="count", fill_value=0)
                .reindex(BAND_ORDER + ["(parse-fail)"]).dropna(how="all")
                .astype(int).reset_index())

    xtab = (out.assign(loi_band=band,
                       dispo=out["user_disposition"].replace("", "(none)"))
            .pivot_table(index="dispo", columns="loi_band",
                         values="call_id", aggfunc="count", fill_value=0))
    xtab = xtab[[c for c in BAND_ORDER + ["(parse-fail)"] if c in xtab.columns]]
    xtab = xtab.loc[xtab.sum(axis=1).sort_values(ascending=False).index].reset_index()

    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        scores.to_excel(xw, sheet_name="scores", index=False)
        summary.to_excel(xw, sheet_name="summary", index=False, startrow=0)
        by_group.to_excel(xw, sheet_name="summary", index=False,
                          startrow=len(summary) + 3)
        xtab.to_excel(xw, sheet_name="disposition_x_band", index=False)
    print(f"[saved] {xlsx}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="*", help="prediction csv(s) to merge in")
    a = ap.parse_args()
    raise SystemExit(main([Path(p) for p in a.preds]))
