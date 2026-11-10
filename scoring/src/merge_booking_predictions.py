"""Merge Kaggle predictions into the Google Sheet booking score feed.

The combined daily Kaggle batch can contain sales and booking calls, but
booking predictions must stay separate from the sales master. This command
validates every incoming call against calls.csv, then appends/deduplicates the
predictions in predictions_qwen3_8b_gsheet_booking.csv.

Usage:
  python -m src.merge_booking_predictions predictions_kaggle_latest.csv
  python -m src.merge_booking_predictions predictions_kaggle_latest.csv --dry-run
"""
from __future__ import annotations

import argparse
import shutil

import pandas as pd

from .config import DATA, ROOT
from .merge_predictions import attach_leadid


MASTER = ROOT / "predictions_qwen3_8b_gsheet_booking.csv"
CALLS = DATA / "calls.csv"
BANDS = {
    "Freddo",
    "Tiepido",
    "Calduccio",
    "Caldo",
    "Non interessato",
    "Non idoneo",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new_csv", help="Kaggle predictions CSV containing booking calls")
    ap.add_argument("--dry-run", action="store_true", help="validate/report only")
    a = ap.parse_args(argv)

    new = pd.read_csv(a.new_csv)
    if "call_id" not in new.columns:
        raise SystemExit("incoming predictions are missing call_id")
    new["call_id"] = new["call_id"].astype(str)

    calls = pd.read_csv(CALLS, low_memory=False, dtype={"call_id": str})
    zoom = calls[calls["source"] == "zoom"].copy()
    call_types = (zoom.drop_duplicates("call_id", keep="last")
                  .set_index("call_id")["call_type"])
    incoming_types = new["call_id"].map(call_types)
    unknown = new.loc[incoming_types.isna(), "call_id"].tolist()
    non_booking = new.loc[
        incoming_types.notna() & incoming_types.ne("booking"), "call_id"
    ].tolist()
    if unknown or non_booking:
        raise SystemExit(
            "refusing booking merge: "
            f"{len(unknown)} unknown call_ids, {len(non_booking)} non-booking call_ids"
        )

    if MASTER.exists():
        master = pd.read_csv(MASTER, dtype={"call_id": str})
    else:
        master = pd.DataFrame(columns=new.columns)

    before = len(master)
    overlap = len(set(master["call_id"]) & set(new["call_id"]))
    if master.empty:
        combined = new.copy()
    else:
        combined = pd.concat([master, new], ignore_index=True, sort=False)
    combined = (combined.drop_duplicates("call_id", keep="last")
                .reset_index(drop=True))
    combined = attach_leadid(combined)
    parse_fails = (
        int((~new["pred"].astype(str).isin(BANDS)).sum())
        if "pred" in new.columns else 0
    )

    print(f"booking master rows : {before}")
    print(
        f"incoming rows       : {len(new)} "
        f"(overlap {overlap}, parse-fails {parse_fails})"
    )
    print(
        f"=> merged total     : {len(combined)} "
        f"(net new {len(combined) - before}, {overlap} updated)"
    )

    if a.dry_run:
        print("(dry-run: nothing written)")
        return 0

    if MASTER.exists():
        backup = MASTER.with_suffix(".csv.bak")
        shutil.copy(MASTER, backup)
        print(f"[backup] {backup}")
    combined.to_csv(MASTER, index=False)
    print(f"[saved] {MASTER} ({len(combined)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
