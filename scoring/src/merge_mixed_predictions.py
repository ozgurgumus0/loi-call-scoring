"""Split a mixed CSV batch by call_type and merge each score feed safely."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DATA, ROOT
from . import merge_booking_predictions, merge_predictions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    incoming = pd.read_csv(args.predictions_csv, dtype={"call_id": str})
    calls = pd.read_csv(DATA / "calls.csv", low_memory=False, dtype={"call_id": str})
    types = (
        calls.drop_duplicates("call_id", keep="last")
        .set_index("call_id")["call_type"]
    )
    incoming["_call_type"] = incoming["call_id"].map(types)
    unknown = incoming[incoming["_call_type"].isna()]
    if not unknown.empty:
        raise SystemExit(f"{len(unknown)} prediction rows have unknown call_id")

    sales = incoming[incoming["_call_type"] == "sales"].drop(columns="_call_type")
    booking = incoming[incoming["_call_type"] == "booking"].drop(columns="_call_type")
    other = incoming[~incoming["_call_type"].isin(["sales", "booking"])]
    if not other.empty:
        raise SystemExit(f"{len(other)} prediction rows have unsupported call_type")

    print(f"incoming: {len(incoming)} = {len(sales)} sales + {len(booking)} booking")
    suffix = "_dry" if args.dry_run else ""
    if not sales.empty:
        sales_path = ROOT / f"predictions_kaggle_latest_sales{suffix}.csv"
        sales.to_csv(sales_path, index=False)
        merge_predictions.main(
            [str(sales_path)] + (["--dry-run"] if args.dry_run else [])
        )
    if not booking.empty:
        booking_path = ROOT / f"predictions_kaggle_latest_booking{suffix}.csv"
        booking.to_csv(booking_path, index=False)
        merge_booking_predictions.main(
            [str(booking_path)] + (["--dry-run"] if args.dry_run else [])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
