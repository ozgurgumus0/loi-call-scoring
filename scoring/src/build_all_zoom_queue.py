"""Build one scoring queue for every usable unscored Zoom call.

Unlike ``build_score_queue`` (sales only) and ``build_gsheet_queue``
(sales plus sheet-matched booking calls), this command covers the full Zoom
corpus for both supported call types. Predictions must be ingested with
``merge_mixed_predictions`` so sales and booking masters stay separate.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .config import DATA, ROOT


SALES_MASTER = ROOT / "predictions_qwen3_8b_sales_full.csv"
BOOKING_MASTER = ROOT / "predictions_qwen3_8b_gsheet_booking.csv"
MIN_VTT = 500


def _ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["call_id"])["call_id"].astype(str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(DATA / "train" / f"ids_zoom_all_unscored_{date.today():%Y%m%d}.json"),
    )
    ap.add_argument("--since", help="only calls on/after YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    calls = pd.read_csv(DATA / "calls.csv", low_memory=False)
    zoom = calls[
        (calls["source"] == "zoom")
        & (calls["call_type"].isin(["sales", "booking"]))
    ].copy()
    zoom["call_id"] = zoom["call_id"].astype(str)
    zoom = zoom.drop_duplicates("call_id", keep="last")
    if args.since:
        call_dates = pd.to_datetime(zoom["call_date"], errors="coerce")
        zoom = zoom[call_dates >= pd.Timestamp(args.since)].copy()

    sales_scored = _ids(SALES_MASTER)
    booking_scored = _ids(BOOKING_MASTER)
    silent_path = DATA / "whisper_silent_calls.json"
    silent = (
        set(json.loads(silent_path.read_text(encoding="utf-8"))["call_ids"])
        if silent_path.exists()
        else set()
    )

    def is_scored(row) -> bool:
        master = sales_scored if row["call_type"] == "sales" else booking_scored
        return row["call_id"] in master

    zoom["is_scored"] = zoom.apply(is_scored, axis=1)
    transcript_bytes = pd.to_numeric(zoom["transcript_bytes"], errors="coerce").fillna(0)
    zoom["usable_transcript"] = transcript_bytes >= MIN_VTT
    zoom["is_silent"] = zoom["call_id"].isin(silent)

    queue_frame = zoom[
        ~zoom["is_scored"] & zoom["usable_transcript"] & ~zoom["is_silent"]
    ].copy()
    queue_frame = queue_frame.sort_values(["call_date", "start_time", "call_id"])
    queue = queue_frame["call_id"].tolist()

    print(f"zoom calls (sales + booking): {len(zoom)}")
    print("corpus by call type:")
    print(zoom["call_type"].value_counts().to_string())
    print("scoreable unscored by call type:")
    if queue_frame.empty:
        print("none")
    else:
        print(queue_frame["call_type"].value_counts().to_string())
    unusable = zoom[~zoom["is_scored"] & ~zoom["usable_transcript"]]
    print(f"unscored without usable transcript: {len(unusable)}")
    print(f"=> GLOBAL MIXED QUEUE: {len(queue)}")

    if args.dry_run:
        print("(dry-run: nothing written)")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "call_ids": queue,
                "source": "zoom",
                "note": (
                    "all usable unscored Zoom sales + booking calls; "
                    f"since={args.since or 'all'}"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[saved] {out} ({len(queue)} ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
