"""Join CRM lead profile + NETSC1 outcome from the user-provided merged xlsx
onto `data/calls.csv`, producing `data/calls_enriched.csv`.

Source file (received 2026-06-02):
    zoom_calendar_event_merged.xlsx — sheet `merged`, 5,136 VD events x Zoom
    meetings in window 2026-03-26..2026-06-04, 100% Zoom-calendar match.

Join key: calls.csv.meeting_id (Zoom rows only)  <->  merged.meetingId.
Genesys rows get NaN for all added columns.

Usage:
    python -m src.join_crm                    # uses default Downloads path
    python -m src.join_crm --xlsx <path>      # override

The added columns (CRM_COLS) carry an `lead_` prefix in the output to make
provenance obvious downstream.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import CALLS_CSV, DATA

DEFAULT_XLSX = Path(
    str(Path.home() / "Downloads" / "zoom_calendar_event_merged.xlsx")
)
OUT_CSV = DATA / "calls_enriched.csv"

# Columns from the xlsx that already duplicate calls.csv (we have these on
# the Zoom-meeting side). Dropping avoids the suffix-mangling that pandas
# does on overlapping column names.
XLSX_DROP_DUP = {
    "id", "calendarEventId", "startTime", "createdAt", "deletedAt",
}


def _rename(c: str) -> str:
    if c == "meetingId":
        return "_meeting_id_xlsx"
    if c == "NETSC1":
        return "crm_netsc1"
    if c == "GROSSSC1":
        return "crm_grosssc1"
    if c == "TSSTART_DATE":
        return "crm_appt_date"
    if c == "LEADID":
        return "crm_lead_id"
    if c.startswith("LEAD_"):
        return "lead_" + c[len("LEAD_"):].lower()
    return c.lower()


def join(xlsx_path: Path) -> dict:
    calls = pd.read_csv(CALLS_CSV)
    m = pd.read_excel(xlsx_path, sheet_name="merged")
    m = m.drop(columns=[c for c in XLSX_DROP_DUP if c in m.columns])
    m = m.rename(columns={c: _rename(c) for c in m.columns})

    # Nullable Int64 on both sides for clean equi-join.
    calls["meeting_id"] = pd.to_numeric(calls["meeting_id"], errors="coerce").astype("Int64")
    m["_meeting_id_xlsx"] = pd.to_numeric(m["_meeting_id_xlsx"], errors="coerce").astype("Int64")

    # Dedup xlsx side defensively (file already 1:1 but cheap insurance).
    m = m.drop_duplicates(subset="_meeting_id_xlsx", keep="first")

    enriched = calls.merge(
        m, how="left",
        left_on="meeting_id", right_on="_meeting_id_xlsx",
        validate="m:1",
    ).drop(columns="_meeting_id_xlsx")

    enriched.to_csv(OUT_CSV, index=False)

    zoom_mask = enriched["source"] == "zoom"
    matched = zoom_mask & enriched["crm_netsc1"].notna()
    return {
        "rows_total":          len(enriched),
        "rows_zoom":           int(zoom_mask.sum()),
        "rows_genesys":        int((enriched["source"] == "genesys").sum()),
        "zoom_matched_crm":    int(matched.sum()),
        "zoom_unmatched_crm":  int((zoom_mask & ~matched).sum()),
        "match_rate_zoom_pct": round(100 * matched.sum() / max(1, zoom_mask.sum()), 2),
        "netsc1_pos_rate_matched": round(
            float(enriched.loc[matched, "crm_netsc1"].mean()), 3
        ),
        "out_file": str(OUT_CSV),
    }


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX,
                    help=f"Path to the merged xlsx (default: {DEFAULT_XLSX}).")
    args = ap.parse_args()
    if not args.xlsx.exists():
        raise SystemExit(f"xlsx not found: {args.xlsx}")
    r = join(args.xlsx)
    for k, v in r.items():
        print(f"{k:24s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
