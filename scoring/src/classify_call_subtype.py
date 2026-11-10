"""Add a `call_subtype` column to calls.csv: within sales calls, split
Sales-Call-1 (first consultation) vs follow-up, across all 3 platforms.

Two methods, best-available per source:

  - Zoom    : LEAD-SEQUENCING (the reliable method). Join `LEADID` from the CRM
              xlsx, order each lead's sales calls by date — first = SC1, the
              rest = follow_up. (An earlier version assumed "every Zoom call =
              SC1"; repeat/follow-up video calls are possible.)
              Zoom sales with no LEADID (newer calls not yet in the CRM export)
              default to SC1, flagged in the summary.
  - xCally / Genesys : DISPOSITION (no lead key joined yet — phone is PII and
              not stored). Human wrap-up code:
                FU_*            -> follow_up
                SC_VIA_TELEFONO -> SC1  (consultation by phone)
                SC_POST_FIRMA   -> post_sale
                else            -> unknown
              Genesys mostly carries system codes (ININ-*/GUID) -> unknown.

When the CRM phone join arrives (CRM owner), switch xCally/Genesys to cross-source
lead-sequencing too (a seller call after the lead's SC1 is a follow-up).

Only `call_type == 'sales'` rows get a subtype; booking/other stay blank.

Usage:  python -m src.classify_call_subtype
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict

import pandas as pd

from .config import DATA, CALLS_CSV

CRM_XLSX = DATA / "crm" / "calls2_zoom_merged.xlsx"


def _disposition_subtype(wrapup: str) -> str:
    w = (wrapup or "").strip().upper()
    if w.startswith("FU_"):
        return "follow_up"
    if w == "SC_VIA_TELEFONO":
        return "SC1"
    if w == "SC_POST_FIRMA":
        return "post_sale"
    return "unknown"


def _zoom_leadid_map() -> dict[str, str]:
    """call_id -> LEADID from the CRM xlsx (for Zoom lead-sequencing)."""
    if not CRM_XLSX.exists():
        return {}
    df = pd.read_excel(CRM_XLSX, sheet_name="merged")
    if "call_id" not in df.columns or "LEADID" not in df.columns:
        return {}
    out = {}
    for cid, lead in zip(df["call_id"], df["LEADID"]):
        if pd.notna(cid) and pd.notna(lead):
            out[str(cid)] = str(lead)
    return out


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = list(reader.fieldnames or [])
    if "call_subtype" not in cols:
        cols.insert(cols.index("call_type") + 1, "call_subtype")

    leadmap = _zoom_leadid_map()

    # 1) non-Zoom sales: disposition-based; booking/other: blank.
    for r in rows:
        if r.get("call_type") != "sales":
            r["call_subtype"] = ""
        elif r.get("source") == "zoom":
            r["call_subtype"] = ""   # filled by the sequencing pass below
        else:
            r["call_subtype"] = _disposition_subtype(r.get("wrapup_code", ""))

    # 2) Zoom sales: lead-sequence. First call per LEADID (by date) = SC1,
    #    the rest = follow_up. No LEADID -> SC1 (can't sequence; flagged).
    zoom_sales = [r for r in rows
                  if r.get("source") == "zoom" and r.get("call_type") == "sales"]
    by_lead: dict[str, list] = defaultdict(list)
    no_lead = 0
    for r in zoom_sales:
        lead = leadmap.get(r["call_id"])
        if lead is None:
            r["call_subtype"] = "SC1"   # unverified default
            no_lead += 1
        else:
            by_lead[lead].append(r)
    for lead, rs in by_lead.items():
        rs.sort(key=lambda r: (r.get("call_date") or "", r.get("start_time") or ""))
        for i, r in enumerate(rs):
            r["call_subtype"] = "SC1" if i == 0 else "follow_up"

    with CALLS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # summary
    by = defaultdict(Counter)
    for r in rows:
        if r.get("call_type") == "sales":
            by[r["source"]][r["call_subtype"]] += 1
    print(f"[updated] {CALLS_CSV}  (call_subtype on {len(rows)} rows)")
    print(f"Zoom lead-sequencing: {len(leadmap)} call→LEADID; "
          f"{len(zoom_sales)} zoom sales, {no_lead} had no LEADID (defaulted SC1)")
    subtypes = ["SC1", "follow_up", "post_sale", "unknown"]
    print("\nSALES call_subtype by source:")
    print(f"  {'source':9s} " + "".join(f"{s:>11s}" for s in subtypes) + f"{'total':>8s}")
    for src in ("zoom", "genesys", "xcally"):
        c = by[src]
        print(f"  {src:9s} " + "".join(f"{c.get(s,0):>11d}" for s in subtypes)
              + f"{sum(c.values()):>8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
