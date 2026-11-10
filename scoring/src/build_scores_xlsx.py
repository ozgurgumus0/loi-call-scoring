"""Build outputs/reports/loi_scores.xlsx from outputs/scores/*.json + calls.csv.

Two sheets:
- "Scores": one row per scored call, all useful fields, color-coded by outcome.
- "Summary": distribution counts vs the authored 30/50/15/5.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import CALLS_CSV, EXPECTED_DIST, OUT_REPORTS, OUT_SCORES

OUTCOME_COLORS = {
    "Caldo":           "C6EFCE",  # green
    "Calduccio":       "FFEB9C",  # yellow
    "Tiepido":         "FFD8A8",  # orange
    "Freddo":          "F4CCCC",  # red
    "Non interessato": "D9D9D9",  # gray
    "Non idoneo":      "B7B7B7",  # darker gray
}
HEADER_FILL = PatternFill("solid", fgColor="0B3D66")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def _load_calls_meta() -> dict[tuple[str, str], dict]:
    out = {}
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[(r["source"], r["call_id"])] = r
    return out


def _transcript_chars(meta: dict) -> int:
    """Best-effort transcript size from the on-disk file."""
    src = meta.get("source")
    tf = meta.get("transcript_file") or ""
    from .config import DATA, GENESYS_TRANSCRIPT_DIR
    if src == "zoom" and tf:
        p = DATA / tf
        return p.stat().st_size if p.exists() else 0
    if src == "genesys":
        p = GENESYS_TRANSCRIPT_DIR / f"{meta['call_id']}.txt"
        return p.stat().st_size if p.exists() else 0
    return 0


def build() -> Path:
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    meta = _load_calls_meta()

    rows = []
    for f in sorted(OUT_SCORES.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        s = d.get("score", {})
        tp = s.get("tree_path") or {}
        m = meta.get((d["source"], d["call_id"]), {})
        ev = s.get("evidence_quotes") or []
        rows.append({
            "source":             d["source"],
            "call_id":            d["call_id"],
            "call_date":          m.get("call_date", ""),
            "start_time":         m.get("start_time", ""),
            "duration_min":       round(int(m.get("duration_sec") or 0) / 60, 1),
            "call_type":          m.get("call_type", ""),
            "rep_id":             m.get("rep_id", ""),
            "outcome":            s.get("outcome", ""),
            "score":              s.get("score"),
            "q1":                 tp.get("q1"),
            "q2":                 tp.get("q2"),
            "q3":                 tp.get("q3"),
            "q4":                 tp.get("q4"),
            "time_signal":        s.get("time_signal"),
            "behavior_mod":       "Y" if s.get("behavior_modifier_applied") else "",
            "behavior_mod_reason": s.get("behavior_modifier_reason") or "",
            "confidence":         s.get("confidence"),
            "evidence_1":         ev[0] if len(ev) > 0 else "",
            "evidence_2":         ev[1] if len(ev) > 1 else "",
            "transcript_chars":   _transcript_chars(m),
            "model":              d.get("model"),
            "rubric_version":     d.get("rubric_version"),
            "cost_usd":           round(d.get("cost_usd", 0), 5),
            "scored_at":          d.get("scored_at"),
            "audio_file":         m.get("audio_file", ""),
            "transcript_file":    m.get("transcript_file", "")
                                  or (f"transcripts/genesys/{d['call_id']}.txt"
                                      if d["source"] == "genesys" else ""),
        })

    # sort: most-recent first
    rows.sort(key=lambda r: (r["start_time"] or "", r["call_date"] or ""),
              reverse=True)

    wb = Workbook()

    # --------------------------------------------------------- Scores sheet
    ws = wb.active
    ws.title = "Scores"
    headers = list(rows[0].keys()) if rows else []
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(rows)+1}"

    for r in rows:
        ws.append(list(r.values()))

    # outcome color
    out_col = headers.index("outcome") + 1
    for ridx, r in enumerate(rows, start=2):
        color = OUTCOME_COLORS.get(r["outcome"])
        if color:
            for c in range(1, len(headers) + 1):
                ws.cell(row=ridx, column=c).fill = PatternFill("solid", fgColor=color)

    # column widths
    widths = {
        "source": 10, "call_id": 40, "call_date": 12, "start_time": 22,
        "duration_min": 10, "call_type": 10, "rep_id": 40, "outcome": 16,
        "score": 7, "q1": 8, "q2": 22, "q3": 18, "q4": 14,
        "time_signal": 13, "behavior_mod": 6, "behavior_mod_reason": 30,
        "confidence": 11, "evidence_1": 60, "evidence_2": 60,
        "transcript_chars": 12, "model": 22, "rubric_version": 16,
        "cost_usd": 10, "scored_at": 22, "audio_file": 50, "transcript_file": 50,
    }
    for i, h in enumerate(headers, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(h, 14)

    # wrap evidence cells
    for ridx in range(2, len(rows) + 2):
        for h in ("evidence_1", "evidence_2", "behavior_mod_reason"):
            cell = ws.cell(row=ridx, column=headers.index(h) + 1)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[ridx].height = 60

    # --------------------------------------------------------- Summary sheet
    summary = wb.create_sheet("Summary")
    summary.append(["LOI Distribution — observed vs authored target"])
    summary["A1"].font = Font(bold=True, size=14)
    summary.append([])
    summary.append(["Outcome", "N", "Observed %", "Authored target %", "Δ"])
    for c in range(1, 6):
        cell = summary.cell(row=3, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    outcomes = Counter(r["outcome"] for r in rows)
    total_core = sum(outcomes[k] for k in EXPECTED_DIST)
    rownum = 4
    for k in ("Freddo", "Tiepido", "Calduccio", "Caldo"):
        n = outcomes.get(k, 0)
        pct = n / total_core if total_core else 0
        target = EXPECTED_DIST[k]
        summary.append([k, n, round(pct * 100, 1), round(target * 100, 1),
                        round((pct - target) * 100, 1)])
        summary.cell(row=rownum, column=1).fill = PatternFill(
            "solid", fgColor=OUTCOME_COLORS[k])
        rownum += 1
    summary.append([])
    rownum += 1
    summary.append(["Excluded outcomes"])
    summary.cell(row=rownum, column=1).font = Font(bold=True)
    rownum += 1
    for k in ("Non interessato", "Non idoneo"):
        summary.append([k, outcomes.get(k, 0)])
        summary.cell(row=rownum, column=1).fill = PatternFill(
            "solid", fgColor=OUTCOME_COLORS[k])
        rownum += 1
    summary.append([])
    rownum += 1
    summary.append(["Total scored", len(rows)])
    summary.cell(row=rownum, column=1).font = Font(bold=True)
    rownum += 1
    summary.append(["Total cost USD", round(sum(r["cost_usd"] for r in rows), 4)])
    summary.cell(row=rownum, column=1).font = Font(bold=True)

    summary.column_dimensions["A"].width = 22
    summary.column_dimensions["B"].width = 10
    summary.column_dimensions["C"].width = 14
    summary.column_dimensions["D"].width = 20
    summary.column_dimensions["E"].width = 8

    out_path = OUT_REPORTS / "loi_scores.xlsx"
    wb.save(out_path)
    return out_path


if __name__ == "__main__":
    p = build()
    print(f"wrote {p}")
