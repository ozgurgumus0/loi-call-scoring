"""First-pass LOI analysis on the calls we currently have Sonnet-labeled.

Answers the questions the user actually cares about:
  - Which channels / regions / segments / product types yield higher LOI?
  - Does Sonnet's post-call LOI correlate with the CRM's pre-call LEAD_SCORE
    at the bigger sample size (n=203) we now have?

Reads:
    data/master.csv

Writes:
    outputs/reports/04_loi_analysis.md
    outputs/loi_analysis.xlsx   (same content, multi-sheet, sortable in Excel)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA, OUTPUTS, OUT_REPORTS

MASTER_CSV = DATA / "master.csv"
REPORT_MD = OUT_REPORTS / "04_loi_analysis.md"
REPORT_XLSX = OUTPUTS / "loi_analysis.xlsx"

IN_BAND = ["Freddo", "Tiepido", "Calduccio", "Caldo"]
EXCL = ["Non interessato", "Non idoneo"]
OUTCOME_TO_NUM = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7}


def _cohort_table(df: pd.DataFrame, col: str, min_n: int = 5) -> pd.DataFrame:
    """For each non-null value of `col`, compute LOI summary stats."""
    rows = []
    for v, g in df.groupby(col, dropna=True):
        n = len(g)
        if n < min_n:
            continue
        in_band = g[g["sonnet_outcome"].isin(IN_BAND)]
        excl = g[g["sonnet_outcome"].isin(EXCL)]
        mean_loi = in_band["sonnet_score"].mean() if len(in_band) else None
        mean_lead = pd.to_numeric(g["lead_score"], errors="coerce").mean()
        rows.append({
            col:               v,
            "n":               n,
            "excl_rate":       round(len(excl) / n, 2),
            "mean_loi_inband": round(mean_loi, 2) if mean_loi is not None else None,
            "pct_freddo":      round((g["sonnet_outcome"] == "Freddo").mean(), 2),
            "pct_tiepido":     round((g["sonnet_outcome"] == "Tiepido").mean(), 2),
            "pct_calduccio":   round((g["sonnet_outcome"] == "Calduccio").mean(), 2),
            "pct_caldo":       round((g["sonnet_outcome"] == "Caldo").mean(), 2),
            "mean_lead_score": round(mean_lead, 0) if pd.notna(mean_lead) else None,
        })
    out = pd.DataFrame(rows)
    if not out.empty and "mean_loi_inband" in out.columns:
        out = out.sort_values("mean_loi_inband", ascending=False, na_position="last")
    return out


def _md_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"### {title}\n\n_no rows with n >= 5_\n"
    out = [f"### {title}", ""]
    out.append("| " + " | ".join(df.columns) + " |")
    out.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, r in df.iterrows():
        out.append("| " + " | ".join(
            ("" if pd.isna(v) else str(v)) for v in r.values
        ) + " |")
    return "\n".join(out) + "\n"


def analyze() -> None:
    df_all = pd.read_csv(MASTER_CSV)
    df = df_all[df_all["sonnet_outcome"].notna() & (df_all["source"] == "zoom")].copy()
    # Coerce numeric
    df["lead_score"] = pd.to_numeric(df["lead_score"], errors="coerce")
    df["sonnet_score"] = pd.to_numeric(df["sonnet_score"], errors="coerce")

    n_total = len(df)
    n_with_crm = int(df["lead_score"].notna().sum())

    # --- Headline distribution ---
    band_counts = df["sonnet_outcome"].value_counts(dropna=False)
    excl = int(df["sonnet_outcome"].isin(EXCL).sum())
    in_band = df[df["sonnet_outcome"].isin(IN_BAND)]

    lines = [
        "# 04 — LOI analysis (Sonnet, Zoom)",
        "",
        f"_n = {n_total} Sonnet-scored Zoom calls; {n_with_crm} have CRM features._",
        "",
        "## Headline distribution",
        "",
        "| Outcome | n | % |",
        "|---|---:|---:|",
    ]
    for o in IN_BAND + EXCL:
        c = int(band_counts.get(o, 0))
        lines.append(f"| {o} | {c} | {round(100*c/max(1, n_total), 1)}% |")
    lines.append("")
    lines.append("Authored target (in-band only): **30/50/15/5**. "
                 f"Observed in-band (n={len(in_band)}): "
                 f"**{round(100*(in_band['sonnet_outcome']=='Freddo').sum()/max(1,len(in_band)))}**/"
                 f"**{round(100*(in_band['sonnet_outcome']=='Tiepido').sum()/max(1,len(in_band)))}**/"
                 f"**{round(100*(in_band['sonnet_outcome']=='Calduccio').sum()/max(1,len(in_band)))}**/"
                 f"**{round(100*(in_band['sonnet_outcome']=='Caldo').sum()/max(1,len(in_band)))}** "
                 "— strong warm-tilt vs target.")
    lines.append("")

    # --- Cohort breakdowns ---
    cohort_cols = [
        ("lead_channel",        "By lead channel (paid source)"),
        ("lead_utmsource",      "By UTM source (sanity check vs channel)"),
        ("lead_segmentation",   "By CRM lead segmentation (A-Gold ... F-NotServed)"),
        ("lead_region",         "By region"),
        ("lead_producttype",    "By product type"),
        ("lead_evaluationtime", "By self-reported lead evaluation time"),
        ("call_type",           "By call type (sales vs booking)"),
    ]
    cohort_tables: dict[str, pd.DataFrame] = {}
    for col, label in cohort_cols:
        if col not in df.columns:
            continue
        t = _cohort_table(df, col)
        cohort_tables[col] = t
        lines.append(_md_table(t, label))
        lines.append("")

    # --- Sonnet LOI vs LEAD_SCORE correlation ---
    num = df[df["sonnet_score"].isin([1, 4, 6, 7]) & df["lead_score"].notna()]
    if len(num) >= 30:
        from scipy.stats import pearsonr, spearmanr
        r_p, p_p = pearsonr(num["sonnet_score"], num["lead_score"])
        r_s, p_s = spearmanr(num["sonnet_score"], num["lead_score"])
        lines.append("## Sonnet LOI score vs CRM LEAD_SCORE")
        lines.append("")
        lines.append(f"- n = **{len(num)}** in-band Sonnet labels with CRM lead score")
        lines.append(f"- Pearson r   = **{r_p:+.3f}**  (p = {p_p:.4f})")
        lines.append(f"- Spearman rho = **{r_s:+.3f}**  (p = {p_s:.4f})")
        lines.append("")
        by_out = num.groupby("sonnet_outcome")["lead_score"].agg(["count", "mean"]).round(0)
        by_out = by_out.reindex([o for o in IN_BAND if o in by_out.index])
        lines.append("### Mean LEAD_SCORE per Sonnet outcome")
        lines.append("")
        lines.append("| outcome | n | mean LEAD_SCORE |")
        lines.append("|---|---:|---:|")
        for idx, row in by_out.iterrows():
            lines.append(f"| {idx} | {int(row['count'])} | {row['mean']:.0f} |")
        lines.append("")

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    # Excel multi-sheet
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as w:
        for col, t in cohort_tables.items():
            t.to_excel(w, sheet_name=col[:31], index=False)

    print(f"[saved] {REPORT_MD}")
    print(f"[saved] {REPORT_XLSX}")
    # Quick stdout preview
    print()
    print("=== Top of each cohort (sorted by mean_loi_inband) ===")
    for col, t in cohort_tables.items():
        if t.empty:
            continue
        print(f"\n--- {col} ---")
        print(t.head(10).to_string(index=False))


if __name__ == "__main__":
    analyze()
