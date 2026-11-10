"""Analyse LOI x NETSC1 (treated as the contract/closed flag per user).

Answers:
  - Do low-LOI calls coincide with NETSC1=0 (lost) more than high-LOI ones?
  - What is the "contract" rate per LOI band? (lift table)
  - Is the LOI x NETSC1 association statistically significant?
  - By channel, does the LOI -> NETSC1 effect change?

Usage:
    python -m src.analyze_loi_vs_contracts
    python -m src.analyze_loi_vs_contracts --since 2026-05-04 --until 2026-05-10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DATA, OUTPUTS, OUT_REPORTS

MASTER_CSV = DATA / "master.csv"
REPORT_MD = OUT_REPORTS / "05_loi_vs_contracts.md"
REPORT_XLSX = OUTPUTS / "loi_vs_contracts.xlsx"

IN_BAND = ["Freddo", "Tiepido", "Calduccio", "Caldo"]
EXCL = ["Non interessato", "Non idoneo"]
OUTCOME_ORDER = ["Non idoneo", "Non interessato", "Freddo", "Tiepido",
                 "Calduccio", "Caldo"]


def _md_section(df: pd.DataFrame, title: str, include_index: bool = False) -> str:
    if df.empty:
        return f"### {title}\n\n_(empty)_\n"
    out = [f"### {title}", ""]
    cols = ([df.index.name or ""] if include_index else []) + list(df.columns)
    out.append("| " + " | ".join(str(c) for c in cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for idx, row in df.iterrows():
        cells = [str(idx)] if include_index else []
        for v in row.values:
            cells.append("" if pd.isna(v) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def analyze(since: str | None, until: str | None) -> None:
    df = pd.read_csv(MASTER_CSV)
    df = df[(df["source"] == "zoom")
            & df["sonnet_outcome"].notna()
            & df["crm_netsc1"].notna()].copy()
    df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")
    if since:
        df = df[df["call_date"] >= pd.Timestamp(since)]
    if until:
        df = df[df["call_date"] <= pd.Timestamp(until)]
    df["crm_netsc1"] = df["crm_netsc1"].astype(int)
    df["sonnet_score"] = pd.to_numeric(df["sonnet_score"], errors="coerce")
    df["lead_score"] = pd.to_numeric(df["lead_score"], errors="coerce")

    n = len(df)
    n0 = int((df["crm_netsc1"] == 0).sum())
    n1 = int((df["crm_netsc1"] == 1).sum())

    lines = [
        "# 05 - LOI x NETSC1 (treated as contracts)",
        "",
        f"_n = {n} Sonnet-labeled Zoom calls with CRM (date range: "
        f"{since or 'all'} .. {until or 'all'}).  "
        f"NETSC1=1: {n1}  /  NETSC1=0: {n0}._",
        "",
    ]

    if n0 < 5:
        lines.append("**Warning:** fewer than 5 NETSC1=0 cases in this slice. "
                     "Per-band rates will be noisy; treat any lift as tentative.")
        lines.append("")

    # 1) Crosstab outcome x NETSC1
    ct = pd.crosstab(df["sonnet_outcome"], df["crm_netsc1"])
    ct = ct.reindex([o for o in OUTCOME_ORDER if o in ct.index])
    for col in [0, 1]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[[0, 1]]
    ct.columns = ["netsc1_0", "netsc1_1"]
    ct["total"] = ct.sum(axis=1)
    ct["contract_rate"] = (ct["netsc1_1"] / ct["total"].replace(0, pd.NA)).round(3)
    ct["lift_vs_overall"] = (ct["contract_rate"] / (n1 / n)).round(2)
    lines.append("## Contract rate (NETSC1=1 rate) per LOI band")
    lines.append("")
    ct.index.name = "outcome"
    lines.append(_md_section(ct, "Outcome x NETSC1", include_index=True))
    lines.append("")
    lines.append(f"_Overall contract rate: {n1}/{n} = {round(100*n1/max(1,n),1)}%. "
                 "Lift > 1 means above average, lift < 1 means below._")
    lines.append("")

    # 2) Mean LOI score, NETSC1=1 vs NETSC1=0 (in-band only)
    inb = df[df["sonnet_outcome"].isin(IN_BAND)]
    if not inb.empty:
        means = inb.groupby("crm_netsc1")["sonnet_score"].agg(["count", "mean"]).round(2)
        means.index.name = "netsc1"
        lines.append("## Mean LOI score by NETSC1 (in-band only)")
        lines.append("")
        lines.append(_md_section(means, "Mean LOI by NETSC1", include_index=True))
        lines.append("")
        # t-test
        s0 = inb.loc[inb["crm_netsc1"] == 0, "sonnet_score"].dropna()
        s1 = inb.loc[inb["crm_netsc1"] == 1, "sonnet_score"].dropna()
        if len(s0) >= 3 and len(s1) >= 3:
            from scipy.stats import ttest_ind, mannwhitneyu
            t, p_t = ttest_ind(s0, s1, equal_var=False)
            u, p_u = mannwhitneyu(s0, s1, alternative="two-sided")
            lines.append(f"- Welch t-test (NETSC1=0 vs =1): t = {t:+.2f}, p = {p_t:.4f}")
            lines.append(f"- Mann-Whitney U: U = {u:.0f}, p = {p_u:.4f}")
            lines.append("")

    # 3) Chi-square / Fisher
    try:
        from scipy.stats import chi2_contingency
        chi_table = ct[["netsc1_0", "netsc1_1"]].values
        if chi_table.sum() and chi_table.shape == (len([o for o in OUTCOME_ORDER if o in ct.index]), 2):
            chi2, p_chi, dof, _ = chi2_contingency(chi_table)
            lines.append(f"- Chi-square (outcome x NETSC1): chi2 = {chi2:.2f}, "
                         f"dof = {dof}, p = {p_chi:.4f}")
            lines.append("")
    except Exception as e:
        lines.append(f"_Chi-square skipped: {e}_")
        lines.append("")

    # 4) By channel: contract rate per channel, plus LOI mean by NETSC1 within channel
    if "lead_channel" in df.columns:
        ch = df.groupby("lead_channel").agg(
            n=("call_id", "count"),
            n_netsc1_1=("crm_netsc1", "sum"),
        )
        ch["contract_rate"] = (ch["n_netsc1_1"] / ch["n"]).round(3)
        inb_ch = inb.groupby("lead_channel")["sonnet_score"].mean().round(2)
        ch["mean_loi_inband"] = inb_ch
        ch = ch[ch["n"] >= 5].sort_values("contract_rate", ascending=False)
        ch.index.name = "channel"
        lines.append("## By channel: contract rate + mean LOI")
        lines.append("")
        lines.append(_md_section(ch, "Channel x contract rate (n>=5)", include_index=True))
        lines.append("")

    # 5) The NETSC1=0 list — qualitative inspection candidates
    losers = df[df["crm_netsc1"] == 0][
        ["call_id", "call_date", "sonnet_outcome", "sonnet_score",
         "sonnet_confidence", "lead_channel", "lead_segmentation",
         "lead_score", "transcript_file"]
    ].copy()
    losers = losers.sort_values("sonnet_score", ascending=False)
    lines.append("## Every NETSC1=0 call (sorted by Sonnet score desc)")
    lines.append("")
    lines.append(f"_{len(losers)} rows. The high-LOI cases here are the most "
                 "interesting — calls Sonnet judged warm that did NOT convert._")
    lines.append("")
    lines.append(_md_section(losers, "NETSC1=0 calls"))
    lines.append("")

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    # Excel multi-sheet
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as w:
        ct.reset_index().to_excel(w, sheet_name="outcome_x_netsc1", index=False)
        if "ch" in locals():
            ch.reset_index().to_excel(w, sheet_name="by_channel", index=False)
        losers.to_excel(w, sheet_name="netsc1_0_calls", index=False)

    print(f"[saved] {REPORT_MD}")
    print(f"[saved] {REPORT_XLSX}")
    print()
    print("=== Crosstab ===")
    print(ct.to_string())
    print()
    print("=== NETSC1=0 calls ({} rows) ===".format(len(losers)))
    print(losers.head(15).to_string(index=False))


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="YYYY-MM-DD lower bound on call_date")
    ap.add_argument("--until", help="YYYY-MM-DD upper bound on call_date")
    a = ap.parse_args()
    analyze(since=a.since, until=a.until)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
