"""
Diagnose the user's two questions:
  (1) Why is better-AUC (per-lead) LOI not better in the (per-cell) allocator?
  (2) Is our LOI penalised because its calls are too recent (unmatured)?

Tests: date distributions (our vs agent), maturity sweep (calibration gradient,
lead-level AUC, cell gate, in-sample allocation efficiency vs base_rate), and a
coverage-matched comparison (cells we actually scored well).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import common as C
from our_loi_verify import join_our_loi
from gate_check import weighted_spearman

BAND = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7}


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    ok = np.isfinite(score)
    score, y = score[ok], y[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    try:
        u = mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").statistic
        return u / (y.sum() * (len(y) - y.sum()))
    except Exception:
        return np.nan


def shrink(c, n, prior, k=12.0):
    return (k * prior + c) / (k + n)


def cell_eff(cell, signal_col, budget, cap_share=0.18):
    """In-sample greedy: rank cells by signal/cpl, fill budget to caps, score
    realized contracts captured per €10k. Comparing the GAP across maturity."""
    d = cell.dropna(subset=[signal_col, "cpl_sc1"]).copy()
    d = d[d["cpl_sc1"] > 0]
    d["cpe_est"] = d[signal_col] / d["cpl_sc1"]
    d["cpe_real"] = d["realized_rate"] / d["cpl_sc1"]
    d["cap"] = np.minimum(d["spend"] * 2.5, cap_share * budget)
    d = d.sort_values("cpe_est", ascending=False)
    rem, cap_contracts = budget, 0.0
    for _, r in d.iterrows():
        if rem <= 0:
            break
        s = min(r["cap"], rem)
        cap_contracts += s * r["cpe_real"]
        rem -= s
    return cap_contracts / budget * 1e4


def build_cells(matjoin, agent_leads, cutoff):
    """Per-cell base_rate (agent), realized_rate (agent), our_cont (our LOI), spend, cpl."""
    # agent realized + base_rate
    a = agent_leads[(agent_leads["chan"].notna()) & (agent_leads["sc1_date"] <= cutoff)]
    o = a.groupby(["chan", "region"]).agg(n_leads=("signed", "size"), n_signed=("signed", "sum")).reset_index()
    o["realized_rate"] = o["n_signed"] / o["n_leads"]
    chrate = a.groupby("chan")["signed"].mean().to_dict(); g = a["signed"].mean()
    o["base_rate"] = [shrink(s, n, chrate.get(ch, g)) for ch, n, s in zip(o["chan"], o["n_leads"], o["n_signed"])]
    # spend
    start = a["sc1_date"].min()
    sp = C.load_panel_spend(start, cutoff)
    o = o.merge(sp, on=["chan", "region"], how="left")
    o["spend"] = o["spend"].fillna(0.0)
    o = o[(o["spend"] > 0) & (o["n_leads"] >= 5)].copy()
    o["cpl_sc1"] = o["spend"] / o["n_leads"]
    # our cont quality
    mj = matjoin[matjoin["sc1_date"] <= cutoff].copy()
    mj["cr"] = pd.to_numeric(mj["closing_readiness"], errors="coerce")
    cj = mj.dropna(subset=["cr"])
    if len(cj) > 30:
        cj = cj.assign(crbin=pd.qcut(cj["cr"].rank(method="first"), 10, labels=False))
        binrate = cj.groupby("crbin")["signed"].mean()
        cj = cj.assign(p_cr=cj["crbin"].map(binrate))
        gP = cj["p_cr"].mean(); chP = cj.groupby("chan")["p_cr"].mean().to_dict()
        rows = []
        for (ch, rg), s in cj.groupby(["chan", "region"]):
            prior = (chP.get(ch, gP) + gP) / 2
            rows.append(dict(chan=ch, region=rg, n_our=len(s),
                             our_cont=(10 * ((chP.get(ch, gP) + gP) / 2) + s["p_cr"].sum()) / (10 + len(s))))
        qc = pd.DataFrame(rows)
        o = o.merge(qc, on=["chan", "region"], how="left")
    else:
        o["our_cont"] = np.nan; o["n_our"] = 0
    return o, cj if len(cj) > 30 else mj


def main():
    mj_all, _ = join_our_loi()
    mj_all = mj_all[mj_all["chan"].notna() & mj_all["region"].notna()].copy()
    mj_all["band"] = mj_all["pred"].map(BAND)
    agent = C.load_leads().leads  # has sc1_date, matured etc.

    print("=" * 80); print("(2a) DATE DISTRIBUTION — is our set more recent than agent's?"); print("=" * 80)
    cutoff = C.TODAY - pd.Timedelta(days=35)
    ag = agent[(agent["chan"].notna()) & (agent["sc1_date"] <= cutoff)]
    ou = mj_all[mj_all["sc1_date"] <= cutoff]
    for lab, d in [("agent matured SC1", ag), ("our matured SC1", ou)]:
        print(f"  {lab:<20}: n={len(d):>4}  min {d['sc1_date'].min().date()}  median {d['sc1_date'].median().date()}  max {d['sc1_date'].max().date()}")

    print("\n" + "=" * 80); print("(2b) MATURITY SWEEP — does our LOI improve with more maturity?"); print("=" * 80)
    print(f"  {'cutoff(days)':<13}{'our_n':>7}{'signed':>7}{'lead_AUC':>10}{'cal_gradient':>14}{'cell_gate':>11}{'base_eff':>10}{'ourcont_eff':>12}")
    for md in (21, 28, 35, 45, 60):
        cut = C.TODAY - pd.Timedelta(days=md)
        mj = mj_all[mj_all["sc1_date"] <= cut]
        # lead AUC (our cr vs signed)
        la = auc(pd.to_numeric(mj["closing_readiness"], errors="coerce"), mj["signed"])
        # calibration gradient Caldo/Tiepido
        pr = {k: (mj[mj["band"] == k]["signed"].mean() if (mj["band"] == k).any() else np.nan) for k in C.BANDS}
        grad = (pr[7] / pr[4]) if (pr.get(4) and pr[4] > 0 and np.isfinite(pr.get(7, np.nan))) else np.nan
        cell, _ = build_cells(mj_all, agent, cut)
        g = cell[(cell["n_leads"] >= 20)].dropna(subset=["our_cont"])
        gate = weighted_spearman(g["our_cont"], g["realized_rate"], g["n_leads"]) if len(g) >= 5 else np.nan
        budget = float(cell["spend"].sum())
        be = cell_eff(cell, "base_rate", budget)
        oe = cell_eff(cell, "our_cont", budget)
        print(f"  {md:<13}{len(mj):>7}{int(mj['signed'].sum()):>7}{la:>10.3f}{grad:>14.2f}{gate:>11.3f}{be:>10.3f}{oe:>12.3f}")

    print("\n  (lead_AUC = per-lead discrimination of our cr; cell_gate = per-cell rank corr;")
    print("   base_eff/ourcont_eff = in-sample contracts/€10k. If recency hurt us, ourcont_eff")
    print("   would CLIMB toward base_eff as the cutoff grows. base_eff is the same-data ceiling.)")

    print("\n" + "=" * 80); print("(1) PER-LEAD AUC vs PER-CELL SPREAD — the core mechanism"); print("=" * 80)
    cut = C.TODAY - pd.Timedelta(days=35)
    cell, _ = build_cells(mj_all, agent, cut)
    mj = mj_all[mj_all["sc1_date"] <= cut]
    la = auc(pd.to_numeric(mj["closing_readiness"], errors="coerce"), mj["signed"])
    g = cell.dropna(subset=["our_cont"])
    print(f"  our LOI per-LEAD AUC (cr vs signed)     : {la:.3f}   (good ranker of individuals)")
    print(f"  our LOI per-CELL quality spread          : {g['our_cont'].min()*100:.2f}-{g['our_cont'].max()*100:.2f}%  (std {g['our_cont'].std()*100:.3f}pp)")
    print(f"  base_rate per-CELL spread (the target)   : {cell['base_rate'].min()*100:.2f}-{cell['base_rate'].max()*100:.2f}%  (std {cell['base_rate'].std()*100:.3f}pp)")
    print(f"  realized per-CELL spread                 : {cell['realized_rate'].min()*100:.2f}-{cell['realized_rate'].max()*100:.2f}%")
    print("  -> a strong per-LEAD ranker collapses to a near-constant per CELL because the")
    print("     within-cell LOI mix is similar everywhere + Caldo is rare; base_rate keeps full spread.")

    print("\n" + "=" * 80); print("(3) COVERAGE-MATCHED — restrict to cells we scored well (>=15 our calls)"); print("=" * 80)
    cm = cell[(cell["n_our"] >= 15) & (cell["n_leads"] >= 20)].dropna(subset=["our_cont"])
    if len(cm) >= 5:
        rho_our = weighted_spearman(cm["our_cont"], cm["realized_rate"], cm["n_leads"])
        rho_base = weighted_spearman(cm["base_rate"], cm["realized_rate"], cm["n_leads"])
        print(f"  well-covered cells: {len(cm)}  (contracts {int(cm['n_signed'].sum())})")
        print(f"  gate our_cont  vs realized: {rho_our:+.3f}")
        print(f"  gate base_rate vs realized: {rho_base:+.3f}")
    else:
        print(f"  only {len(cm)} well-covered cells — too few to compare (coverage IS the limit).")


if __name__ == "__main__":
    main()
