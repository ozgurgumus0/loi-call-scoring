"""
The nowcasting test — LOI's real comparative advantage.

In production, when you allocate for the coming week your contract history is from
leads created 40+ days ago (STALE). LOI is available immediately (FRESH). So the
fair contest is:
    stale_history (early-period contract rate)   vs   fresh_LOI (late-period LOI)
as predictors of the LATE-period realized contract rate.

If lead quality drifts, fresh LOI should beat stale history even in big cells.
Uses agent LOI (coverage in both periods). NOTE: 79 contracts split in two is
thin -> read direction + CI, not a verdict.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import common as C
from ev_minspend import build_base_table
from backtest_ci import _score_alloc

RNG = np.random.default_rng(20260623)
K = 12.0
SPLIT = pd.Timestamp("2026-04-25")  # early: ..Apr25  | late: Apr26..May18


def rate_table(leads, start, end, prior_strength=K):
    d = leads[(leads["chan"].notna()) & (leads["sc1_date"] >= start) & (leads["sc1_date"] <= end)]
    chrate = d.groupby("chan")["signed"].mean().to_dict(); g = d["signed"].mean()
    o = d.groupby(["chan", "region"]).agg(n=("signed", "size"), c=("signed", "sum")).reset_index()
    o["chrate"] = o["chan"].map(chrate).fillna(g)
    o["rate"] = (prior_strength * o["chrate"] + o["c"]) / (prior_strength + o["n"])
    o["realized"] = o["c"] / o["n"]
    return o[["chan", "region", "n", "c", "rate", "realized"]]


def main():
    leads = C.load_leads().leads
    mat, cell, budget0, _, _ = build_base_table()
    cell = cell.reset_index(drop=True)

    early = rate_table(leads, pd.Timestamp("2026-03-26"), SPLIT)
    late = rate_table(leads, SPLIT + pd.Timedelta(days=1), pd.Timestamp("2026-05-18"))
    # fresh LOI from LATE calls
    dl = leads[(leads["chan"].notna()) & (leads["sc1_date"] > SPLIT) & (leads["sc1_date"] <= pd.Timestamp("2026-05-18"))]
    p_loi = C.global_calibration(leads, matured_only=True)
    q_late = C.build_cell_quality(dl, p_loi, paid_only=True)[["chan", "region", "quality"]]

    print("=" * 80)
    print("NOWCAST: stale history (early) vs fresh LOI (late) -> predict LATE contracts")
    print("=" * 80)
    print(f"  early contracts {int(early['c'].sum())} ({early['n'].sum():.0f} leads) | late contracts {int(late['c'].sum())} ({late['n'].sum():.0f} leads)")

    m = (late.rename(columns={"rate": "late_rate", "realized": "late_realized", "n": "late_n"})
         .merge(early[["chan", "region", "rate"]].rename(columns={"rate": "stale_rate"}), on=["chan", "region"], how="left")
         .merge(q_late.rename(columns={"quality": "fresh_loi"}), on=["chan", "region"], how="left"))
    m["stale_rate"] = m["stale_rate"].fillna(early["c"].sum() / early["n"].sum())
    m["fresh_loi"] = m["fresh_loi"].fillna(m["fresh_loi"].median())
    g = m[m["late_n"] >= 20].copy()
    print(f"  cells with >=20 late leads: {len(g)} (late contracts in them {int(g['c'].sum())})")

    print("\n  predicting LATE realized rate (Spearman):")
    print(f"    stale history (early rate)  : {spearmanr(g['stale_rate'], g['late_realized']).correlation:+.3f}")
    print(f"    fresh LOI (late)            : {spearmanr(g['fresh_loi'], g['late_realized']).correlation:+.3f}")

    # allocation efficiency on the LATE period: rank by signal, fund late-period budget
    late_spend = C.load_panel_spend(SPLIT + pd.Timedelta(days=1), pd.Timestamp("2026-05-18"))
    cc = cell[["chan", "region", "cap"]].merge(late_spend, on=["chan", "region"], how="left")
    m = m.merge(cc, on=["chan", "region"], how="left")
    m["spend"] = m["spend"].fillna(0.0)
    m = m[(m["spend"] > 0) & (m["late_n"] >= 5)].reset_index(drop=True)
    m["cpl"] = m["spend"] / m["late_n"]
    m["cpe_real"] = np.where(m["cpl"] > 0, m["late_realized"] / m["cpl"], 0.0)
    budget = float(m["spend"].sum()); cpl = m["cpl"].to_numpy(); cap = m["cap"].fillna(m["spend"]*2.5).to_numpy()

    def eff(sig): return _score_alloc(np.asarray(sig) / cpl, cpl, cap, budget, m["cpe_real"].to_numpy())
    es, ef = eff(m["stale_rate"]), eff(m["fresh_loi"])
    print(f"\n  LATE-period allocation efficiency (realized contracts/€10k):")
    print(f"    rank by stale history : {es:.3f}")
    print(f"    rank by fresh LOI     : {ef:.3f}")

    # bootstrap the difference (resample late leads)
    late_leads = leads[(leads["chan"].notna()) & (leads["sc1_date"] > SPLIT) & (leads["sc1_date"] <= pd.Timestamp("2026-05-18"))].reset_index(drop=True)
    diffs = []
    keyi = {(r.chan, r.region): i for i, r in m.iterrows()}
    arr = late_leads[["chan", "region", "signed"]].to_numpy(object)
    for _ in range(400):
        s = arr[RNG.integers(0, len(arr), len(arr))]
        db = pd.DataFrame(s, columns=["chan", "region", "signed"]); db["signed"] = pd.to_numeric(db["signed"])
        rb = db.groupby(["chan", "region"])["signed"].mean()
        cpe = np.zeros(len(m))
        for (ch, rg), v in rb.items():
            if (ch, rg) in keyi:
                i = keyi[(ch, rg)]; cpe[i] = v / cpl[i] if cpl[i] > 0 else 0
        d = _score_alloc(m["fresh_loi"].to_numpy()/cpl, cpl, cap, budget, cpe) - _score_alloc(m["stale_rate"].to_numpy()/cpl, cpl, cap, budget, cpe)
        diffs.append(d)
    diffs = np.array(diffs); lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"    fresh LOI - stale history = {diffs.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  P(>0)={(diffs>0).mean():.0%}")
    print("\n  (thin: read direction. A positive, CI-crossing result = 'promising, needs the holdout'.)")


if __name__ == "__main__":
    main()
