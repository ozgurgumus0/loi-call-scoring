"""
Can LOI earn a role in the allocator as the PRIOR for thin/new cells?

Instead of shrinking a cell's sparse contract count toward the flat channel
average, shrink it toward its LOI-implied rate:
    rate_LOIprior = (K * quality_cell + contracts) / (K + n_leads)
vs the incumbent:
    rate_chanprior = (K * channel_rate + contracts) / (K + n_leads)
Same K, only the prior differs -> isolates LOI's value exactly where history is weak.

Also measures how much budget flows to low-history cells (where LOI is the only
current signal) and whether LOI ranks the ZERO-history cells correctly.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import common as C
from ev_minspend import build_base_table, alloc_fixed_budget
from backtest_ci import _score_alloc

RNG = np.random.default_rng(20260623)
N_FOLDS, N_BOOT = 5, 250
K = 12.0  # shrink strength (same for both priors, so only the prior differs)


def fold_estimators(mat, cell, tr):
    """Per-cell train counts, channel rate, quality -> two shrinkage estimators."""
    train = mat.iloc[tr]
    chrate = train.groupby("chan")["signed"].mean().to_dict()
    g = train["signed"].mean()
    cnt = train.groupby(["chan", "region"]).agg(n=("signed", "size"), c=("signed", "sum")).reset_index()
    p_loi = C.global_calibration(train, matured_only=False)
    q = C.build_cell_quality(train, p_loi, paid_only=True)[["chan", "region", "quality"]]
    d = cell.merge(cnt, on=["chan", "region"], how="left").merge(q, on=["chan", "region"], how="left")
    d["n"] = d["n"].fillna(0.0)
    d["c"] = d["c"].fillna(0.0)
    d["chrate"] = d["chan"].map(chrate).fillna(g)
    d["quality"] = d["quality"].fillna(d["quality"].median())
    d["rate_chanprior"] = (K * d["chrate"] + d["c"]) / (K + d["n"])
    d["rate_LOIprior"] = (K * d["quality"] + d["c"]) / (K + d["n"])
    return d


def cv(mat, cell, fold_id):
    n = len(cell); ki = {(r.chan, r.region): i for i, r in cell.iterrows()}
    cpl = cell["cpl_sc1"].to_numpy(); cap = cell["cap"].to_numpy(); budget = float(cell["spend"].sum())
    acc = {"chanprior": [], "LOIprior": []}
    for f in range(N_FOLDS):
        tr = np.where(fold_id != f)[0]; te = mat[fold_id == f]
        d = fold_estimators(mat, cell, tr)
        trate = te.groupby(["chan", "region"])["signed"].agg(["size", "sum"])
        rr = np.zeros(n)
        for (ch, rg), row in trate.iterrows():
            if (ch, rg) in ki and row["size"] > 0:
                rr[ki[(ch, rg)]] = row["sum"] / row["size"]
        cpe_real = np.where(cpl > 0, rr / cpl, 0.0)
        acc["chanprior"].append(_score_alloc(d["rate_chanprior"].to_numpy() / cpl, cpl, cap, budget, cpe_real))
        acc["LOIprior"].append(_score_alloc(d["rate_LOIprior"].to_numpy() / cpl, cpl, cap, budget, cpe_real))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    mat, cell, budget, start, end = build_base_table()
    mat = mat.reset_index(drop=True); cell = cell.reset_index(drop=True)
    print("=" * 80)
    print("LOI AS THE THIN-CELL PRIOR  (chan-average prior vs LOI prior, same K)")
    print("=" * 80)
    print(f"  cells {len(cell)} | contracts {int(mat['signed'].sum())} | budget €{budget:,.0f} | K={K}")

    # how thin are the cells? (full-data history)
    full = fold_estimators(mat, cell, np.arange(len(mat)))
    thin = (full["c"] <= 1).sum()
    zero = (full["c"] == 0).sum()
    print(f"  cells with <=1 historical contract: {thin}/{len(cell)}   (0 contracts: {zero})")

    fid = RNG.integers(0, N_FOLDS, len(mat))
    point = cv(mat, cell, fid)
    boots = {k: [] for k in point}
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(mat), len(mat))
        mb = mat.iloc[idx].reset_index(drop=True); fb = RNG.integers(0, N_FOLDS, len(mb))
        try:
            for k, v in cv(mb, cell, fb).items():
                boots[k].append(v)
        except Exception:
            pass
    def ci(k): a = np.array(boots[k]); return np.percentile(a, [2.5, 97.5])
    print("\n  realized contracts per €10k (point [95% CI]):")
    for k in ("chanprior", "LOIprior"):
        lo, hi = ci(k); print(f"    {k:<12}{point[k]:>7.3f}   [{lo:>6.3f}, {hi:>6.3f}]")
    d = np.array(boots["LOIprior"]) - np.array(boots["chanprior"])
    lo, hi = np.percentile(d, [2.5, 97.5])
    sig = "SIG" if (lo > 0 or hi < 0) else "ns"
    print(f"\n  LOIprior - chanprior = {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  P(>0)={(d>0).mean():.0%}  {sig}")

    # do the ZERO-history cells get ranked better by LOI than by the channel average?
    print("\n  ZERO-history cells (allocator's blind spot) — does LOI rank them right?")
    zc = full[full["c"] == 0].merge(
        C.build_cell_outcomes(C.load_leads().leads, matured_only=True, paid_only=True),
        on=["chan", "region"], how="left")
    zc = zc[zc["n_leads"].fillna(0) >= 10]
    if len(zc) >= 4:
        from scipy.stats import spearmanr
        print(f"    zero-history cells with >=10 leads: {len(zc)}")
        print(f"    Spearman(LOI quality, realized_rate) : {spearmanr(zc['quality'], zc['realized_rate']).correlation:+.3f}")
        print(f"    Spearman(chan avg,    realized_rate) : {spearmanr(zc['chrate'], zc['realized_rate']).correlation:+.3f}")
    else:
        print(f"    only {len(zc)} qualifying zero-history cells — too few.")


if __name__ == "__main__":
    main()
