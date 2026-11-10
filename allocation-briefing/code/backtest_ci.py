"""
Rigorous capstone: CV realized-contract backtest with bootstrap CI, comparing
the value signals that could drive channel×region allocation:

  uniform        - value-blind floor
  production     - the incumbent frozen EV_LOI (glitchy snapshot, ranks by €/lead)
  base_rate      - shrunk realized contracts/consultation  ("LOI off", clean rate)
  quality(LOI)   - Σ p_loi_k·share_loi_k                    (pure LOI signal)
  blend λ*       - (1-λ)·base_rate + λ·quality

Allocator = fixed-budget greedy by the signal's contracts-per-euro ranking, cap
limited. Judge = realized contracts captured per €10k on held-out consultations
(5-fold CV). Bootstrap resamples consultations and redoes the whole CV.
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
from ev_minspend import build_base_table, fold_rates, alloc_fixed_budget, shrink_rate

RNG = np.random.default_rng(20260623)
N_FOLDS = 5
N_BOOT = 250


def _load_production_signal(cell):
    df = pd.read_excel(C.THESIS / "MODELS INPUT" / "model_input.xlsx", sheet_name="Foglio1")
    df = df.rename(columns={"Aggregated": "chan", "ZIPREGION": "rraw"})
    df["region"] = df["rraw"].map(C.nreg)
    df["EV_LOI"] = pd.to_numeric(df["EV_LOI"].astype(str).str.replace("€", "").str.replace(",", "."), errors="coerce")
    m = cell.merge(df[["chan", "region", "EV_LOI"]], on=["chan", "region"], how="left")
    return m["EV_LOI"].to_numpy()  # NaN where incumbent has no cell


def _score_alloc(rank_key, cpl, cap, budget, cpe_real):
    """Greedy fill by rank_key desc (NaN ranked last), score realized cpe."""
    rk = np.where(np.isfinite(rank_key), rank_key, -1e18)
    sp = alloc_fixed_budget(rk, cpl, cap, budget)
    return float((sp * cpe_real).sum()) / budget * 1e4


def cv_run(mat, cell, fold_id, prod_signal, lambdas, dir_strength=C.DIR_STRENGTH):
    """One CV pass → dict signal -> contracts/€10k (mean over folds)."""
    n_cells = len(cell)
    key_index = {(r.chan, r.region): i for i, r in cell.iterrows()}
    cpl = cell["cpl_sc1"].to_numpy()
    cap = cell["cap"].to_numpy()
    budget = float(cell["spend"].sum())

    acc = {f"blend_{l}": [] for l in lambdas}
    acc.update({"uniform": [], "production": []})

    for f in range(N_FOLDS):
        tr = np.where(fold_id != f)[0]
        te = mat[fold_id == f]
        ft = fold_rates(mat, cell, tr)
        # quality with adjustable shrinkage
        if dir_strength != C.DIR_STRENGTH:
            train = mat.iloc[tr]
            p_loi = C.global_calibration(train, matured_only=False)
            q = C.build_cell_quality(train, p_loi, paid_only=True, dir_strength=dir_strength)[["chan", "region", "quality"]]
            ft = ft.drop(columns=["quality"]).merge(q, on=["chan", "region"], how="left")
            ft["quality"] = ft["quality"].fillna(ft["quality"].median())
        base = ft["base_rate"].to_numpy()
        qual = ft["quality"].to_numpy()

        trate = te.groupby(["chan", "region"])["signed"].agg(["size", "sum"])
        rr = np.zeros(n_cells)
        for (ch, rg), row in trate.iterrows():
            if (ch, rg) in key_index and row["size"] > 0:
                rr[key_index[(ch, rg)]] = row["sum"] / row["size"]
        cpe_real = np.where(cpl > 0, rr / cpl, 0.0)

        for l in lambdas:
            rate_est = (1 - l) * base + l * qual
            acc[f"blend_{l}"].append(_score_alloc(rate_est / cpl, cpl, cap, budget, cpe_real))
        # uniform
        spu = np.minimum(np.full(n_cells, budget / n_cells), cap)
        spu = spu * (budget / spu.sum())
        acc["uniform"].append(float((spu * cpe_real).sum()) / budget * 1e4)
        # production (rank by EV_LOI directly)
        acc["production"].append(_score_alloc(prod_signal, cpl, cap, budget, cpe_real))

    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    mat, cell, budget, start, end = build_base_table()
    mat = mat.reset_index(drop=True)
    cell = cell.reset_index(drop=True)
    prod_signal = _load_production_signal(cell)
    lambdas = [0.0, 0.25, 0.5, 1.0]
    n_prod_cells = int(np.isfinite(prod_signal).sum())

    print("=" * 80)
    print("REALIZED-CONTRACT BACKTEST WITH BOOTSTRAP CI")
    print("=" * 80)
    print(f"  consultations {len(mat):,} | contracts {int(mat['signed'].sum())} | cells {len(cell)} | budget €{budget:,.0f}")
    print(f"  incumbent EV_LOI covers {n_prod_cells}/{len(cell)} cells")

    # point estimate
    fold_id = RNG.integers(0, N_FOLDS, len(mat))
    point = cv_run(mat, cell, fold_id, prod_signal, lambdas)

    # bootstrap
    arr = mat[["chan", "region", "signed", "matured"]].copy()
    boots = {k: [] for k in point}
    for b in range(N_BOOT):
        idx = RNG.integers(0, len(mat), len(mat))
        mb = mat.iloc[idx].reset_index(drop=True)
        fb = RNG.integers(0, N_FOLDS, len(mb))
        try:
            r = cv_run(mb, cell, fb, prod_signal, lambdas)
            for k, v in r.items():
                boots[k].append(v)
        except Exception:
            continue

    def ci(key):
        a = np.array(boots[key]); return np.percentile(a, [2.5, 97.5])

    print("\n  realized contracts per €10k  (point [95% CI]):")
    order = ["uniform", "production", "blend_0.0", "blend_0.25", "blend_0.5", "blend_1.0"]
    label = {"uniform": "uniform (blind)", "production": "production EV_LOI",
             "blend_0.0": "base_rate (LOI off)", "blend_0.25": "blend λ=0.25",
             "blend_0.5": "blend λ=0.5", "blend_1.0": "quality λ=1 (pure LOI)"}
    for k in order:
        lo, hi = ci(k)
        print(f"    {label[k]:<26}{point[k]:>7.3f}   [{lo:>6.3f}, {hi:>6.3f}]")

    # paired difference: base_rate vs quality, and base_rate vs production
    def paired(a, b):
        d = np.array(boots[a]) - np.array(boots[b])
        return np.mean(d), np.percentile(d, [2.5, 97.5]), (d > 0).mean()

    print("\n  paired differences (bootstrap):")
    for a, b, txt in [("blend_0.0", "blend_1.0", "base_rate − pure LOI"),
                      ("blend_0.0", "production", "base_rate − production"),
                      ("blend_1.0", "production", "pure LOI − production"),
                      ("blend_0.0", "uniform", "base_rate − uniform")]:
        m, (lo, hi), p = paired(a, b)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        print(f"    {txt:<26}{m:>+7.3f}   [{lo:>+6.3f}, {hi:>+6.3f}]   P(>0)={p:.0%}  {sig}")

    # shrinkage sensitivity for quality (does any setting make LOI competitive?)
    print("\n  quality-shrinkage sensitivity (pure LOI λ=1, point estimate):")
    for ds in (2.0, 5.0, 10.0, 25.0):
        r = cv_run(mat, cell, fold_id, prod_signal, [1.0], dir_strength=ds)
        print(f"    DIR_STRENGTH={ds:>5}: contracts/€10k = {r['blend_1.0']:.3f}  (base_rate={point['blend_0.0']:.3f})")


if __name__ == "__main__":
    main()
