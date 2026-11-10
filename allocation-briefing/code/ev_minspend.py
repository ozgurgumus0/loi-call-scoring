"""
New EV_LOI + min-spend allocator + LOI-weight sweep, judged on REALIZED contracts
out-of-sample (5-fold CV over consultations, so it is not circular).

EV_LOI mechanism (makes LOI an explicit, tunable lever while keeping ROI sane):
    rate_est(lambda) = (1-lambda) * base_rate_cell  +  lambda * quality_cell
    cpe_est(lambda)  = rate_est(lambda) / CPL_SC1_cell        # est. contracts / euro
  - base_rate_cell : realized contracts/consultation, shrunk to channel (the
                     direct but noisy signal; this is "LOI off").
  - quality_cell   : Σ_k p_loi_k · share_loi_k  (the LOI signal; smoother, ranks
                     cells well per the gate — generalizes better OOS).
  - CPL_SC1_cell   : panel spend / #consultations  (cost per consultation) — keeps
                     numerator/denominator consistent so cpe = contracts/euro.

Allocator (two duals, both reported):
  (A) FIXED-BUDGET efficiency: greedily fill cells by cpe_est descending up to
      caps until the budget is spent  ->  realized contracts/€10k  (ROI).
  (B) MIN-SPEND to target: fill cheapest €/contract first until the estimated
      contract target is met  ->  spend required  (the user's locked objective).

Judge: realized contracts captured on the held-out consultations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

RNG = np.random.default_rng(20260622)
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_FOLDS = 5
RATE_SHRINK = 12.0          # base_rate shrink toward channel mean
CAP_BUDGET_SHARE = 0.18     # per-cell ceiling as share of total budget
CAP_GROWTH = 2.5            # a cell can grow to this multiple of its historic spend


def shrink_rate(contracts, n, chan_rate, k=RATE_SHRINK):
    return (k * chan_rate + contracts) / (k + n) if (k + n) > 0 else chan_rate


def build_base_table():
    """Per paid cell: matured consultations, realized contracts, panel spend,
    CPL per consultation, caps. (Outcome columns get recomputed per CV fold.)"""
    ld = C.load_leads()
    leads = ld.leads
    mat = leads[(leads["chan"].notna()) & (leads["matured"])].copy()

    # window for spend = matured consultation window
    start = mat["sc1_date"].min()
    end = mat["sc1_date"].max()
    spend = C.load_panel_spend(start, end)  # chan, region, spend

    sc1 = mat.groupby(["chan", "region"]).agg(
        n_sc1=("signed", "size"), contracts=("signed", "sum")
    ).reset_index()

    cell = sc1.merge(spend, on=["chan", "region"], how="left")
    cell["spend"] = cell["spend"].fillna(0.0)
    # need cost structure: drop cells with no spend or no consultations
    cell = cell[(cell["spend"] > 0) & (cell["n_sc1"] >= 5)].reset_index(drop=True)
    cell["cpl_sc1"] = cell["spend"] / cell["n_sc1"]

    budget = float(cell["spend"].sum())
    cap_share = CAP_BUDGET_SHARE * budget
    cell["cap"] = np.minimum(cell["spend"] * CAP_GROWTH, cap_share)
    return mat, cell, budget, start, end


def alloc_fixed_budget(cpe_est, cpl, cap, budget):
    """(A) Greedy fill by cpe_est desc, up to cap, until budget spent."""
    order = np.argsort(-cpe_est)
    spend = np.zeros(len(cpe_est))
    remaining = budget
    for i in order:
        if remaining <= 0:
            break
        s = min(cap[i], remaining)
        spend[i] = s
        remaining -= s
    return spend


def alloc_min_spend(cpe_est, cpl, cap, target_contracts):
    """(B) Fill cheapest €/contract first until estimated target met."""
    eur_per_contract = np.where(cpe_est > 0, 1.0 / cpe_est, np.inf)
    order = np.argsort(eur_per_contract)
    spend = np.zeros(len(cpe_est))
    got = 0.0
    for i in order:
        if got >= target_contracts:
            break
        need = (target_contracts - got) / cpe_est[i] if cpe_est[i] > 0 else 0.0
        s = min(cap[i], max(need, 0.0))
        spend[i] = s
        got += s * cpe_est[i]
    return spend


def fold_rates(mat, cell, train_idx):
    """Compute base_rate (train, shrunk) and quality (train) per cell;
    realized rate per cell on the FULL set (used as test outcome via held-out)."""
    train = mat.iloc[train_idx]
    # channel-level rate (train) for shrinkage prior
    chrate = train.groupby("chan")["signed"].mean().to_dict()
    glob = train["signed"].mean()
    base = train.groupby(["chan", "region"]).agg(n=("signed", "size"), c=("signed", "sum")).reset_index()
    base["chrate"] = base["chan"].map(chrate).fillna(glob)
    base["base_rate"] = [shrink_rate(c, n, cr) for c, n, cr in zip(base["c"], base["n"], base["chrate"])]

    # quality (train) via the shared calibration + shrinkage
    p_loi = C.global_calibration(train, matured_only=False)  # train already matured
    q = C.build_cell_quality(train, p_loi, paid_only=True)[["chan", "region", "quality"]]

    out = cell.merge(base[["chan", "region", "base_rate"]], on=["chan", "region"], how="left")
    out = out.merge(q, on=["chan", "region"], how="left")
    out["base_rate"] = out["base_rate"].fillna(glob)
    out["quality"] = out["quality"].fillna(out["quality"].median())
    return out


def main():
    mat, cell, budget, start, end = build_base_table()
    n_cells = len(cell)
    print("=" * 80)
    print("MIN-SPEND ALLOCATOR + LOI-WEIGHT SWEEP  (5-fold CV, realized contracts)")
    print("=" * 80)
    print(f"  matured paid consultations : {len(mat):,}  contracts {int(mat['signed'].sum())}")
    print(f"  cells (spend>0, >=5 SC1)   : {n_cells}")
    print(f"  window                     : {start.date()} -> {end.date()}")
    print(f"  reallocatable budget       : €{budget:,.0f}   (per-cell cap {CAP_BUDGET_SHARE:.0%} = €{CAP_BUDGET_SHARE*budget:,.0f})")

    # fold assignment over consultations
    mat = mat.reset_index(drop=True)
    fold_id = RNG.integers(0, N_FOLDS, len(mat))

    # accumulate per-lambda realized contracts/€10k across folds
    results = {lam: [] for lam in LAMBDAS}
    results_uniform = []
    minspend = {lam: [] for lam in LAMBDAS}  # spend to hit fold target

    key_index = {(r.chan, r.region): i for i, r in cell.iterrows()}

    for f in range(N_FOLDS):
        tr = np.where(fold_id != f)[0]
        te_mask = fold_id == f
        ft = fold_rates(mat, cell, tr)
        cpl = ft["cpl_sc1"].to_numpy()
        cap = ft["cap"].to_numpy()

        # realized TEST rate per cell (held-out consultations)
        test = mat[te_mask]
        trate = test.groupby(["chan", "region"])["signed"].agg(["size", "sum"])
        realized_rate_test = np.zeros(n_cells)
        for (ch, rg), row in trate.iterrows():
            if (ch, rg) in key_index and row["size"] > 0:
                realized_rate_test[key_index[(ch, rg)]] = row["sum"] / row["size"]
        # realized cpe on test = realized_rate_test / cpl   (contracts per euro)
        cpe_real = np.where(cpl > 0, realized_rate_test / cpl, 0.0)
        test_target = float(test["signed"].sum()) / len(tr) * len(mat)  # scale-free target

        for lam in LAMBDAS:
            rate_est = (1 - lam) * ft["base_rate"].to_numpy() + lam * ft["quality"].to_numpy()
            cpe_est = np.where(cpl > 0, rate_est / cpl, 0.0)
            # (A) fixed-budget efficiency
            sp = alloc_fixed_budget(cpe_est, cpl, cap, budget)
            captured = float((sp * cpe_real).sum())
            results[lam].append(captured / budget * 10000)
            # (B) min-spend dual: hit the fold's estimated contract target
            est_total = float((sp * cpe_est).sum())  # contracts the budget would buy (est.)
            sp_min = alloc_min_spend(cpe_est, cpl, cap, est_total)
            minspend[lam].append(float(sp_min.sum()))

        # uniform (value-blind) baseline at fixed budget
        sp_u = np.minimum(np.full(n_cells, budget / n_cells), cap)
        # renormalize to budget
        if sp_u.sum() > 0:
            sp_u = sp_u * (budget / sp_u.sum())
        results_uniform.append(float((sp_u * cpe_real).sum()) / budget * 10000)

    print("\n  realized contracts per €10k (fixed budget, mean over folds):")
    print(f"  {'lambda (LOI weight)':<22}{'contracts/€10k':>16}{'min-spend to target €':>24}")
    base0 = np.mean(results[0.0])
    for lam in LAMBDAS:
        m = np.mean(results[lam])
        ms = np.mean(minspend[lam])
        tag = "  (LOI off)" if lam == 0 else (f"  ({(m/base0-1)*100:+.1f}% vs off)" if base0 else "")
        print(f"  {lam:<22}{m:>16.3f}{ms:>24,.0f}{tag}")
    print(f"  {'uniform (value-blind)':<22}{np.mean(results_uniform):>16.3f}")

    # bootstrap CI on the BEST lambda vs lambda=0 (paired over folds+leads)
    best_lam = max(LAMBDAS, key=lambda l: np.mean(results[l]))
    print(f"\n  best lambda = {best_lam}  ->  +{(np.mean(results[best_lam])/base0-1)*100:.1f}% contracts/€ vs LOI-off")
    print("  (CV means; bootstrap CI computed in backtest_ci.py)")

    # save the full-data EV_LOI table at best lambda for the allocator handoff
    ft_full = fold_rates(mat, cell, np.arange(len(mat)))
    for lam in LAMBDAS:
        ft_full[f"rate_est_l{lam}"] = (1 - lam) * ft_full["base_rate"] + lam * ft_full["quality"]
        ft_full[f"ev_loi_l{lam}"] = ft_full[f"rate_est_l{lam}"] / ft_full["cpl_sc1"]
    path = C.OUT_DIR / "ev_minspend_table.csv"
    ft_full.to_csv(path, index=False)
    print(f"\nsaved -> {path}")
    return results, best_lam


if __name__ == "__main__":
    main()
