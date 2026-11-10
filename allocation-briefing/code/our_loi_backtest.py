"""
Does OUR Qwen LOI quality survive the realized-contract backtest where agent LOI
did not?  Same CV framework + bootstrap CI. Signals compared:
  base_rate (CV'd)  | agent-quality (CV'd) | our band-quality | our cont-quality
Our-LOI quality is computed once on all our matured scored calls (static) — mildly
generous, so a LOSS here is decisive; a WIN would need the geo holdout to confirm.
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
from ev_minspend import build_base_table, fold_rates, alloc_fixed_budget
from backtest_ci import _score_alloc, _load_production_signal
from our_loi_verify import join_our_loi

RNG = np.random.default_rng(20260623)
N_FOLDS, N_BOOT = 5, 250


def our_quality_table(cell):
    """Static per-cell our-LOI quality (band + continuous), merged to cell, with
    channel/global fallback for cells we did not score."""
    m, _ = join_our_loi()
    mat = m[m["chan"].notna() & m["region"].notna() & m["matured"]].copy()
    mat["our_band"] = mat["pred"].map({"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7})
    gmean = mat["signed"].mean()
    p_our = {k: (mat[mat["our_band"] == k]["signed"].mean() if (mat["our_band"] == k).any() else gmean) for k in C.BANDS}
    p_our = {k: (v if np.isfinite(v) else gmean) for k, v in p_our.items()}

    def shares(fr):
        v = fr["our_band"].value_counts(); tot = int(v.reindex(C.BANDS).fillna(0).sum())
        return {k: (float(v.get(k, 0)) / tot if tot else 0.0) for k in C.BANDS}, tot
    d = mat[mat["our_band"].isin(C.BANDS)]
    p_nat, _ = shares(d); chs, chn = {}, {}
    for ch, g in d.groupby("chan"):
        chs[ch], chn[ch] = shares(g)
    cc = d.groupby(["chan", "region", "our_band"]).size()
    # continuous cr -> P(sign) via deciles
    cal = mat.dropna(subset=["closing_readiness"]).copy()
    cal["cr"] = pd.to_numeric(cal["closing_readiness"], errors="coerce")
    cal["crbin"] = pd.qcut(cal["cr"].rank(method="first"), 10, labels=False)
    binrate = cal.groupby("crbin")["signed"].mean()
    cal["p_cr"] = cal["crbin"].map(binrate)
    gP = cal["p_cr"].mean(); chP = cal.groupby("chan")["p_cr"].mean().to_dict()

    rows = []
    cells = set(zip(d["chan"], d["region"])) | set(zip(cal["chan"], cal["region"]))
    for (ch, rg) in cells:
        # band
        p_ch, n_ch = chs.get(ch, p_nat), chn.get(ch, 0)
        p_prior = {k: (n_ch * p_ch[k] + C.PRIOR_STRENGTH * p_nat[k]) / (n_ch + C.PRIOR_STRENGTH) for k in C.BANDS}
        cnt = {k: float(cc.get((ch, rg, k), 0)) for k in C.BANDS}; nl = sum(cnt.values())
        bs = {k: (C.DIR_STRENGTH * p_prior[k] + cnt[k]) / (C.DIR_STRENGTH + nl) for k in C.BANDS}
        qband = sum(p_our[k] * bs[k] for k in C.BANDS)
        # cont
        sub = cal[(cal["chan"] == ch) & (cal["region"] == rg)]; n = len(sub)
        prior = chP.get(ch, gP); prior = (n_ch * prior + C.PRIOR_STRENGTH * gP) / (n_ch + C.PRIOR_STRENGTH)
        qcont = (C.DIR_STRENGTH * prior + sub["p_cr"].sum()) / (C.DIR_STRENGTH + n) if (C.DIR_STRENGTH + n) else gP
        rows.append(dict(chan=ch, region=rg, q_band=qband, q_cont=qcont))
    qt = pd.DataFrame(rows)
    out = cell.merge(qt, on=["chan", "region"], how="left")
    # fallback for unscored cells: channel median then global median
    for col in ("q_band", "q_cont"):
        out[col] = out.groupby("chan")[col].transform(lambda s: s.fillna(s.median()))
        out[col] = out[col].fillna(out[col].median())
    return out


def cv_run(mat, cell, fold_id, prod_signal):
    n = len(cell); ki = {(r.chan, r.region): i for i, r in cell.iterrows()}
    cpl = cell["cpl_sc1"].to_numpy(); cap = cell["cap"].to_numpy(); budget = float(cell["spend"].sum())
    qband = cell["q_band"].to_numpy(); qcont = cell["q_cont"].to_numpy()
    acc = {k: [] for k in ("base_rate", "agent_q", "our_band", "our_cont", "production", "uniform")}
    for f in range(N_FOLDS):
        tr = np.where(fold_id != f)[0]; te = mat[fold_id == f]
        ft = fold_rates(mat, cell, tr)
        base = ft["base_rate"].to_numpy(); aq = ft["quality"].to_numpy()
        trate = te.groupby(["chan", "region"])["signed"].agg(["size", "sum"])
        rr = np.zeros(n)
        for (ch, rg), row in trate.iterrows():
            if (ch, rg) in ki and row["size"] > 0:
                rr[ki[(ch, rg)]] = row["sum"] / row["size"]
        cpe_real = np.where(cpl > 0, rr / cpl, 0.0)
        acc["base_rate"].append(_score_alloc(base / cpl, cpl, cap, budget, cpe_real))
        acc["agent_q"].append(_score_alloc(aq / cpl, cpl, cap, budget, cpe_real))
        acc["our_band"].append(_score_alloc(qband / cpl, cpl, cap, budget, cpe_real))
        acc["our_cont"].append(_score_alloc(qcont / cpl, cpl, cap, budget, cpe_real))
        acc["production"].append(_score_alloc(prod_signal, cpl, cap, budget, cpe_real))
        spu = np.minimum(np.full(n, budget / n), cap); spu = spu * (budget / spu.sum())
        acc["uniform"].append(float((spu * cpe_real).sum()) / budget * 1e4)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    mat, cell, budget, start, end = build_base_table()
    mat = mat.reset_index(drop=True); cell = cell.reset_index(drop=True)
    cell = our_quality_table(cell)
    prod = _load_production_signal(cell)
    print("=" * 80)
    print("OUR-LOI REALIZED-CONTRACT BACKTEST (CV + bootstrap CI)")
    print("=" * 80)
    print(f"  cells {len(cell)} | contracts {int(mat['signed'].sum())} | budget €{budget:,.0f}")
    print(f"  our-LOI quality spread: band {cell['q_band'].std()*100:.3f}pp  cont {cell['q_cont'].std()*100:.3f}pp")

    fid = RNG.integers(0, N_FOLDS, len(mat))
    point = cv_run(mat, cell, fid, prod)
    boots = {k: [] for k in point}
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(mat), len(mat))
        mb = mat.iloc[idx].reset_index(drop=True); fb = RNG.integers(0, N_FOLDS, len(mb))
        try:
            for k, v in cv_run(mb, cell, fb, prod).items():
                boots[k].append(v)
        except Exception:
            pass
    def ci(k): a = np.array(boots[k]); return np.percentile(a, [2.5, 97.5])
    print("\n  realized contracts per €10k (point [95% CI]):")
    for k in ("uniform", "production", "agent_q", "our_band", "our_cont", "base_rate"):
        lo, hi = ci(k); print(f"    {k:<14}{point[k]:>7.3f}   [{lo:>6.3f}, {hi:>6.3f}]")
    def paired(a, b):
        d = np.array(boots[a]) - np.array(boots[b]); return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()
    print("\n  paired differences (bootstrap):")
    for a, b in [("base_rate", "our_cont"), ("our_cont", "agent_q"), ("our_cont", "production"),
                 ("our_cont", "uniform"), ("base_rate", "our_band")]:
        mn, (lo, hi), p = paired(a, b)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns"
        print(f"    {a:<10} - {b:<12}{mn:>+7.3f}  [{lo:>+6.3f}, {hi:>+6.3f}]  P(>0)={p:.0%}  {sig}")


if __name__ == "__main__":
    main()
