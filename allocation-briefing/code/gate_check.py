"""
GATE CHECK — does the per-cell LOI quality term rank channel×region cells by
who actually signs?  This is the go/no-go for the whole allocator-LOI rebuild:
if quality does NOT track realized contracts at cell level, weighting LOI up in
the allocator chases noise and ROI drops.

Last time (2026-06-18) this came out Spearman ~ 0 on 41 contracts (underpowered).
Now: 172 contracts, ~4x the LOI coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C

RNG = np.random.default_rng(20260622)
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 60)


def weighted_spearman(x, y, w):
    """Spearman via weighted Pearson on ranks (weights = leads per cell)."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    wm = w / w.sum()
    mx, my = (wm * rx).sum(), (wm * ry).sum()
    cov = (wm * (rx - mx) * (ry - my)).sum()
    vx = (wm * (rx - mx) ** 2).sum()
    vy = (wm * (ry - my) ** 2).sum()
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


def main():
    ld = C.load_leads()
    leads = ld.leads
    print("=" * 78)
    print("DATA")
    print("=" * 78)
    print(f"  file rows           : {ld.n_total:,}")
    print(f"  total signed        : {ld.n_signed}")
    print(f"  maturity cutoff     : SC1 <= {ld.maturity_cutoff.date()} (TODAY-{C.MATURITY_DAYS}d)")
    paid = leads[leads['chan'].notna()]
    print(f"  paid-channel leads  : {len(paid):,}  signed {int(paid['signed'].sum())}")
    mat = paid[paid['matured']]
    print(f"  matured paid leads  : {len(mat):,}  signed {int(mat['signed'].sum())}")
    with_loi = paid[paid['loi'].isin(C.BANDS)]
    print(f"  paid leads w/ LOI   : {len(with_loi):,}  ({len(with_loi)/len(paid):.0%} coverage)")

    # ---- global calibration p_loi_k ----
    p_all = C.global_calibration(leads, matured_only=False)
    p_mat = C.global_calibration(leads, matured_only=True)
    print("\n" + "=" * 78)
    print("GLOBAL CALIBRATION  p_loi_k = P(sign | LOI=k)")
    print("=" * 78)
    print(f"  {'band':<10}{'all-dates':>12}{'matured':>12}")
    names = {1: "Freddo", 4: "Tiepido", 6: "Calduccio", 7: "Caldo"}
    for k in C.BANDS:
        print(f"  {names[k]:<10}{p_all[k]*100:>11.2f}%{p_mat[k]*100:>11.2f}%")
    grad = p_mat[7] / p_mat[1] if p_mat[1] else np.nan
    print(f"  gradient Caldo/Freddo (matured): {grad:.1f}x")

    # use matured calibration as the production calibration
    p_loi = p_mat

    # ---- per-cell quality + realized outcomes ----
    q = C.build_cell_quality(leads, p_loi, paid_only=True)
    o = C.build_cell_outcomes(leads, matured_only=True, paid_only=True)
    cell = q.merge(o, on=["chan", "region"], how="outer")
    cell["n_leads"] = cell["n_leads"].fillna(0).astype(int)
    cell["n_signed"] = cell["n_signed"].fillna(0).astype(int)

    print("\n" + "=" * 78)
    print("QUALITY-TERM SPREAD across cells (this is what the allocator would see)")
    print("=" * 78)
    qv = cell["quality"].dropna()
    print(f"  cells with quality   : {qv.size}")
    print(f"  quality range        : {qv.min()*100:.2f}% .. {qv.max()*100:.2f}%  (spread {(qv.max()-qv.min())*100:.2f} pp)")
    print(f"  quality std          : {qv.std()*100:.3f} pp")
    rr = cell.loc[cell['n_leads'] > 0, 'realized_rate']
    print(f"  realized-rate range  : {rr.min()*100:.2f}% .. {rr.max()*100:.2f}%  (spread {(rr.max()-rr.min())*100:.2f} pp)")

    # ---- THE GATE: cell-level quality vs realized rate ----
    MIN_LEADS = 20
    g = cell[(cell["n_leads"] >= MIN_LEADS) & cell["quality"].notna()].copy()
    print("\n" + "=" * 78)
    print(f"GATE  —  cell-level Spearman(quality, realized_rate), cells >= {MIN_LEADS} matured leads")
    print("=" * 78)
    print(f"  qualifying cells     : {len(g)}  (matured contracts in them: {int(g['n_signed'].sum())})")

    rho_uw = spearmanr(g["quality"], g["realized_rate"]).correlation
    rho_w = weighted_spearman(g["quality"], g["realized_rate"], g["n_leads"])
    print(f"  Spearman (unweighted): {rho_uw:+.3f}")
    print(f"  Spearman (lead-wtd)  : {rho_w:+.3f}")

    # bootstrap CI: resample leads (paired with their cell), recompute rates + rho
    paidm = paid[paid["matured"]].copy()
    cells_idx = {(r.chan, r.region): i for i, r in g.reset_index().iterrows()}
    qmap = dict(zip(zip(g["chan"], g["region"]), g["quality"]))
    keys = list(qmap.keys())
    boots = []
    arr = paidm[["chan", "region", "signed"]].to_numpy(object)
    n = len(arr)
    for _ in range(2000):
        samp = arr[RNG.integers(0, n, n)]
        dfb = pd.DataFrame(samp, columns=["chan", "region", "signed"])
        dfb["signed"] = pd.to_numeric(dfb["signed"])
        gb = dfb.groupby(["chan", "region"])["signed"].agg(["size", "sum"])
        rates, quals, wts = [], [], []
        for key in keys:
            if key in gb.index and gb.loc[key, "size"] >= MIN_LEADS:
                rates.append(gb.loc[key, "sum"] / gb.loc[key, "size"])
                quals.append(qmap[key])
                wts.append(gb.loc[key, "size"])
        if len(rates) >= 5:
            boots.append(weighted_spearman(quals, rates, wts))
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_pos = (boots > 0).mean()
    print(f"  bootstrap 95% CI     : [{lo:+.3f}, {hi:+.3f}]   P(rho>0) = {p_pos:.0%}   (n_boot={len(boots)})")

    # ---- channel-level gate (robust aggregate, 8 channels) ----
    print("\n" + "=" * 78)
    print("CHANNEL-LEVEL GATE  (aggregate to 8 channels — fewer, denser units)")
    print("=" * 78)
    chq = []
    for ch, sub in with_loi.groupby("chan"):
        sh, ntot = C._shares(sub)
        quality = sum(p_loi[k] * sh[k] for k in C.BANDS)
        subm = mat[mat["chan"] == ch]
        rate = subm["signed"].mean() if len(subm) else np.nan
        chq.append(dict(chan=ch, n_loi=ntot, n_mat=len(subm), signed=int(subm["signed"].sum()),
                        quality=quality, realized_rate=rate))
    chdf = pd.DataFrame(chq).sort_values("quality", ascending=False)
    print(chdf.assign(quality=(chdf["quality"]*100).round(2),
                      realized_rate=(chdf["realized_rate"]*100).round(2)).to_string(index=False))
    chv = chdf.dropna(subset=["realized_rate"])
    if len(chv) >= 4:
        rho_ch = spearmanr(chv["quality"], chv["realized_rate"]).correlation
        print(f"  channel Spearman(quality, realized_rate) = {rho_ch:+.3f}  (n={len(chv)})")

    # ---- tercile readout (business framing of the gate) ----
    print("\n" + "=" * 78)
    print("TERCILE READOUT  —  do high-quality cells actually sign more?")
    print("=" * 78)
    gg = g.sort_values("quality")
    gg["tercile"] = pd.qcut(gg["quality"].rank(method="first"), 3, labels=["bottom", "mid", "top"])
    for t in ["bottom", "mid", "top"]:
        s = gg[gg["tercile"] == t]
        rate = s["n_signed"].sum() / s["n_leads"].sum()
        print(f"  {t:<7} third: {len(s):>2} cells | {int(s['n_leads'].sum()):>4} leads | "
              f"{int(s['n_signed'].sum()):>3} signed | realized rate {rate*100:.2f}% | "
              f"mean quality {s['quality'].mean()*100:.2f}%")

    # save merged table
    out = cell.sort_values("quality", ascending=False)
    path = C.OUT_DIR / "gate_cell_table.csv"
    out.to_csv(path, index=False)
    print(f"\nsaved -> {path}")

    print("\n" + "=" * 78)
    print("READING THE GATE")
    print("=" * 78)
    verdict = "PASS" if (np.isfinite(rho_w) and rho_w > 0.15 and p_pos >= 0.80) else \
              ("WEAK/DIRECTIONAL" if (np.isfinite(rho_w) and rho_w > 0) else "FAIL")
    print(f"  lead-weighted cell Spearman = {rho_w:+.3f}, P(rho>0) = {p_pos:.0%}  ->  {verdict}")


if __name__ == "__main__":
    main()
