"""
Re-run the legacy model's 50/50 model: WITH his 'LOI' half vs WITHOUT, scored on realized
contracts (fresh 172-contract file + JUN11 panel, same matured cells as all our
other analyses). Poisson-bootstrap CI on the difference.

CRUCIAL: Tom's 'LOI' = EV filtered by a binary 'Not Interested' salesperson tag
(per _no_loi_backtest_tom.py line 8) — an EV/CONTRACT-ECONOMICS signal, NOT the
graded warmth LOI (1/4/6/7) or our Qwen closing_readiness.
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
from ev_minspend import build_base_table

RNG = np.random.default_rng(20260623)
RUNS = C.THESIS / "evaluation" / "results" / "notebook_runs"

# realized contracts-per-euro per cell (same matured window/cells as everything else)
mat, cell, budget0, start, end = build_base_table()
cell = cell.reset_index(drop=True)
cell["cpe"] = np.where(cell["spend"] > 0, cell["contracts"] / cell["spend"], 0.0)
cpe_map = {(r.chan, r.region): r.cpe for r in cell.itertuples()}
contr_map = {(r.chan, r.region): r.contracts for r in cell.itertuples()}
spend_map = {(r.chan, r.region): r.spend for r in cell.itertuples()}
print(f"realized cells: {len(cell)} | matured contracts {int(cell['contracts'].sum())} | window {start.date()}..{end.date()}")


def load_alloc(fn):
    a = pd.read_csv(RUNS / fn)
    a = a[["Channel", "Region", "Allocated_Budget"]].dropna(subset=["Allocated_Budget"]).copy()
    a["chan"] = a["Channel"].astype(str)
    a["region"] = a["Region"].map(C.nreg)
    a["bud"] = pd.to_numeric(a["Allocated_Budget"], errors="coerce").fillna(0)
    a["cpe"] = a.apply(lambda r: cpe_map.get((r["chan"], r["region"]), np.nan), axis=1)
    return a


def score(a):
    tot = a["bud"].sum()
    bought = (a["bud"] * a["cpe"].fillna(0)).sum()
    cov = a.loc[a["cpe"].notna(), "bud"].sum() / tot * 100 if tot else 0
    return bought / tot * 1e4, tot, cov  # contracts per EUR10k


runs = {"with LOI (50/50)": "tom_LOI.csv", "LOI half removed": "tom_NOLOI.csv", "uniform (blind)": "tom_UNIFORM.csv"}
allocs = {k: load_alloc(v) for k, v in runs.items()}
print("\n" + "=" * 72)
print("TOM MODEL on realized contracts (fresh 172-contract data)")
print("=" * 72)
print(f"  {'variant':<22}{'contracts/EUR10k':>18}{'budget':>10}{'coverage':>10}")
pts = {}
for k, a in allocs.items():
    s, tot, cov = score(a)
    pts[k] = s
    print(f"  {k:<22}{s:>18.3f}{tot:>10,.0f}{cov:>9.0f}%")

L, N = pts["with LOI (50/50)"], pts["LOI half removed"]
print(f"\n  with-LOI vs no-LOI: {L-N:+.3f} contracts/EUR10k ({(L/N-1)*100:+.1f}%)")

# Poisson-bootstrap CI: resample each cell's contract count, recompute cpe + both scores
aL, aN = allocs["with LOI (50/50)"], allocs["LOI half removed"]
keys = list(cpe_map.keys())
base_contr = np.array([contr_map[k] for k in keys])
spend_arr = np.array([spend_map[k] for k in keys])
diffs, Ls, Ns = [], [], []
for _ in range(5000):
    cb = RNG.poisson(base_contr)
    cpe_b = {k: (cb[i] / spend_arr[i] if spend_arr[i] > 0 else 0.0) for i, k in enumerate(keys)}
    sL = (aL["bud"] * aL.apply(lambda r: cpe_b.get((r["chan"], r["region"]), 0.0), axis=1)).sum() / aL["bud"].sum() * 1e4
    sN = (aN["bud"] * aN.apply(lambda r: cpe_b.get((r["chan"], r["region"]), 0.0), axis=1)).sum() / aN["bud"].sum() * 1e4
    diffs.append(sL - sN)
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [5, 95])
print(f"  Poisson bootstrap: 90% CI [{lo:+.3f}, {hi:+.3f}]  P(LOI helps) = {(diffs>0).mean():.0%}")

print("\n" + "=" * 72)
print("WHAT THIS MEANS")
print("=" * 72)
print("  Tom's 'LOI' = EV filtered by a binary Not-Interested tag = a CONTRACT-ECONOMICS")
print("  signal (cell-shaped), which is exactly the kind of signal that DOES help (= our")
print("  Option A contract-rate signal). It is NOT the graded warmth LOI. Swapping our")
print("  graded/Qwen LOI into Tom's quality half went FLAT (prior STEP 2). So Tom helping")
print("  with 'LOI' supports Option A, it does NOT mean warmth-LOI helps the allocator.")
