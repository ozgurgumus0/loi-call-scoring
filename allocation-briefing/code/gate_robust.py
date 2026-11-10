"""Robustness of the gate: MIN_LEADS sensitivity, drop-biggest-cell, and the
within-channel (region) signal isolated from the Google>Meta channel ordering."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import common as C
from gate_check import weighted_spearman

ld = C.load_leads()
leads = ld.leads
p_loi = C.global_calibration(leads, matured_only=True)
q = C.build_cell_quality(leads, p_loi, paid_only=True)
o = C.build_cell_outcomes(leads, matured_only=True, paid_only=True)
cell = q.merge(o, on=["chan", "region"], how="inner")

print("MIN_LEADS sensitivity (lead-weighted Spearman):")
for ml in (10, 20, 30, 50):
    g = cell[cell["n_leads"] >= ml]
    rho = weighted_spearman(g["quality"], g["realized_rate"], g["n_leads"])
    print(f"  >= {ml:>3} leads : n_cells={len(g):>2}  contracts={int(g['n_signed'].sum()):>3}  rho={rho:+.3f}")

# drop-biggest-cell robustness on the >=20 set
g = cell[cell["n_leads"] >= 20].sort_values("n_leads", ascending=False).reset_index(drop=True)
base = weighted_spearman(g["quality"], g["realized_rate"], g["n_leads"])
print(f"\nDrop-biggest-cell (>=20 set, base rho={base:+.3f}):")
for i in range(5):
    gg = g.drop(index=i)
    rho = weighted_spearman(gg["quality"], gg["realized_rate"], gg["n_leads"])
    print(f"  drop {g.loc[i,'chan']:<8}/{g.loc[i,'region']:<16} (n={int(g.loc[i,'n_leads'])}, signed={int(g.loc[i,'n_signed'])}): rho={rho:+.3f}")

# within-channel signal: demean quality & rate by channel, then correlate
print("\nWithin-channel (region) signal — demean by channel mean, then Spearman:")
g2 = cell[cell["n_leads"] >= 20].copy()
g2["q_dm"] = g2["quality"] - g2.groupby("chan")["quality"].transform("mean")
g2["r_dm"] = g2["realized_rate"] - g2.groupby("chan")["realized_rate"].transform("mean")
# keep only channels with >=2 qualifying cells (so a within-channel contrast exists)
keep = g2.groupby("chan")["chan"].transform("size") >= 2
g2 = g2[keep]
rho_within = spearmanr(g2["q_dm"], g2["r_dm"]).correlation
print(f"  channels with >=2 cells: {g2['chan'].nunique()}  cells={len(g2)}")
print(f"  within-channel Spearman(quality_demeaned, rate_demeaned) = {rho_within:+.3f}")
print("  -> if ~0, the gate is really the Google>Meta channel ordering, not regional LOI.")
print("  -> if >0, regional LOI mix carries genuine extra signal.")

# also: how much of the ranking is Google vs Meta?
print("\nGoogle vs Meta cells (the two dense channels):")
for ch in ("Google", "Meta"):
    s = cell[(cell["chan"] == ch) & (cell["n_leads"] >= 20)]
    print(f"  {ch:<7}: {len(s)} cells | mean quality {s['quality'].mean()*100:.2f}% | "
          f"lead-wtd realized {s['n_signed'].sum()/s['n_leads'].sum()*100:.2f}% | "
          f"quality range {s['quality'].min()*100:.2f}-{s['quality'].max()*100:.2f}%")
