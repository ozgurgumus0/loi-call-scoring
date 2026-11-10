"""
Same quality-decay-vs-spend test, but on OUR Qwen LOI (continuous closing_readiness).
Detrend time within channel, correlate residuals. Coverage is thinner/shorter than
agent LOI, so read with that caveat.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import common as C
from our_loi_verify import join_our_loi

MINW = 6  # min leads per channel-week for a stable mean
MINWEEKS = 5

m0, _ = join_our_loi()
m0 = m0[m0["chan"].notna()].copy()
m0["cr"] = pd.to_numeric(m0["closing_readiness"], errors="coerce")
m0["created"] = pd.to_datetime(m0["sc1_date"], errors="coerce")  # consultation date (lead_created not in join)
m0 = m0.dropna(subset=["cr", "created"])
m0 = m0[m0["created"] >= pd.Timestamp("2026-01-01")]
m0["week"] = m0["created"].dt.to_period("W").dt.start_time
wk = m0.groupby(["chan", "week"]).agg(mean_cr=("cr", "mean"), n=("cr", "size")).reset_index()
wk = wk[wk["n"] >= MINW]
print(f"channel-weeks (n>= {MINW}): {len(wk)} | leads scored+joined: {len(m0)} | weeks span "
      f"{m0['week'].min().date()}..{m0['week'].max().date()}")

# weekly spend
p = pd.read_excel(C.PANEL_XLSX, sheet_name="raw_data")
p["date"] = pd.to_datetime(dict(year=pd.to_numeric(p["Year"], errors="coerce"),
                                month=pd.to_numeric(p["Month"], errors="coerce"),
                                day=pd.to_numeric(p["Day"], errors="coerce")), errors="coerce")
p["week"] = p["date"].dt.to_period("W").dt.start_time
p["spend"] = pd.to_numeric(p["mktg_spend"], errors="coerce").fillna(0)
pw = p.groupby(["Aggregated", "week"])["spend"].sum().reset_index().rename(columns={"Aggregated": "chan"})
m = wk.merge(pw, on=["chan", "week"], how="inner").sort_values(["chan", "week"])
m["wk_idx"] = m.groupby("chan")["week"].rank().astype(int)


def detrend(g, col):
    x = g["wk_idx"].to_numpy(float); y = g[col].to_numpy(float)
    return y - np.polyval(np.polyfit(x, y, 1), x)


print("=" * 70)
print("QWEN LOI (closing_readiness) quality-decay: RAW vs DETRENDED")
print("=" * 70)
print(f"  {'channel':<10}{'weeks':>6}{'raw corr':>11}{'detrended':>12}")
res = []
for ch, g in m.groupby("chan"):
    if len(g) < MINWEEKS:
        print(f"  {ch:<10}{len(g):>6}   (too few weeks)")
        continue
    raw = spearmanr(g["spend"], g["mean_cr"]).correlation
    rs, rl = detrend(g, "spend"), detrend(g, "mean_cr")
    res.append((ch, rs, rl))
    print(f"  {ch:<10}{len(g):>6}{raw:>+11.3f}{pearsonr(rs, rl)[0]:>+12.3f}")

if res:
    RS = np.concatenate([r[1] for r in res]); RL = np.concatenate([r[2] for r in res])
    dt = pearsonr(RS, RL)[0]
    rng = np.random.default_rng(20260623)
    bs = [pearsonr(RS[i], RL[i])[0] for i in (rng.integers(0, len(RS), len(RS)) for _ in range(2000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n  POOLED detrended: {dt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(<0)={(np.array(bs)<0).mean():.0%}  (n={len(RS)})")
else:
    print("\n  No channel had enough weeks — our Qwen coverage is too thin/short for this test.")
