"""
Validate the quality-decay-vs-spend signal: is it real, or a time-trend artifact?
Method: within each channel, remove a linear TIME trend from BOTH weekly spend and
weekly mean-LOI, then correlate the residuals. If spend still predicts lower LOI
after detrending, the decay is credible. If it collapses to ~0, it was just a
shared drift (spend up + quality down over the months for unrelated reasons).
Contract-free.
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

# weekly mean LOI per channel (agent LOI = best coverage)
leads = C.load_leads().leads
d = leads[leads["chan"].notna() & leads["loi"].isin(C.BANDS)].copy()
d["created"] = pd.to_datetime(d["lead_created"], errors="coerce")
d = d[d["created"] >= pd.Timestamp("2026-01-01")]
d["week"] = d["created"].dt.to_period("W").dt.start_time
wk = d.groupby(["chan", "week"]).agg(mean_loi=("loi", "mean"), n=("loi", "size")).reset_index()
wk = wk[wk["n"] >= 8]

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
    """residual of col after removing a linear trend in week index."""
    if len(g) < 4:
        return None
    x = g["wk_idx"].to_numpy(float)
    y = g[col].to_numpy(float)
    b = np.polyfit(x, y, 1)
    return y - np.polyval(b, x)


print("=" * 76)
print("QUALITY-DECAY: RAW vs DETRENDED (within channel)")
print("=" * 76)
print(f"  {'channel':<10}{'weeks':>6}{'raw corr':>11}{'detrended corr':>16}")
res_all = []
for ch, g in m.groupby("chan"):
    if len(g) < 5:
        continue
    raw = spearmanr(g["spend"], g["mean_loi"]).correlation
    rs, rl = detrend(g, "spend"), detrend(g, "mean_loi")
    dt = pearsonr(rs, rl)[0] if rs is not None else np.nan
    res_all.append((ch, rs, rl))
    print(f"  {ch:<10}{len(g):>6}{raw:>+11.3f}{dt:>+16.3f}")

# pooled detrended (stack residuals across channels)
RS = np.concatenate([r[1] for r in res_all])
RL = np.concatenate([r[2] for r in res_all])
pooled_raw = spearmanr(m["spend"] - m.groupby("chan")["spend"].transform("mean"),
                       m["mean_loi"] - m.groupby("chan")["mean_loi"].transform("mean")).correlation
pooled_dt_p = pearsonr(RS, RL)[0]
pooled_dt_s = spearmanr(RS, RL).correlation
# bootstrap CI on pooled detrended
rng = np.random.default_rng(20260623)
bs = []
for _ in range(2000):
    idx = rng.integers(0, len(RS), len(RS))
    bs.append(pearsonr(RS[idx], RL[idx])[0])
lo, hi = np.percentile(bs, [2.5, 97.5])
print("\n  POOLED within-channel:")
print(f"    raw (channel-demeaned)      : {pooled_raw:+.3f}")
print(f"    DETRENDED (time removed)    : {pooled_dt_p:+.3f} (Pearson) / {pooled_dt_s:+.3f} (Spearman)")
print(f"    detrended 95% CI            : [{lo:+.3f}, {hi:+.3f}]   P(<0) = {(np.array(bs)<0).mean():.0%}")
print("\n  READ: if DETRENDED stays clearly negative (CI excludes 0) -> decay is real,")
print("  worth wiring into the Tier-1 curve. If it collapses toward 0 -> it was a time")
print("  artifact, and even this last LOI-in-allocator angle closes.")
