"""
The one untested 'LOI in the allocator' angle: does lead QUALITY decay as you
SPEND MORE on a channel?  If yes, LOI bends the saturation curve (a real Tier-2/
TFT input) — used as a slope (quality vs spend), NOT a flat per-cell level.
Contract-free: uses only LOI + spend.
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

leads = C.load_leads().leads
d = leads[leads["chan"].notna() & leads["loi"].isin(C.BANDS)].copy()
d["created"] = pd.to_datetime(d["lead_created"], errors="coerce")
d = d[d["created"] >= pd.Timestamp("2026-01-01")]
d["week"] = d["created"].dt.to_period("W").dt.start_time

# weekly mean LOI per channel
wk = d.groupby(["chan", "week"]).agg(mean_loi=("loi", "mean"), n=("loi", "size")).reset_index()
wk = wk[wk["n"] >= 8]  # need enough leads for a stable weekly mean

# weekly spend per channel
pan = C.load_panel_spend(d["created"].min(), d["created"].max())  # not weekly; recompute weekly
import openpyxl  # noqa
p = pd.read_excel(C.PANEL_XLSX, sheet_name="raw_data")
p["date"] = pd.to_datetime(dict(year=pd.to_numeric(p["Year"], errors="coerce"),
                                month=pd.to_numeric(p["Month"], errors="coerce"),
                                day=pd.to_numeric(p["Day"], errors="coerce")), errors="coerce")
p["week"] = p["date"].dt.to_period("W").dt.start_time
p["spend"] = pd.to_numeric(p["mktg_spend"], errors="coerce").fillna(0)
pw = p.groupby(["Aggregated", "week"])["spend"].sum().reset_index().rename(columns={"Aggregated": "chan"})

m = wk.merge(pw, on=["chan", "week"], how="inner")
print("=" * 74)
print("DOES LEAD QUALITY (mean LOI) DROP AS WEEKLY SPEND RISES? (within channel)")
print("=" * 74)
print(f"  channel-weeks: {len(m)}  channels: {m['chan'].nunique()}")
print(f"\n  {'channel':<10}{'weeks':>7}{'spend-vs-LOI corr':>20}")
rows = []
for ch, g in m.groupby("chan"):
    if len(g) >= 5:
        r = spearmanr(g["spend"], g["mean_loi"]).correlation
        rows.append((ch, len(g), r))
        print(f"  {ch:<10}{len(g):>7}{r:>+20.3f}")
# pooled within-channel: demean both by channel, then correlate
m["sp_dm"] = m["spend"] - m.groupby("chan")["spend"].transform("mean")
m["loi_dm"] = m["mean_loi"] - m.groupby("chan")["mean_loi"].transform("mean")
keep = m.groupby("chan")["chan"].transform("size") >= 5
mm = m[keep]
r_pool = spearmanr(mm["sp_dm"], mm["loi_dm"]).correlation
print(f"\n  POOLED within-channel (spend-demeaned vs LOI-demeaned): {r_pool:+.3f}  (n={len(mm)})")
print("\n  READ: a clear NEGATIVE corr = scaling a channel cools its leads -> LOI is a")
print("  real saturation-curve input for Tier 2/TFT. ~0 or mixed = even this angle is")
print("  flat, and LOI has no allocator role. (Directional only — weekly, ~6 months, agent LOI.)")
