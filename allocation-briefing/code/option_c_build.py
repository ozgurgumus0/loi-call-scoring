"""
Build the LOCAL, buildable-tonight pieces of Option C (value-based bidding):
  1. The proxy-value schedule: LOI band -> relative conversion value (no euros;
     proportional to P(sign|band), so Google/Meta bidding weights leads by quality).
  2. A SAMPLE enhanced-conversions-for-leads upload template from our scored leads
     (GCLID is a PLACEHOLDER — the real click-ID must come from the CRM/ad layer).
  3. Corrected geo-lift power: Option C optimizes/measures on a DENSER signal
     (qualified leads), so its experiment is far more feasible than a contract test.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import common as C
from our_loi_verify import join_our_loi

BAND = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7}
NAME = {1: "Freddo", 4: "Tiepido", 6: "Calduccio", 7: "Caldo"}

# 1) proxy-value schedule from matured calibration, relative to the coldest band
p = C.global_calibration(C.load_leads().leads, matured_only=True)  # P(sign|band)
base = p[1]
proxy = {k: round(p[k] / base, 2) for k in C.BANDS}
print("=" * 74)
print("1) PROXY-VALUE SCHEDULE  (relative conversion value per LOI band; no euros)")
print("=" * 74)
print(f"  {'band':<11}{'P(sign|band)':>14}{'proxy value (x Freddo)':>24}")
for k in C.BANDS:
    print(f"  {NAME[k]:<11}{p[k]*100:>13.2f}%{proxy[k]:>22.2f}x")
print("  -> Google needs >=2 distinct values; this gives 4. A Caldo lead is fed as")
print(f"     ~{proxy[7]:.1f}x the value of a Freddo lead, so bidding chases warm leads harder.")

# 2) sample upload template from OUR scored leads
m, _ = join_our_loi()
m = m[m["chan"].notna()].copy()
m["band"] = m["pred"].map(BAND)
m = m.dropna(subset=["band"])
m["band"] = m["band"].astype(int)
up = pd.DataFrame({
    "Google Click ID (GCLID)": ["PLACEHOLDER_NEEDS_CRM_" + str(int(x)) for x in m["meetingId"].fillna(0)],
    "Conversion Name": "Qualified lead (LOI)",
    "Conversion Time": pd.to_datetime(m["sc1_date"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
    "Conversion Value": m["band"].map(proxy),
    "Conversion Currency": "EUR",
    "_LOI_band": m["band"].map(NAME),
    "_closing_readiness": m["closing_readiness"],
})
outp = C.OUT_DIR / "option_c_sample_upload.csv"
up.to_csv(outp, index=False)
print("\n" + "=" * 74)
print("2) SAMPLE UPLOAD TEMPLATE  (enhanced conversions for leads format)")
print("=" * 74)
print(f"  rows: {len(up)}  ->  {outp}")
print("  columns:", ", ".join(c for c in up.columns if not c.startswith("_")))
print("  value mix sent:", {NAME[k]: int((m['band']==k).sum()) for k in C.BANDS})
print("  *** GCLID is a PLACEHOLDER — real click-IDs (or hashed email) must come from")
print("      the CRM/ad layer; they are NOT in our current data exports. ***")

# 3) corrected geo-lift power on denser targets
def weeks_for(rate_per_week, delta, alpha=0.05, power=0.80):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    rc = rate_per_week / 2.0
    N = (z / np.log(1 + delta)) ** 2 * (2 + delta) / ((1 + delta) * rc)
    return int(np.ceil(N))

leads = C.load_leads().leads
d = leads.dropna(subset=["sc1_date"]).copy()
d["week"] = d["sc1_date"].dt.to_period("W").dt.start_time
stable = d[d["week"] <= (C.TODAY - pd.Timedelta(days=35))]
all_leads_wk = stable.groupby("week").size().mean()
qual = stable[stable["loi"].isin([6, 7])]  # Calduccio+Caldo = "qualified"
qual_wk = qual.groupby("week").size().mean()
contr_wk = stable.groupby("week")["signed"].sum().mean()

print("\n" + "=" * 74)
print("3) GEO-LIFT POWER by optimization target  (weeks to 80% power)")
print("=" * 74)
print(f"  {'target (per week)':<34}{'+20% lift':>11}{'+10% lift':>11}")
for lab, r in [(f"signed contracts (~{contr_wk:.0f}/wk)", contr_wk),
               (f"qualified leads C+H (~{qual_wk:.0f}/wk)", qual_wk),
               (f"all leads (~{all_leads_wk:.0f}/wk)", all_leads_wk)]:
    print(f"  {lab:<34}{weeks_for(r,0.20):>9}wk{weeks_for(r,0.10):>9}wk")
print("\n  => Option C optimizes to the LOI/qualified-lead signal, which is ~10x denser")
print("     than contracts, so a geo-lift read is WEEKS not ~10 months. Caveat (Goodhart):")
print("     a fast qualified-lead read must still be confirmed against signed contracts.")
