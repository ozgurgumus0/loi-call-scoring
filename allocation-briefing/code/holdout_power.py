"""
Geo-holdout feasibility / power analysis, grounded in the real contract volume.

Question: if we run LOI-informed allocation vs contract-history allocation in a
controlled experiment, how many weeks until we can detect a plausible lift in
contracts with 80% power?  Also: how concentrated are contracts across regions
(does naive region-split work, or do we need a switchback / matched design)?
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

RNG = np.random.default_rng(20260623)


def weekly_volume():
    ld = C.load_leads(); leads = ld.leads
    d = leads.dropna(subset=["sc1_date"]).copy()
    d["week"] = d["sc1_date"].dt.to_period("W").dt.start_time
    wk = d.groupby("week").agg(leads=("signed", "size"), contracts=("signed", "sum"))
    # use a stable, matured stretch for the run-rate (exclude the freshest unmatured weeks)
    stable = wk[wk.index <= (C.TODAY - pd.Timedelta(days=35))]
    return wk, stable


def region_concentration():
    ld = C.load_leads(); leads = ld.leads
    sg = leads[leads["signed"] == 1].groupby("region").size().sort_values(ascending=False)
    tot = sg.sum()
    return sg, tot


def power_curve(contracts_per_week, deltas, max_weeks=60, alpha=0.05, power=0.80):
    """Switchback / balanced design: each arm accrues ~half the weekly contracts.
    Analytic two-sample Poisson rate-ratio power (Wald on log-rate):
        SE(log RR) = sqrt(1/mt + 1/mc),  need log(1+d)/SE >= z_a/2 + z_power.
    Returns weeks needed for 80% power (None if > max_weeks)."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)  # ~2.80 for 0.05/0.80
    rc = contracts_per_week / 2.0
    out = {}
    for d in deltas:
        # N >= (z/log(1+d))^2 * (2+d)/((1+d)*rc)
        N = (z / np.log(1 + d)) ** 2 * (2 + d) / ((1 + d) * rc)
        out[d] = int(np.ceil(N)) if N <= max_weeks else None
    return out


def main():
    print("=" * 78)
    print("GEO-HOLDOUT POWER / FEASIBILITY")
    print("=" * 78)
    wk, stable = weekly_volume()
    cpw = stable["contracts"].mean()
    print("  weekly contract volume (matured stretch):")
    print(stable.tail(8).to_string())
    print(f"\n  -> contracts/week (mean, matured) = {cpw:.1f}")
    print(f"     leads/week (mean)              = {stable['leads'].mean():.0f}")

    sg, tot = region_concentration()
    top4 = sg.head(4).sum() / tot
    print("\n  region concentration of contracts (top regions):")
    print("   " + ", ".join(f"{r} {n}" for r, n in sg.head(6).items()))
    print(f"  -> top-4 regions hold {top4:.0%} of contracts  (n_regions={len(sg)})")
    print("     => naive region-split is LUMPY/imbalanced; prefer switchback or matched-pair.")

    print("\n" + "=" * 78)
    print("WEEKS TO 80% POWER (switchback / balanced; both arms same budget)")
    print("=" * 78)
    deltas = [0.10, 0.15, 0.20, 0.30, 0.50]
    res = power_curve(cpw, deltas)
    print(f"  {'detectable lift in contracts':<32}{'weeks needed':>14}")
    for d in deltas:
        w = res[d]
        print(f"  +{int(d*100)}% vs control{'':<18}{(str(w) if w else '>40'):>14}")
    # sensitivity: if the real run-rate is higher (all channels / growth)
    print("\n  sensitivity at higher run-rates (weeks for +20% lift):")
    for mult, lab in [(1.0, "current"), (1.5, "+50% volume"), (2.0, "2x volume")]:
        r = power_curve(cpw * mult, [0.20])
        print(f"    {lab:<14}: {cpw*mult:.0f} contracts/wk -> {r[0.20] or '>40'} weeks")


if __name__ == "__main__":
    main()
