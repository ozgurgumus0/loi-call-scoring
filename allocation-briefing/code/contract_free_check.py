"""
CONTRACT-FREE proof of whether LOI can drive channel x region allocation.
No contracts used anywhere. Pure question: does the LOI score VARY across the
cells the allocator chooses between? A signal that's ~constant across cells
cannot steer budget between them, regardless of how value is measured.

Variance decomposition of the LOI score:
   between-channel  +  between-region-within-channel  +  within-cell(lead-to-lead)
If almost all the variance is lead-to-lead (within cell), LOI is a per-LEAD
signal that flattens at the cell level -> can't allocate. Contracts irrelevant.
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
from our_loi_verify import join_our_loi


def decompose(df, val, label):
    """Nested SS decomposition: total = between-channel + region-within-channel + within-cell."""
    d = df.dropna(subset=[val, "chan", "region"]).copy()
    grand = d[val].mean()
    ss_total = ((d[val] - grand) ** 2).sum()
    # cell means
    cellm = d.groupby(["chan", "region"])[val].transform("mean")
    ss_within = ((d[val] - cellm) ** 2).sum()
    ss_between_cell = ss_total - ss_within
    # channel means
    chanm = d.groupby("chan")[val].transform("mean")
    ss_between_chan = ((chanm - grand) ** 2).sum()
    ss_region_in_chan = ss_between_cell - ss_between_chan
    print(f"\n=== {label}  (n={len(d)}, leads with the score) ===")
    print(f"  variance split:")
    print(f"    between CHANNEL              : {ss_between_chan/ss_total*100:5.1f}%")
    print(f"    between REGION within channel: {ss_region_in_chan/ss_total*100:5.1f}%")
    print(f"    within CELL (lead-to-lead)   : {ss_within/ss_total*100:5.1f}%")
    print(f"  => cell-level (channel+region) carries {ss_between_cell/ss_total*100:.1f}% of LOI's variation;")
    print(f"     {ss_within/ss_total*100:.1f}% is between individual leads (where LOI's AUC lives).")
    # cell-mean spread vs lead-level spread
    cm = d.groupby(["chan", "region"]).filter(lambda g: len(g) >= 20).groupby(["chan", "region"])[val].mean()
    lead_sd = d[val].std()
    print(f"  cell-mean range (cells>=20 leads): {cm.min():.1f} .. {cm.max():.1f}  (std across cells {cm.std():.2f})")
    print(f"  lead-level SD of the score       : {lead_sd:.2f}   (cell means vary {cm.std()/lead_sd:.2f}x of one lead's SD)")


def main():
    print("=" * 78)
    print("CONTRACT-FREE: does LOI vary across the cells the allocator chooses between?")
    print("=" * 78)

    # (1) OUR continuous closing_readiness (our model's best signal)
    m, _ = join_our_loi()
    m = m[m["chan"].notna()].copy()
    m["cr"] = pd.to_numeric(m["closing_readiness"], errors="coerce")
    decompose(m, "cr", "OUR Qwen closing_readiness (0-100)")

    # (2) agent LOI band (full coverage, ordinal 1/4/6/7)
    leads = C.load_leads().leads
    leads = leads[leads["chan"].notna() & leads["loi"].isin(C.BANDS)].copy()
    decompose(leads, "loi", "Agent LOI band (1/4/6/7)")

    print("\n" + "=" * 78)
    print("READ")
    print("=" * 78)
    print("  If cell-level variation is small and lead-to-lead variation dominates, LOI is")
    print("  intrinsically a per-LEAD signal that is ~flat across channel x region cells.")
    print("  A ~flat signal cannot rank the cells -> cannot steer budget between them, no")
    print("  matter how you score value. This needs NO contracts to conclude.")


if __name__ == "__main__":
    main()
