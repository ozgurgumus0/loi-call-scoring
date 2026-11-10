"""
VERIFY ON OUR QWEN LOI — does our model's LOI (AUC 0.68 per-lead, plus the
continuous closing_readiness) carry cell-level allocation signal that the agent
LOI lacked?  Same gate + same spread test, our-LOI quality vs agent-LOI quality
vs base_rate.

Join: pred.call_id -> calls.csv.meeting_id (float->int) -> contract file.meetingId
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
from gate_check import weighted_spearman

MVP = C.THESIS.parent / "loi-audio-mvp"
BAND_MAP = {"Freddo": 1, "Tiepido": 4, "Calduccio": 6, "Caldo": 7}


def join_our_loi():
    pred = pd.read_csv(MVP / "predictions_qwen3_8b_sales_full.csv")
    calls = pd.read_csv(MVP / "data" / "calls.csv", low_memory=False)
    czoom = calls[calls["source"] == "zoom"][["call_id", "meeting_id"]].dropna(subset=["meeting_id"])
    czoom["meetingId"] = czoom["meeting_id"].astype("int64")
    pred = pred.merge(czoom[["call_id", "meetingId"]], on="call_id", how="left")

    fr = pd.read_excel(C.CONTRACTS_XLSX, sheet_name="merged")
    fr["signed"] = pd.to_numeric(fr["CONTRACT_SIGNED"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    fr["chan"] = fr["LEAD_CHANNEL"].map(C.map_channel)
    reg_src = fr["LEAD_ZIPREGION"].where(fr["LEAD_ZIPREGION"].notna(), fr["LEAD_REGION"])
    fr["region"] = reg_src.map(C.nreg)
    sc1 = pd.to_datetime(fr.get("TSSTART_DATE"), errors="coerce").fillna(pd.to_datetime(fr.get("startTime"), errors="coerce"))
    fr["sc1_date"] = sc1
    fr["matured"] = fr["sc1_date"] <= (C.TODAY - pd.Timedelta(days=C.MATURITY_DAYS))

    m = pred.merge(fr[["meetingId", "chan", "region", "signed", "matured", "sc1_date"]],
                   on="meetingId", how="left")
    m["our_band"] = m["pred"].map(BAND_MAP)
    m["cr"] = pd.to_numeric(m["closing_readiness"], errors="coerce")
    return m, pred


def main():
    m, pred = join_our_loi()
    joined = m[m["chan"].notna() & m["region"].notna()].copy()
    print("=" * 80)
    print("OUR QWEN LOI — JOIN COVERAGE")
    print("=" * 80)
    print(f"  our scored sales calls       : {len(pred):,}")
    print(f"  joined to a meetingId        : {m['meetingId'].notna().sum():,}")
    print(f"  joined w/ paid chan+region   : {len(joined):,}")
    print(f"  of those, matured            : {int(joined['matured'].sum()):,}  signed {int(joined.loc[joined['matured'],'signed'].sum())}")
    mat = joined[joined["matured"]].copy()

    # ---- our calibration P(sign | our band) ----
    print("\n  P(sign | OUR band), matured:")
    for name, k in BAND_MAP.items():
        sub = mat[mat["our_band"] == k]
        r = sub["signed"].mean() * 100 if len(sub) else float("nan")
        print(f"    {name:<10} n={len(sub):>4}  P(sign)={r:5.2f}%")
    p_our = {k: (mat[mat["our_band"] == k]["signed"].mean() if (mat["our_band"] == k).any() else np.nan) for k in C.BANDS}
    # fill any nan band with global mean
    gmean = mat["signed"].mean()
    p_our = {k: (v if np.isfinite(v) else gmean) for k, v in p_our.items()}

    # ---- per-cell our-LOI quality: BAND-based (share machinery) ----
    def shares(frame):
        v = frame["our_band"].value_counts(); tot = int(v.reindex(C.BANDS).fillna(0).sum())
        return {k: (float(v.get(k, 0)) / tot if tot else 0.0) for k in C.BANDS}, tot
    d = mat[mat["our_band"].isin(C.BANDS)]
    p_nat, _ = shares(d)
    chs, chn = {}, {}
    for ch, g in d.groupby("chan"):
        chs[ch], chn[ch] = shares(g)
    cc = d.groupby(["chan", "region", "our_band"]).size()
    band_rows = []
    for (ch, rg), g in d.groupby(["chan", "region"]):
        p_ch, n_ch = chs.get(ch, p_nat), chn.get(ch, 0)
        p_prior = {k: (n_ch * p_ch[k] + C.PRIOR_STRENGTH * p_nat[k]) / (n_ch + C.PRIOR_STRENGTH) for k in C.BANDS}
        cnt = {k: float(cc.get((ch, rg, k), 0)) for k in C.BANDS}
        nl = sum(cnt.values())
        bs = {k: (C.DIR_STRENGTH * p_prior[k] + cnt[k]) / (C.DIR_STRENGTH + nl) for k in C.BANDS}
        band_rows.append(dict(chan=ch, region=rg, n_our=int(nl),
                              q_band=sum(p_our[k] * bs[k] for k in C.BANDS)))
    qb = pd.DataFrame(band_rows)

    # ---- per-cell our-LOI quality: CONTINUOUS closing_readiness ----
    # global logistic-ish calibration cr -> P(sign): use isotonic-free simple bins
    cal = mat.dropna(subset=["cr"]).copy()
    # map cr to P(sign) via global mean within deciles, then per-cell mean of mapped P
    cal["crbin"] = pd.qcut(cal["cr"].rank(method="first"), 10, labels=False)
    binrate = cal.groupby("crbin")["signed"].mean()
    cal["p_cr"] = cal["crbin"].map(binrate)
    # shrink per-cell continuous quality toward channel/global
    gP = cal["p_cr"].mean()
    chP = cal.groupby("chan")["p_cr"].mean().to_dict()
    cont_rows = []
    for (ch, rg), g in cal.groupby(["chan", "region"]):
        n = len(g); prior = chP.get(ch, gP)
        prior = (chn.get(ch, 0) * prior + C.PRIOR_STRENGTH * gP) / (chn.get(ch, 0) + C.PRIOR_STRENGTH)
        q = (C.DIR_STRENGTH * prior + g["p_cr"].sum()) / (C.DIR_STRENGTH + n)
        cont_rows.append(dict(chan=ch, region=rg, n_cr=n, q_cont=q))
    qc = pd.DataFrame(cont_rows)

    # ---- realized outcomes per cell (full matured agent set = the truth) ----
    o = C.build_cell_outcomes(C.load_leads().leads, matured_only=True, paid_only=True)
    cell = o.merge(qb, on=["chan", "region"], how="left").merge(qc, on=["chan", "region"], how="left")

    print("\n" + "=" * 80)
    print("SPREAD of OUR-LOI quality across cells (vs agent flat 3.29-4.54%)")
    print("=" * 80)
    for col, lab in [("q_band", "our band-quality"), ("q_cont", "our cont-quality (cr)")]:
        v = cell[col].dropna()
        print(f"  {lab:<24}: n={v.size:>3}  range {v.min()*100:.2f}-{v.max()*100:.2f}%  spread {(v.max()-v.min())*100:.2f}pp  std {v.std()*100:.3f}pp")

    print("\n" + "=" * 80)
    print("GATE — Spearman(our quality, realized_rate), cells >=20 matured leads")
    print("=" * 80)
    g = cell[(cell["n_leads"] >= 20)].copy()
    for col, lab in [("q_band", "our band-quality"), ("q_cont", "our cont-quality")]:
        gg = g.dropna(subset=[col])
        if len(gg) >= 5:
            rho = spearmanr(gg[col], gg["realized_rate"]).correlation
            rhow = weighted_spearman(gg[col], gg["realized_rate"], gg["n_leads"])
            print(f"  {lab:<22}: n={len(gg):>2}  Spearman {rho:+.3f}  lead-wtd {rhow:+.3f}")
    print("  (agent-LOI reference: lead-wtd +0.437)")

    cell.to_csv(C.OUT_DIR / "our_loi_cell_table.csv", index=False)
    print(f"\nsaved -> {C.OUT_DIR / 'our_loi_cell_table.csv'}")


if __name__ == "__main__":
    main()
