"""
LOI-aware EV redesign — shared data + calibration layer.

This module is the single source of truth for:
  - the channel×region cell universe (8 allocator channels)
  - the per-lead contract file (zoom_calendar_event_merged ... .xlsx)
  - the realized SPEND panel (cpl_model_input ... raw_data)
  - the LOI calibration  p_loi_k = P(sign | LOI=k)   (global)
  - the per-cell shrunk LOI shares  share_loi_k       (hierarchical + Dirichlet,
    replicating the production model: PRIOR_STRENGTH=50 channel→national,
    DIR_STRENGTH=10 cell→channel)
  - the per-cell LOI QUALITY term  quality = Σ_k p_loi_k · share_loi_k

Everything downstream (gate check, two-axis EV_LOI, allocator, backtest) imports
from here so the cell definitions and calibration never drift between steps.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
THESIS = Path(__file__).resolve().parent.parent.parent
CONTRACTS_XLSX = THESIS / "source_data" / "crm_contracts" / "zoom_calendar_event_merged_2026-06-22.xlsx"
PANEL_XLSX = THESIS / "MODELS INPUT" / "cpl_model_input.xlsx"
OUT_DIR = THESIS / "evaluation" / "loi_ev" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Cell universe
# --------------------------------------------------------------------------- #
# The 8 PAID allocator channels (what the budget model actually controls).
PAID_CHANNELS = ["Google", "MediaGo", "Meta", "Outbrain", "Tik Tok", "Youtube", "Display", "PMax"]

# raw LEAD_CHANNEL  ->  allocator channel
CHANNEL_MAP = {
    "facebook": "Meta", "fb": "Meta", "ig": "Meta", "instagram": "Meta",
    "search ads": "Google", "bing ads": "Google",
    "performancemax": "PMax",
    "outbrain": "Outbrain",
    "youtube": "Youtube",
    "mediago": "MediaGo",
    "display": "Display",
    "tiktok": "Tik Tok",
    # everything else (organic, partnership, referral, email, chatgpt.com, ...)
    # is NOT a paid budget channel -> mapped to None and excluded from the cell
    # universe (but still counted in the GLOBAL calibration if it carries LOI).
}

BANDS = [1, 4, 6, 7]  # Freddo, Tiepido, Calduccio, Caldo

# Production shrinkage constants (verified in scratch_recompute_ev.py)
PRIOR_STRENGTH = 50.0   # channel -> national
DIR_STRENGTH = 10.0     # cell -> channel

# Maturity: a consultation needs ~this many days for a contract to be observable.
MATURITY_DAYS = 35
TODAY = pd.Timestamp("2026-06-22")


def nreg(s) -> str:
    """Normalize a region label: strip accents/spaces/punctuation, lowercase."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def map_channel(c) -> str | None:
    return CHANNEL_MAP.get(str(c).strip().lower())


# --------------------------------------------------------------------------- #
# Per-lead contract file
# --------------------------------------------------------------------------- #
@dataclass
class LeadData:
    leads: pd.DataFrame          # all leads (mapped), one row per SC1
    maturity_cutoff: pd.Timestamp
    n_total: int
    n_signed: int


def load_leads(maturity_days: int = MATURITY_DAYS) -> LeadData:
    """Load the per-lead file, map channel/region, parse dates, flag maturity.

    Maturity clock = TSSTART (the SC1 consultation date); a lead is 'matured'
    if its consultation is at least `maturity_days` before TODAY, so a contract
    had time to land. Falls back to startTime / LEAD_CREATEDAT if TSSTART null.
    """
    df = pd.read_excel(CONTRACTS_XLSX, sheet_name="merged")

    df["signed"] = pd.to_numeric(df["CONTRACT_SIGNED"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    df["loi"] = pd.to_numeric(df["LOI"], errors="coerce")
    df["chan"] = df["LEAD_CHANNEL"].map(map_channel)
    # region: prefer ZIPREGION (matches the panel/allocator), fall back to REGION
    reg_src = df["LEAD_ZIPREGION"].where(df["LEAD_ZIPREGION"].notna(), df["LEAD_REGION"])
    df["region"] = reg_src.map(nreg)
    df["region_label"] = reg_src.astype(str)

    # consultation date (maturity clock)
    sc1 = pd.to_datetime(df.get("TSSTART_DATE"), errors="coerce")
    sc1 = sc1.fillna(pd.to_datetime(df.get("startTime"), errors="coerce"))
    sc1 = sc1.fillna(pd.to_datetime(df.get("LEAD_CREATEDAT"), errors="coerce"))
    df["sc1_date"] = sc1
    df["lead_created"] = pd.to_datetime(df.get("LEAD_CREATEDAT"), errors="coerce")

    cutoff = TODAY - pd.Timedelta(days=maturity_days)
    df["matured"] = df["sc1_date"] <= cutoff

    return LeadData(
        leads=df,
        maturity_cutoff=cutoff,
        n_total=len(df),
        n_signed=int(df["signed"].sum()),
    )


# --------------------------------------------------------------------------- #
# LOI calibration  p_loi_k = P(sign | LOI=k)
# --------------------------------------------------------------------------- #
def global_calibration(leads: pd.DataFrame, matured_only: bool = True) -> dict[int, float]:
    """Global P(sign | LOI=k) over leads carrying an in-scale LOI band.

    Returns {band: probability}. Matured-only by default (avoids the downward
    bias from recent leads that have not had time to sign).
    """
    d = leads[leads["loi"].isin(BANDS)].copy()
    if matured_only:
        d = d[d["matured"]]
    out = {}
    for k in BANDS:
        sub = d[d["loi"] == k]
        out[k] = float(sub["signed"].mean()) if len(sub) else np.nan
    return out


# --------------------------------------------------------------------------- #
# Per-cell shrunk LOI shares + quality term
# --------------------------------------------------------------------------- #
def _shares(frame: pd.DataFrame) -> tuple[dict[int, float], int]:
    v = frame["loi"].value_counts()
    tot = int(v.reindex(BANDS).fillna(0).sum())
    return ({k: (float(v.get(k, 0)) / tot if tot else 0.0) for k in BANDS}, tot)


def build_cell_quality(
    leads: pd.DataFrame,
    p_loi: dict[int, float],
    paid_only: bool = True,
    prior_strength: float = PRIOR_STRENGTH,
    dir_strength: float = DIR_STRENGTH,
) -> pd.DataFrame:
    """Per (channel, region) LOI quality via hierarchical + Dirichlet shrinkage.

    Replicates the production recipe:
      national shares  ->  channel shares (smoothed to national, PRIOR_STRENGTH)
                       ->  cell shares     (Dirichlet-blended, DIR_STRENGTH)
      quality = Σ_k p_loi_k · cell_share_k     (in PROBABILITY units, not %)

    Only leads carrying an in-scale LOI band feed the share computation.
    Returns one row per cell that has at least one LOI-bearing lead.
    """
    d = leads[leads["loi"].isin(BANDS)].copy()
    if paid_only:
        d = d[d["chan"].notna()]

    p_nat, n_nat = _shares(d)
    chan_share, chan_n = {}, {}
    for ch, g in d.groupby("chan"):
        chan_share[ch], chan_n[ch] = _shares(g)

    cell_counts = d.groupby(["chan", "region", "loi"]).size()

    rows = []
    for (ch, rg), g in d.groupby(["chan", "region"]):
        p_ch = chan_share.get(ch, p_nat)
        n_ch = chan_n.get(ch, 0)
        # level-1 hierarchical prior: channel smoothed toward national
        p_prior = {k: (n_ch * p_ch[k] + prior_strength * p_nat[k]) / (n_ch + prior_strength) for k in BANDS}
        # level-2 cell Dirichlet blend
        c = {k: float(cell_counts.get((ch, rg, k), 0)) for k in BANDS}
        n_loi = sum(c.values())
        bs = {k: (dir_strength * p_prior[k] + c[k]) / (dir_strength + n_loi) for k in BANDS}
        w_obs = n_loi / (dir_strength + n_loi)
        quality = sum(p_loi[k] * bs[k] for k in BANDS)
        rows.append(dict(
            chan=ch, region=rg, n_loi=int(n_loi), reliability=round(w_obs, 3),
            share1=bs[1], share4=bs[4], share6=bs[6], share7=bs[7],
            quality=quality,
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Realized outcomes per cell (the honest anchor)
# --------------------------------------------------------------------------- #
def build_cell_outcomes(leads: pd.DataFrame, matured_only: bool = True, paid_only: bool = True) -> pd.DataFrame:
    """Per (channel, region): n_leads, n_signed, realized contract RATE.

    Counts ALL leads in the cell (not just LOI-bearing) — the realized rate is
    a property of the cell, independent of whether the agent graded the call.
    """
    d = leads.copy()
    if paid_only:
        d = d[d["chan"].notna()]
    if matured_only:
        d = d[d["matured"]]
    g = d.groupby(["chan", "region"]).agg(
        n_leads=("signed", "size"),
        n_signed=("signed", "sum"),
    ).reset_index()
    g["realized_rate"] = g["n_signed"] / g["n_leads"]
    return g


def load_panel_spend(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Realized marketing spend per (channel, region) over [start, end]."""
    pan = pd.read_excel(PANEL_XLSX, sheet_name="raw_data")
    pan["date"] = pd.to_datetime(dict(
        year=pd.to_numeric(pan["Year"], errors="coerce"),
        month=pd.to_numeric(pan["Month"], errors="coerce"),
        day=pd.to_numeric(pan["Day"], errors="coerce"),
    ), errors="coerce")
    pan["spend"] = pd.to_numeric(pan["mktg_spend"], errors="coerce").fillna(0.0)
    pan["chan"] = pan["Aggregated"].astype(str)
    pan["region"] = pan["ZIPREGION"].map(nreg)
    sub = pan[(pan["date"] >= start) & (pan["date"] <= end)]
    g = sub.groupby(["chan", "region"])["spend"].sum().reset_index()
    return g
