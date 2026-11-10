"""Weekly reason feed: the Report-15 conversation markers, counted per week.

Why these and not the Qwen topic labels: the three markers that survived the
Report-15 four-yardstick test (financing discussed, "must ask partner/family",
product/eligibility mismatch) are LITERAL text patterns, validated against
signed contracts and the rep's own LOI. They run locally, in seconds, on the
transcripts already on this machine -- no model, no Kaggle, nothing uploaded.

The patterns and the call loading are imported from the Report-15 script
itself (outputs/sc1_drivers_2026-09-11/analyze.py), so the feed can never
drift from the definitions that were validated.

PERSONAL DATA: transcripts are read in this process only. The output holds
weekly counts and shares, groups under MIN_GROUP calls are suppressed, and no
call id, speaker, host or lead id is written. The control center's LOI Agent
reads the output; nothing else leaves this folder.

Usage (weekly, after Part 1):
    python -m src.export_reason_aggregates
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from datetime import date, datetime, timezone

import pandas as pd

from .config import ROOT

REPORT15 = ROOT / "outputs" / "sc1_drivers_2026-09-11" / "analyze.py"
OUT_DIR = ROOT / "outputs" / "reason_aggregates"

# Validated on contracts in Report 15 (direction of the effect on signing).
PRIMARY = {"financing": "raises", "decision_deferred": "raises", "product_mismatch": "lowers"}
# Shown for context only: they did NOT survive the four-yardstick test.
SECONDARY = ["roof_renovation_block", "timing_delay", "price_objection", "competitor_comparison",
             "no_funds", "trust_doubt", "think_about_it", "early_exploration"]
LABELS = {"financing": "financing / instalments discussed", "decision_deferred": "must ask partner or family",
          "product_mismatch": "product or eligibility mismatch", "roof_renovation_block": "roof works or renovation first",
          "timing_delay": "not now / later", "price_objection": "price objection",
          "competitor_comparison": "comparing other quotes", "no_funds": "no funds", "trust_doubt": "trust doubt",
          "think_about_it": "needs to think about it", "early_exploration": "just exploring"}
# Report 15's all-speaker topic patterns: what the consultation covered.
TOPICS = ["savings", "financing", "warranty_service", "installation_next_steps", "incentives",
          "roof_sizing", "battery", "contract_terms"]
# A FIXED watchlist, matched on normalised text (lower-case, no accents). Deliberately
# not a free "rising words" list: open vocabulary would surface names and addresses.
WATCH_TERMS = {
    "incentives": {
        "Conto Termico": r"conto termico", "Superbonus / bonus casa": r"superbonus|bonus casa|bonus ristrutturazion",
        "tax deduction 50%": r"detrazion\w*.{0,20}50|50.{0,5}(per ?cento|%).{0,20}detrazion",
        "Reddito Energetico": r"reddito energetico", "energy community (CER)": r"comunita energetic\w*|\bcer\b",
        "PNRR": r"\bpnrr\b"},
    "products": {
        "heat pump": r"pompa di calore|pompe di calore", "air conditioning": r"condizionator\w*|climatizzator\w*",
        "battery / storage": r"batteri\w*|accumul\w*", "EV charger / electric car": r"wallbox|colonnin\w*|auto elettric\w*"},
    "energy_costs": {
        "bills": r"bollett\w*", "energy price rises": r"(aument\w*|rincar\w*|caro).{0,20}(energi|luce|gas|bollett)"},
    # Customers often name their CURRENT supplier (Enel above all): a brand mention is
    # not a competing offer. The group name says so, so no one reads it as competition.
    "energy_brands": {
        "Enel": r"\benel\b", "Sorgenia": r"sorgenia", "Edison": r"\bedison\b", "Plenitude / Eni": r"plenitude|\beni\b",
        "Iren": r"\biren\b", "A2A": r"\ba2a\b", "Hera": r"\bhera\b", "Tesla": r"\btesla\b", "Engie": r"\bengie\b",
        "E.ON": r"\be\.?on\b"},
}
WATCH_RX = {(group, name): re.compile(pattern) for group, terms in WATCH_TERMS.items()
            for name, pattern in terms.items()}
MOVER_Z = 2.0           # two-proportion z against the pooled prior 4 weeks
MOVER_POINTS = 3.0      # and at least this many percentage points
MIN_WORDS = 1000        # Report 15's "real consultation" condition
MIN_GROUP = 20
WEEKS = 12


def _report15():
    spec = importlib.util.spec_from_file_location("report15_markers", REPORT15)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_markers() -> pd.DataFrame:
    """One row per sales consultation: week, words, band, role-resolved flag, marker booleans.

    Kept in memory only; main() reduces it to weekly shares before anything is written.
    """
    r15 = _report15()
    corpus, _ = r15.load_corpus()
    r15.role_map(corpus)
    rows = []
    for call in corpus:
        if call.get("call_type") != "sales":
            continue
        turns = [(t.get("role"), r15.normtext(t.get("text", ""))) for t in call["turns"]]
        row = {"call_date": call.get("call_date"), "words": call["words"], "band": call.get("pred"),
               "roles_resolved": call.get("role_method") != "unresolved"}
        for key in list(PRIMARY) + SECONDARY + [t for t in TOPICS if t not in PRIMARY]:
            scope = r15.FEATURES[key][0]
            row[key] = any(r15.RXS[key].search(text) for role, text in turns
                           if scope == "all" or role == "customer")
        whole = " ".join(text for _, text in turns)
        for (group, name), rx in WATCH_RX.items():
            row[f"term:{group}:{name}"] = bool(rx.search(whole))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["call_date"], errors="coerce")
    frame["week"] = (frame["date"] - pd.to_timedelta(frame["date"].dt.weekday, unit="D")).dt.normalize()
    return frame


def weekly_shares(frame: pd.DataFrame, today: pd.Timestamp | None = None) -> dict:
    """Marker shares per week over real consultations; customer-only markers over role-resolved calls."""
    r15 = _report15()
    consultations = frame[(frame["words"] >= MIN_WORDS) & (frame["band"] != "Non idoneo") & frame["week"].notna()]
    today = (today or pd.Timestamp.now()).normalize()
    this_monday = today - pd.Timedelta(days=today.weekday())
    last = this_monday - pd.Timedelta(days=7)
    weeks = []
    for week in pd.date_range(end=last, periods=WEEKS, freq="7D"):
        group = consultations[consultations["week"] == week]
        resolved = group[group["roles_resolved"]]
        entry = {"week": str(week.date()), "consultations": int(len(group)), "roles_resolved": int(len(resolved)),
                 "share_pct": {}}
        for key in list(PRIMARY) + SECONDARY:
            base = group if r15.FEATURES[key][0] == "all" else resolved
            entry["share_pct"][key] = round(100 * base[key].mean(), 1) if len(base) >= MIN_GROUP else None
        weeks.append(entry)
    change = {}
    for key in PRIMARY:
        now = weeks[-1]["share_pct"][key]
        prior = [w["share_pct"][key] for w in weeks[-5:-1] if w["share_pct"][key] is not None]
        if now is not None and prior:
            change[key] = {"now": now, "prior_4w": round(sum(prior) / len(prior), 1),
                           "points": round(now - sum(prior) / len(prior), 1)}
    return {"through_week": str(last.date()), "weeks": weeks, "primary_last_vs_prior_4w": change}


def _z(k1: float, n1: int, k0: float, n0: int) -> float | None:
    if not n1 or not n0:
        return None
    pooled = (k1 + k0) / (n1 + n0)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    return (k1 / n1 - k0 / n0) / se if se else None


def weekly_digest(frame: pd.DataFrame, today: pd.Timestamp | None = None) -> dict:
    """Last complete week of consultations: what was talked about, and what moved vs the 4 weeks before."""
    consultations = frame[(frame["words"] >= MIN_WORDS) & (frame["band"] != "Non idoneo") & frame["week"].notna()]
    today = (today or pd.Timestamp.now()).normalize()
    last = today - pd.Timedelta(days=today.weekday()) - pd.Timedelta(days=7)
    now = consultations[consultations["week"] == last]
    prior = consultations[consultations["week"].between(last - pd.Timedelta(days=28), last - pd.Timedelta(days=1))]
    if len(now) < MIN_GROUP or len(prior) < MIN_GROUP:
        return {"week": str(last.date()), "consultations": int(len(now)), "items": [], "movers": [],
                "note": f"Fewer than {MIN_GROUP} consultations in the week or its baseline."}
    items = []
    columns = [c for c in consultations.columns if c.startswith("term:")] + TOPICS
    for column in columns:
        if column.startswith("term:"):
            _, group, name = column.split(":", 2)
        else:
            group, name = "topics", column.replace("_", " ")
        k1, k0 = float(now[column].sum()), float(prior[column].sum())
        share, base = 100 * k1 / len(now), 100 * k0 / len(prior)
        z = _z(k1, len(now), k0, len(prior))
        items.append({"group": group, "name": name, "share_pct": round(share, 1), "prior_4w_pct": round(base, 1),
                      "points": round(share - base, 1), "z": round(z, 2) if z is not None else None})
    movers = [i for i in items if i["z"] is not None and abs(i["z"]) >= MOVER_Z and abs(i["points"]) >= MOVER_POINTS]
    movers.sort(key=lambda i: -abs(i["z"]))
    return {"week": str(last.date()), "consultations": int(len(now)), "baseline_consultations": int(len(prior)),
            "items": sorted(items, key=lambda i: (i["group"], -i["share_pct"])), "movers": movers,
            "rule": f"mover = |z| >= {MOVER_Z} and >= {MOVER_POINTS} points vs the 4 weeks before",
            "note": ("Share of real consultations mentioning each item, by any speaker (the rep included). "
                     "energy_brands = the customer's current supplier OR a competitor; the text cannot tell "
                     "them apart. A mover says the conversation shifted, not why.")}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    frame = call_markers()
    shares = weekly_shares(frame)
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Report-15 literal markers over local transcripts (no model)",
        "definitions": {key: {"label": LABELS[key], "effect_on_signing": PRIMARY.get(key, "not validated")}
                        for key in list(PRIMARY) + SECONDARY},
        "condition": f"sales consultations with >= {MIN_WORDS} words, band not Non idoneo; "
                     "customer-only markers over calls whose speakers were resolved",
        "min_group": MIN_GROUP,
        "note": "National: the corpus carries no channel for recent calls. Primary markers were validated "
                "on contracts (Report 15); read the others as context. Directions, not causes.",
        **shares,
        "digest": weekly_digest(frame),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"reason_aggregates_{date.today().isoformat()}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}: {len(frame)} sales calls scanned, weeks through {shares['through_week']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
