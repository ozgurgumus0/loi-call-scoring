"""Topic / objection / buyer-signal analysis on top of LOI scores.

Answers:
  - Which topics, when discussed, raise the mean LOI? Which hurt?
  - Which objections are LOI-killers and which are signs of engagement?
  - Which buyer signals are strongest predictors of warm calls?
  - Per-rep: what topics is each rep strongest/weakest on?
  - Per-channel: what topics do Outbrain vs Facebook vs Search Ads
    conversations cover?

Reads:
    outputs/topics/zoom__*.json   (from src.extract_topics)
    outputs/scores/zoom__*.json   (Sonnet LOI scores)
    data/master.csv               (CRM features + rep identity)

Writes:
    outputs/reports/08_topic_analysis.md
    outputs/topic_analysis.xlsx
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .config import DATA, OUTPUTS, OUT_REPORTS, OUT_SCORES

TOPICS_DIR = OUTPUTS / "topics"
REPORT_MD = OUT_REPORTS / "08_topic_analysis.md"
REPORT_XLSX = OUTPUTS / "topic_analysis.xlsx"

IN_BAND = ["Freddo", "Tiepido", "Calduccio", "Caldo"]
DEAD_BAND = ["Non interessato", "Non idoneo"]

# Italian keyword buckets for clustering free-text customer_questions WITHOUT
# an LLM call. Each pattern is matched case-insensitively against the joined
# questions of a call. Buckets are not mutually exclusive (a call can ask about
# both price and financing).
QUESTION_BUCKETS = {
    "price_cost":     r"prezz|costo|cost[ai]|quanto (cost|si spend|verr|viene)|spesa|importo|chiavi in mano|tutto (compreso|incluso)|compreso nel prezzo",
    "incentives_tax": r"incentiv|detra|detrarre|bonus|agevolazion|conto termico|pnrr|cashback|ecobonus|50%|36%|rimborso|recuper",
    "financing":      r"finanziament|finanziaria|\brata\b|\brate\b|tasso|mutuo|interess|pagament",
    "battery":        r"batteri|accumulo|azzurro|fox|capacit. (della|di accumulo)",
    "heat_pump":      r"pompa di calore|caldaia|ibrid",
    "install_timing": r"installa|montare|montagg|tempistic|tempi(?!d)|quando|allacci|posa|cantiere|sopralluogo",
    "roof_space":     r"tetto|metri quadr|\bmq\b|superficie|falda|pergola|spazio|ombr|esposizion|copertura",
    "warranty_dur":   r"garanzia|vita utile|durata|durano|\banni\b|degrado|manutenzion",
    "ev_wallbox":     r"auto elettrica|colonnin|wall ?box|ricaric",
    "grid_cer":       r"\brete\b|enel|comunit. energetica|\bcer\b|rivendita|immission|gse|stacc",
    "product_comp":   r"inverter|pannell|ottimizzat|marca|modello|watt|\bkw\b|potenza",
    "contract_proc":  r"contratt|pratich|burocra|document|firm|recess|assicurazion",
}


GLOSSARY_MD = """## 12. Glossary — what each label means

_Short definitions of the controlled vocabulary used above. Topics/objections/
signals are the exact tokens Sonnet was allowed to emit; question-themes are our
own keyword buckets from §9._

### Topics discussed
| label | meaning (short) |
|---|---|
| `financing` | Loan / instalments / deposit / payment plan |
| `system_size` | kW sizing, number of panels |
| `savings` | Bill savings, self-consumption, payback / ROI |
| `battery` | Storage batteries, capacity |
| `heat_pump` | Heat pump, heating & cooling |
| `incentives` | State bonuses, tax deduction, ecobonus, CER, GSE |
| `installation` | Install timing, work phases, site survey |
| `warranty` | Product warranty / coverage |
| `after_sales` | Service, maintenance, post-install monitoring |
| `roof_assessment` | Roof exposure, shading, structure |
| `grid_export` | Selling energy, grid feed-in, net metering |
| `renovation` | Renovation works in progress (windows, insulation) |
| `other_quotes` | Competing quotes / comparison shopping |
| `decision_makers` | Spouse / family / co-owners to consult |
| `contract_terms` | Contract terms, conditions, duration |

### Objections raised
| label | meaning (short) |
|---|---|
| `price_too_high` | Too expensive / over budget |
| `timing_not_right` | "Not now", waiting — **the one real LOI-killer** |
| `decision_maker_absent` | Must speak to partner/family first |
| `existing_quote` | Already holds other quotes |
| `house_situation` | Renting, selling soon, complicated building |
| `skeptical_savings` | Doubts the promised savings |
| `system_complexity` | "Too complicated", doesn't follow the tech |
| `trust_concerns` | Doubts about the company / bad past experience |
| `waiting_better_offer` | Shopping the market for a better deal |
| `renovation_in_progress` | Must finish ongoing works first |

### Buyer signals
| label | meaning (short) |
|---|---|
| `ready_to_decide` | About to decide / nearly decided |
| `urgency` | Wants to install soon |
| `comparing_actively` | Asks for specific competitor comparison |
| `technical_questions` | Detailed technical questions, knows the topic |
| `mentions_timeline` | Cites specific dates for decision/install |
| `requests_quote` | Asks for a formal quote |
| `requests_visit` | Asks for a site survey |
| `mentions_financing` | Asks about financing details |
| `references_research` | Has already researched, knows the market |
| `proactive_interest` | Spontaneous interest, proposes solutions |

### Customer-question themes (§9, our keyword buckets)
| label | meaning (short) |
|---|---|
| `price_cost` | Total price, "all-inclusive?", what it costs |
| `incentives_tax` | Bonuses, deductions, ecobonus, PNRR, conto termico |
| `financing` | Instalments, rate, interest, loan, payment |
| `battery` | Storage battery, capacity, brands |
| `heat_pump` | Heat pump / boiler / hybrid |
| `install_timing` | Install dates, lead times, site survey, grid hook-up |
| `roof_space` | Roof, m², exposure, shading, pergola |
| `warranty_dur` | Warranty, lifespan, degradation, maintenance |
| `ev_wallbox` | EV charging, wallbox |
| `grid_cer` | Grid, energy community (CER), feed-in, GSE |
| `product_comp` | Inverter, panels, optimisers, brand/model, kW |
| `contract_proc` | Contract, paperwork, bureaucracy, signing |
"""


def _band_stats(g: pd.DataFrame) -> dict:
    """Mean in-band LOI + dead-rate for a cohort. Free; no LLM."""
    gi = g[g["sonnet_outcome"].isin(IN_BAND)]
    loi = pd.to_numeric(gi["sonnet_score"], errors="coerce").dropna()
    dead = g["sonnet_outcome"].isin(DEAD_BAND)
    return {
        "n": len(g),
        "mean_loi": round(loi.mean(), 2) if len(loi) else float("nan"),
        "pct_dead": round(100 * dead.mean()) if len(g) else float("nan"),
    }


def _load_topics(topics_dir: Path = TOPICS_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(topics_dir.glob("zoom__*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("parse_fail"):
            continue                  # Qwen call whose JSON didn't parse
        e = d.get("extraction") or {}
        rows.append({
            "call_id":             d.get("call_id"),
            "topics":              tuple(e.get("topics_discussed") or []),
            "objections":          tuple(e.get("objections_raised") or []),
            "buyer_signals":       tuple(e.get("buyer_signals") or []),
            "rep_name_extracted":  e.get("rep_name"),
            "rep_strengths":       e.get("rep_strengths") or [],
            "rep_weaknesses":      e.get("rep_weaknesses") or [],
            "customer_questions":  e.get("customer_questions") or [],
            "call_summary_it":     e.get("call_summary_it") or "",
        })
    return pd.DataFrame(rows)


def _load_sonnet() -> pd.DataFrame:
    rows = []
    for p in sorted(OUT_SCORES.glob("zoom__*.json")):
        if "__ollama_" in p.name or p.name.startswith("upload__") or p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (d.get("model") or "").lower().startswith("claude-sonnet"):
            continue
        sc = d.get("score") or {}
        if not sc.get("outcome"):
            continue
        rows.append({
            "call_id":         d.get("call_id"),
            "sonnet_outcome":  sc.get("outcome"),
            "sonnet_score":    sc.get("score"),
        })
    # One call is saved under two filenames (with/without the base64 '=='
    # suffix) — dedupe so the topics merge can't double-count it.
    return pd.DataFrame(rows).drop_duplicates(subset="call_id", keep="first")


def _feature_lift_table(df: pd.DataFrame, feature_col: str,
                        min_n: int = 5) -> pd.DataFrame:
    """For each token in df[feature_col] (a Series of tuples), compute mean
    LOI when token present vs absent, with sample sizes and t-test p-value."""
    from scipy.stats import ttest_ind

    df_in = df[df["sonnet_outcome"].isin(IN_BAND)].copy()
    df_in["sonnet_score"] = pd.to_numeric(df_in["sonnet_score"], errors="coerce")
    all_tokens = sorted({t for tup in df_in[feature_col] for t in tup})
    rows = []
    overall_mean = df_in["sonnet_score"].mean()
    for tok in all_tokens:
        mask = df_in[feature_col].apply(lambda tup: tok in tup)
        n_with = int(mask.sum())
        n_without = int((~mask).sum())
        if n_with < min_n or n_without < min_n:
            continue
        loi_with = df_in.loc[mask, "sonnet_score"].dropna()
        loi_without = df_in.loc[~mask, "sonnet_score"].dropna()
        try:
            t_stat, p_val = ttest_ind(loi_with, loi_without, equal_var=False)
        except Exception:
            t_stat, p_val = float("nan"), float("nan")
        rows.append({
            feature_col[:-1] if feature_col.endswith("s") else feature_col:
                tok,
            "n_with":     n_with,
            "mean_loi_with":    round(loi_with.mean(), 2),
            "mean_loi_without": round(loi_without.mean(), 2),
            "delta":      round(loi_with.mean() - loi_without.mean(), 2),
            "p_value":    round(p_val, 4),
        })
    # Always carry the columns so callers indexing ["delta"] don't KeyError
    # on a small/partial corpus where nothing reaches min_n.
    name = feature_col[:-1] if feature_col.endswith("s") else feature_col
    out = pd.DataFrame(rows, columns=[name, "n_with", "mean_loi_with",
                                      "mean_loi_without", "delta", "p_value"])
    if not out.empty:
        out = out.sort_values("delta", ascending=False)
    return out


def _md_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"### {title}\n\n_(no rows)_\n"
    L = [f"### {title}", "",
         "| " + " | ".join(df.columns) + " |",
         "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        L.append("| " + " | ".join("" if pd.isna(v) else str(v)
                                     for v in r.values) + " |")
    return "\n".join(L) + "\n"


# Labels whose Qwen tags failed the lift-replication audit on the 282-call
# Claude overlap (2026-06-11). Two are lift-DISCORDANT (technical_questions:
# Claude lift +2.22 vs Qwen -0.13; timing_not_right: -0.46 vs +0.05 — the 8B
# under/mis-tags them so badly the with/without comparison breaks); the rest
# are rare labels the 8B essentially never tags right (R <= 0.12 or P <= 0.09
# in the v3 agreement table). The other 27 measurable labels' lifts replicate
# (sign-concordant). These stay Claude-282-only in the main report 08.
QWEN_EXCLUDED_LABELS = {
    "objections": {"timing_not_right", "waiting_better_offer",
                   "trust_concerns", "system_complexity"},
    "buyer_signals": {"technical_questions", "ready_to_decide", "urgency"},
}


def run(extractor: str = "claude") -> None:
    """extractor='claude' -> outputs/topics (the original 282-call audit set);
    extractor='qwen'   -> outputs/topics_qwen (the full Qwen3-8B sweep) and the
    report/xlsx get an `_qwen` suffix so the two stay side-by-side until the
    Qwen version is promoted."""
    if extractor == "qwen":
        topics_dir = OUTPUTS / "topics_qwen"
        report_md = OUT_REPORTS / "08_topic_analysis_qwen.md"
        report_xlsx = OUTPUTS / "topic_analysis_qwen.xlsx"
    else:
        topics_dir = TOPICS_DIR
        report_md, report_xlsx = REPORT_MD, REPORT_XLSX
    topics_df = _load_topics(topics_dir)
    if extractor == "qwen" and not topics_df.empty:
        for col, bad in QWEN_EXCLUDED_LABELS.items():
            topics_df[col] = topics_df[col].map(
                lambda t: tuple(x for x in t if x not in bad))
    sonnet_df = _load_sonnet()
    if topics_df.empty:
        print(f"No topic files in {topics_dir}. Run `python -m "
              f"src.extract_topics` (claude) or ingest the Kaggle output "
              f"(`python -m src.ingest_qwen_topics ...`) first.")
        return

    master = pd.read_csv(DATA / "master.csv", low_memory=False)
    master = master[master["source"] == "zoom"][[
        "call_id", "call_date", "call_type", "lead_channel",
        "lead_segmentation", "lead_region", "lead_producttype",
        "lead_score", "crm_netsc1",
        "lead_utmsource", "lead_utmcampaign", "lead_utmcontent", "lead_utmterm",
    ]]
    df = topics_df.merge(sonnet_df, on="call_id", how="left")
    df = df.merge(master, on="call_id", how="left")
    df["sonnet_score"] = pd.to_numeric(df["sonnet_score"], errors="coerce")
    df["lead_score"] = pd.to_numeric(df["lead_score"], errors="coerce")

    # Booking-contamination guard (2026-06-09): the topic extraction ran on the
    # mixed corpus, where appointment-setting (booking) calls are scored
    # with the sales rubric. Take call_type from the authoritative calls.csv
    # (master.csv is stale on newer rows) and keep sales consultations only.
    calls = pd.read_csv(DATA / "calls.csv", low_memory=False)
    ctype = calls[calls["source"] == "zoom"].set_index("call_id")["call_type"]
    df["call_type"] = df["call_id"].map(ctype).fillna(df["call_type"])
    n_mixed = len(df)
    df = df[df["call_type"] == "sales"].copy()
    n_booking_dropped = n_mixed - len(df)

    n = len(df)
    in_band_n = int(df["sonnet_outcome"].isin(IN_BAND).sum())

    # Date span of the analysed calls (free — call_date already in master).
    dates = pd.to_datetime(df["call_date"], errors="coerce").dropna()
    if len(dates):
        date_min = dates.min().date()
        date_max = dates.max().date()
        n_days = (date_max - date_min).days + 1
        date_line = (f"**Calls span {date_min:%d %b %Y} → {date_max:%d %b %Y}** "
                     f"({n_days} calendar days). "
                     f"{len(dates)} of {n} calls carry a usable date.")
    else:
        date_line = "_(no usable call dates found)_"

    L = ["# 08 - Topic / objection / buyer-signal x LOI analysis",
         "",
         ("_Topic extractor: **Claude Sonnet** (original 2026-06-03 run)._"
          if extractor == "claude" else
          "_Topic extractor: **Qwen3-8B, local & free**, audited against the "
          "282-call Claude overlap via lift replication (27/29 measurable "
          "labels sign-concordant, 2026-06-11). **Seven labels are excluded** "
          "because their Qwen tags failed the audit and they stay in the "
          "Claude-282 report only: `timing_not_right`, `technical_questions` "
          "(lift-discordant), `waiting_better_offer`, `trust_concerns`, "
          "`system_complexity`, `ready_to_decide`, `urgency` (too rarely / "
          "wrongly tagged). CAVEAT: Qwen lift MAGNITUDES attenuate vs Claude "
          "(e.g. financing +1.86 → +0.55 on the same calls) — read lift "
          "DIRECTIONS and ranks, not absolute sizes._"),
         "",
         date_line,
         "",
         f"_Built from {n} SALES consultations with both topic extraction and "
         f"Sonnet LOI scores ({in_band_n} of which are in-band "
         "Freddo/Tiepido/Calduccio/Caldo). "
         f"**{n_booking_dropped} booking (appointment-setting) calls were "
         "excluded** after the 2026-06-09 call_type cleanup — they were scored "
         "with the sales rubric and polluted earlier versions of this report. "
         "All effect sizes use in-band only because Non interessato / Non idoneo "
         "don't have a numeric LOI score._",
         ""]

    # --- Section 1: topics helping vs hurting LOI ---
    topics_lift = _feature_lift_table(df, "topics", min_n=5)
    helpers = topics_lift[topics_lift["delta"] > 0]
    hurters = topics_lift[topics_lift["delta"] < 0]

    L += ["## 1. Topics that HELP the LOI score",
          "",
          "_Mean LOI on calls where this topic was discussed, vs calls where it "
          "wasn't. Sorted by `delta` (positive = warmer when discussed)._",
          "",
          _md_table(helpers, "Helping topics"),
          "",
          "## 2. Topics that HURT the LOI score",
          "",
          _md_table(hurters.sort_values("delta"), "Hurting topics"),
          ""]

    # --- Section 3: objections ---
    obj_lift = _feature_lift_table(df, "objections", min_n=5)
    L += ["## 3. Objections — which are LOI-killers vs signs of engagement",
          "",
          "_Note: some objections (`existing_quote`, `decision_maker_absent`) "
          "are often raised by ENGAGED prospects who are seriously evaluating. "
          "Negative delta = LOI-killer; positive delta = engagement signal._",
          "",
          _md_table(obj_lift, "Objections × LOI"),
          ""]

    # --- Section 4: buyer signals ---
    sig_lift = _feature_lift_table(df, "buyer_signals", min_n=5)
    L += ["## 4. Buyer signals × LOI",
          "",
          _md_table(sig_lift, "Buyer signals × LOI"),
          ""]

    # --- Section 5: prevalence (which topics dominate calls overall?) ---
    tc = Counter(t for tup in df["topics"] for t in tup)
    oc = Counter(t for tup in df["objections"] for t in tup)
    sc = Counter(t for tup in df["buyer_signals"] for t in tup)
    L += ["## 5. Overall prevalence",
          "",
          "### Most-discussed topics",
          "",
          "| topic | n_calls | % of corpus |",
          "|---|---:|---:|",
          *[f"| {t} | {c} | {100*c/n:.0f}% |"
            for t, c in tc.most_common()],
          "",
          "### Most-frequent objections",
          "",
          "| objection | n_calls | % of corpus |",
          "|---|---:|---:|",
          *[f"| {t} | {c} | {100*c/n:.0f}% |"
            for t, c in oc.most_common()],
          "",
          "### Most-frequent buyer signals",
          "",
          "| signal | n_calls | % of corpus |",
          "|---|---:|---:|",
          *[f"| {t} | {c} | {100*c/n:.0f}% |"
            for t, c in sc.most_common()],
          ""]

    # --- Section 6: by channel ---
    L += ["## 6. Topic mix by channel",
          "",
          "_What do prospects from different sources actually talk about?_",
          ""]
    for col in ("lead_channel",):
        sub = df.dropna(subset=[col])
        for ch, g in sub.groupby(col):
            if len(g) < 5:
                continue
            tc_ch = Counter(t for tup in g["topics"] for t in tup)
            sig_ch = Counter(t for tup in g["buyer_signals"] for t in tup)
            obj_ch = Counter(t for tup in g["objections"] for t in tup)
            L += [f"### {ch}  (n={len(g)})", "",
                  "Top topics: " + ", ".join(
                      f"`{t}` ({100*c/len(g):.0f}%)"
                      for t, c in tc_ch.most_common(8)),
                  "",
                  "Top signals: " + ", ".join(
                      f"`{t}` ({100*c/len(g):.0f}%)"
                      for t, c in sig_ch.most_common(5)),
                  "",
                  "Top objections: " + (", ".join(
                      f"`{t}` ({100*c/len(g):.0f}%)"
                      for t, c in obj_ch.most_common(5)) or "_(none common)_"),
                  ""]

    # --- Section 7: per-rep topic skills (only reps with n>=8) ---
    df["rep"] = df["rep_name_extracted"].fillna("(unknown)")
    rep_topic_skill = []
    for rep, g in df.groupby("rep"):
        if len(g) < 8 or rep == "(unknown)":
            continue
        g_in = g[g["sonnet_outcome"].isin(IN_BAND)].copy()
        g_in["sonnet_score"] = pd.to_numeric(g_in["sonnet_score"], errors="coerce")
        rep_mean = g_in["sonnet_score"].mean() if len(g_in) else None
        # For each topic the rep covers, the mean LOI on calls of theirs
        # that included it
        topic_means = {}
        for tup, sc_ in zip(g_in["topics"], g_in["sonnet_score"]):
            if pd.isna(sc_):
                continue
            for t in tup:
                topic_means.setdefault(t, []).append(sc_)
        skills = sorted(
            [(t, round(sum(vs)/len(vs), 2), len(vs))
             for t, vs in topic_means.items() if len(vs) >= 3],
            key=lambda x: -x[1],
        )
        rep_topic_skill.append({
            "rep": rep,
            "n_calls": len(g),
            "mean_loi": round(rep_mean, 2) if rep_mean is not None else None,
            "top_topics": ", ".join(f"{t}({m})" for t, m, _ in skills[:5]),
            "weakest_topics": ", ".join(f"{t}({m})" for t, m, _ in skills[-3:]),
        })
    rep_skill_df = pd.DataFrame(rep_topic_skill)
    if not rep_skill_df.empty:
        rep_skill_df = rep_skill_df.sort_values(
            "mean_loi", ascending=False, na_position="last")
    L += ["## 7. Per-rep topic skill (mean LOI on their calls when topic was discussed)",
          "",
          "_Reps with >= 8 calls only. The numbers in parentheses are the mean "
          "LOI score on that rep's calls when the topic appeared. Caveat: "
          "topics co-occur, so this is not pure 'topic skill' — it's "
          "'mean LOI on calls of this rep where this topic came up'._",
          ""]
    if not rep_skill_df.empty:
        L.append(_md_table(rep_skill_df, "Per-rep topic skill"))
        L.append("")

    # --- Section 8: aggregate rep strengths / weaknesses (free text) ---
    # Cluster the per-call observations into rep-level summaries.
    # The 8B PARROTS the prompt's example phrase here ("buona gestione delle
    # obiezioni sul prezzo" appears verbatim on call after call) instead of
    # observing the call — open-ended coaching prose doesn't distill at 8B
    # scale (anticipated in the v3 program: narrative stays Sonnet). The Qwen
    # edition therefore sources THIS SECTION ONLY from the audited Claude
    # extractions (282 calls); all structured tables stay Qwen-892.
    if extractor == "qwen":
        prose_df = _load_topics(TOPICS_DIR)
        if not prose_df.empty:
            prose_df = prose_df[prose_df["call_id"].map(ctype) == "sales"].copy()
        prose_df["rep"] = prose_df["rep_name_extracted"].fillna("(unknown)")
        prose_note = ("_Source for this section: **Claude Sonnet's 282 audited "
                      "extractions** (sales-only), NOT the Qwen sweep — the 8B's "
                      "free-text observations are generic/parroted and not "
                      "coaching-grade. New calls after 2026-06-03 have no "
                      "coaching prose until a stronger open model runs on the "
                      "server._")
    else:
        prose_df = df
        prose_note = None
    rep_strengths_agg = defaultdict(list)
    rep_weak_agg = defaultdict(list)
    for _, r in prose_df.iterrows():
        if r["rep"] == "(unknown)":
            continue
        for s in r["rep_strengths"] or []:
            rep_strengths_agg[r["rep"]].append(s)
        for s in r["rep_weaknesses"] or []:
            rep_weak_agg[r["rep"]].append(s)

    L += ["## 8. Aggregated rep observations from Sonnet",
          "",
          "_Sonnet's per-call qualitative notes, aggregated. First 5 strengths "
          "and 5 weaknesses per rep with >= 8 calls. These are raw observations "
          "across many calls — read for patterns rather than one-shot judgments._",
          ""]
    if prose_note:
        L += [prose_note, ""]
    eligible_reps = sorted(rep_skill_df["rep"].tolist())
    for rep in eligible_reps:
        L += [f"### {rep}  (n={int(rep_skill_df[rep_skill_df['rep']==rep]['n_calls'].iloc[0])})", ""]
        s_top = rep_strengths_agg.get(rep, [])[:5]
        w_top = rep_weak_agg.get(rep, [])[:5]
        L.append("**Strengths (sample):**")
        for s in s_top:
            L.append(f"- {s}")
        if not s_top:
            L.append("- _(none recorded)_")
        L.append("")
        L.append("**Weaknesses (sample):**")
        for s in w_top:
            L.append(f"- {s}")
        if not w_top:
            L.append("- _(none recorded)_")
        L.append("")

    # --- Section 9: what customers ask (free-text question buckets) ---
    from scipy.stats import ttest_ind

    # Same parroting problem as section 8: the 8B copies the calibration's
    # example questions ("Quanti pannelli servono?", "La batteria quanto
    # dura?") verbatim into customer_questions on call after call. The Qwen
    # edition sources this section from the Claude extractions too.
    if extractor == "qwen":
        _cq = dict(zip(prose_df["call_id"], prose_df["customer_questions"]))
        dfq = df[df["call_id"].isin(_cq)].copy()
        dfq["customer_questions"] = dfq["call_id"].map(_cq)
    else:
        dfq = df

    dfq["_q_blob"] = dfq["customer_questions"].apply(
        lambda qs: " ".join(qs).lower() if isinstance(qs, list) else "")
    dfq["_n_q"] = dfq["customer_questions"].apply(
        lambda qs: len(qs) if isinstance(qs, list) else 0)
    n_with_q = int((dfq["_n_q"] > 0).sum())
    total_q = int(dfq["_n_q"].sum())

    di = dfq[dfq["sonnet_outcome"].isin(IN_BAND)].copy()
    di["sonnet_score"] = pd.to_numeric(di["sonnet_score"], errors="coerce")
    overall_in_band_mean = di["sonnet_score"].mean()
    q_rows = []
    for bucket, pat in QUESTION_BUCKETS.items():
        rx = re.compile(pat, re.IGNORECASE)
        mask_all = dfq["_q_blob"].apply(lambda b: bool(rx.search(b)))
        n_calls = int(mask_all.sum())
        mask_in = di["_q_blob"].apply(lambda b: bool(rx.search(b)))
        loi_w = di.loc[mask_in, "sonnet_score"].dropna()
        loi_wo = di.loc[~mask_in, "sonnet_score"].dropna()
        if len(loi_w) < 5:
            continue
        try:
            _, p_val = ttest_ind(loi_w, loi_wo, equal_var=False)
        except Exception:
            p_val = float("nan")
        q_rows.append({
            "question_theme": bucket,
            "n_calls": n_calls,
            "% of calls w/ Qs": f"{100*n_calls/max(n_with_q,1):.0f}%",
            "mean_loi_with": round(loi_w.mean(), 2),
            "delta_vs_rest": round(loi_w.mean() - loi_wo.mean(), 2),
            "p_value": round(p_val, 4),
        })
    q_df = pd.DataFrame(q_rows)
    if not q_df.empty:
        q_df = q_df.sort_values("n_calls", ascending=False)

    L += ["## 9. What customers actually ask",
          "",
          f"_Free-text `customer_questions` from {n_with_q} calls "
          f"({total_q} questions total), bucketed by Italian keyword matching "
          "(no extra model cost). A call can fall in several buckets. "
          "`delta_vs_rest` = mean in-band LOI on calls that raised this theme "
          "minus calls that didn't._"
          + (" _Source: **Claude's 282 audited extractions** — the 8B parrots "
             "example questions instead of quoting the call (same issue as "
             "section 8)._" if extractor == "qwen" else ""),
          "",
          _md_table(q_df, "Customer-question themes"),
          "",
          "**Read:** every theme has a *positive* LOI delta — customers who ask "
          "questions at all are more engaged, so question-asking is itself a "
          "warmth signal rather than a worry. The strongest are `warranty_dur`, "
          "`battery`, `contract_proc` and `price_cost` — concrete buying-stage "
          "questions, not vague exploration.",
          ""]

    # --- Section 10: channel + sub-category LOI drill-down ---
    ch_rows = []
    for ch, g in df.dropna(subset=["lead_channel"]).groupby("lead_channel"):
        s = _band_stats(g)
        if s["n"] < 3:
            continue
        ch_rows.append({"lead_channel": ch, **s})
    ch_df = pd.DataFrame(ch_rows)
    if not ch_df.empty:
        ch_df = ch_df.sort_values("mean_loi", ascending=False,
                                  na_position="last")

    L += ["## 10. Where the leads come from — channel & sub-category LOI",
          "",
          "_Mean in-band LOI and dead-rate (% scored Non interessato / Non idoneo) "
          "per acquisition source. `pct_dead` is the share of calls that never "
          "even got a numeric LOI — a blunt 'wasted appointment' proxy._",
          "",
          _md_table(ch_df, "LOI by channel"),
          "",
          "**What happens with Organic:** it is the strongest large channel "
          "(top mean LOI, lowest dead-rate among channels with real volume), but "
          "by definition organic / direct traffic carries **no UTM tags**, so it "
          "can't be split into sub-campaigns — there is nothing to optimise inside "
          "it except to get more of it. The tunable money is in the paid channels "
          "below.",
          ""]

    # Sub-category drill-down on the paid channels (organic has no UTM tags).
    for col, label in (("lead_utmcampaign", "campaign"),
                       ("lead_utmcontent", "ad creative / content"),
                       ("lead_utmterm", "search term / keyword")):
        rows = []
        for val, g in df.dropna(subset=[col]).groupby(col):
            s = _band_stats(g)
            if s["n"] < 5:
                continue
            rows.append({col.replace("lead_utm", "utm_"): val, **s})
        sub = pd.DataFrame(rows)
        if sub.empty:
            continue
        sub = sub.sort_values("mean_loi", na_position="last")
        L += [_md_table(
                  sub.head(8),
                  f"Weakest paid sub-segments by {label} (n≥5, coldest first)"),
              ""]

    # --- Section 11: consolidated "what drags LOI down" ---
    L += ["## 11. What drags LOI down — consolidated",
          "",
          "_Pulling together every genuinely cooling factor across the report. "
          "Everything here has a negative or near-zero association with warmth — "
          "unlike most 'objections', which are actually engagement signals._",
          ""]

    hurt_topics = topics_lift[topics_lift["delta"] < 0].sort_values("delta")
    kill_obj = obj_lift[obj_lift["delta"] < 0].sort_values("delta")
    cold_ch = ch_df[(ch_df["n"] >= 5) & (ch_df["mean_loi"] < overall_in_band_mean)]

    L += ["**Topics that cool the call:**", ""]
    if hurt_topics.empty:
        L.append("- _(none with negative delta)_")
    for _, r in hurt_topics.iterrows():
        L.append(f"- `{r['topic']}` — LOI {r['mean_loi_with']} vs "
                 f"{r['mean_loi_without']} (Δ {r['delta']}, p={r['p_value']})")
    L += ["",
          "**Objections that are real LOI-killers (not engagement):**", ""]
    if kill_obj.empty:
        L.append("- _(none with negative delta)_")
    for _, r in kill_obj.iterrows():
        L.append(f"- `{r['objection']}` — LOI {r['mean_loi_with']} vs "
                 f"{r['mean_loi_without']} (Δ {r['delta']}, p={r['p_value']})")
    L += ["",
          f"**Channels below the corpus mean ({overall_in_band_mean:.2f}):**", ""]
    if cold_ch.empty:
        L.append("- _(none)_")
    for _, r in cold_ch.iterrows():
        L.append(f"- {r['lead_channel']} — LOI {r['mean_loi']}, "
                 f"{r['pct_dead']:.0f}% dead (n={r['n']})")
    L += ["",
          "**Takeaway:** the only on-call levers that actually cool a prospect are "
          "`timing_not_right` and an in-progress `renovation`. Far more LOI is "
          "lost *before* the call — in weak paid sub-segments (see §10) — than "
          "during it.",
          ""]

    # --- Section 12: glossary of the controlled vocabulary ---
    L += [GLOSSARY_MD, ""]

    # --- Save ---
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(L), encoding="utf-8")

    with pd.ExcelWriter(report_xlsx, engine="openpyxl") as w:
        topics_lift.to_excel(w, sheet_name="topics_x_loi", index=False)
        obj_lift.to_excel(w, sheet_name="objections_x_loi", index=False)
        sig_lift.to_excel(w, sheet_name="signals_x_loi", index=False)
        rep_skill_df.to_excel(w, sheet_name="per_rep_topic_skill", index=False)
        pd.DataFrame({
            "topic": [t for t, _ in tc.most_common()],
            "n_calls": [c for _, c in tc.most_common()],
        }).to_excel(w, sheet_name="topic_prevalence", index=False)

    print(f"[saved] {report_md}")
    print(f"[saved] {report_xlsx}")
    print()
    print(f"Coverage: {n} calls with topics + LOI; "
          f"{in_band_n} in-band for effect-size analysis.")
    if not topics_lift.empty:
        print()
        print("=== Topics that HELP most (top 5 by delta) ===")
        print(topics_lift.head(5).to_string(index=False))
        print()
        print("=== Topics that HURT most (bottom 5 by delta) ===")
        print(topics_lift.tail(5).sort_values("delta").to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["claude", "qwen"], default="claude",
                    help="which topic-extraction set to analyze "
                         "(claude = outputs/topics, qwen = outputs/topics_qwen)")
    a = ap.parse_args()
    run(extractor=a.extractor)
