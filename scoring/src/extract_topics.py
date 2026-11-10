"""Topic / objection / buyer-signal extraction from Italian sales-call
transcripts. Uses Claude Sonnet with a separate prompt from the LOI
scoring rubric — outputs go to a NEW directory (`outputs/topics/`) so
the existing LOI score files are untouched.

Controlled vocabulary in English (for clean downstream aggregation).
Rubric prompt is in Italian (the call language) so the model can match
phrasing precisely.

Usage:
    python -m src.extract_topics --limit 3       # pilot on 3 calls
    python -m src.extract_topics                 # full run on every
                                                 # Sonnet-scored Zoom call
    python -m src.extract_topics --max-cost 20   # cost cap (safety)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import time
from collections import Counter
from pathlib import Path

from .config import (
    CALLS_CSV, DATA, OUT_LOGS, OUTPUTS, SCORING_MODEL, ensure_dirs, env,
)
from .transcribe import CallRow, load_existing_transcript

log = logging.getLogger("extract_topics")
TOPICS_DIR = OUTPUTS / "topics"

_INTERRUPTED = False


def _handle_sigint(_sig, _frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    log.warning("SIGINT received — will exit after current call.")


# ----------------------------------------------------------------- prompt
TOPIC_PROMPT = """\
Sei un analista esperto di vendite per l'azienda (azienda di fotovoltaico e \
pompe di calore). Devi estrarre etichette strutturate da una trascrizione di \
una chiamata di vendita italiana (consulente dell'azienda + cliente).

Restituisci SOLO un JSON valido con questi campi (no markdown, no preambolo):

{
  "topics_discussed": [...],
  "objections_raised": [...],
  "buyer_signals": [...],
  "rep_name": "...",
  "rep_strengths": [...],
  "rep_weaknesses": [...],
  "customer_questions": [...],
  "call_summary_it": "..."
}

REGOLE:
- Usa SOLO i token del vocabolario controllato per topics_discussed, \
objections_raised, buyer_signals. NIENTE token inventati.
- rep_name = nome COGNOME del consulente estratto dalla trascrizione \
(es. "Serena Pisano"). Se non identificabile, usa null.
- rep_strengths e rep_weaknesses: 0-3 osservazioni concise in italiano \
(es. "buona gestione delle obiezioni sul prezzo"). null se non applicabile.
- customer_questions: 1-3 domande significative poste dal cliente, in italiano, \
ESTRATTE letteralmente o parafrasate brevemente. Non inventare.
- call_summary_it: 2-3 frasi in italiano che riassumono la chiamata.

VOCABOLARIO CONTROLLATO:

topics_discussed (cosa si è discusso):
- financing       (finanziamento, rate, prestito, anticipo, dilazione)
- system_size     (kW, dimensione impianto, numero pannelli)
- savings         (risparmio bolletta, autoconsumo, ritorno investimento)
- battery         (batterie, accumulo, capacità di accumulo)
- heat_pump       (pompa di calore, riscaldamento, raffrescamento)
- incentives      (bonus, detrazione, ecobonus, CER, GSE, incentivi statali)
- installation    (installazione, tempistiche di installazione, fasi lavoro, sopralluogo)
- warranty        (garanzia, copertura)
- after_sales     (assistenza, manutenzione, monitoraggio post-installazione)
- roof_assessment (tetto, esposizione, ombre, struttura)
- grid_export     (vendita energia, immissione rete, scambio sul posto)
- renovation      (lavori di ristrutturazione in corso, cambio infissi, isolamento)
- other_quotes    (altri preventivi, confronto con concorrenti)
- decision_makers (moglie, marito, famiglia, condomini da consultare)
- contract_terms  (termini contrattuali, condizioni, durata)

objections_raised (obiezioni sollevate dal cliente):
- price_too_high          (troppo caro, oltre budget, costoso)
- timing_not_right        (non ora, momento sbagliato, aspetto)
- decision_maker_absent   (devo parlare con coniuge/partner/figli prima)
- existing_quote          (ho già altri preventivi)
- house_situation         (affitto, vendita imminente, condominio complicato)
- skeptical_savings       (non credo nei risparmi promessi, dubbi sulle stime)
- system_complexity       (troppo complicato, non capisco la tecnologia)
- trust_concerns          (dubbi sull'azienda, esperienze negative passate)
- waiting_better_offer    (aspetto offerta migliore, sondo il mercato)
- renovation_in_progress  (devo finire i lavori in corso prima)

buyer_signals (segnali positivi dal cliente):
- ready_to_decide          (sta per decidere, è quasi deciso)
- urgency                  (vuole installare presto, ha urgenza)
- comparing_actively       (chiede confronto specifico con concorrenti)
- technical_questions      (fa domande tecniche dettagliate, conosce l'argomento)
- mentions_timeline        (cita date specifiche per decisione/installazione)
- requests_quote           (chiede preventivo formale)
- requests_visit           (chiede sopralluogo)
- mentions_financing       (chiede dettagli su finanziamento)
- references_research      (ha già fatto ricerca, conosce il mercato)
- proactive_interest       (mostra interesse spontaneo, propone soluzioni)
"""


def _build_messages(transcript: str) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            "Trascrizione della call (italiano, già diarizzata):\n\n"
            f"<transcript>\n{transcript}\n</transcript>\n\n"
            "Estrai le etichette strutturate secondo il vocabolario."
        ),
    }]


def _parse_json(raw: str) -> dict | None:
    """Robust JSON parse — tolerates markdown fences."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        s = s.rstrip("`").rstrip()
        if s.endswith("```"):
            s = s[:-3].rstrip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        return None


def _output_path(call_id: str) -> Path:
    stem = call_id.replace("/", "_").replace("+", "-").replace("=", "")
    return TOPICS_DIR / f"zoom__{stem}.json"


def _sonnet_scored_zoom_call_ids() -> set[str]:
    """Return the set of zoom call_ids that already have a Sonnet LOI score —
    these are the calls we want topic-extracted."""
    out: set[str] = set()
    from .config import OUT_SCORES
    for p in OUT_SCORES.glob("zoom__*.json"):
        name = p.name
        if "__ollama_" in name or name.startswith("upload__") or name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (d.get("model") or "").lower().startswith("claude-sonnet"):
            continue
        if d.get("call_id"):
            out.add(d["call_id"])
    return out


def _load_zoom_calls() -> dict[str, dict]:
    out = {}
    with CALLS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source") == "zoom" and r.get("transcript_file"):
                out[r["call_id"]] = r
    return out


def extract_one(client, transcript: str, model: str) -> tuple[dict, float, dict]:
    """Single Anthropic call. Returns (parsed_dict, cost_usd, meta)."""
    messages = _build_messages(transcript)
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=TOPIC_PROMPT,
        messages=messages,
    )
    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text += block.text
    parsed = _parse_json(text) or {}
    # Pricing rough (Sonnet 4.6): $3/MTok input, $15/MTok output
    iu = resp.usage.input_tokens
    ou = resp.usage.output_tokens
    cost = (iu * 3 / 1_000_000) + (ou * 15 / 1_000_000)
    meta = {
        "input_tokens": iu, "output_tokens": ou,
        "cost_usd": round(cost, 6), "raw": text,
    }
    return parsed, cost, meta


def run(limit: int | None, max_cost: float | None, model: str) -> dict:
    ensure_dirs()
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    logfile = OUT_LOGS / f"extract_topics_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(logfile, encoding="utf-8"),
                  logging.StreamHandler()],
        force=True,
    )
    signal.signal(signal.SIGINT, _handle_sigint)

    targets = _sonnet_scored_zoom_call_ids()
    zoom_calls = _load_zoom_calls()
    work = []
    for cid in sorted(targets):
        if cid not in zoom_calls:
            continue
        if _output_path(cid).exists():
            continue  # idempotent skip
        work.append(cid)
    if limit is not None:
        work = work[:limit]
    log.info("worklist: %d calls (already done not in list)", len(work))

    import anthropic  # type: ignore
    client = anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY", required=True))

    scored = 0
    errors = 0
    cost = 0.0
    topic_counter: Counter = Counter()
    obj_counter: Counter = Counter()
    sig_counter: Counter = Counter()
    t0 = time.monotonic()
    for i, cid in enumerate(work, 1):
        if _INTERRUPTED:
            log.info("stopping at user request after %d calls", scored)
            break
        if max_cost is not None and cost >= max_cost:
            log.info("hit max-cost cap ($%.2f) after %d calls", max_cost, scored)
            break
        row = zoom_calls[cid]
        cr = CallRow(source="zoom", call_id=cid, audio_file=None,
                     transcript_file=row.get("transcript_file"),
                     duration_sec=None)
        loaded = load_existing_transcript(cr)
        if loaded is None:
            log.warning("no transcript for %s", cid)
            continue
        transcript, _ = loaded
        try:
            parsed, c, meta = extract_one(client, transcript, model)
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", cid, e)
            errors += 1
            continue
        out_payload = {
            "call_id": cid,
            "source": "zoom",
            "model": model,
            "topic_schema_version": "v1-2026-06-02",
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cost_usd": meta["cost_usd"],
            "input_tokens": meta["input_tokens"],
            "output_tokens": meta["output_tokens"],
            "extraction": parsed,
            "raw": meta["raw"],
        }
        _output_path(cid).write_text(
            json.dumps(out_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        scored += 1
        cost += c
        topic_counter.update(parsed.get("topics_discussed") or [])
        obj_counter.update(parsed.get("objections_raised") or [])
        sig_counter.update(parsed.get("buyer_signals") or [])
        if i % 25 == 0 or i == len(work):
            rate = scored / max(time.monotonic() - t0, 1e-9)
            log.info(
                "progress %d/%d  cost=$%.4f  rate=%.2f calls/s  "
                "top_topics=%s",
                i, len(work), cost, rate,
                dict(topic_counter.most_common(5)),
            )

    wall = time.monotonic() - t0
    log.info(
        "done — scored=%d errors=%d cost=$%.4f wall=%.1fs",
        scored, errors, cost, wall,
    )
    log.info("top topics:     %s", topic_counter.most_common(10))
    log.info("top objections: %s", obj_counter.most_common(10))
    log.info("top signals:    %s", sig_counter.most_common(10))
    return {
        "scored": scored, "errors": errors, "cost_usd": cost,
        "wall_sec": wall,
    }


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-cost", type=float, default=25.0,
                    help="USD safety cap (default 25).")
    ap.add_argument("--model", default=SCORING_MODEL)
    a = ap.parse_args()
    run(limit=a.limit, max_cost=a.max_cost, model=a.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
