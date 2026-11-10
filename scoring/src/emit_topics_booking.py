"""Emit xCally booking-call transcripts for Qwen TOPIC extraction on Kaggle.

Booking-edition counterpart of emit_topics_prompting: same 1,497-call scope as
the LOI scoring run (emit_xcally filter, so topic labels and LOI scores join
1:1), but the system prompt carries a BOOKING-specific controlled vocabulary —
these are short qualification / appointment / reminder phone calls, not sales
consultations.

Output: data/train/topics_booking_b<N>.jsonl  (--batch-size, default 750 —
        output ~200-400 tok/call puts a single 1,497-call session near the cap)

Rows: {call_id, source='xcally', messages=[system(BOOKING_TOPIC_PROMPT),
user(transcript)]} — no gold (no Claude labels exist for xCally; validation is
post-hoc vs agent dispositions in analyze_xcally_booking).

Usage:  python -m src.emit_topics_booking --csv data/xcally/OMNIBUS_2026-07-10-1435.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DATA
from .emit_xcally import load_omnibus, _skip_reason
from .transcript_prep import head_tail_truncate

# Controlled vocabulary MUST match VOCAB in kaggle/extract_topics_booking.py.
BOOKING_TOPIC_PROMPT = """\
Sei un analista di call-center per l'azienda (fotovoltaico e pompe di \
calore). Devi estrarre etichette strutturate da una BREVE telefonata di \
booking/qualificazione (operatore dell'azienda + potenziale cliente): sono chiamate \
per fissare o confermare un primo appuntamento di consulenza, NON consulenze \
di vendita.

Restituisci SOLO un JSON valido con questi campi (no markdown, no preambolo):

{
  "topics_discussed": [...],
  "objections_raised": [...],
  "booking_signals": [...],
  "call_context": [...],
  "customer_questions": [...]
}

REGOLE:
- Usa SOLO i token del vocabolario controllato qui sotto. NIENTE token inventati.
- Etichetta in modo INCLUSIVO i fatti: basta UNA menzione anche breve \
(topics, segnali fattuali come mentions_high_bill o asks_product_questions).
- Etichetta con RIGORE le interpretazioni: not_interested SOLO se il cliente \
lo dice chiaramente (un "ora non posso" è no_time_now, non disinteresse); \
accepts_appointment SOLO se un appuntamento viene effettivamente accettato o \
confermato nella chiamata.
- customer_questions: 0-2 domande significative poste dal CLIENTE, brevi, \
estratte quasi letteralmente. Se il cliente non fa domande, lista vuota.

VOCABOLARIO CONTROLLATO:

topics_discussed (di cosa si è parlato):
- appointment_scheduling  (fissare/spostare un appuntamento: data, ora, disponibilità)
- appointment_reminder    (promemoria/conferma di un appuntamento GIÀ fissato)
- zoom_logistics          (link Zoom, videochiamata, WhatsApp/email col link, come collegarsi)
- eligibility_screening   (domande di qualificazione: proprietario o affitto, tipo di casa, tetto, riscaldamento, reddito, familiari)
- bill_energy_costs       (bolletta, consumi, costi dell'energia)
- product_overview        (pannelli, batteria, pompa di calore, come funziona l'impianto o l'offerta dell'azienda)
- savings                 (risparmio in bolletta, convenienza)
- incentives              (bonus, detrazioni, PNRR, conto termico, CER, incentivi statali)
- financing               (finanziamento, rate, anticipo, prestito)
- price                   (prezzo, costo dell'impianto)
- existing_pv_or_quotes   (impianto fotovoltaico già presente, o preventivi/contatti con altre aziende)
- callback_scheduling     (accordo per richiamare in un altro momento)
- data_confirmation       (conferma o raccolta di dati: indirizzo, email, telefono)

objections_raised (obiezioni/resistenze del cliente):
- not_interested          (non mi interessa, non voglio niente — detto esplicitamente)
- no_time_now             (non ho tempo adesso, sono al lavoro, sto uscendo)
- already_has_system      (ha già un impianto fotovoltaico)
- already_evaluating_others (sta già valutando o ha preventivi di altre aziende)
- renter_or_moving        (in affitto, non proprietario, casa in vendita, trasloco imminente)
- too_expensive           (troppo caro, non può permetterselo, paura del costo)
- distrust_phone_sales    (diffidenza: "come avete il mio numero", paura di truffe, non compra al telefono)
- decision_maker_absent   (deve parlarne con coniuge/famiglia prima)
- age_or_health           (età avanzata o problemi di salute propri o di famiglia)
- never_requested         (dice di non aver mai fatto richiesta / non ricorda di essersi registrato)
- home_unsuitable         (dice che la sua casa o il tetto non sono adatti)

booking_signals (segnali positivi verso l'appuntamento):
- accepts_appointment     (accetta di fissare l'appuntamento nella chiamata)
- proposes_own_time       (propone lui stesso una data/ora)
- confirms_attendance     (conferma che parteciperà a un appuntamento già fissato)
- asks_product_questions  (fa domande su prodotto, tecnologia, installazione)
- asks_savings_incentives (chiede di risparmio, bonus, incentivi)
- mentions_high_bill      (dice che la bolletta è alta o che vuole ridurla)
- requests_info_material  (chiede materiale/informazioni via email o WhatsApp)
- mentions_timeline       (cita tempistiche concrete per decidere o installare)

call_context (circostanze della chiamata):
- gatekeeper_or_third_party  (risponde una persona diversa dall'interessato)
- wrong_number               (numero sbagliato, persona estranea)
- language_barrier           (difficoltà di lingua)
- broken_audio               (audio pessimo, non si sentono, linea che cade)
- missed_appointment_followup (chiamata dopo un appuntamento saltato/no-show)
"""


def main(csv_path: Path, batch_size: int) -> int:
    rows = load_omnibus(csv_path)
    print(f"OMNIBUS rows: {len(rows)}")

    emit = []
    for row in sorted(rows, key=lambda r: r.get("CREATEDAT", "")):
        if _skip_reason(row) is not None:
            continue
        transcript = head_tail_truncate(row["FULL_TRANSCRIPT"])
        emit.append({
            "call_id": row["UNIQUEID"],
            "source": "xcally",
            "messages": [
                {"role": "system", "content": BOOKING_TOPIC_PROMPT},
                {"role": "user", "content": (
                    "Trascrizione della telefonata (italiano, già diarizzata):\n\n"
                    f"<transcript>\n{transcript}\n</transcript>\n\n"
                    "Estrai le etichette strutturate secondo il vocabolario."
                )},
            ],
        })
    print(f"in scope: {len(emit)} (same emit_xcally filter)")

    batches = [emit[i:i + batch_size] for i in range(0, len(emit), batch_size)] \
        if batch_size else [emit]
    for i, batch in enumerate(batches, 1):
        out = DATA / "train" / f"topics_booking_b{i}.jsonl"
        out.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in batch),
            encoding="utf-8")
        print(f"[saved] {out}  ({len(batch)} rows)")
    print("Run each batch: python -m src.kaggle_topics_run --jsonl <batch> --go")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="OMNIBUS export csv")
    ap.add_argument("--batch-size", type=int, default=750)
    a = ap.parse_args()
    raise SystemExit(main(Path(a.csv), a.batch_size))
