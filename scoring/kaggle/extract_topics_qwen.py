# %% Qwen3-8B topic extraction — the $0 replacement for Claude topic extraction.
#
# Runs the SAME Italian TOPIC_PROMPT Claude used (controlled English vocab) over
# the Sonnet-scored sales calls, on a free T4. Two-phase protocol:
#
#   PHASE 1 (VALIDATE_ONLY=True, ~110 calls, ~1h): score only calls that already
#     have Claude topic labels ('gold' in the input rows) and print per-label
#     precision/recall/F1 + prevalence ratios. GATE before burning 10h:
#     micro-F1 >= ~0.75 on topics/objections/signals AND no high-volume label
#     with prevalence ratio outside ~[0.7, 1.4].
#   PHASE 2 (VALIDATE_ONLY=False): extract everything (gold rows come first in
#     the file, so the audit overlap is always covered). Use START/END to split
#     across sessions if the run would exceed the 12h cap (~35-50s/call ->
#     ~900 calls ~ 10-12h; START=0,END=500 then START=500,END=None is safe).
#
# INPUT  (Kaggle dataset): topics_prompting_sales.jsonl
#        (built locally by `python -m src.emit_topics_prompting`)
# OUTPUT: /kaggle/working/topics_qwen.jsonl — download, then locally run
#        python -m src.ingest_qwen_topics <downloaded file>
#        python -m src.analyze_topics --extractor qwen

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "unsloth", "unsloth_zoo"], check=True)

import os, json, time
from pathlib import Path
from collections import Counter, defaultdict
from unsloth import FastLanguageModel

MODEL = "unsloth/Qwen3-8B"
THINKING = False
# v2 (after round-1 CHECK): 12288 window for the 26k-char transcripts
# (~10.3k tok input worst case + 1024 gen). KV cache at 12k on a 4-bit 8B
# ~= 1.8 GB — fits the T4. Expect ~50-60s/call (was 41s at 8k): validation
# ~1.7-2h; full 892 ~13-15h -> split in 2-3 sessions via START/END.
MAX_SEQ = 12288
# Claude's outputs for this SAME schema ran median 714 / p90 809 / max 930
# tokens — 768 would truncate ~20% of calls into spurious parse-fails.
MAX_NEW = 1024

VALIDATE_ONLY = True   # PHASE 1. Flip to False for the full run (PHASE 2).
VALIDATE_N = 110       # how many gold rows the validation phase scores
# Round 2 validates on FRESH gold rows (110..220 of the 282): the calibration
# below was tuned on rows 0..110, so re-grading on them would flatter it.
# Set to 0 for a direct same-slice A/B against round 1 instead.
VALIDATE_SKIP = 110
START, END = 0, None   # full-run slicing across sessions (row indices)

# Round-1 validation (110 gold calls) showed the gap to Claude is mostly
# TAGGING-STANDARD drift, not capability: the 8B under-tags factual labels
# Claude tags inclusively (technical_questions gold-prevalence 85%, Qwen
# recall 0.34 at perfect precision) and over-tags interpretive ones
# (ready_to_decide: Claude 2/110, Qwen 43/110). This block encodes Claude's
# standards explicitly — same fix pattern as the LOI CALIBRATION. Italian to
# match TOPIC_PROMPT; ~0.6k tok.
CALIBRATION = (
    "CALIBRAZIONE DELLE SOGLIE (integra il vocabolario; non lo sostituisce).\n"
    "\nA) ETICHETTA IN MODO INCLUSIVO — basta UNA SOLA menzione, anche breve "
    "o di passaggio. NON serve che sia un tema centrale della chiamata:\n"
    "- topics_discussed: un argomento toccato anche solo per qualche battuta "
    "va etichettato (vale in particolare per financing, grid_export, "
    "after_sales, warranty, other_quotes: se nella chiamata compaiono altri "
    "preventivi o concorrenti, etichetta SEMPRE other_quotes).\n"
    "- technical_questions: il cliente fa ANCHE UNA SOLA domanda sul prodotto "
    "o sulla tecnologia, ANCHE SEMPLICE O BANALE (\"quanti pannelli "
    "servono?\", \"la batteria quanto dura?\", \"funziona quando piove?\", "
    "\"che potenza ha?\"). IGNORA il requisito \"dettagliate\" del "
    "vocabolario: NON serve che sia esperto né che le domande siano "
    "approfondite. Nella grande maggioranza delle chiamate il cliente fa "
    "almeno una domanda del genere.\n"
    "- references_research: QUALSIASI segno che il cliente si sia informato "
    "(ha letto online, conosce prezzi o concorrenti, cita incentivi o "
    "normative, ha già visto altre offerte).\n"
    "- mentions_financing: il cliente nomina o chiede rate/finanziamento/"
    "anticipo anche di sfuggita.\n"
    "- requests_quote: chiede il preventivo, la proposta scritta o l'invio "
    "di documenti.\n"
    "- decision_maker_absent: accenna che deve parlarne con moglie/marito/"
    "famiglia, anche senza presentarlo come blocco.\n"
    "- renovation_in_progress: QUALSIASI menzione di lavori in corso o "
    "previsti sulla casa (ristrutturazione, rifacimento tetto, infissi, "
    "isolamento), anche se il cliente non la presenta come un blocco; in "
    "quel caso etichetta anche il topic renovation.\n"
    "- timing_not_right: il cliente RIMANDA a un momento successivo "
    "(\"ne riparliamo a settembre\", \"dopo le vacanze\", \"l'anno "
    "prossimo\", \"adesso ho altre priorità\"). ATTENZIONE: il semplice "
    "\"devo pensarci\" / \"devo valutare\" / \"ne parlo con la famiglia\" "
    "NON è timing_not_right — quella è normale esitazione (eventualmente "
    "decision_maker_absent se cita la famiglia).\n"
    "\nB) ETICHETTA CON RIGORE — SOLO se esplicito nel testo:\n"
    "- ready_to_decide: SOLO se il cliente dice chiaramente di essere pronto "
    "a decidere/firmare ORA. È RARISSIMO. Accettare un sopralluogo, un "
    "follow-up o dirsi interessato NON è ready_to_decide.\n"
    "- comparing_actively: SOLO se il cliente chiede un confronto specifico "
    "con i concorrenti durante la chiamata. Avere semplicemente altri "
    "preventivi NON basta (quello è other_quotes ed eventualmente "
    "existing_quote).\n"
    "- proactive_interest: SOLO se il cliente guida la conversazione "
    "(domande spontanee, propone lui i prossimi passi).\n"
    "- house_situation: SOLO se il cliente la solleva come OSTACOLO "
    "(affitto, vendita imminente, condominio): descrivere la propria casa "
    "non è un'obiezione.\n"
    "- skeptical_savings: SOLO se esprime dubbi sulle stime di risparmio; "
    "le semplici domande di chiarimento non contano.\n"
    "- existing_quote (obiezione): SOLO se i preventivi esistenti sono usati "
    "come freno (\"ho già altri preventivi, devo valutare\")."
)

# Controlled vocabulary (MUST match TOPIC_PROMPT in src/extract_topics.py) —
# used to drop invented tokens and to print the per-label agreement table.
VOCAB = {
    "topics_discussed": {
        "financing", "system_size", "savings", "battery", "heat_pump",
        "incentives", "installation", "warranty", "after_sales",
        "roof_assessment", "grid_export", "renovation", "other_quotes",
        "decision_makers", "contract_terms"},
    "objections_raised": {
        "price_too_high", "timing_not_right", "decision_maker_absent",
        "existing_quote", "house_situation", "skeptical_savings",
        "system_complexity", "trust_concerns", "waiting_better_offer",
        "renovation_in_progress"},
    "buyer_signals": {
        "ready_to_decide", "urgency", "comparing_actively",
        "technical_questions", "mentions_timeline", "requests_quote",
        "requests_visit", "mentions_financing", "references_research",
        "proactive_interest"},
}
LIST_FIELDS = list(VOCAB.keys())

# --- locate the input ---------------------------------------------------------
TEST = None
for root, _d, files in os.walk("/kaggle/input"):
    if "topics_prompting_sales.jsonl" in files:
        TEST = str(Path(root, "topics_prompting_sales.jsonl")); break
assert TEST, "topics_prompting_sales.jsonl not found under /kaggle/input"
print(f"Model: {MODEL}\nInput: {TEST}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
)
FastLanguageModel.for_inference(model)


def _as_parts(msgs):
    return [{"role": m["role"],
             "content": ([{"type": "text", "text": m["content"]}]
                         if isinstance(m["content"], str) else m["content"])}
            for m in msgs]


def _norm(enc):
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    elif isinstance(enc, dict):
        enc = enc["input_ids"]
    return enc.to("cuda")


def chat_ids(msgs):
    def apply(m, **kw):
        return tokenizer.apply_chat_template(
            m, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", **kw)
    try:
        return _norm(apply(msgs, enable_thinking=THINKING))
    except TypeError:
        pass
    try:
        return _norm(apply(_as_parts(msgs), enable_thinking=THINKING))
    except TypeError:
        pass
    try:
        return _norm(apply(msgs))
    except TypeError:
        return _norm(apply(_as_parts(msgs)))


def parse_extraction(text):
    """JSON object out of the model output (after any </think>). Returns dict
    or None."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        d = json.loads(text[s:e + 1])
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def enforce_vocab(ex):
    """Keep only controlled-vocab tokens in the three list fields; return the
    cleaned extraction + count of dropped (invented) tokens."""
    dropped = 0
    for f in LIST_FIELDS:
        vals = ex.get(f)
        if not isinstance(vals, list):
            ex[f] = []
            continue
        keep = [v for v in vals if isinstance(v, str) and v in VOCAB[f]]
        dropped += len(vals) - len(keep)
        ex[f] = sorted(set(keep))
    return ex, dropped


# --- work list ----------------------------------------------------------------
with open(TEST, encoding="utf-8") as f:
    rows_in = [json.loads(l) for l in f]
n_gold_total = sum(1 for r in rows_in if r.get("gold"))
if VALIDATE_ONLY:
    work = [r for r in rows_in
            if r.get("gold")][VALIDATE_SKIP:VALIDATE_SKIP + VALIDATE_N]
    print(f"VALIDATION phase: {len(work)} of {n_gold_total} gold calls "
          f"(rows {VALIDATE_SKIP}..{VALIDATE_SKIP + VALIDATE_N})")
else:
    work = rows_in[START:END]
    print(f"FULL phase: rows {START}..{END if END is not None else len(rows_in)}"
          f" -> {len(work)} calls ({n_gold_total} gold in file)")

# --- run ----------------------------------------------------------------------
OUT = "/kaggle/working/topics_qwen.jsonl"
tp = defaultdict(Counter); fp = defaultdict(Counter); fn = defaultdict(Counter)
gold_prev = defaultdict(Counter); pred_prev = defaultdict(Counter)
jac = defaultdict(list)
fails, dropped_total, n_gold_seen = 0, 0, 0
t0 = time.time()

thought = 0
overflows = 0
with open(OUT, "w", encoding="utf-8") as out_f:
    for i, ex_row in enumerate(work):
        # Append the calibration to the system prompt without mutating the
        # source rows (the emitted file stays Claude's prompt verbatim).
        msgs = [dict(m) for m in ex_row["messages"]]
        msgs[0] = {**msgs[0],
                   "content": msgs[0]["content"] + "\n\n" + CALIBRATION}
        ids = chat_ids(msgs)
        if ids.shape[1] + MAX_NEW > MAX_SEQ:
            # Skip-and-record instead of assert: one freak tokenization must
            # not kill a multi-hour run. Headroom is ~1k tok so this should
            # stay at 0.
            overflows += 1
            print(f"  WARNING row {i} ({ex_row.get('call_id','')}): input "
                  f"{ids.shape[1]} tok overflows MAX_SEQ — skipped")
            text = ""
        else:
            out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW,
                                 use_cache=True, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.batch_decode(out[:, ids.shape[1]:],
                                          skip_special_tokens=True)[0]
        if "<think>" in text or "</think>" in text:
            thought += 1
        ex = parse_extraction(text)
        parse_fail = ex is None
        if parse_fail:
            fails += 1
            ex = {}
        ex, dropped = enforce_vocab(ex)
        dropped_total += dropped

        gold = ex_row.get("gold")
        if gold:
            n_gold_seen += 1
            for f in LIST_FIELDS:
                g, p = set(gold.get(f) or []), set(ex.get(f) or [])
                for lab in p & g:
                    tp[f][lab] += 1
                for lab in p - g:
                    fp[f][lab] += 1
                for lab in g - p:
                    fn[f][lab] += 1
                gold_prev[f].update(g); pred_prev[f].update(p)
                jac[f].append(len(p & g) / len(p | g) if (p | g) else 1.0)

        out_f.write(json.dumps({
            "call_id": ex_row.get("call_id", ""),
            "source": ex_row.get("source", "zoom"),
            "model": "qwen3-8b-topics",
            "extraction": ex,
            "vocab_dropped": dropped,
            "parse_fail": parse_fail,
            "raw": text,
        }, ensure_ascii=False) + "\n")

        if (i + 1) % 25 == 0 or (i + 1) == len(work):
            out_f.flush()
            pace = (time.time() - t0) / (i + 1)
            eta = pace * (len(work) - i - 1) / 3600
            print(f"  {i+1}/{len(work)}  {pace:.1f}s/call  ETA {eta:.1f}h  "
                  f"(parse-fails {fails}, vocab-dropped {dropped_total})")

print(f"\n[saved] {OUT}  ({len(work)} rows, {fails} parse-fails, "
      f"{dropped_total} invented tokens dropped, {overflows} overflow-skipped)")
print(f"THINKING flag = {THINKING}; <think> seen on {thought}/{len(work)} calls")

# --- agreement vs Claude on the gold overlap -----------------------------------
if n_gold_seen:
    print(f"\n=== Agreement vs Claude topics (n={n_gold_seen} overlap calls) ===")
    gate_ok = True
    for f in LIST_FIELDS:
        TP = sum(tp[f].values()); FP = sum(fp[f].values()); FN = sum(fn[f].values())
        P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        mj = sum(jac[f]) / len(jac[f])
        print(f"\n[{f}]  micro P={P:.2f} R={R:.2f} F1={F1:.2f}  "
              f"mean-Jaccard={mj:.2f}")
        if F1 < 0.75 and (TP + FP + FN) > 0:
            gate_ok = False
        print(f"  {'label':24s} {'gold':>4s} {'pred':>4s} {'P':>5s} {'R':>5s}")
        for lab in sorted(VOCAB[f], key=lambda x: -gold_prev[f][x]):
            g_n, p_n = gold_prev[f][lab], pred_prev[f][lab]
            if g_n == 0 and p_n == 0:
                continue
            lp = tp[f][lab] / max(tp[f][lab] + fp[f][lab], 1)
            lr = tp[f][lab] / max(tp[f][lab] + fn[f][lab], 1)
            flag = ""
            if g_n >= 10 and (p_n > 1.4 * g_n or p_n < 0.7 * g_n):
                flag = "  <-- prevalence shift"
                gate_ok = False
            print(f"  {lab:24s} {g_n:4d} {p_n:4d} {lp:5.2f} {lr:5.2f}{flag}")
    verdict = ("PASS — flip VALIDATE_ONLY=False and run the full sweep"
               if gate_ok else
               "CHECK — a field is under F1 0.75 or a common label has a "
               "prevalence shift; inspect before the full run")
    print(f"\nGATE: {verdict}")
else:
    print("\n(no gold rows in this slice — agreement table skipped)")

print("\nDownload topics_qwen.jsonl, then locally:")
print("  python -m src.ingest_qwen_topics <downloaded file>")
print("  python -m src.analyze_topics --extractor qwen")
