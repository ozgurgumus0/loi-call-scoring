"""Transcript preprocessing for the Qwen LoRA path (v1).

Why this exists
---------------
Qwen-LoRA v0 trained at 4096 ctx underperformed (15% exact overall, but 43% on
the calls short enough to fit). Diagnosis: the rubric system prompt costs
~1100 tokens, leaving only ~2700 tokens for the transcript inside a 4096 window,
yet the *median* Zoom call is ~6100 tokens. 76% of calls overflowed, so during
training the assistant label (at the end of the sequence) was at risk of being
truncated away, and at inference the model only ever saw the *opening* of long
calls — where there is little LOI signal — and defaulted to "Non interessato".

This module fixes that deterministically, at build time, for free:

  1. collapse_turns()      — merge consecutive same-speaker VTT captions into one
                             turn (drops repeated "Name: " prefixes; ~7% smaller).
  2. head_tail_truncate()  — if still over budget, keep the OPENING (Q1 product
                             fit, Q2 blockers) and the CLOSE (Q4 price reaction /
                             next steps), dropping the middle product-pitch. Both
                             ends carry LOI signal; the middle rarely does.

Sonnet (200k ctx) never needed this — it only applies to the Qwen path, so it
lives here rather than in rubric.build_messages().
"""
from __future__ import annotations

# Char budget for the transcript portion. Calibrated with the real Qwen
# tokenizer (Italian ≈ 3.0 chars/token, worst-case ~2.71): with the CONDENSED
# QWEN_RUBRIC_PROMPT (~490 tok) + wrapper (~40) + chat overhead (~14) +
# assistant JSON (~250), a 7500-char transcript (~2770 tok worst-case) keeps the
# FULL training sequence under ~3600 tokens — safely below the 4096 window, so
# nothing (least of all the assistant label) gets truncated. Verified by
# re-encoding the rebuilt JSONL with the tokenizer.
DEFAULT_MAX_CHARS = 7500
HEAD_FRACTION = 0.40  # share of the budget kept from the opening; rest from the close
GAP_MARKER = "\n[... parte centrale della call omessa per lunghezza ...]\n"


def collapse_turns(segments: list[dict]) -> str:
    """Merge consecutive captions from the same speaker into one line.

    `segments` is the list returned by transcribe._parse_vtt (each has
    'speaker' and 'text'). Returns diarized plain text, one line per *turn*.
    """
    turns: list[tuple[str | None, str]] = []
    cur_speaker: str | None = None
    buf: list[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if speaker == cur_speaker:
            buf.append(text)
        else:
            if buf:
                turns.append((cur_speaker, " ".join(buf)))
            cur_speaker = speaker
            buf = [text]
    if buf:
        turns.append((cur_speaker, " ".join(buf)))
    return "\n".join((f"{sp}: " if sp else "") + txt for sp, txt in turns)


def head_tail_truncate(text: str, max_chars: int = DEFAULT_MAX_CHARS,
                       head_fraction: float = HEAD_FRACTION) -> str:
    """Keep the opening and the close of a long transcript, drop the middle.

    Short transcripts pass through untouched. Truncation snaps to line
    boundaries so we never cut a turn mid-sentence.
    """
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(GAP_MARKER)
    head_budget = int(budget * head_fraction)
    tail_budget = budget - head_budget

    lines = text.split("\n")

    head: list[str] = []
    used = 0
    for ln in lines:
        if used + len(ln) + 1 > head_budget:
            break
        head.append(ln)
        used += len(ln) + 1

    tail: list[str] = []
    used = 0
    for ln in reversed(lines):
        if used + len(ln) + 1 > tail_budget:
            break
        tail.append(ln)
        used += len(ln) + 1
    tail.reverse()

    # Guard against head/tail overlapping on very small budgets, and against
    # single-line transcripts (e.g. undiarized Whisper output collapses into
    # one giant turn) where line-granular selection yields nothing at all —
    # fall back to plain character slicing in both cases.
    head_set_len = len(head)
    if (not head and not tail) or head_set_len + len(tail) >= len(lines):
        return text[:head_budget] + GAP_MARKER + text[-tail_budget:]

    return "\n".join(head) + GAP_MARKER + "\n".join(tail)


def prepare_transcript(segments: list[dict],
                       max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Full Qwen-path preprocessing: collapse turns, then head/tail truncate."""
    return head_tail_truncate(collapse_turns(segments), max_chars=max_chars)
