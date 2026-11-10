"""The parser handles valid / code-fenced / malformed JSON per v3 schema."""
import json

import pytest

from src.score_llm import ParseError, parse_score


VALID = {
    "outcome": "Tiepido",
    "score": 4,
    "tree_path": {"q1": "si", "q2": "si", "q3": "raccolta_info", "q4": None},
    "time_signal": "tiepido",
    "behavior_modifier_applied": False,
    "behavior_modifier_reason": None,
    "confidence": "medium",
    "evidence_quotes": ["sa di che parliamo", "cerca la bolletta"],
}


def test_parses_clean_json():
    out = parse_score(json.dumps(VALID))
    assert out["outcome"] == "Tiepido"
    assert out["score"] == 4


def test_parses_code_fenced_json():
    raw = "```json\n" + json.dumps(VALID) + "\n```"
    assert parse_score(raw)["outcome"] == "Tiepido"


def test_rejects_malformed():
    with pytest.raises(ParseError):
        parse_score("not json at all")


def test_rejects_missing_keys():
    bad = {"outcome": "Tiepido", "score": 4}  # missing required keys
    with pytest.raises(ParseError):
        parse_score(json.dumps(bad))


def test_rejects_invalid_outcome():
    bad = dict(VALID, outcome="Lukewarm")
    with pytest.raises(ParseError):
        parse_score(json.dumps(bad))


def test_flags_score_mismatch_without_rejecting():
    # Declared Caldo (canonical 7) but score 6 — keep the parse, flag it.
    bad = dict(VALID, outcome="Caldo", score=6)
    out = parse_score(json.dumps(bad))
    assert out["outcome"] == "Caldo"
    assert out["_score_mismatch"] == {"declared": 6, "canonical": 7}
