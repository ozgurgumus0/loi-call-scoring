"""Offline tests for the weekly reason feed: shares, movers, suppression. No transcripts."""
import json
import unittest

import pandas as pd

from src import export_reason_aggregates as era

TODAY = pd.Timestamp("2026-09-30")          # last complete week = 2026-09-21


def frame(week_counts: dict, hits: dict, column: str = "term:energy_brands:Enel") -> pd.DataFrame:
    """week -> number of consultations; week -> how many of them mention `column`."""
    rows = []
    for week, n in week_counts.items():
        for i in range(n):
            row = {"words": 2000, "band": "Tiepido", "roles_resolved": True, "week": pd.Timestamp(week),
                   "call_date": week}
            for key in list(era.PRIMARY) + era.SECONDARY + era.TOPICS:
                row[key] = False
            for group, name in era.WATCH_RX:
                row[f"term:{group}:{name}"] = False
            row[column] = i < hits.get(week, 0)
            rows.append(row)
    return pd.DataFrame(rows)


PRIOR = {"2026-08-24": 100, "2026-08-31": 100, "2026-09-07": 100, "2026-09-14": 100}


class DigestTests(unittest.TestCase):
    def test_a_real_drop_is_a_mover(self):
        d = era.weekly_digest(frame({**PRIOR, "2026-09-21": 100},
                                    {**{w: 35 for w in PRIOR}, "2026-09-21": 15}), TODAY)
        self.assertEqual(d["week"], "2026-09-21")
        mover = d["movers"][0]
        self.assertEqual((mover["name"], mover["share_pct"], mover["prior_4w_pct"]), ("Enel", 15.0, 35.0))
        self.assertLess(mover["z"], -2)

    def test_noise_is_not_a_mover(self):
        d = era.weekly_digest(frame({**PRIOR, "2026-09-21": 100},
                                    {**{w: 35 for w in PRIOR}, "2026-09-21": 33}), TODAY)
        self.assertEqual(d["movers"], [])

    def test_thin_weeks_are_suppressed(self):
        d = era.weekly_digest(frame({**PRIOR, "2026-09-21": era.MIN_GROUP - 1}, {}), TODAY)
        self.assertEqual(d["items"], [])

    def test_short_or_ineligible_calls_do_not_count(self):
        f = frame({**PRIOR, "2026-09-21": 100}, {"2026-09-21": 100})
        f.loc[f["week"] == pd.Timestamp("2026-09-21"), "band"] = "Non idoneo"
        self.assertEqual(era.weekly_digest(f, TODAY)["consultations"], 0)

    def test_output_holds_no_call_level_fields(self):
        d = era.weekly_digest(frame({**PRIOR, "2026-09-21": 100}, {"2026-09-21": 10}), TODAY)
        self.assertNotIn("call_id", json.dumps(d))

    def test_watchlist_patterns_match_normalised_text(self):
        rx = dict(((g, n), r) for (g, n), r in era.WATCH_RX.items())
        self.assertTrue(rx[("incentives", "Conto Termico")].search("il conto termico 3.0"))
        self.assertTrue(rx[("energy_brands", "Enel")].search("sono con enel da anni"))
        self.assertFalse(rx[("energy_brands", "Enel")].search("pannelli"))


if __name__ == "__main__":
    unittest.main()
