"""Unit tests of the weight analysis."""

import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

import common as c
import config
import weight_analysis as wa

NORMAL = ["alice", "bob smith", "red shoes", "42", "hello world", "blue sky", "invoice 2021",
          "green tea", "new york", "john", "summer sale", "contact us"]
ATTACK = ["' OR 1=1 --", "1 UNION SELECT a,b FROM users", "admin' --", "1; DROP TABLE users",
          "' OR 'a'='a", "1 AND SLEEP(5)", "x' UNION SELECT NULL --", "1 OR 1=1",
          "'; EXEC xp_cmdshell --", "1' AND '1'='1", "0 UNION SELECT 1,2", "a' OR 1=1 #"]


def toy_detector(model: str, extra: list[str] | None = None, normalised: bool = False):
    texts = NORMAL + ATTACK + (extra or [])
    labels = [0] * len(NORMAL) + [1] * (len(ATTACK) + len(extra or []))
    val = pd.DataFrame({"payload": NORMAL + ATTACK, "y": [0] * len(NORMAL) + [1] * len(ATTACK)})
    return wa.Detector(texts, labels, val, normalised=normalised, model=model, seed=0,
                       params={"C": 1.0})


class GroupTests(unittest.TestCase):
    def test_each_rule_of_the_grouping(self):
        cases = {" ": "word boundary", "%2": "percent", " %": "percent", " or": "lower-case letters",
                 "OR ": "upper-case letters", "oR": "mixed-case letters", "12": "digits",
                 "a1": "letters and digits", "1=": "other symbols", "'": "other symbols",
                 " -": "other symbols", "a%1": "percent"}
        for ngram, group in cases.items():
            self.assertEqual(wa.group_of(ngram), group, ngram)

    def test_every_ngram_of_a_detector_has_exactly_one_group(self):
        det = toy_detector("logreg")
        groups = [wa.group_of(n) for n in det.names]
        self.assertTrue(all(g in wa.GROUPS for g in groups))
        self.assertEqual(det.parts.getnnz(), np.count_nonzero(det.weights))
        self.assertTrue((det.parts.getnnz(axis=1) <= 1).all())


class DecompositionTests(unittest.TestCase):
    def test_threshold_on_the_scale_of_the_decision(self):
        self.assertAlmostEqual(wa.to_logit(0.5), 0.0)
        self.assertAlmostEqual(float(expit(wa.to_logit(0.1564))), 0.1564)

    def test_three_parts_add_up_to_the_models_own_decision(self):
        for model in wa.MODELS:
            det = toy_detector(model)
            reading = det.read(NORMAL + ATTACK + ["JyBPUiAxPTEgLS0=", "%27%20OR%201%3D1"])
            own = det.estimator.decision_function(
                det.vectoriser.transform(NORMAL + ATTACK + ["JyBPUiAxPTEgLS0=", "%27%20OR%201%3D1"]))
            np.testing.assert_allclose(reading.margin, own - det.tau, atol=1e-10)
            self.assertLess(reading.residual, 1e-10)
            self.assertTrue((reading.attack_evidence >= 0).all())
            self.assertTrue((reading.normal_evidence <= 0).all())
            row = wa.run_row(det, reading)
            for sign in ("attack", "normal"):
                by_group = sum(row[wa.group_column(sign, g)] for g in wa.GROUPS)
                self.assertAlmostEqual(by_group, row[f"{sign}_evidence"], places=5)

    def test_flagged_means_margin_at_or_above_zero(self):
        det = toy_detector("logreg")
        reading = det.read(NORMAL + ATTACK)
        away = np.abs(reading.margin) > 1e-9
        np.testing.assert_array_equal(reading.flagged[away], reading.margin[away] >= 0)

    def test_an_empty_input_sits_at_the_baseline(self):
        det = toy_detector("logreg")
        self.assertAlmostEqual(float(det.read([""]).margin[0]), det.baseline)

    def test_an_unknown_word_keeps_only_the_word_boundary(self):
        det = toy_detector("logreg")
        reading = det.read(["\u4e2d\u6587"], coverage=True)
        boundary = float(det.weights[list(det.names).index(" ")])
        self.assertAlmostEqual(float(reading.margin[0]), det.baseline + boundary)
        self.assertGreater(reading.coverage, 0.0)
        self.assertLess(reading.coverage, 1.0)

    def test_gap_between_two_models_splits_into_groups_intercept_and_threshold(self):
        x, y = toy_detector("logreg", extra=["' oR 1=1 --", "1 uNiOn SeLeCt a"]), toy_detector("logreg")
        inputs = ["JyBPUiAxPTEgLS0=", "aGVsbG8gd29ybGQ=", "' OR 1=1 --"]
        rx, ry = x.read(inputs), y.read(inputs)
        by_group = (rx.group_means() - ry.group_means()).sum()
        gap = by_group + (x.intercept - y.intercept) - (x.tau - y.tau)
        self.assertAlmostEqual(float(rx.margin.mean() - ry.margin.mean()), float(gap), places=10)

    def test_normalised_detector_scores_the_normalised_text(self):
        det = toy_detector("logreg", normalised=True)
        a, b = det.read(["%27%20OR%201%3D1%20--"]), det.read(["' OR 1=1 --"])
        self.assertAlmostEqual(float(a.margin[0]), float(b.margin[0]))


class ReproductionTests(unittest.TestCase):
    def test_random_stream_matches_the_position_in_experiment_py(self):
        wa.experiment_stream(3, "A")
        got = np.random.randint(10**6)
        np.random.seed(3)
        for _ in range(4 * c.FAMILIES.index(wa.HELD_OUT)):
            np.random.randint(np.iinfo("i").max)
        self.assertEqual(got, np.random.randint(10**6))

    def test_normaliser_reverts_three_families_and_leaves_base64_unchanged(self):
        tampers = c.load_tampers()
        self.assertEqual(sorted(wa.REVERTED), sorted(c.NORMALISER_COVERED))
        for payload in ATTACK:
            for family in wa.REVERTED:
                self.assertEqual(c.normalise(c.apply_tamper(tampers, family, payload)),
                                 c.normalise(payload), (family, payload))
            encoded = c.apply_tamper(tampers, wa.HELD_OUT, payload)
            self.assertEqual(c.normalise(encoded), encoded)


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.grid = config.PARAM_GRID["logreg"]["C"]
        rows, tests = [], []

        def run(seed, part, condition, flagged, C=None, input_set=f"{wa.HELD_OUT} attacks", **more):
            rows.append({"model": "logreg", "seed": seed, "part": part, "condition": condition,
                         "C": C, "input_set": input_set, "flagged_pct": flagged,
                         "baseline": more.get("baseline", -1.0), "intercept": 0.0, "tau": 0.0})

        for seed in (0, 1):
            run(seed, 1, "A", 1.0, input_set=wa.PLAIN_NORMAL, baseline=-1.0)
            run(seed, 2, "A", 40.0)
            run(seed, 2, "B", 90.0)
            for g in wa.OTHERS:
                run(seed, 2, f"only {g}", 85.0 if g == "randomcase" else 42.0)
                run(seed, 2, f"without {g}", 45.0 if g == "randomcase" else 89.0)
            for condition, flagged in (("B", 90.0), ("B without reverted", 89.0),
                                       ("D without reverted", 87.0), ("D", 78.0)):
                run(seed, 3, condition, flagged)
            for C in self.grid:
                run(seed, 4, "A", 40.0, C)
                run(seed, 4, "B", 90.0, C)
                run(seed, 4, "D", 78.0 if C == self.grid[0] else 90.0, C)
        self.runs = pd.DataFrame(rows)

        def test(comparison, difference, significant="2/2"):
            tests.append({"comparison": comparison, "difference_pp": difference,
                          "significant_seeds": significant})

        test("B vs A", 50.0)
        for g in wa.OTHERS:
            test(f"only {g} vs A", 45.0 if g == "randomcase" else 2.0)
            test(f"without {g} vs B", -45.0 if g == "randomcase" else -1.0)
        test(wa.STEPS[0], -1.0, "0/2")
        test(wa.STEPS[1], -2.0)
        test(wa.STEPS[2], -9.0)
        test("D vs B", -12.0)
        for C in self.grid:
            test(f"B vs A at C={C}", 50.0)
            test(f"D vs B at C={C}", -12.0 if C == self.grid[0] else 0.0,
                 "2/2" if C == self.grid[0] else "0/2")
        self.tests = pd.DataFrame(tests)

        gaps = []
        for seed in (0, 1):
            for f in c.ENCODING_FAMILIES:
                name = f"rise: {f} normal minus plain normal"
                gaps += [{"seed": seed, "comparison": name, "part": "added attack evidence",
                          "value": 0.8, "share_pct": 80.0},
                         {"seed": seed, "comparison": name, "part": "lost normal evidence",
                          "value": 0.2, "share_pct": 20.0},
                         {"seed": seed, "comparison": name, "part": "(gap)", "value": 1.0,
                          "share_pct": 100.0}]
            name = f"gap: B minus A on {wa.HELD_OUT} attacks"
            gaps += [{"seed": seed, "comparison": name, "part": "(intercept)", "value": 0.4,
                      "share_pct": 40.0},
                     {"seed": seed, "comparison": name, "part": "(threshold)", "value": -0.1,
                      "share_pct": -10.0},
                     {"seed": seed, "comparison": name, "part": "(gap)", "value": 1.0,
                      "share_pct": 100.0}]
        self.gaps = pd.DataFrame(gaps)
        self.verdicts = wa.verdicts("logreg", self.runs, self.gaps, self.tests)

    def verdict(self, rule, subject):
        rows = self.verdicts[(self.verdicts["rule"] == rule) & (self.verdicts["subject"] == subject)]
        self.assertEqual(len(rows), 1, (rule, subject))
        return rows.iloc[0]["verdict"]

    def test_rules_give_the_expected_verdicts(self):
        self.assertTrue(self.verdict("R1", "charencode").startswith("H1b supported"))
        self.assertEqual(self.verdict("R1 default", "condition A"), "does not flag by default")
        self.assertEqual(self.verdict("R2 family", "randomcase"), "sufficient and necessary")
        self.assertEqual(self.verdict("R2 family", "percentage"), "neither")
        self.assertTrue(self.verdict("R2 source", wa.HELD_OUT).startswith("H2a supported"))
        self.assertEqual(self.verdict("R2 location", wa.HELD_OUT), "in the n-grams")
        self.assertEqual(self.verdict("R3", wa.HELD_OUT), "main cause: H3c, duplicated rows")
        self.assertEqual(self.verdict("R4", "B above A"), "stable across C")
        self.assertEqual(self.verdict("R4", "D below B"),
                         f"depends on C: holds at [{self.grid[0]}]")

    def test_a_step_that_is_not_significant_cannot_be_the_main_cause(self):
        tests = self.tests.copy()
        tests.loc[tests["comparison"] == wa.STEPS[2], "significant_seeds"] = "1/2"
        self.assertEqual(self.again(tests=tests, rule="R3"), "no single cause")

    def again(self, rule, subject=None, runs=None, gaps=None, tests=None):
        verdicts = wa.verdicts("logreg", self.runs if runs is None else runs,
                               self.gaps if gaps is None else gaps,
                               self.tests if tests is None else tests)
        rows = verdicts[verdicts["rule"] == rule]
        if subject is not None:
            rows = rows[rows["subject"] == subject]
        self.assertEqual(len(rows), 1, (rule, subject))
        return rows.iloc[0]["verdict"]

    def rise(self, family, added, lost, seed=None):
        gaps = self.gaps.copy()
        name = f"rise: {family} normal minus plain normal"
        where = (gaps["comparison"] == name) & (True if seed is None else gaps["seed"] == seed)
        for part, value in (("added attack evidence", added), ("lost normal evidence", lost),
                            ("(gap)", added + lost)):
            rows = where & (gaps["part"] == part)
            gaps.loc[rows, "value"] = value
            gaps.loc[rows, "share_pct"] = 100 * value / (added + lost)
        return gaps

    def test_r1_other_branches(self):
        self.assertTrue(self.again("R1", "charencode", gaps=self.rise("charencode", 0.2, 0.8))
                        .startswith("H1a supported"))
        self.assertEqual(self.again("R1", "charencode",
                                    gaps=self.rise("charencode", 0.2, 0.8, seed=0)), "mixed")
        self.assertEqual(self.again("R1", "charencode", gaps=self.rise("charencode", -0.8, -0.2)),
                         "rule does not apply")
        runs = self.runs.copy()
        runs.loc[runs["part"] == 1, "baseline"] = 0.3
        self.assertEqual(self.again("R1 default", runs=runs), "flags by default")
        runs.loc[(runs["part"] == 1) & (runs["seed"] == 0), "baseline"] = -0.3
        self.assertEqual(self.again("R1 default", runs=runs), "does not flag by default")

    def flagged(self, part, condition, value):
        runs = self.runs.copy()
        runs.loc[(runs["part"] == part) & (runs["condition"] == condition), "flagged_pct"] = value
        return runs

    def test_r2_source_other_branches(self):
        runs = self.flagged(2, "without randomcase", 88.0)
        self.assertTrue(self.again("R2 source", runs=runs).startswith("H2b supported"))
        self.assertEqual(self.again("R2 family", "randomcase", runs=runs), "sufficient")
        runs = self.flagged(2, "only randomcase", 42.0)
        self.assertTrue(self.again("R2 source", runs=runs).startswith("H2c supported"))
        self.assertEqual(self.again("R2 family", "randomcase", runs=runs), "necessary")
        tests = self.tests.copy()
        tests.loc[tests["comparison"] == "only randomcase vs A", "significant_seeds"] = "1/2"
        self.assertEqual(self.again("R2 family", "randomcase", tests=tests), "necessary")
        tests = self.tests.copy()
        tests.loc[tests["comparison"] == "B vs A", "significant_seeds"] = "1/2"
        self.assertEqual(self.again("R2 source", tests=tests), "rule does not apply")

    def test_r2_location_other_branches(self):
        name = f"gap: B minus A on {wa.HELD_OUT} attacks"

        def offset(intercept, threshold, seed=None):
            gaps = self.gaps.copy()
            where = (gaps["comparison"] == name) & (True if seed is None else gaps["seed"] == seed)
            gaps.loc[where & (gaps["part"] == "(intercept)"), "share_pct"] = intercept
            gaps.loc[where & (gaps["part"] == "(threshold)"), "share_pct"] = threshold
            return gaps

        self.assertEqual(self.again("R2 location", gaps=offset(90.0, -20.0)), "in the offset")
        self.assertEqual(self.again("R2 location", gaps=offset(90.0, -20.0, seed=0)), "mixed")

    def test_r3_does_not_apply_without_a_consistent_difference(self):
        tests = self.tests.copy()
        tests.loc[tests["comparison"] == "D vs B", "significant_seeds"] = "1/2"
        self.assertTrue(self.again("R3", tests=tests).startswith("rule does not apply"))

    def test_r4_other_branches(self):
        tests = self.tests.copy()
        rows = tests["comparison"].str.startswith("D vs B at C=")
        tests.loc[rows, "significant_seeds"] = "2/2"
        tests.loc[rows, "difference_pp"] = -12.0
        runs = self.runs.copy()
        runs.loc[(runs["part"] == 4) & (runs["condition"] == "D"), "flagged_pct"] = 78.0
        self.assertEqual(self.again("R4", "D below B", runs=runs, tests=tests), "stable across C")
        last = self.grid[-1]
        runs.loc[(runs["part"] == 4) & (runs["condition"] == "D") & (runs["seed"] == 0)
                 & np.isclose(runs["C"].astype(float), last), "flagged_pct"] = 95.0
        self.assertEqual(self.again("R4", "D below B", runs=runs, tests=tests),
                         f"depends on C: holds at {self.grid[:-1]}")
        tests.loc[rows, "significant_seeds"] = "0/2"
        self.assertEqual(self.again("R4", "D below B", runs=runs, tests=tests),
                         "depends on C: holds at no value")


if __name__ == "__main__":
    unittest.main()
