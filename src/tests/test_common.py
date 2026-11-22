"""Unit tests of the pipeline."""

import csv
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import analysis
import common as c
import config
import experiment
import split_dataset


class NormaliserTests(unittest.TestCase):
    def test_decodes_single_and_double_url_encoding(self):
        self.assertEqual(c.normalise("%27%20OR%201%3D1"), "' OR 1=1")
        self.assertEqual(c.normalise("%2527%2520OR"), "' OR")

    def test_strips_sql_comments_and_collapses_whitespace(self):
        self.assertEqual(c.normalise("UNION/**/SELECT   a"), "UNION SELECT a")

    def test_leaves_benign_text_unchanged(self):
        self.assertEqual(c.normalise("alice"), "alice")


class TamperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tampers = c.load_tampers()

    def test_all_eight_families_are_loaded(self):
        self.assertEqual(sorted(self.tampers), sorted(c.FAMILIES))

    def test_randomised_families_change_the_payload(self):
        payload = "' UNION SELECT pass FROM users"
        c.set_seed(0)
        for family in ("randomcase", "randomcomments"):
            self.assertNotEqual(c.apply_tamper(self.tampers, family, payload), payload, family)

    def test_seeding_makes_variants_reproducible(self):
        payload = "' UNION SELECT pass FROM users"
        c.set_seed(3)
        first = c.apply_tamper(self.tampers, "randomcase", payload)
        c.set_seed(3)
        self.assertEqual(first, c.apply_tamper(self.tampers, "randomcase", payload))

    def test_normaliser_coverage_matches_its_behaviour(self):
        attacks = c.load_source("test").query("y == 1")["payload"].tolist()[:300]
        c.set_seed(0)
        for family in c.FAMILIES:
            pairs = [(p, c.apply_tamper(self.tampers, family, p)) for p in attacks]
            pairs = [(p, v) for p, v in pairs if v != p]
            reverted = sum(c.normalise(v) == c.normalise(p) for p, v in pairs) / len(pairs)
            if family in c.NORMALISER_COVERED:
                self.assertEqual(reverted, 1.0, family)
            else:
                self.assertLess(reverted, 0.5, family)

    def test_control_variants_leave_the_random_state_unchanged(self):
        payload = "' UNION SELECT pass FROM users"
        c.set_seed(3)
        expected = c.apply_tamper(self.tampers, "randomcase", payload)
        c.set_seed(3)
        variants = experiment.normal_variants(self.tampers, ["alice", "bob smith"])
        self.assertEqual(c.apply_tamper(self.tampers, "randomcase", payload), expected)
        self.assertEqual(set(variants), set(c.FAMILIES) | {"urlquote"})
        self.assertEqual(variants["urlquote"], ["alice", "bob%20smith"])

    def test_failing_tamper_returns_payload_unchanged(self):
        def broken(_payload):
            raise ValueError("boom")
        self.assertEqual(c.apply_tamper({"broken": broken}, "broken", "abc"), "abc")


class DataTests(unittest.TestCase):
    def test_source_keeps_only_normal_and_sqli_rows(self):
        train, test = c.load_source("train"), c.load_source("test")
        self.assertEqual(len(train), 20105)
        self.assertEqual(len(test), 10051)
        self.assertEqual(set(train["label"]), {"norm", "sqli"})
        self.assertEqual(len(c.load_source_all()), len(train) + len(test))

    def test_split_is_70_15_15_stratified_and_deterministic(self):
        full = c.load_source_all()
        sets = split_dataset.split(full, 0)
        again = split_dataset.split(full, 0)
        n = len(full)
        self.assertEqual(sum(len(v) for v in sets.values()), n)
        self.assertAlmostEqual(len(sets["train"]) / n, 0.70, places=2)
        self.assertAlmostEqual(len(sets["val"]) / n, 0.15, places=2)
        self.assertAlmostEqual(len(sets["test"]) / n, 0.15, places=2)
        share = full["y"].mean()
        for v in sets.values():
            self.assertAlmostEqual(v["y"].mean(), share, places=2)
        for k in sets:
            self.assertTrue(sets[k]["payload"].equals(again[k]["payload"]))
        seen = pd.concat([v["payload"] for v in sets.values()])
        self.assertEqual(len(seen), len(full))
        self.assertEqual(seen.nunique(), len(full))
        self.assertEqual(full["payload"].nunique(), len(full))

    def test_load_data_reads_the_three_explicit_sets(self):
        if not all(os.path.exists(c.split_path(k)) for k in c.SPLITS):
            self.skipTest("run split_dataset.py first")
        sizes = {k: len(c.load_data(k)) for k in c.SPLITS}
        self.assertGreater(sizes["train"], sizes["val"])
        self.assertGreater(sizes["train"], sizes["test"])
        with self.assertRaises(ValueError):
            c.load_data("validation")


class ThresholdTests(unittest.TestCase):
    def test_validation_fpr_does_not_exceed_target(self):
        texts = ["' OR 1=1 --", "1 UNION SELECT a,b", "admin' --", "x' AND 1=1"] * 30 + \
                ["alice", "bob", "hello world", "search term", "page 2", "o'neil"] * 40
        labels = np.array([1] * 120 + [0] * 240)
        vectoriser, model = c.build_detector(texts, labels)
        thr = c.threshold_at_fpr(vectoriser, model, texts, labels)
        scores = c.score(vectoriser, model, texts)
        self.assertLessEqual((scores[labels == 0] >= thr).mean(), 0.01)


class DatasetAndModelTests(unittest.TestCase):
    def test_single_file_dataset_is_loaded_whole_and_has_no_official_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "Second")
            os.makedirs(folder)
            with open(os.path.join(folder, "data.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["query", "label"])
                for i in range(40):
                    w.writerow([f"benign {i}", "0"])
                    w.writerow([f"' OR {i}=1 --", "1"])
                w.writerow(["ignored row", "2"])
            config.DATASET_SPECS["second_test"] = {
                "folder": folder, "single_file": "data.csv", "text_col": "query",
                "label_col": "label", "sqli_labels": ["1"], "normal_labels": ["0"],
            }
            orig_locate = c._locate
            c._locate = lambda name: folder if name == folder else orig_locate(name)
            os.environ["DATASET"] = "second_test"
            try:
                full = c.load_source_all()
                with self.assertRaises(ValueError):
                    c.load_source("train")
            finally:
                del os.environ["DATASET"]
                c._locate = orig_locate
                config.DATASET_SPECS.pop("second_test")
            self.assertEqual(len(full), 80)
            self.assertEqual(list(full.columns), ["payload", "label", "y"])
            self.assertEqual(int(full["y"].sum()), 40)

    def test_score_falls_back_to_sigmoid_without_predict_proba(self):
        class MarginOnly:
            def decision_function(self, X):
                return np.array([-2.0, 0.0, 2.0])

        class FakeVec:
            def transform(self, texts):
                return texts

        out = c.score(FakeVec(), MarginOnly(), ["a", "b", "d"])
        self.assertTrue(np.all((out > 0) & (out < 1)))
        self.assertAlmostEqual(out[1], 0.5, places=6)
        self.assertTrue(out[0] < out[1] < out[2])

    def test_model_factory_builds_each_backend(self):
        self.assertIsInstance(c._make_model("logreg"), LogisticRegression)
        self.assertIsInstance(c._make_model("linsvc"), LinearSVC)
        self.assertIsInstance(c._make_model("rf"), RandomForestClassifier)
        self.assertEqual(c._make_model("logreg", {"C": 3.0}).C, 3.0)
        with self.assertRaises(ValueError):
            c._make_model("nope")


class StatisticsTests(unittest.TestCase):
    def test_paired_significance_detects_a_real_difference(self):
        new = np.array([1] * 90 + [0] * 10)
        old = np.array([1] * 10 + [0] * 90)
        spec = [("B vs A", lambda s: new, lambda s: old)]
        out = analysis.paired_significance(spec, [0, 1])
        self.assertEqual(out.loc[0, "significant_seeds"], "2/2")
        self.assertEqual(out.loc[0, "difference_pp"], 80.0)
        self.assertGreater(out.loc[0, "ci95_low_pp"], 0)

    def test_interval_does_not_depend_on_other_comparisons(self):
        rng = np.random.RandomState(7)
        a = rng.randint(0, 2, 300)
        b = rng.randint(0, 2, 300)
        target = ("B vs A", lambda s: b, lambda s: a)
        other = ("C vs A", lambda s: a, lambda s: b)
        alone = analysis.paired_significance([target], [0])
        with_other = analysis.paired_significance([other, target], [0])
        row = with_other[with_other["comparison"] == "B vs A"].iloc[0]
        self.assertEqual(alone.loc[0, "ci95_low_pp"], row["ci95_low_pp"])
        self.assertEqual(alone.loc[0, "ci95_high_pp"], row["ci95_high_pp"])

    def test_identical_predictions_give_no_difference(self):
        same = np.array([1, 0] * 50)
        spec = [("B vs A", lambda s: same, lambda s: same)]
        out = analysis.paired_significance(spec, [0])
        self.assertEqual(out.loc[0, "difference_pp"], 0.0)
        self.assertTrue(np.isnan(out.loc[0, "odds_ratio"]))


if __name__ == "__main__":
    unittest.main()
