"""Step 14 (weight analysis): what the weights of the linear detectors respond to."""

from __future__ import annotations

import argparse
import functools
import os
import random
import zlib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

import analysis
import common as c
import config

MODELS = ("logreg", "linsvc")
HELD_OUT = "base64encode"
OTHERS = [f for f in c.FAMILIES if f != HELD_OUT]
REVERTED = [f for f in OTHERS if f in c.NORMALISER_COVERED]
KEPT = [f for f in OTHERS if f not in REVERTED]
MAJORITY = 0.5
TOP_NGRAMS = 20
EXAMPLE = "' OR 1=1 --"
SAME_DETECTION, SAME_FPR = 0.011, 0.0011
SAME_SUM = 1e-8

GROUPS = ["word boundary", "percent", "lower-case letters", "upper-case letters",
          "mixed-case letters", "digits", "letters and digits", "other symbols"]

PLAIN_NORMAL, PLAIN_ATTACK = "plain normal", "plain attacks"
STEPS = ["step 1: B without reverted vs B",
         "step 2: D without reverted vs B without reverted",
         "step 3: D vs D without reverted"]
STEP_CONDITIONS = [("B without reverted", "B"),
                   ("D without reverted", "B without reverted"),
                   ("D", "D without reverted")]
STEP_HYPOTHESES = ["H3a, lost variants", "H3b, normalised training text",
                   "H3c, duplicated rows"]
FILES = {"runs": "weight_analysis.csv", "ngrams": "weight_analysis_ngrams.csv",
         "facts": "weight_analysis_facts.csv", "tests": "weight_analysis_tests.csv"}


@functools.lru_cache(maxsize=None)
def group_of(ngram: str) -> str:
    core = ngram.replace(" ", "")
    if not core:
        return "word boundary"
    if "%" in core:
        return "percent"
    if core.isalpha():
        upper, lower = any(ch.isupper() for ch in core), any(ch.islower() for ch in core)
        if upper and lower:
            return "mixed-case letters"
        return "upper-case letters" if upper else "lower-case letters"
    if core.isdigit():
        return "digits"
    if core.isalnum():
        return "letters and digits"
    return "other symbols"


def shown(ngram: str) -> str:
    return f"[{ngram}]"


def group_column(sign: str, group: str) -> str:
    return f"{sign}_{group.replace(' ', '_').replace('-', '_')}"


def to_logit(threshold: float) -> float:
    return float(np.log(threshold / (1.0 - threshold)))


def prediction_key(part: int, condition: str, C: float | None = None) -> str:
    return f"part{part}__{condition}__C={C}"


def experiment_stream(seed: int, condition: str) -> None:
    fits_before = 4 * c.FAMILIES.index(HELD_OUT) + "ABCD".index(condition)
    np.random.seed(seed)
    for _ in range(fits_before):
        np.random.randint(np.iinfo("i").max)


def own_stream(seed: int, label: str) -> None:
    np.random.seed(zlib.crc32(f"{seed}|{label}".encode()) % 2**32)


class Detector:
    def __init__(self, texts: list[str], labels: list[int], val: pd.DataFrame, *,
                 normalised: bool, model: str, seed: int, params: dict | None = None):
        self.normalised = normalised
        self.train_rows = len(texts)
        self.vectoriser, self.estimator = c.build_detector(
            texts, np.array(labels), normalised=normalised, model=model, params=params, seed=seed)
        self.threshold = c.threshold_at_fpr(self.vectoriser, self.estimator, val["payload"],
                                            val["y"].values, normalised=normalised)
        if not 0.0 < self.threshold < 1.0:
            raise SystemExit(f"Threshold {self.threshold} is not inside (0, 1); "
                             "the margin cannot be computed.")
        self.tau = to_logit(self.threshold)
        self.weights = self.estimator.coef_.ravel()
        self.intercept = float(self.estimator.intercept_[0])
        self.baseline = self.intercept - self.tau
        self.names = self.vectoriser.get_feature_names_out()
        boundary = self.vectoriser.vocabulary_.get(" ")
        self.boundary_weight = float(self.weights[boundary]) if boundary is not None else 0.0
        group = np.array([GROUPS.index(group_of(n)) for n in self.names])
        column = 2 * group + (self.weights < 0)
        self.parts = csr_matrix((self.weights, (np.arange(len(self.names)), column)),
                                shape=(len(self.names), 2 * len(GROUPS)))

    def prepare(self, texts: list[str]) -> list[str]:
        return [c.normalise(t) for t in texts] if self.normalised else list(texts)

    def read(self, texts: list[str], coverage: bool = False) -> "Reading":
        return Reading(self, texts, coverage)


class Reading:
    def __init__(self, det: Detector, texts: list[str], coverage: bool):
        prepared = det.prepare(texts)
        features = det.vectoriser.transform(prepared)
        self.n = len(texts)
        self.scores = c.score(det.vectoriser, det.estimator, texts, normalised=det.normalised)
        self.flagged = self.scores >= det.threshold
        self.contributions = np.asarray((features @ det.parts).todense())
        self.attack_evidence = self.contributions[:, 0::2].sum(axis=1)
        self.normal_evidence = self.contributions[:, 1::2].sum(axis=1)
        self.margin = det.baseline + self.attack_evidence + self.normal_evidence
        own = det.estimator.decision_function(features) - det.tau
        self.residual = float(np.max(np.abs(self.margin - own))) if self.n else 0.0
        self.mean_tfidf = np.asarray(features.mean(axis=0)).ravel()
        self.coverage = self._coverage(det, prepared) if coverage else float("nan")

    @staticmethod
    def _coverage(det: Detector, prepared: list[str]) -> float:
        analyse, known = det.vectoriser.build_analyzer(), det.vectoriser.vocabulary_
        shares = []
        for text in prepared:
            grams = analyse(text)
            if grams:
                shares.append(sum(g in known for g in grams) / len(grams))
        return float(np.mean(shares)) if shares else float("nan")

    def group_means(self) -> np.ndarray:
        return self.contributions.mean(axis=0).reshape(len(GROUPS), 2)


def run_row(det: Detector, reading: Reading, **keys) -> dict:
    means = reading.group_means()
    return {**keys, "n": reading.n,
            "flagged_pct": round(float(reading.flagged.mean()) * 100, 2),
            "mean_margin": round(float(reading.margin.mean()), 6),
            "baseline": round(det.baseline, 6),
            "attack_evidence": round(float(reading.attack_evidence.mean()), 6),
            "normal_evidence": round(float(reading.normal_evidence.mean()), 6),
            "coverage_pct": round(reading.coverage * 100, 2),
            "boundary_weight": round(det.boundary_weight, 6),
            "intercept": round(det.intercept, 6), "tau": round(det.tau, 6),
            "threshold": round(det.threshold, 4), "vocabulary": len(det.names),
            "train_rows": det.train_rows, "residual": reading.residual,
            **{group_column(sign, g): round(float(means[i, j]), 6)
               for i, g in enumerate(GROUPS) for j, sign in enumerate(("attack", "normal"))}}


def ngram_rows(det_x: Detector, read_x: Reading, det_y: Detector, read_y: Reading,
               **keys) -> list[dict]:
    def table(det, reading):
        used = np.flatnonzero(reading.mean_tfidf)
        return {det.names[j]: (float(reading.mean_tfidf[j]),
                               float(det.weights[j] * reading.mean_tfidf[j])) for j in used}
    x, y = table(det_x, read_x), table(det_y, read_y)
    wx, wy = dict(zip(det_x.names, det_x.weights)), dict(zip(det_y.names, det_y.weights))
    none = (0.0, 0.0)
    diff = {g: x.get(g, none)[1] - y.get(g, none)[1] for g in set(x) | set(y)}
    top = sorted(diff, key=lambda g: (-abs(round(diff[g], 6)), g))[:TOP_NGRAMS]
    return [{**keys, "rank": rank, "ngram": shown(g), "group": group_of(g),
             "weight_x": round(float(wx[g]), 4) if g in wx else np.nan,
             "tfidf_x": round(x.get(g, none)[0], 4),
             "contribution_x": round(x.get(g, none)[1], 4),
             "weight_y": round(float(wy[g]), 4) if g in wy else np.nan,
             "tfidf_y": round(y.get(g, none)[0], 4),
             "contribution_y": round(y.get(g, none)[1], 4),
             "difference": round(diff[g], 4)} for rank, g in enumerate(top, 1)]


def example_rows(det: Detector, payload: str, **keys) -> list[dict]:
    features = det.vectoriser.transform(det.prepare([payload]))
    rows = [{"ngram": shown(det.names[j]), "group": group_of(det.names[j]),
             "weight_x": round(float(det.weights[j]), 4), "tfidf_x": round(float(v), 4),
             "contribution_x": round(float(v * det.weights[j]), 4),
             "weight_y": np.nan, "tfidf_y": np.nan, "contribution_y": np.nan,
             "difference": round(float(v * det.weights[j]), 4)}
            for j, v in zip(features.indices, features.data)]
    rows.sort(key=lambda r: (-abs(r["contribution_x"]), r["ngram"]))
    return [{**keys, "comparison": f"example: {payload}", "rank": rank, **row}
            for rank, row in enumerate(rows, 1)]


def fact_rows(subject: str, measures: dict, **keys) -> list[dict]:
    return [{**keys, "subject": subject, "measure": measure, "value": value}
            for measure, value in measures.items() if not pd.isna(value)]


def dataset_facts(tampers: dict) -> pd.DataFrame:
    patterns = {"upper_case_letter_pct": r"[A-Z]", "percent_sign_pct": r"%",
                "url_escape_pct": r"%[0-9A-Fa-f]{2}", "equals_sign_pct": r"=",
                "digit_pct": r"[0-9]", "whitespace_pct": r"\s"}

    def describe(name: str, label: str, payloads: pd.Series) -> list[dict]:
        text = "".join(payloads)
        return fact_rows(f"{name} {label}", {
            "rows": len(payloads),
            **{column: round(100 * float(payloads.str.contains(rx, regex=True).mean()), 2)
               for column, rx in patterns.items()},
            "upper_case_chars_pct": round(100 * sum(ch.isupper() for ch in text)
                                          / max(len(text), 1), 2)}, model=None, seed=None)

    rows = []
    for name in c.SPLITS:
        data = c.load_data(name)
        for y, label in ((0, "normal"), (1, "sqli")):
            payloads = data[data["y"] == y]["payload"]
            rows.extend(describe(name, label, payloads))
            if name == "test":
                encoded = pd.Series([c.apply_tamper(tampers, HELD_OUT, p) for p in payloads])
                rows.extend(describe(f"test in {HELD_OUT}", label, encoded))
    return pd.DataFrame(rows)


def make_inputs(tampers: dict, seed: int):
    c.set_seed(seed)
    train, val, test = c.load_data("train"), c.load_data("val"), c.load_data("test")
    test_attacks = test[test["y"] == 1]["payload"].tolist()
    test_normal = test[test["y"] == 0]["payload"].tolist()
    rng = np.random.RandomState(seed)
    train_attacks = train[train["y"] == 1]["payload"].tolist()
    base = list(rng.choice(train_attacks, min(config.N_AUGMENT, len(train_attacks)),
                           replace=False))

    state_py, state_np = random.getstate(), np.random.get_state()
    encoded = {f: {"normal": [c.apply_tamper(tampers, f, p) for p in test_normal],
                   "attacks": [c.apply_tamper(tampers, f, p) for p in test_attacks]}
               for f in c.ENCODING_FAMILIES}
    random.setstate(state_py)
    np.random.set_state(state_np)

    for name in c.FAMILIES:
        attack_eval = [c.apply_tamper(tampers, name, p) for p in test_attacks]
        variants = {g: [c.apply_tamper(tampers, g, p) for p in base]
                    for g in c.FAMILIES if g != name}
        if name == HELD_OUT:
            break
    encoded[HELD_OUT]["attacks"] = attack_eval
    return train, val, test_normal, test_attacks, base, variants, encoded


def run_seed(tampers: dict, seed: int, model: str) -> dict[str, pd.DataFrame]:
    train, val, test_normal, test_attacks, base, variants, encoded = make_inputs(tampers, seed)
    held_attacks, held_normal = encoded[HELD_OUT]["attacks"], encoded[HELD_OUT]["normal"]
    key = {"model": model, "seed": seed}
    runs, ngrams, predictions = [], [], {}

    def fit(families: list[str], normalised: bool, label: str, like: str | None = None,
            params: dict | None = None) -> Detector:
        texts = train["payload"].tolist() + [v for g in families for v in variants[g]]
        labels = train["y"].tolist() + [1] * (len(texts) - len(train))
        experiment_stream(seed, like) if like else own_stream(seed, label)
        return Detector(texts, labels, val, normalised=normalised, model=model, seed=seed,
                        params=params)

    def held(det: Detector, part: int, condition: str,
             C: float | None = None) -> tuple[Reading, Reading]:
        attacks, normal = det.read(held_attacks), det.read(held_normal)
        auc = roc_auc_score(np.r_[np.ones(attacks.n), np.zeros(normal.n)],
                            np.r_[attacks.scores, normal.scores])
        clean = float(det.read(test_normal).flagged.mean()) * 100
        where = {**key, "part": part, "condition": condition, "C": C}
        for name, reading in ((f"{HELD_OUT} attacks", attacks), (f"{HELD_OUT} normal", normal)):
            runs.append({**run_row(det, reading, **where, input_set=name),
                         "auc": round(float(auc), 4), "clean_fpr_pct": round(clean, 3)})
        predictions[prediction_key(part, condition, C)] = attacks.flagged.astype(int)
        return attacks, normal

    det_a = fit([], False, "A", like="A")
    sets = {PLAIN_NORMAL: test_normal, PLAIN_ATTACK: test_attacks}
    for f in c.ENCODING_FAMILIES:
        sets[f"{f} normal"], sets[f"{f} attacks"] = encoded[f]["normal"], encoded[f]["attacks"]
    part1 = {name: det_a.read(texts, coverage=True) for name, texts in sets.items()}
    for name, reading in part1.items():
        where = {**key, "part": 1, "condition": "A", "C": None, "input_set": name}
        runs.append({**run_row(det_a, reading, **where), "auc": np.nan, "clean_fpr_pct": np.nan})
    for f in c.ENCODING_FAMILIES:
        ngrams.extend(ngram_rows(det_a, part1[f"{f} normal"], det_a, part1[PLAIN_NORMAL], **key,
                                 comparison=f"rise: {f} normal minus plain normal"))
    if seed == config.SEEDS[0]:
        for payload in (EXAMPLE, c.apply_tamper(tampers, HELD_OUT, EXAMPLE)):
            ngrams.extend(example_rows(det_a, payload, **key))

    a_att, a_norm = held(det_a, 2, "A")
    det_b = fit(OTHERS, False, "B", like="B")
    b_att, b_norm = held(det_b, 2, "B")
    ngrams.extend(ngram_rows(det_b, b_att, det_a, a_att, **key,
                             comparison=f"gap: B minus A on {HELD_OUT} attacks"))
    ngrams.extend(ngram_rows(det_b, b_norm, det_a, a_norm, **key,
                             comparison=f"gap: B minus A on {HELD_OUT} normal"))
    for g in OTHERS:
        for label, families in ((f"only {g}", [g]),
                                (f"without {g}", [f for f in OTHERS if f != g])):
            det = fit(families, False, label)
            held(det, 2, label)
            del det

    det_d = fit(OTHERS, True, "D", like="D")
    d_att, _ = held(det_d, 3, "D")
    held(det_b, 3, "B")
    ngrams.extend(ngram_rows(det_d, d_att, det_b, b_att, **key,
                             comparison=f"gap: D minus B on {HELD_OUT} attacks"))
    for label, normalised in (("B without reverted", False), ("D without reverted", True)):
        det = fit(KEPT, normalised, label)
        held(det, 3, label)
        del det
    reverted = [(v, p) for g in REVERTED for v, p in zip(variants[g], base)]
    share = lambda flags: round(100 * float(np.mean(list(flags))), 2)
    has_upper = lambda text: any(ch.isupper() for ch in text)
    facts = fact_rows("normaliser", {
        "reverted_variants": len(reverted),
        "reverted_equal_original_pct": share(c.normalise(v) == c.normalise(p) for v, p in reverted),
        "held_attacks_unchanged_pct": share(c.normalise(t) == t for t in held_attacks),
        "held_normal_unchanged_pct": share(c.normalise(t) == t for t in held_normal),
        "train_normal_changed_pct": share(c.normalise(t) != t
                                          for t in train[train["y"] == 0]["payload"]),
        "train_sqli_changed_pct": share(c.normalise(t) != t
                                        for t in train[train["y"] == 1]["payload"])}, **key)
    for det, name in ((det_a, "A"), (det_b, "B")):
        weights = np.array([w for n, w in zip(det.names, det.weights) if has_upper(n)])
        facts += fact_rows("letter case", {
            f"upper_case_ngrams_in_{name}": len(weights),
            f"upper_case_ngrams_in_{name}_positive_pct":
                share(weights > 0) if len(weights) else np.nan}, **key)
    for g in OTHERS:
        facts += fact_rows(f"variants of {g}", {
            "with_upper_case_pct": share(has_upper(v) for v in variants[g]),
            "changed_by_normaliser_pct": share(c.normalise(v) != v for v in variants[g])}, **key)
    for f in c.ENCODING_FAMILIES:
        facts += fact_rows(f"normal values in {f}", {
            "changed_pct": share(a != b for a, b in zip(encoded[f]["normal"], test_normal))}, **key)

    selected = c.model_params(model)["C"]
    for C in config.PARAM_GRID[model]["C"]:
        for condition, families, normalised in (("A", [], False), ("B", OTHERS, False),
                                                ("D", OTHERS, True)):
            if C == selected:
                det = {"A": det_a, "B": det_b, "D": det_d}[condition]
            else:
                det = fit(families, normalised, f"{condition} C={C}", like=condition,
                          params={"C": C})
            held(det, 4, condition, C=C)
    print(f"seed {seed} | {model}: {len(predictions)} conditions scored", flush=True)

    np.savez(c.results_path("predictions", f"weight_{model}_{seed}.npz"), **predictions)
    return {"runs": pd.DataFrame(runs), "ngrams": pd.DataFrame(ngrams),
            "facts": pd.DataFrame(facts)}


def save(name: str, table: pd.DataFrame, model: str | None,
         seeds: list[int] | None = None) -> None:
    if table.empty:
        return
    path = c.results_path(name)
    if os.path.exists(path):
        previous = pd.read_csv(path)
        same = previous["model"].isna() if model is None else previous["model"] == model
        if seeds is not None:
            same &= previous["seed"].isin(seeds)
        table = pd.concat([previous[~same], table], ignore_index=True)
    if "seed" in table.columns:
        table["seed"] = table["seed"].astype("Int64")
    table.to_csv(path, index=False, lineterminator="\n")


def check_seed(tables: dict[str, pd.DataFrame], model: str, seed: int) -> tuple[bool, list[str]]:
    runs, lines, passed = tables["runs"], [], True

    residual = runs["residual"].max()
    ok = residual < SAME_SUM
    passed &= ok
    lines.append(f"  parts add up to the model's decision (largest difference {residual:.1e}): "
                 f"{'ok' if ok else 'FAILED'}")

    path = c.results_path("experiment.csv")
    if not os.path.exists(path):
        return False, lines + ["  results/experiment.csv not found: run the main experiment first."]
    ref = pd.read_csv(path).drop_duplicates(
        subset=["model", "protocol", "held_out", "seed", "condition"], keep="last")
    ref = ref[(ref["model"] == model) & (ref["protocol"] == "family") & (ref["seed"] == seed)
              & (ref["held_out"] == HELD_OUT)].set_index("condition")
    mine = runs[(runs["input_set"] == f"{HELD_OUT} attacks") & runs["part"].isin([2, 3])]
    mine = mine.drop_duplicates(subset="condition").set_index("condition")
    for cond in ("A", "B", "D"):
        if cond not in ref.index:
            passed = False
            lines.append(f"  condition {cond}: no row in results/experiment.csv for seed {seed}, "
                         f"model {model}")
            continue
        got = (mine.loc[cond, "flagged_pct"], mine.loc[cond, "clean_fpr_pct"])
        want = (ref.loc[cond, "detection_pct"], ref.loc[cond, "clean_fpr_pct"])
        ok = abs(got[0] - want[0]) < SAME_DETECTION and abs(got[1] - want[1]) < SAME_FPR
        passed &= ok
        lines.append(f"  condition {cond}: detection {got[0]:.2f}% (experiment.csv {want[0]:.2f}%), "
                     f"clean FPR {got[1]:.3f}% ({want[1]:.3f}%): {'ok' if ok else 'DIFFERENT'}")

    if "A" in ref.index:
        part1 = runs[runs["part"] == 1].set_index("input_set")["flagged_pct"]
        same = [abs(part1[f"{f} normal"] - ref.loc["A", f"normal_{f}"]) < SAME_DETECTION
                for f in c.ENCODING_FAMILIES]
        passed &= all(same)
        lines.append(f"  encoded normal values flagged, against experiment.csv: "
                     f"{sum(same)} of {len(same)} encodings match")
    return bool(passed), lines


def load(name: str, model: str) -> pd.DataFrame:
    path = c.results_path(FILES[name])
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run the analysis first.")
    table = pd.read_csv(path)
    return table[table["model"] == model].reset_index(drop=True)


def load_plain(name: str) -> pd.DataFrame:
    path = c.results_path(FILES[name])
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run the analysis first.")
    return pd.read_csv(path)


def gap_tables(runs: pd.DataFrame) -> pd.DataFrame:
    out = []

    def add(seed, comparison, parts):
        total = sum(parts.values())
        for name, value in parts.items():
            out.append({"seed": seed, "comparison": comparison, "part": name,
                        "value": round(value, 4),
                        "share_pct": round(100 * value / total, 1) if abs(total) > 1e-12 else np.nan})
        out.append({"seed": seed, "comparison": comparison, "part": "(gap)",
                    "value": round(total, 4), "share_pct": 100.0})

    def by_group(x: pd.Series, y: pd.Series) -> dict:
        return {g: float(x[group_column("attack", g)] + x[group_column("normal", g)]
                         - y[group_column("attack", g)] - y[group_column("normal", g)])
                for g in GROUPS}

    for seed in sorted(runs["seed"].unique()):
        r = runs[runs["seed"] == seed]
        one = r[r["part"] == 1].set_index("input_set")
        plain = one.loc[PLAIN_NORMAL]
        for f in c.ENCODING_FAMILIES:
            enc = one.loc[f"{f} normal"]
            add(seed, f"rise: {f} normal minus plain normal",
                {"added attack evidence": enc["attack_evidence"] - plain["attack_evidence"],
                 "lost normal evidence": enc["normal_evidence"] - plain["normal_evidence"]})
        for f in c.ENCODING_FAMILIES:
            add(seed, f"rise by group: {f} normal minus plain normal",
                by_group(one.loc[f"{f} normal"], plain))
            add(seed, f"separation: {f} attacks minus {f} normal",
                by_group(one.loc[f"{f} attacks"], one.loc[f"{f} normal"]))
        add(seed, "separation: plain attacks minus plain normal",
            by_group(one.loc[PLAIN_ATTACK], plain))

        def between(part, x, y, input_set, label):
            rows = r[(r["part"] == part) & (r["input_set"] == input_set) & r["C"].isna()]
            rows = rows.drop_duplicates(subset="condition").set_index("condition")
            parts = by_group(rows.loc[x], rows.loc[y])
            parts["(intercept)"] = float(rows.loc[x, "intercept"] - rows.loc[y, "intercept"])
            parts["(threshold)"] = -float(rows.loc[x, "tau"] - rows.loc[y, "tau"])
            add(seed, label, parts)

        between(2, "B", "A", f"{HELD_OUT} attacks", f"gap: B minus A on {HELD_OUT} attacks")
        between(2, "B", "A", f"{HELD_OUT} normal", f"gap: B minus A on {HELD_OUT} normal")
        between(3, "D", "B", f"{HELD_OUT} attacks", f"gap: D minus B on {HELD_OUT} attacks")
        for step, (x, y) in zip(STEPS, STEP_CONDITIONS, strict=True):
            between(3, x, y, f"{HELD_OUT} attacks", f"gap of {step} on {HELD_OUT} attacks")
    return pd.DataFrame(out)


def detection(runs: pd.DataFrame, part: int, C: float | None = None) -> pd.DataFrame:
    r = runs[(runs["part"] == part) & (runs["input_set"] == f"{HELD_OUT} attacks")]
    r = r[r["C"].isna()] if C is None else r[np.isclose(r["C"], C)]
    return r.pivot_table(index="seed", columns="condition", values="flagged_pct")


def paired_tests(model: str, seeds: list[int]) -> pd.DataFrame:
    stored = {s: np.load(c.results_path("predictions", f"weight_{model}_{s}.npz")) for s in seeds}

    def spec(label, part, new, old, C=None):
        new, old = prediction_key(part, new, C), prediction_key(part, old, C)
        return (label, lambda s, k=new: stored[s][k], lambda s, k=old: stored[s][k])

    families = {
        2: [spec("B vs A", 2, "B", "A")]
           + [spec(f"only {g} vs A", 2, f"only {g}", "A") for g in OTHERS]
           + [spec(f"without {g} vs B", 2, f"without {g}", "B") for g in OTHERS],
        3: [spec(STEPS[0], 3, "B without reverted", "B"),
            spec(STEPS[1], 3, "D without reverted", "B without reverted"),
            spec(STEPS[2], 3, "D", "D without reverted"),
            spec("D vs B", 3, "D", "B")],
        4: [spec(f"{new} vs {old} at C={C}", 4, new, old, C)
            for C in config.PARAM_GRID[model]["C"] for new, old in (("B", "A"), ("D", "B"))],
    }
    parts = []
    for part, rows in families.items():
        table = analysis.paired_significance(rows, seeds)
        table.insert(0, "part", part)
        parts.append(table)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "model", model)
    return out


def every_seed(significant_seeds: str) -> bool:
    done, total = significant_seeds.split("/")
    return done == total


def per_seed(values) -> str:
    return ", ".join(f"{v:.1f}" for v in values)


def verdicts(model: str, runs: pd.DataFrame, gaps: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    out = []
    test = tests.set_index("comparison")

    def add(rule, subject, verdict, evidence):
        out.append({"model": model, "rule": rule, "subject": subject, "verdict": verdict,
                    "evidence": evidence})

    def shares(comparison, part):
        g = gaps[(gaps["comparison"] == comparison) & (gaps["part"] == part)]
        return g.sort_values("seed")["share_pct"].to_numpy()

    def values(comparison, part="(gap)"):
        g = gaps[(gaps["comparison"] == comparison) & (gaps["part"] == part)]
        return g.sort_values("seed")["value"].to_numpy()

    for f in c.ENCODING_FAMILIES:
        name = f"rise: {f} normal minus plain normal"
        rise, lost, added = values(name), shares(name, "lost normal evidence"), \
            shares(name, "added attack evidence")
        evidence = (f"rise per seed: {per_seed(rise)}; lost normal evidence, % of the rise: "
                    f"{per_seed(lost)}; added attack evidence, %: {per_seed(added)}")
        if not (rise > 0).all():
            add("R1", f, "rule does not apply", evidence)
        elif (lost > 100 * MAJORITY).all():
            add("R1", f, "H1a supported: unrecognised input", evidence)
        elif (added > 100 * MAJORITY).all():
            add("R1", f, "H1b supported: learnt encoding cue", evidence)
        else:
            add("R1", f, "mixed", evidence)
    base = runs[(runs["part"] == 1) & (runs["input_set"] == PLAIN_NORMAL)].sort_values("seed")
    add("R1 default", "condition A",
        "flags by default" if (base["baseline"] > 0).all() else "does not flag by default",
        f"baseline per seed: {', '.join(f'{v:.3f}' for v in base['baseline'])}")

    det = detection(runs, 2)
    gain = det["B"] - det["A"]
    if not every_seed(test.loc["B vs A", "significant_seeds"]):
        add("R2 source", HELD_OUT, "rule does not apply", "B vs A is not significant in every seed")
    else:
        sufficient, necessary = [], []
        for g in OTHERS:
            reach, loss = det[f"only {g}"] - det["A"], det["B"] - det[f"without {g}"]
            s = bool((reach > MAJORITY * gain).all()
                     and every_seed(test.loc[f"only {g} vs A", "significant_seeds"]))
            n = bool((loss > MAJORITY * gain).all()
                     and every_seed(test.loc[f"without {g} vs B", "significant_seeds"]))
            sufficient += [g] if s else []
            necessary += [g] if n else []
            add("R2 family", g, ("sufficient and necessary" if s and n else "sufficient" if s
                                 else "necessary" if n else "neither"),
                f"gain of B per seed: {per_seed(gain)}; alone it reaches: {per_seed(reach)}; "
                f"removing it loses: {per_seed(loss)}")
        both = [g for g in sufficient if g in necessary]
        verdict = (f"H2a supported: single source ({', '.join(both)})" if both else
                   f"H2b supported: redundant sources ({', '.join(sufficient)})" if sufficient
                   else "H2c supported: joint effect")
        add("R2 source", HELD_OUT, verdict,
            f"sufficient: {sufficient or 'none'}; necessary: {necessary or 'none'}")
        name = f"gap: B minus A on {HELD_OUT} attacks"
        offset = shares(name, "(intercept)") + shares(name, "(threshold)")
        content = 100 - offset
        add("R2 location", HELD_OUT,
            "in the n-grams" if (values(name) > 0).all() and (content > 100 * MAJORITY).all()
            else "in the offset" if (values(name) > 0).all() and (offset > 100 * MAJORITY).all()
            else "mixed",
            f"gap per seed: {per_seed(values(name))}; n-gram groups, % of the gap: "
            f"{per_seed(content)}; intercept and threshold, %: {per_seed(offset)}")

    det = detection(runs, 3)
    total = det["D"] - det["B"]
    steps = dict(zip(STEPS, (det["B without reverted"] - det["B"],
                             det["D without reverted"] - det["B without reverted"],
                             det["D"] - det["D without reverted"])))
    names = dict(zip(STEPS, STEP_HYPOTHESES))
    evidence = f"D minus B per seed: {per_seed(total)}; " + "; ".join(
        f"{k.split(':')[0]}: {per_seed(v)}" for k, v in steps.items())
    if not every_seed(test.loc["D vs B", "significant_seeds"]):
        add("R3", HELD_OUT, "rule does not apply: D vs B is not significant in every seed", evidence)
    else:
        main = [k for k, v in steps.items()
                if (v / total > MAJORITY).all() and every_seed(test.loc[k, "significant_seeds"])]
        add("R3", HELD_OUT,
            f"main cause: {names[main[0]]}" if main else "no single cause", evidence)

    for new, old, expected in (("B", "A", 1), ("D", "B", -1)):
        holds, lines = [], []
        for C in config.PARAM_GRID[model]["C"]:
            row = test.loc[f"{new} vs {old} at C={C}"]
            det = detection(runs, 4, C)
            same_sign = bool((np.sign(det[new] - det[old]) == expected).all())
            ok = same_sign and every_seed(row["significant_seeds"])
            holds += [C] if ok else []
            lines.append(f"C={C}: {row['difference_pp']:+.2f} points, significant in "
                         f"{row['significant_seeds']} seeds")
        every = len(holds) == len(config.PARAM_GRID[model]["C"])
        direction = "above" if expected > 0 else "below"
        add("R4", f"{new} {direction} {old}",
            "stable across C" if every else
            f"depends on C: holds at {holds}" if holds else "depends on C: holds at no value",
            "; ".join(lines))
    return pd.DataFrame(out)


def mean_sd(frame: pd.DataFrame, by: list[str], column: str) -> pd.Series:
    stats = frame.groupby(by, sort=False)[column].agg(["mean", "std"]).fillna(0)
    return stats.apply(lambda r: f"{r['mean']:.1f} ± {r['std']:.1f}", axis=1)


def summarise(model: str, new_tests: bool) -> None:
    runs = load("runs", model)
    seeds = sorted(int(s) for s in runs["seed"].unique())
    gaps = gap_tables(runs)
    if new_tests:
        save(FILES["tests"], paired_tests(model, seeds), model)
    tests = load("tests", model)
    rules = verdicts(model, runs, gaps, tests)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    print(f"\n================ {model}, seeds {seeds} ================")

    one = runs[runs["part"] == 1]
    order = [PLAIN_NORMAL, PLAIN_ATTACK] + [f"{f} {kind}" for f in c.ENCODING_FAMILIES
                                            for kind in ("normal", "attacks")]
    table = one.groupby("input_set")[["flagged_pct", "coverage_pct", "mean_margin", "baseline",
                                      "attack_evidence", "normal_evidence"]].mean().reindex(order)
    print("\nPart 1. Condition A on plain and encoded test inputs (mean over seeds)")
    print(table.round(2).to_string())
    first = one[one["input_set"] == PLAIN_NORMAL]
    print(f"  word-boundary n-gram: weight {first['boundary_weight'].mean():.2f}; a word the "
          f"detector does not know has margin baseline + that weight = "
          f"{(first['baseline'] + first['boundary_weight']).mean():+.2f}")
    for kind in ("rise by group", "separation"):
        g = gaps[gaps["comparison"].str.startswith(kind + ":")]
        wide = g.pivot_table(index="comparison", columns="part", values="value", sort=False)
        wide.index = [i.split(": ", 1)[1] for i in wide.index]
        print(f"\nPart 1. {kind.capitalize()}, in margin units (mean over seeds)")
        print(wide[GROUPS + ["(gap)"]].round(2).to_string())
    every = load_plain("facts")
    data = every[every["model"].isna()].pivot(index="subject", columns="measure", values="value")
    order = list(dict.fromkeys(every[every["model"].isna()]["subject"]))
    print("\nThe data: share of rows that contain each kind of character (%), and share of "
          "characters that are upper-case letters")
    print(data.reindex(order)[["rows", "upper_case_letter_pct", "percent_sign_pct",
                               "url_escape_pct", "equals_sign_pct", "digit_pct",
                               "whitespace_pct", "upper_case_chars_pct"]].to_string())

    two = runs[(runs["part"] == 2)]
    wide = pd.DataFrame({
        "attacks detected %": mean_sd(two[two["input_set"] == f"{HELD_OUT} attacks"],
                                      ["condition"], "flagged_pct"),
        "normal flagged %": mean_sd(two[two["input_set"] == f"{HELD_OUT} normal"],
                                    ["condition"], "flagged_pct"),
        "AUC": two[two["input_set"] == f"{HELD_OUT} attacks"].groupby(
            "condition", sort=False)["auc"].mean().round(3)})
    print(f"\nPart 2. {HELD_OUT} unseen: knock-in and knock-out (mean ± SD over seeds)")
    print(wide.to_string())

    shown_gaps = [f"gap: B minus A on {HELD_OUT} attacks", f"gap: B minus A on {HELD_OUT} normal",
                  f"gap: D minus B on {HELD_OUT} attacks"] \
        + [f"gap of {step} on {HELD_OUT} attacks" for step in STEPS]
    g = gaps[gaps["comparison"].isin(shown_gaps)]
    wide = g.pivot_table(index="comparison", columns="part", values="value", sort=False)
    print("\nGaps in mean margin, split into their parts (mean over seeds, margin units)")
    wide = wide[GROUPS + ["(intercept)", "(threshold)", "(gap)"]].reindex(shown_gaps)
    wide.index = ["B-A attacks", "B-A normal", "D-B attacks", "step 1", "step 2", "step 3"]
    print(wide.round(3).T.to_string())

    three = runs[(runs["part"] == 3)]
    wide = pd.DataFrame({
        "attacks detected %": mean_sd(three[three["input_set"] == f"{HELD_OUT} attacks"],
                                      ["condition"], "flagged_pct"),
        "normal flagged %": mean_sd(three[three["input_set"] == f"{HELD_OUT} normal"],
                                    ["condition"], "flagged_pct")})
    print("\nPart 3. From B to D in three steps (mean ± SD over seeds)")
    print(wide.reindex(["B", "B without reverted", "D without reverted", "D"]).to_string())
    facts = load("facts", model).pivot_table(index="seed", columns=["subject", "measure"],
                                             values="value")
    fact = lambda subject, measure: facts[(subject, measure)]
    print(f"  variants of {', '.join(REVERTED)} equal to the original attack after "
          f"normalisation: {fact('normaliser', 'reverted_equal_original_pct').min():.1f}% "
          f"({int(fact('normaliser', 'reverted_variants').iloc[0])} variants per seed)")
    print(f"  {HELD_OUT} inputs left unchanged by the normaliser: attacks "
          f"{fact('normaliser', 'held_attacks_unchanged_pct').min():.1f}%, normal values "
          f"{fact('normaliser', 'held_normal_unchanged_pct').min():.1f}%")
    print("  training rows rewritten by the normaliser: normal values "
          f"{fact('normaliser', 'train_normal_changed_pct').mean():.1f}%, attacks "
          f"{fact('normaliser', 'train_sqli_changed_pct').mean():.1f}%; variants: " + ", ".join(
              f"{g} {fact(f'variants of {g}', 'changed_by_normaliser_pct').mean():.1f}%"
              for g in OTHERS))
    print("\nLetter case (descriptive, mean over seeds)")
    print("  n-grams with an upper-case letter: "
          f"{fact('letter case', 'upper_case_ngrams_in_A').mean():.0f} in condition A, "
          f"{fact('letter case', 'upper_case_ngrams_in_B').mean():.0f} in condition B, of which "
          f"{fact('letter case', 'upper_case_ngrams_in_B_positive_pct').mean():.1f}% have a "
          "positive weight")
    print("  training variants that contain an upper-case letter: " + ", ".join(
        f"{g} {fact(f'variants of {g}', 'with_upper_case_pct').mean():.1f}%" for g in OTHERS))
    print("  normal test values changed by each encoding: " + ", ".join(
        f"{f} {fact(f'normal values in {f}', 'changed_pct').mean():.1f}%"
        for f in c.ENCODING_FAMILIES))

    four = runs[(runs["part"] == 4)]
    att = four[four["input_set"] == f"{HELD_OUT} attacks"]
    norm = four[four["input_set"] == f"{HELD_OUT} normal"]
    print("\nPart 4. Every value of C: attacks detected % (mean over seeds)")
    print(att.pivot_table(index="C", columns="condition", values="flagged_pct").round(2).to_string())
    print("Part 4. Every value of C: normal values flagged % (mean over seeds)")
    print(norm.pivot_table(index="C", columns="condition", values="flagged_pct").round(2).to_string())

    print("\nPaired tests")
    print(tests.drop(columns="model").to_string(index=False))
    print("\nVerdicts of the decision rules, each with the figures it rests on")
    for rule in rules.itertuples():
        print(f"{rule.rule:12} {rule.subject:17} {rule.verdict}\n{'':12} {rule.evidence}")
    print(f"\nStored tables, in results/: {', '.join(FILES.values())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, default=config.PRIMARY_MODEL,
                        help="linear model to analyse (default: %(default)s)")
    parser.add_argument("--seeds", type=int, nargs="+", choices=config.SEEDS,
                        default=list(config.SEEDS), help="seeds to run (default: all)")
    parser.add_argument("--summary", action="store_true",
                        help="print the tables and verdicts from the saved files, and exit")
    args = parser.parse_args()
    if args.summary:
        summarise(args.model, new_tests=False)
        return

    tampers = c.load_tampers()
    save(FILES["facts"], dataset_facts(tampers), None)
    passed = 0
    for seed in args.seeds:
        tables = run_seed(tampers, seed, args.model)
        ok, lines = check_seed(tables, args.model, seed)
        print("\n".join(lines), flush=True)
        passed += ok
        for name in ("runs", "ngrams", "facts"):
            save(FILES[name], tables[name], args.model, [seed])
    summarise(args.model, new_tests=True)
    if passed == len(args.seeds):
        print(f"\nWEIGHT ANALYSIS CHECK OK ({passed}/{len(args.seeds)} seeds reproduce "
              "results/experiment.csv)")
    else:
        print(f"\nWEIGHT ANALYSIS CHECK FAILED ({passed}/{len(args.seeds)} seeds passed). "
              "Read the lines marked DIFFERENT or FAILED above.")


if __name__ == "__main__":
    main()
