"""Steps 11 and 12: the four conditions on held-out tamper families, with the control."""

from __future__ import annotations

import argparse
import csv
import os
import random
import urllib.parse

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

import common as c
import config

CONDITIONS = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}
KEYS = ["model", "protocol", "held_out", "seed", "condition"]
CSV_HEADER = KEYS + ["detection_pct", "detection_t05_pct", "clean_f1", "clean_fpr_pct",
                     "normal_held_out_flagged_pct", "auc_held_out",
                     "normal_urlquote_flagged_pct", "urlquote_changed_pct"] \
    + [f"normal_{family}" for family in c.FAMILIES]


def prediction_file(model: str, seed: int) -> str:
    return c.results_path("predictions", f"experiment_{model}_{seed}.npz")


def prediction_key(held_out: str, condition: str) -> str:
    return f"{held_out}__{condition}"


def normal_variants(tampers: dict, normal: list[str]) -> dict[str, list[str]]:
    state_py, state_np = random.getstate(), np.random.get_state()
    variants = {f: [c.apply_tamper(tampers, f, p) for p in normal] for f in c.FAMILIES}
    variants["urlquote"] = [urllib.parse.quote(p, safe="") for p in normal]
    random.setstate(state_py)
    np.random.set_state(state_np)
    return variants


def run(seed: int, by_groups: bool = False, model: str = config.PRIMARY_MODEL) -> None:
    path = c.results_path("experiment.csv")
    new_file = not os.path.exists(path)
    if not new_file:
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != CSV_HEADER:
            raise SystemExit(f"{path} has an older format. Delete it (and results/predictions) "
                             "and run the experiment again.")
    tampers = c.load_tampers()
    c.set_seed(seed)
    train, val, test = c.load_data("train"), c.load_data("val"), c.load_data("test")
    test_attacks = test[test["y"] == 1]["payload"].tolist()
    test_normal = test[test["y"] == 0]["payload"].tolist()
    rng = np.random.RandomState(seed)
    train_attacks = train[train["y"] == 1]["payload"].tolist()
    augment_base = list(rng.choice(train_attacks, min(config.N_AUGMENT, len(train_attacks)),
                                   replace=False))
    harmless = normal_variants(tampers, test_normal)
    changed = np.mean([a != b for a, b in zip(harmless["urlquote"], test_normal, strict=True)])

    protocol = "group" if by_groups else "family"
    if by_groups:
        units = {"encoding_group": c.ENCODING_FAMILIES, "rewriting_group": c.REWRITING_FAMILIES}
    else:
        units = {f: [f] for f in c.FAMILIES}

    rows, predictions = [], {}
    for name, excluded in units.items():
        others = [f for f in c.FAMILIES if f not in excluded]
        attack_eval = [c.apply_tamper(tampers, f, p) for f in excluded for p in test_attacks]
        augment_variants = [c.apply_tamper(tampers, g, p) for g in others for p in augment_base]
        normal_eval = [p for f in excluded for p in harmless[f]]

        for cond, (augment, normalised) in CONDITIONS.items():
            X = train["payload"].tolist()
            y = train["y"].tolist()
            if augment:
                X += augment_variants
                y += [1] * len(augment_variants)

            vectoriser, estimator = c.build_detector(X, np.array(y), normalised=normalised,
                                                     model=model, seed=seed)
            thr = c.threshold_at_fpr(vectoriser, estimator, val["payload"], val["y"].values,
                                     normalised=normalised)

            def score(texts):
                return c.score(vectoriser, estimator, texts, normalised=normalised)

            def flagged_pct(texts, decimals=2):
                return round((score(texts) >= thr).mean() * 100, decimals)

            s_attack, s_normal_eval = score(attack_eval), score(normal_eval)
            s_clean = score(test["payload"])
            labels = np.r_[np.ones(len(s_attack)), np.zeros(len(s_normal_eval))]
            rows.append([model, protocol, name, seed, cond,
                         round((s_attack >= thr).mean() * 100, 2),
                         round((s_attack >= config.DEFAULT_THRESHOLD).mean() * 100, 2),
                         round(f1_score(test["y"], (s_clean >= thr).astype(int)), 4),
                         flagged_pct(test_normal, 3),
                         round((s_normal_eval >= thr).mean() * 100, 2),
                         round(roc_auc_score(labels, np.r_[s_attack, s_normal_eval]), 4),
                         flagged_pct(harmless["urlquote"]),
                         round(changed * 100, 1)]
                        + [flagged_pct(harmless[family]) for family in c.FAMILIES])
            predictions[prediction_key(name, cond)] = (s_attack >= thr).astype(int)
            del vectoriser, estimator
        print(f"seed {seed} | {name}: done", flush=True)

    with open(path, "a", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerows(rows)
    stored = {}
    if os.path.exists(prediction_file(model, seed)):
        with np.load(prediction_file(model, seed)) as previous:
            stored = {key: previous[key] for key in previous.files}
    np.savez_compressed(prediction_file(model, seed), **{**stored, **predictions})
    print(f"seed {seed}: {len(rows)} rows appended to results/experiment.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seed", type=int, choices=config.SEEDS, help="random seed for this run")
    parser.add_argument("--groups", action="store_true",
                        help="withhold whole transformation groups instead of single families")
    parser.add_argument("--model", choices=config.MODELS, default=config.PRIMARY_MODEL,
                        help="classifier to use (default: the primary model, %(default)s)")
    args = parser.parse_args()
    run(args.seed, by_groups=args.groups, model=args.model)


if __name__ == "__main__":
    main()
