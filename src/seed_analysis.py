"""Supporting analysis: the experiment with other seeds."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import common as c
import config
from experiment import CONDITIONS

OUTPUT = "seed_analysis.csv"
SEED_SETS = {
    "project": tuple(config.SEEDS),
    "next": tuple(range(len(config.SEEDS), 2 * len(config.SEEDS))),
    "large": (101, 2027, 30011, 400009, 5000011),
}
TARGETS = {"family": "base64encode", "group": "encoding_group"}
SOURCES = ("sample", "tamper", "model")
KEYS = ["model", "protocol", "held_out", "seed", "condition"]


def run_one(model: str, protocol: str, sample_seed: int, tamper_seed: int, model_seed: int,
            data: tuple, tampers: dict) -> list[dict]:
    train, val, test = data
    c.set_seed(tamper_seed)
    test_attacks = test[test["y"] == 1]["payload"].tolist()
    test_normal = test[test["y"] == 0]["payload"].tolist()
    rng = np.random.RandomState(sample_seed)
    train_attacks = train[train["y"] == 1]["payload"].tolist()
    augment_base = list(rng.choice(train_attacks, min(config.N_AUGMENT, len(train_attacks)),
                                   replace=False))
    if protocol == "group":
        units = {"encoding_group": c.ENCODING_FAMILIES, "rewriting_group": c.REWRITING_FAMILIES}
    else:
        units = {f: [f] for f in c.FAMILIES}

    rows = []
    for name, excluded in units.items():
        others = [f for f in c.FAMILIES if f not in excluded]
        attack_eval = [c.apply_tamper(tampers, f, p) for f in excluded for p in test_attacks]
        augment_variants = [c.apply_tamper(tampers, g, p) for g in others for p in augment_base]
        if name != TARGETS[protocol]:
            continue
        for cond, (augment, normalised) in CONDITIONS.items():
            X = train["payload"].tolist()
            y = train["y"].tolist()
            if augment:
                X += augment_variants
                y += [1] * len(augment_variants)
            vectoriser, estimator = c.build_detector(X, np.array(y), normalised=normalised,
                                                     model=model, seed=model_seed)
            thr = c.threshold_at_fpr(vectoriser, estimator, val["payload"], val["y"].values,
                                     normalised=normalised)
            s_attack = c.score(vectoriser, estimator, attack_eval, normalised=normalised)
            s_clean = c.score(vectoriser, estimator, test["payload"], normalised=normalised)
            s_normal = c.score(vectoriser, estimator, test_normal, normalised=normalised)
            rows.append({"model": model, "protocol": protocol, "held_out": name,
                         "sample_seed": sample_seed, "tamper_seed": tamper_seed,
                         "model_seed": model_seed, "condition": cond,
                         "detection_pct": round((s_attack >= thr).mean() * 100, 2),
                         "clean_f1": round(f1_score(test["y"], (s_clean >= thr).astype(int)), 4),
                         "clean_fpr_pct": round((s_normal >= thr).mean() * 100, 3)})
            del vectoriser, estimator
        break
    return rows


def load_done() -> pd.DataFrame:
    path = c.results_path(OUTPUT)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def plan(model: str) -> list[tuple]:
    runs = [(p, s, s, s) for seeds in SEED_SETS.values() for s in seeds for p in TARGETS]
    for source in SOURCES:
        for value in config.SEEDS:
            triple = tuple(value if source == name else 0 for name in SOURCES)
            runs.append(("family",) + triple)
    return list(dict.fromkeys(runs))


def run_model(model: str) -> None:
    tampers = c.load_tampers()
    data = (c.load_data("train"), c.load_data("val"), c.load_data("test"))
    runs = plan(model)
    for i, (protocol, sample, tamper, mseed) in enumerate(runs, 1):
        done = load_done()
        if len(done) and ((done["model"] == model) & (done["protocol"] == protocol)
                          & (done["sample_seed"] == sample) & (done["tamper_seed"] == tamper)
                          & (done["model_seed"] == mseed)).any():
            continue
        rows = pd.DataFrame(run_one(model, protocol, sample, tamper, mseed, data, tampers))
        pd.concat([done, rows], ignore_index=True).to_csv(c.results_path(OUTPUT), index=False)
        print(f"{model} | run {i} of {len(runs)}: {protocol}, seeds sample={sample} "
              f"tamper={tamper} model={mseed}: done", flush=True)
    check_against_experiment(model)


def check_against_experiment(model: str) -> None:
    path = c.results_path("experiment.csv")
    done = load_done()
    if not os.path.exists(path) or done.empty:
        print("results/experiment.csv not found: the check against the main experiment was skipped.")
        return
    same_seed = (done["sample_seed"] == done["tamper_seed"]) & (done["tamper_seed"] == done["model_seed"])
    mine = done[(done["model"] == model) & same_seed
                & done["sample_seed"].isin(SEED_SETS["project"])].rename(columns={"sample_seed": "seed"})
    both = mine.merge(pd.read_csv(path), on=KEYS, suffixes=("", "_experiment"))
    if both.empty:
        print("results/experiment.csv has no rows to compare with: check skipped.")
        return
    same = ((both["detection_pct"] - both["detection_pct_experiment"]).abs() < 0.011) \
        & ((both["clean_fpr_pct"] - both["clean_fpr_pct_experiment"]).abs() < 0.0011)
    print(f"Check against results/experiment.csv ({model}): {int(same.sum())} of {len(both)} rows match.")
    if not same.all():
        print(both.loc[~same, KEYS + ["detection_pct", "detection_pct_experiment"]].to_string(index=False))


def summary() -> None:
    done = load_done()
    if done.empty:
        raise SystemExit("results/seed_analysis.csv not found. Run the analysis first.")
    pd.set_option("display.width", 200)
    same_seed = (done["sample_seed"] == done["tamper_seed"]) & (done["tamper_seed"] == done["model_seed"])
    sets = done[same_seed].copy()
    label = {s: name for name, seeds in SEED_SETS.items() for s in seeds}
    sets["seed_set"] = sets["sample_seed"].map(label)
    print("Part 1 - detection (%) by seed set: mean and standard deviation over five seeds")
    table = sets.dropna(subset=["seed_set"]).groupby(
        ["model", "held_out", "condition", "seed_set"])["detection_pct"].agg(["mean", "std"])
    print(table.unstack("seed_set").round(2).to_string())
    print("\nAll fifteen seeds together")
    print(sets.groupby(["model", "held_out", "condition"])["detection_pct"]
          .agg(["count", "mean", "std", "min", "max"]).round(2).to_string())
    print("\nPart 2 - one source varied at a time (held out: base64encode): standard deviation")
    family = done[done["protocol"] == "family"]
    rows = []
    for source in SOURCES:
        others = [f"{s}_seed" for s in SOURCES if s != source]
        part = family[(family[others[0]] == 0) & (family[others[1]] == 0)
                      & family[f"{source}_seed"].isin(config.SEEDS)]
        sd = part.groupby(["model", "condition"])["detection_pct"].std().rename(source)
        rows.append(sd)
    print(pd.concat(rows, axis=1).round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence for the choice of seeds.")
    parser.add_argument("--model", choices=config.MODELS, default=config.PRIMARY_MODEL)
    parser.add_argument("--summary", action="store_true",
                        help="print the tables for the runs made so far, and exit")
    args = parser.parse_args()
    if args.summary:
        summary()
        return
    run_model(args.model)


if __name__ == "__main__":
    main()
