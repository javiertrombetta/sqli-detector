"""Step 7: trains and compares the three models."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import common as c
import config

O1_TARGET_F1 = 0.95
TOP_WEIGHTS = 20


def evaluate(vec, est, thr, test: pd.DataFrame) -> dict:
    scores = c.score(vec, est, test["payload"])
    y = test["y"].values
    at_thr = (scores >= thr).astype(int)
    return {
        "f1_test": round(float(f1_score(y, at_thr)), 4),
        "f1_test_t05": round(float(f1_score(y, (scores >= config.DEFAULT_THRESHOLD).astype(int))),
                             4),
        "detection_test_pct": round(float((scores[y == 1] >= thr).mean()) * 100, 2),
        "fpr_test_pct": round(float(at_thr[y == 0].mean()) * 100, 3),
    }


def f1_at(vec, est, thr, frame: pd.DataFrame) -> float:
    scores = c.score(vec, est, frame["payload"])
    return round(float(f1_score(frame["y"].values, (scores >= thr).astype(int))), 4)


def top_weights(kind: str, vec, est) -> list[dict]:
    names = vec.get_feature_names_out()
    rows = []
    if hasattr(est, "coef_"):
        w = est.coef_[0]
        order = np.argsort(w)
        for rank, i in enumerate(order[::-1][:TOP_WEIGHTS], 1):
            rows.append({"model": kind, "measure": "weight", "direction": "attack", "rank": rank,
                         "ngram": names[i], "value": round(float(w[i]), 4)})
        for rank, i in enumerate(order[:TOP_WEIGHTS], 1):
            rows.append({"model": kind, "measure": "weight", "direction": "normal", "rank": rank,
                         "ngram": names[i], "value": round(float(w[i]), 4)})
    else:
        imp = est.feature_importances_
        for rank, i in enumerate(np.argsort(imp)[::-1][:TOP_WEIGHTS], 1):
            rows.append({"model": kind, "measure": "importance", "direction": "unsigned",
                         "rank": rank, "ngram": names[i], "value": round(float(imp[i]), 5)})
    return rows


def official_split() -> dict | None:
    try:
        train_full, test = c.load_source("train"), c.load_source("test")
    except ValueError:
        return None
    val_share = config.SPLIT[1] / (config.SPLIT[0] + config.SPLIT[1])
    train, val = train_test_split(train_full, test_size=val_share, stratify=train_full["y"],
                                  random_state=config.SEEDS[0])
    params = c.model_params(config.PRIMARY_MODEL)
    vec, est = c.build_detector(train["payload"], train["y"].values, model=config.PRIMARY_MODEL,
                                params=params, seed=config.SEEDS[0])
    thr = c.threshold_at_fpr(vec, est, val["payload"], val["y"].values)
    return {"model": config.PRIMARY_MODEL, "split": "official", "params": json.dumps(params),
            "threshold": round(thr, 4), "f1_train": f1_at(vec, est, thr, train),
            "f1_val": f1_at(vec, est, thr, val), **evaluate(vec, est, thr, test)}


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    c.set_seed(config.SEEDS[0])
    train, val, test = c.load_data("train"), c.load_data("val"), c.load_data("test")

    rows, weights, primary_metrics = [], [], None
    print("Test set (threshold chosen on val):")
    print(f"{'model':7} {'params':34} {'thr@1%':>7} {'det@1%':>7} {'F1@1%':>7} "
          f"{'F1@0.5':>7} {'FPR':>7}")
    learning = []
    for kind in config.MODELS:
        params = c.model_params(kind)
        vec, est = c.build_detector(train["payload"], train["y"].values, model=kind,
                                    params=params, seed=config.SEEDS[0])
        thr = c.threshold_at_fpr(vec, est, val["payload"], val["y"].values)
        m = evaluate(vec, est, thr, test)
        f1_train, f1_val = f1_at(vec, est, thr, train), f1_at(vec, est, thr, val)
        rows.append({"model": kind, "split": "70/15/15", "params": json.dumps(params),
                     "threshold": round(thr, 4), "f1_train": f1_train, "f1_val": f1_val, **m})
        learning.append((kind, f1_train, f1_val, m["f1_test"]))
        weights += top_weights(kind, vec, est)
        print(f"{kind:7} {json.dumps(params):34} {thr:>7.4f} {m['detection_test_pct']:>6.1f}% "
              f"{m['f1_test']:>7.4f} {m['f1_test_t05']:>7.4f} {m['fpr_test_pct']:>6.2f}%")
        if kind == config.PRIMARY_MODEL:
            primary_metrics = m

    print("\nDid each model learn in a right way? F1 at the same threshold on the three sets:")
    print(f"{'model':7} {'train':>7} {'val':>7} {'test':>7}")
    for kind, a, b, t in learning:
        print(f"{kind:7} {a:>7.4f} {b:>7.4f} {t:>7.4f}")
    print("Similar values on the three sets mean the model generalises; a train value much "
          "higher than val and test would mean overfitting. Val is slightly optimistic, because "
          "the threshold and the hyperparameters were chosen on it; test is the unbiased figure.")

    official = official_split()
    if official is None:
        print("\nThe dataset is a single file, so there is no official split to report.")
    else:
        rows.append(official)
        print(f"\nPrimary model on the split distributed with the dataset: F1 at 1% FPR "
              f"{official['f1_test']:.4f}, detection {official['detection_test_pct']:.1f}%, "
              f"observed FPR {official['fpr_test_pct']:.2f}%")

    pd.DataFrame(rows).to_csv(c.results_path("model_comparison.csv"), index=False)
    pd.DataFrame(weights).to_csv(c.results_path("feature_weights.csv"), index=False)
    primary = [w for w in weights if w["model"] == config.PRIMARY_MODEL]
    attack = [repr(w["ngram"]) for w in primary if w["direction"] == "attack"][:10]
    normal = [repr(w["ngram"]) for w in primary if w["direction"] == "normal"][:10]
    print(f"\nLargest weights of {config.PRIMARY_MODEL} "
          "(all models in results/feature_weights.csv):")
    print("  towards attack:", ", ".join(attack))
    print("  towards normal:", ", ".join(normal))
    m = primary_metrics
    print(f"\nPrimary model ({config.PRIMARY_MODEL}): clean F1 at the 1% FPR threshold "
          f"{m['f1_test']:.4f}   (O1 requires >= {O1_TARGET_F1})")
    print("O1 MET" if m["f1_test"] >= O1_TARGET_F1 else "O1 NOT MET")


if __name__ == "__main__":
    main()
