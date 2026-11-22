"""Step 8: exports the primary model to the demonstration website."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

import common as c
import config

PARITY_OUTPUT = os.path.join(c.ROOT, "demo", "tests", "parity_cases.json")
OUTPUT = os.path.join(c.ROOT, "demo", "security", "model.json")


def primary_model():
    kind = config.PRIMARY_MODEL
    c.set_seed(config.SEEDS[0])
    train, val = c.load_data("train"), c.load_data("val")
    vectoriser, model = c.build_detector(train["payload"], train["y"].values, model=kind,
                                         params=c.model_params(kind), seed=config.SEEDS[0])
    thr = c.threshold_at_fpr(vectoriser, model, val["payload"], val["y"].values)
    path = c.results_path("model_comparison.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run train_models.py first.")
    stored = pd.read_csv(path)
    stored = stored[(stored["model"] == kind) & (stored["split"] == "70/15/15")]["threshold"]
    if len(stored) != 1 or round(thr, 4) != round(float(stored.iloc[0]), 4):
        raise SystemExit(f"The threshold of the model trained here ({thr:.4f}) is not the one in "
                         f"{path}. Run train_models.py again, then this script.")
    return vectoriser, model, thr


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    vectoriser, model, thr = primary_model()

    exported = {
        "ngram_min": config.NGRAM_RANGE[0],
        "ngram_max": config.NGRAM_RANGE[1],
        "analyzer": "char_wb",
        "lowercase": False,
        "threshold": thr,
        "intercept": float(model.intercept_[0]),
        "vocabulary": {k: int(v) for k, v in vectoriser.vocabulary_.items()},
        "idf": [float(x) for x in vectoriser.idf_],
        "coef": [float(x) for x in model.coef_[0]],
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as fh:
        json.dump(exported, fh)
    print(f"Model exported to {OUTPUT}")
    print(f"  vocabulary: {len(exported['vocabulary'])} n-grams | threshold: {thr:.4f}")

    test = c.load_data("test")
    hand_picked = ["' OR 1=1 --", "researcher", "o'neil"]
    texts = test["payload"].sample(300, random_state=1).tolist() + hand_picked
    scores = c.score(vectoriser, model, texts)
    with open(PARITY_OUTPUT, "w") as fh:
        json.dump([{"text": t, "py": float(sc)} for t, sc in zip(texts, scores, strict=True)], fh)
    print(f"{len(texts)} parity cases written to {PARITY_OUTPUT}")


if __name__ == "__main__":
    main()
