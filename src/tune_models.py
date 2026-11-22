"""Step 6: chooses the hyperparameters of each model on val."""

from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd
from sklearn.metrics import f1_score

import common as c
import config


def combinations(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    c.set_seed(config.SEEDS[0])
    train, val = c.load_data("train"), c.load_data("val")
    val_attacks = val[val["y"] == 1]["payload"].tolist()

    rows, chosen = [], {}
    for kind in config.MODELS:
        best = None
        for params in combinations(config.PARAM_GRID[kind]):
            vec, est = c.build_detector(train["payload"], train["y"].values, model=kind,
                                        params=params, seed=config.SEEDS[0])
            thr = c.threshold_at_fpr(vec, est, val["payload"], val["y"].values)
            detection = float((c.score(vec, est, val_attacks) >= thr).mean())
            f1 = float(f1_score(val["y"], (c.score(vec, est, val["payload"]) >= thr).astype(int)))
            rows.append({"model": kind, "params": json.dumps(params), "threshold": round(thr, 4),
                         "val_detection_pct": round(detection * 100, 2), "val_f1": round(f1, 4)})
            print(f"{kind:7} {json.dumps(params):40} detection {detection:6.2%}  F1 {f1:.4f}")
            if best is None or (detection, f1) > (best[0], best[1]):
                best = (detection, f1, params)
        chosen[kind] = best[2]
        print(f"  -> {kind}: {json.dumps(best[2])}\n")

    table = pd.DataFrame(rows)
    table["selected"] = [json.loads(p) == chosen[m]
                         for m, p in zip(table["model"], table["params"], strict=True)]
    table.to_csv(c.results_path("model_selection.csv"), index=False)
    print(f"All {len(rows)} combinations saved to results/model_selection.csv; the chosen one per "
          "model is marked selected = True.")


if __name__ == "__main__":
    main()
