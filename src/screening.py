"""Step 9: effect of each tamper family on each model."""

from __future__ import annotations

import argparse
import csv

import numpy as np

import common as c
import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", action="store_true",
                        help="also write the disguised attacks to results/variants.csv")
    args = parser.parse_args()
    tampers = c.load_tampers()
    c.set_seed(config.SEEDS[0])
    train, val, test = c.load_data("train"), c.load_data("val"), c.load_data("test")
    attacks = test[test["y"] == 1]["payload"].tolist()

    variants = {family: [c.apply_tamper(tampers, family, p) for p in attacks]
                for family in c.FAMILIES}
    if args.variants:
        with open(c.results_path("variants.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["family", "seed", "original", "variant"])
            for family in c.FAMILIES:
                for original, variant in zip(attacks, variants[family], strict=True):
                    w.writerow([family, config.SEEDS[0], original, variant])
        print(f"{len(attacks) * len(c.FAMILIES)} disguised attacks written to "
              "results/variants.csv")

    rows = []
    for kind in config.MODELS:
        vec, est = c.build_detector(train["payload"], train["y"].values, model=kind,
                                    seed=config.SEEDS[0])
        thr = c.threshold_at_fpr(vec, est, val["payload"], val["y"].values)
        clean = c.score(vec, est, attacks)
        clean_target = (clean >= thr).mean() * 100
        clean_half = (clean >= config.DEFAULT_THRESHOLD).mean() * 100
        print(f"\n{kind}: clean detection {clean_target:.1f}% at 1% FPR, {clean_half:.1f}% at 0.5")
        print(f"{'family':18} {'%mod':>6} {'det@1%':>7} {'drop':>6} {'det@0.5':>8} {'drop':>6}")
        model_rows = []
        for family in c.FAMILIES:
            tampered = variants[family]
            modified = np.mean([v != p for v, p in zip(tampered, attacks, strict=True)]) * 100
            s = c.score(vec, est, tampered)
            det_target = (s >= thr).mean() * 100
            det_half = (s >= config.DEFAULT_THRESHOLD).mean() * 100
            model_rows.append([kind, family, round(modified, 1), round(det_target, 1),
                               round(clean_target - det_target, 1), round(det_half, 1),
                               round(clean_half - det_half, 1)])
        model_rows.sort(key=lambda r: -r[4])
        for r in model_rows:
            print(f"{r[1]:18} {r[2]:>6.1f} {r[3]:>7.1f} {r[4]:>6.1f} {r[5]:>8.1f} {r[6]:>6.1f}")
        rows += model_rows

    with open(c.results_path("screening.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["model", "family", "modified_pct", "detection_pct", "drop_pp",
                    "detection_t05_pct", "drop_t05_pp"])
        w.writerows(rows)
    print("\nSaved to results/screening.csv")


if __name__ == "__main__":
    main()
