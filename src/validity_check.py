"""Step 10: checks whether tampered payloads remain valid attacks."""

from __future__ import annotations

import argparse
import csv
import urllib.parse

import numpy as np
import sqlglot

import common as c
import config


def parses_ok(text: str) -> bool:
    try:
        sqlglot.parse_one(text)
        return True
    except Exception:
        return False


def reaches_backend_as_attack(family: str, original: str, variant: str) -> bool:
    if family in c.ENCODING_FAMILIES:
        return urllib.parse.unquote(variant) == original
    return True


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    tampers = c.load_tampers()
    c.set_seed(config.SEEDS[0])
    test = c.load_data("test")
    attacks = test[test["y"] == 1]["payload"].tolist()
    rng = np.random.RandomState(config.SEEDS[0])
    sample = [attacks[i] for i in rng.choice(len(attacks), config.VALIDITY_SAMPLE, replace=False)]

    original_rate = np.mean([parses_ok(p) for p in sample]) * 100
    rows = [["original (no tamper)", round(original_rate, 1), 100.0]]
    for family in c.FAMILIES:
        syntactic, backend = [], []
        for p in sample:
            v = c.apply_tamper(tampers, family, p)
            syntactic.append(parses_ok(v))
            backend.append(reaches_backend_as_attack(family, p, v))
        rows.append([family, round(np.mean(syntactic) * 100, 1), round(np.mean(backend) * 100, 1)])

    with open(c.results_path("validity.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["family", "syntactically_valid_pct", "reaches_backend_pct"])
        w.writerows(rows)

    print(f"{'family':22} {'syntactic':>10} {'backend':>9}")
    for r in rows:
        print(f"{r[0]:22} {r[1]:>9}% {r[2]:>8}%")
    print("\nThe first row is the reference: untampered payloads are injection fragments, not")
    print("complete queries, so their syntactic rate is low too. A family with a low backend")
    print("rate evades the detector but may not reach the backend as an attack.")


if __name__ == "__main__":
    main()
