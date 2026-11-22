"""Compares the tables of weight_analysis.py with those of another run."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import common as c

FILES = {
    "weight_analysis.csv": ["model", "seed", "part", "condition", "C", "input_set"],
    "weight_analysis_ngrams.csv": ["model", "seed", "comparison", "ngram"],
    "weight_analysis_facts.csv": ["model", "seed", "subject", "measure"],
    "weight_analysis_tests.csv": ["model", "part", "comparison"],
}
SKIP = {"rank", "residual"}
RANKED = {"weight_analysis_ngrams.csv"}
TOLERANCE_PCT, TOLERANCE_OTHER = 0.011, 0.001


def tolerance(column: str) -> float:
    return TOLERANCE_PCT if "pct" in column or column.endswith("_pp") else TOLERANCE_OTHER


def compare(name: str, keys: list[str], mine: pd.DataFrame, ref: pd.DataFrame) -> list[str]:
    problems = []
    fill = {k: (-1 if mine[k].dtype.kind in "fiu" or ref[k].dtype.kind in "fiu" else "")
            for k in keys}
    mine, ref = mine.fillna(fill), ref.fillna(fill)
    both = mine.merge(ref, on=keys, how="outer", suffixes=("", "__ref"), indicator=True)
    only_mine = int((both["_merge"] == "left_only").sum())
    only_ref = int((both["_merge"] == "right_only").sum())
    if (only_mine or only_ref) and name in RANKED:
        print(f"   note: {only_mine} n-grams at the edge of a top-20 list are only in this run, "
              f"{only_ref} only in the reference")
    elif only_mine or only_ref:
        problems.append(f"{only_mine} rows only in this run, {only_ref} rows only in the reference")
    both = both[both["_merge"] == "both"]
    for column in [col for col in mine.columns if col not in keys and col not in SKIP]:
        if column + "__ref" not in both.columns:
            problems.append(f"column {column} is not in the reference")
            continue
        a, b = both[column], both[column + "__ref"]
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            a, b = a.astype(float), b.astype(float)
            differs = ~(np.isclose(a, b, rtol=0, atol=tolerance(column))
                        | (a.isna() & b.isna()) | (np.isinf(a) & np.isinf(b) & (a == b)))
        else:
            differs = ~((a.astype(str) == b.astype(str)) | (a.isna() & b.isna()))
        if differs.any():
            first = both.loc[differs, keys + [column, column + "__ref"]].head(3)
            problems.append(f"column {column}: {int(differs.sum())} rows differ, for example\n"
                            + first.to_string(index=False))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True,
                        help="folder with the tables of the other run")
    args = parser.parse_args()
    if not os.path.isdir(args.reference):
        raise SystemExit(f"Reference folder not found: {args.reference}")

    clean = True
    for name, keys in FILES.items():
        mine_path, ref_path = c.results_path(name), os.path.join(args.reference, name)
        if not os.path.exists(mine_path) or not os.path.exists(ref_path):
            missing = "results/" if not os.path.exists(mine_path) else "the reference folder"
            print(f"{name}: not found in {missing}")
            clean = False
            continue
        mine, ref = pd.read_csv(mine_path), pd.read_csv(ref_path)
        ref = ref[ref["model"].isin(mine["model"].unique()) | ref["model"].isna()]
        problems = compare(name, keys, mine, ref)
        print(f"{name}: {len(mine)} rows, " + ("match" if not problems else "DIFFERENT"))
        for line in problems:
            print("   " + line.replace("\n", "\n   "))
        clean &= not problems
    print("\nAll files match the reference" if clean else
          "\nSome files differ from the reference: see the lines marked DIFFERENT above.")


if __name__ == "__main__":
    main()
