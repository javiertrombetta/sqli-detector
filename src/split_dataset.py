"""Step 5: splits the dataset into train, val and test."""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

import common as c
import config


def split(full: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    p_train, p_val, p_test = config.SPLIT
    train, rest = train_test_split(full, test_size=p_val + p_test, stratify=full["y"],
                                   random_state=seed)
    val, test = train_test_split(rest, test_size=p_test / (p_val + p_test), stratify=rest["y"],
                                 random_state=seed)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    full = c.load_source_all()
    before = len(full)
    full = full.drop_duplicates(subset="payload").reset_index(drop=True)
    print(f"{before} rows read; {before - len(full)} duplicate payloads removed")
    sets = split(full, config.SEEDS[0])
    seen = [set(sets[name]["payload"]) for name in c.SPLITS]
    if seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]:
        raise SystemExit("The three sets share payloads; they must be disjoint.")

    print(f"{'set':6} {'rows':>6} {'normal':>7} {'sqli':>6} {'share':>6}")
    for name in c.SPLITS:
        df = sets[name]
        os.makedirs(os.path.dirname(c.split_path(name)), exist_ok=True)
        df.to_csv(c.split_path(name), index=False)
        print(f"{name:6} {len(df):>6} {int((df.y == 0).sum()):>7} {int((df.y == 1).sum()):>6} "
              f"{len(df) / len(full):>6.1%}")
    print(f"\n{len(full)} rows in total, disjoint sets, written to {c.DATA}")


if __name__ == "__main__":
    main()
