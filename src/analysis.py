"""Step 13: summary tables and paired tests of the main experiment."""

from __future__ import annotations

import argparse
import functools
import os
import zlib

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

import common as c
import config
from experiment import CSV_HEADER, prediction_file, prediction_key

CONDITIONS = ["A", "B", "C", "D"]
COMPARISONS = [("B", "A"), ("C", "A"), ("D", "B"), ("C", "B")]
GROUPS = ["encoding_group", "rewriting_group"]
PROTOCOLS = (("family", None), ("group", GROUPS))


@functools.lru_cache(maxsize=None)
def stored_predictions(model: str, seed: int) -> dict[str, np.ndarray]:
    path = prediction_file(model, seed)
    if not os.path.exists(path):
        return {}
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_predictions(model: str, name: str, seed: int, cond: str) -> np.ndarray:
    return stored_predictions(model, seed)[prediction_key(name, cond)]


def have_predictions(model: str, units: list[str], seeds: list[int],
                     conds=tuple(CONDITIONS)) -> bool:
    return all(prediction_key(u, k) in stored_predictions(model, s)
               for u in units for s in seeds for k in conds)


def load_results(name: str = "experiment.csv") -> pd.DataFrame:
    path = c.results_path(name)
    df = pd.read_csv(path)
    if list(df.columns) != CSV_HEADER:
        raise SystemExit(f"{path} has an older format. "
                         "Delete it (and results/predictions) and run experiment.py again.")
    return df.drop_duplicates(subset=["model", "protocol", "held_out", "seed", "condition"],
                              keep="last")


def models_in(df: pd.DataFrame) -> list[str]:
    present = set(df["model"])
    return [m for m in config.MODELS if m in present] + sorted(present - set(config.MODELS))


def mean_sd_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    stats = df.groupby(["held_out", "condition"])[metric].agg(["mean", "std"]).fillna(0)
    cells = stats.apply(lambda r: f"{r['mean']:.1f} ± {r['std']:.1f}", axis=1)
    return cells.unstack().reindex(columns=CONDITIONS)


def discordant(old: np.ndarray, new: np.ndarray) -> tuple[int, int]:
    return int(((old == 0) & (new == 1)).sum()), int(((old == 1) & (new == 0)).sum())


def mcnemar_p(n01: int, n10: int) -> float:
    return 1.0 if n01 + n10 == 0 else float(mcnemar([[0, n10], [n01, 0]], exact=True).pvalue)


def bootstrap_ci(differences: np.ndarray, rng: np.random.RandomState) -> tuple[float, float]:
    n = len(differences)
    means = [differences[rng.randint(0, n, n)].mean() for _ in range(config.BOOTSTRAP_RESAMPLES)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_significance(rows_spec, seeds: list[int]) -> pd.DataFrame:
    corrected = {spec[0]: [] for spec in rows_spec}
    pooled = {spec[0]: [0, 0] for spec in rows_spec}
    for seed in seeds:
        pvalues = []
        for label, new_loader, old_loader in rows_spec:
            n01, n10 = discordant(old_loader(seed), new_loader(seed))
            pooled[label][0] += n01
            pooled[label][1] += n10
            pvalues.append(mcnemar_p(n01, n10))
        for spec, p in zip(rows_spec, multipletests(pvalues, method="holm")[1], strict=True):
            corrected[spec[0]].append(float(p))

    rows = []
    for label, new_loader, old_loader in rows_spec:
        diff = np.mean([new_loader(s).astype(int) - old_loader(s).astype(int) for s in seeds],
                       axis=0)
        rng = np.random.RandomState((config.SEEDS[0] + zlib.crc32(label.encode())) % 2**32)
        low, high = bootstrap_ci(diff, rng)
        n01, n10 = pooled[label]
        significant = sum(p < config.ALPHA for p in corrected[label])
        rows.append({
            "comparison": label,
            "difference_pp": round(diff.mean() * 100, 2),
            "ci95_low_pp": round(low * 100, 2),
            "ci95_high_pp": round(high * 100, 2),
            "odds_ratio": ((round(n01 / n10, 2) if n10 else float("inf")) if n01 + n10
                           else float("nan")),
            "p_holm_median": round(float(np.median(corrected[label])), 4),
            "significant_seeds": f"{significant}/{len(seeds)}",
        })
    return pd.DataFrame(rows)


def condition_tests(model: str, units: list[str], seeds: list[int]) -> pd.DataFrame:
    spec = [(f"{unit} | {new} vs {old}",
             (lambda s, u=unit, n=new: load_predictions(model, u, s, n)),
             (lambda s, u=unit, o=old: load_predictions(model, u, s, o)))
            for unit in units for new, old in COMPARISONS]
    out = paired_significance(spec, seeds)
    out[["unit", "comparison"]] = out["comparison"].str.split(" | ", expand=True, regex=False)
    return out[["unit", "comparison", "difference_pp", "ci95_low_pp", "ci95_high_pp",
                "odds_ratio", "p_holm_median", "significant_seeds"]]


def functional_families() -> set[str]:
    path = c.results_path("validity.csv")
    if not os.path.exists(path):
        return set()
    validity = pd.read_csv(path)
    return set(validity.loc[validity["reaches_backend_pct"] > 0, "family"]) & set(c.FAMILIES)


def main_table(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for model in models_in(df):
        for threshold, metric in (("1% FPR", "detection_pct"), ("0.5", "detection_t05_pct")):
            for protocol, _ in PROTOCOLS:
                sub = df[(df["model"] == model) & (df["protocol"] == protocol)]
                if sub.empty:
                    continue
                t = mean_sd_table(sub, metric).reset_index()
                t.insert(0, "protocol", protocol)
                t.insert(0, "threshold", threshold)
                t.insert(0, "model", model)
                parts.append(t)
    return pd.concat(parts, ignore_index=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    functional = functional_families()
    subsets = {
        "detection_all_pct": c.FAMILIES,
        "detection_encoding_pct": c.ENCODING_FAMILIES,
        "detection_rewriting_pct": c.REWRITING_FAMILIES,
        "detection_normaliser_covered_pct": c.NORMALISER_COVERED,
        "detection_normaliser_not_covered_pct": [f for f in c.FAMILIES
                                                 if f not in c.NORMALISER_COVERED],
        "detection_reaches_backend_pct": sorted(functional),
    }
    rows = []
    for model in models_in(df):
        fam = df[(df["model"] == model) & (df["protocol"] == "family")]
        grp = df[(df["model"] == model) & (df["protocol"] == "group")]
        for cond in CONDITIONS:
            d = fam[fam["condition"] == cond]
            if d.empty:
                continue
            row = {"model": model, "condition": cond, "seeds": d["seed"].nunique()}
            for name, families in subsets.items():
                sel = d[d["held_out"].isin(families)]["detection_pct"]
                row[name] = round(sel.mean(), 1) if len(sel) else float("nan")
            for group in GROUPS:
                sel = grp[(grp["condition"] == cond) & (grp["held_out"] == group)]["detection_pct"]
                row[f"detection_{group}_pct"] = round(sel.mean(), 1) if len(sel) else float("nan")
            row.update({"clean_f1_mean": round(d["clean_f1"].mean(), 4),
                        "clean_f1_sd": round(d["clean_f1"].std(), 4),
                        "clean_fpr_pct_mean": round(d["clean_fpr_pct"].mean(), 4),
                        "clean_fpr_pct_sd": round(d["clean_fpr_pct"].std(), 4)})
            rows.append(row)
    return pd.DataFrame(rows)


def all_tests(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    parts, missing = [], []
    for model in models_in(df):
        for protocol, units in PROTOCOLS:
            sub = df[(df["model"] == model) & (df["protocol"] == protocol)]
            if sub.empty:
                continue
            units = units or c.FAMILIES
            seeds = sorted(sub["seed"].unique().tolist())
            if not have_predictions(model, units, seeds):
                missing.append(f"{model}/{protocol}")
                continue
            t = condition_tests(model, units, seeds).rename(columns={"unit": "held_out"})
            t.insert(0, "protocol", protocol)
            t.insert(0, "model", model)
            parts.append(t)
    return (pd.concat(parts, ignore_index=True) if parts else None), missing


def control_table(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    shown = ["detection_pct", "normal_held_out_flagged_pct", "auc_held_out",
             "normal_urlquote_flagged_pct", "clean_fpr_pct"]
    encoding_units = c.ENCODING_FAMILIES + ["encoding_group"]

    def mean(part: pd.DataFrame) -> pd.DataFrame:
        return part.groupby(["model", "condition"])[shown].mean().round(2)

    tables = [(f"Held out: {unit}", mean(df[df["held_out"] == unit]))
              for unit in encoding_units if (df["held_out"] == unit).any()]
    rest = df[~df["held_out"].isin(encoding_units)]
    if len(rest):
        tables.append(("Rewriting units (families and group), mean over units and seeds",
                       mean(rest)))
    return tables


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    df = load_results()
    for model in models_in(df):
        for protocol, _ in PROTOCOLS:
            n = df[(df["model"] == model) & (df["protocol"] == protocol)]["seed"].nunique()
            if 0 < n < len(config.SEEDS):
                print(f"NOTE: {model}/{protocol} has {n} seed(s); "
                      f"the protocol uses {len(config.SEEDS)}.")

    table = main_table(df)
    for model in models_in(df):
        for threshold, label in (("1% FPR", "the 1% FPR threshold"), ("0.5", "the threshold 0.5")):
            view = table[(table["model"] == model) & (table["threshold"] == threshold)]
            print(f"\n=== {model}: detection at {label}, mean ± SD (A none, B adversarial, "
                  "C normalisation, D both) ===")
            print(view.drop(columns=["model", "threshold"]).to_string(index=False))

    summary = summary_table(df)
    print("\n=== Summary by model and condition (detection at 1% FPR; clean F1 and FPR in %) ===")
    print(summary.to_string(index=False))

    print("\n=== Control: attacks detected and harmless disguised values flagged (in %), and "
          "AUC; mean over seeds ===")
    for title, part in control_table(df):
        print(f"\n{title}")
        print(part.to_string())

    tests, missing = all_tests(df)
    if tests is not None:
        tests.to_csv(c.results_path("tests.csv"), index=False)
        print(f"\nTests saved to results/tests.csv ({len(tests)} comparisons; Holm within each "
              "seed, separately per model and protocol)")
    for item in missing:
        print(f"NOTE: per-payload predictions for {item} not found in results/predictions/ "
              "(they are not kept in the repository); its tests were skipped.")


if __name__ == "__main__":
    main()
