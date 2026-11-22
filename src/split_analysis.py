"""Supporting analysis: dataset profile and checks of the train/val/test split."""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import train_test_split

import common as c
import config
from split_dataset import split

SYMBOLS = ["'", '"', "(", ")", "=", ",", "-", ";", "%", " "]
SIMILARITY_BINS = [0.0, 0.5, 0.7, 0.9, 0.99, 1.0001]
SIMILARITY_LABELS = ["below 0.50", "0.50 to 0.70", "0.70 to 0.90", "0.90 to 0.99", "0.99 to 1.00"]
MIN_STRATUM = 20


def corner_threshold(vec, model, val: pd.DataFrame) -> float:
    fpr, _, thresholds = roc_curve(val["y"], c.score(vec, model, val["payload"]))
    idx = np.where(fpr <= config.TARGET_FPR)[0]
    return float(thresholds[idx[-1]]) if len(idx) else 1.0


def both_thresholds(vec, model, val: pd.DataFrame) -> dict:
    return {"ROC corners (old rule)": corner_threshold(vec, model, val),
            "exact 1% FPR": c.threshold_at_fpr(vec, model, val["payload"], val["y"])}


def normal_kind(text: str) -> str:
    if re.fullmatch(r"\d+", text):
        return "digits only"
    if "@" in text:
        return "e-mail"
    if re.fullmatch(r"[^\W\d_]+", text):
        return "one word"
    if re.fullmatch(r"[^\W\d_]+([ '\-.][^\W\d_]+)+\.?", text):
        return "several words"
    if re.search(r"\d", text) and re.search(r"[^\W\d_]", text) and re.search(r"[ ,./]", text):
        return "address-like"
    if re.fullmatch(r"[0-9a-zA-Z]+", text):
        return "letters and digits"
    return "other"


def sqli_kind(text: str) -> str:
    s = text.lower()
    if "union" in s:
        return "union"
    if re.search(r";\s*(select|create|declare|begin|waitfor|call|exec)", s):
        return "stacked queries"
    if re.search(r"sleep|benchmark|waitfor|pg_sleep|dbms_|generate_series|randomblob|"
                 r"count\(\*\) from", s):
        return "time-based"
    if re.search(r"0x7171|char\(113\)|chr\(113\)|extractvalue|updatexml|xmltype|utl_inaddr|"
                 r"ctxsys|exp\(~", s):
        return "error-based"
    return "boolean-based"


def add_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["payload"] = df["payload"].astype(str)
    df["length"] = df["payload"].str.len()
    df["kind"] = [sqli_kind(p) if y == 1 else normal_kind(p) for p, y in zip(df["payload"], df["y"])]
    return df


def light_key(text: str) -> str:
    return re.sub(r"\d+", "0", re.sub(r"\s+", " ", text.lower())).strip()


def profile(full: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(section, item, value, share=np.nan):
        rows.append({"section": section, "item": item, "value": value, "share_pct": share})

    name, spec = c._active_dataset()
    if "single_file" not in spec:
        folder = c._dataset_folder(name, spec)
        raw = pd.concat([pd.read_csv(os.path.join(folder, spec[k]), keep_default_na=False)
                         .assign(source=k) for k in ("train_file", "test_file")], ignore_index=True)
        add("rows", "rows in the source files", len(raw))
        add("rows", "rows that are normal or SQL injection", len(full), 100 * len(full) / len(raw))
        add("rows", "rows of other attack types, left out", len(raw) - len(full),
            100 * (len(raw) - len(full)) / len(raw))
        add("rows", "exact duplicate payloads", int(raw[spec["text_col"]].duplicated().sum()))
        add("rows", "empty payloads", int((raw[spec["text_col"]].astype(str).str.len() == 0).sum()))
    for y, label in ((0, "normal"), (1, "SQL injection")):
        part = full[full["y"] == y]
        add("classes", label, len(part), 100 * len(part) / len(full))
        q = part["length"].quantile([0, 0.25, 0.5, 0.75, 1]).round(0).astype(int).tolist()
        for item, value in zip(("shortest", "first quartile", "median", "third quartile", "longest"), q):
            add(f"length of {label} (characters)", item, value)
        for kind, n in part["kind"].value_counts().items():
            add(f"kinds of {label}", kind, int(n), 100 * n / len(part))
        keys = part["payload"].map(light_key)
        add("near-duplicates", f"{label}: rows that share their text with another row once case, "
            "numbers and spacing are ignored", int(keys.duplicated(keep=False).sum()),
            100 * keys.duplicated(keep=False).mean())
    for symbol in SYMBOLS:
        shown = "space" if symbol == " " else symbol
        for y, label in ((0, "normal"), (1, "SQL injection")):
            part = full[full["y"] == y]
            has = part["payload"].str.contains(symbol, regex=False)
            add(f"payloads that contain the symbol {shown}", label, int(has.sum()), 100 * has.mean())
    return pd.DataFrame(rows)


def balance(sets: dict) -> pd.DataFrame:
    rows = []
    names = list(c.SPLITS)

    def categorical(characteristic, column, subset=None):
        frames = {n: (sets[n] if subset is None else sets[n][sets[n]["y"] == subset]) for n in names}
        table = pd.DataFrame({n: frames[n][column].value_counts() for n in names}).fillna(0)
        p = chi2_contingency(table.values)[1]
        for value, counts in table.iterrows():
            rows.append({"section": characteristic, "item": value,
                         **{n: 100 * counts[n] / len(frames[n]) for n in names},
                         "unit": "% of the set", "p_value": p,
                         "present_in_all_sets": bool((counts > 0).all())})

    for n in names:
        sets[n]["class"] = np.where(sets[n]["y"] == 1, "SQL injection", "normal")
    categorical("classes", "class")
    categorical("kinds of normal", "kind", 0)
    categorical("kinds of SQL injection", "kind", 1)
    for y, label in ((0, "normal"), (1, "SQL injection")):
        lengths = {n: sets[n][sets[n]["y"] == y]["length"] for n in names}
        p_val = ks_2samp(lengths["train"], lengths["val"]).pvalue
        p_test = ks_2samp(lengths["train"], lengths["test"]).pvalue
        for item, q in (("first quartile", 0.25), ("median", 0.5), ("third quartile", 0.75)):
            rows.append({"section": f"length of {label} (characters)", "item": item,
                         **{n: float(lengths[n].quantile(q)) for n in names},
                         "unit": "characters", "p_value": min(p_val, p_test),
                         "p_train_vs_val": p_val, "p_train_vs_test": p_test,
                         "present_in_all_sets": True})
    return pd.DataFrame(rows)


def similarity(sets: dict, kind: str, seed: int) -> pd.DataFrame:
    train, val, test = (sets[n] for n in c.SPLITS)
    vec, model = c.build_detector(train["payload"], train["y"], model=kind, seed=seed)
    test_scores = c.score(vec, model, test["payload"])
    rows = []
    for rule, thr in both_thresholds(vec, model, val).items():
        flagged = test_scores >= thr
        for y, label in ((1, "SQL injection"), (0, "normal")):
            mask = (test["y"] == y).values
            a = vec.transform(test["payload"][mask])
            b = vec.transform(train["payload"][train["y"] == y])
            nearest = np.asarray((a @ b.T).max(axis=1).todense()).ravel()
            keys = set(train["payload"][train["y"] == y].map(light_key))
            twin = test["payload"][mask].map(light_key).isin(keys).values
            hits = flagged[mask]
            bins = pd.cut(nearest, SIMILARITY_BINS, labels=SIMILARITY_LABELS, right=False)
            groups = [("all test rows", np.ones(len(nearest), bool)),
                      ("has a near-duplicate in train (same text but for case, numbers, spacing)", twin),
                      ("has no near-duplicate in train", ~twin)]
            groups += [(f"nearest training row at similarity {b}", np.asarray(bins == b))
                       for b in SIMILARITY_LABELS]
            for name, pick in groups:
                rows.append({"model": kind, "threshold_rule": rule, "threshold": thr,
                             "class": label, "group": name, "rows": int(pick.sum()),
                             "share_pct": 100 * pick.mean(),
                             "median_similarity": float(np.median(nearest[pick])) if pick.any() else np.nan,
                             "flagged_as_attack_pct": 100 * hits[pick].mean() if pick.any() else np.nan})
    return pd.DataFrame(rows)


def unseen_kind(sets: dict, kind: str, seed: int) -> pd.DataFrame:
    train, val, test = (sets[n] for n in c.SPLITS)
    vec, model = c.build_detector(train["payload"], train["y"], model=kind, seed=seed)
    reference_scores = c.score(vec, model, test["payload"])
    reference_thr = both_thresholds(vec, model, val)
    rows = []
    for held in sorted(train[train["y"] == 1]["kind"].unique()):
        keep_train = train[~((train["y"] == 1) & (train["kind"] == held))]
        keep_val = val[~((val["y"] == 1) & (val["kind"] == held))]
        v, m = c.build_detector(keep_train["payload"], keep_train["y"], model=kind, seed=seed)
        scores = c.score(v, m, test["payload"])
        target = ((test["y"] == 1) & (test["kind"] == held)).values
        for rule, thr in both_thresholds(v, m, keep_val).items():
            hits = scores >= thr
            rows.append({"model": kind, "threshold_rule": rule,
                         "kind_removed_from_training": held,
                         "training_attacks_removed": int(len(train) - len(keep_train)),
                         "test_attacks_of_that_kind": int(target.sum()),
                         "detected_when_seen_pct": 100 * (reference_scores >= reference_thr[rule])[target].mean(),
                         "detected_when_unseen_pct": 100 * hits[target].mean(),
                         "false_positive_rate_pct": 100 * hits[(test["y"] == 0).values].mean()})
    return pd.DataFrame(rows)


def richer_split(full: pd.DataFrame, seed: int) -> dict:
    quartile = full.groupby("y")["length"].transform(lambda s: pd.qcut(s, 4, labels=False,
                                                                         duplicates="drop"))
    strata = full["y"].astype(str) + "|" + full["kind"] + "|" + quartile.astype(str)
    small = strata.map(strata.value_counts()) < MIN_STRATUM
    strata = strata.where(~small, full["y"].astype(str) + "|merged")
    still_small = strata.map(strata.value_counts()) < MIN_STRATUM
    for y in full.loc[still_small, "y"].unique():
        largest = strata[(full["y"] == y) & ~still_small].value_counts().index[0]
        strata = strata.where(~(still_small & (full["y"] == y)), largest)
    p_train, p_val, p_test = config.SPLIT
    train, rest, _, rest_strata = train_test_split(full, strata, test_size=p_val + p_test,
                                                   stratify=strata, random_state=seed)
    val, test = train_test_split(rest, test_size=p_test / (p_val + p_test),
                                 stratify=rest_strata, random_state=seed)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def repeats(full: pd.DataFrame, kind: str, tampers: dict) -> pd.DataFrame:
    strategies = {"by class (the project's split)": split,
                  "by class, kind of value and length": richer_split}
    rows = []
    for strategy, make in strategies.items():
        for seed in config.SEEDS:
            c.set_seed(seed)
            parts = make(full, seed)
            train, val, test = (parts[n] for n in c.SPLITS)
            vec, model = c.build_detector(train["payload"], train["y"], model=kind, seed=seed)
            test_scores = c.score(vec, model, test["payload"])
            val_scores = c.score(vec, model, val["payload"])
            attacks = test[test["y"] == 1]["payload"].tolist()
            family_scores = {f: c.score(vec, model, [c.apply_tamper(tampers, f, p) for p in attacks])
                             for f in c.FAMILIES}
            for rule, thr in both_thresholds(vec, model, val).items():
                pred = (test_scores >= thr).astype(int)
                row = {"model": kind, "strategy": strategy, "split_seed": seed,
                       "threshold_rule": rule, "threshold": thr,
                       "val_fpr_pct": 100 * (val_scores >= thr)[val["y"].values == 0].mean(),
                       "clean_f1": f1_score(test["y"], pred),
                       "clean_fpr_pct": 100 * pred[test["y"].values == 0].mean(),
                       "clean_detection_pct": 100 * pred[test["y"].values == 1].mean()}
                for family in c.FAMILIES:
                    row[family] = 100 * (family_scores[family] >= thr).mean()
                row["encoding_mean_pct"] = float(np.mean([row[f] for f in c.ENCODING_FAMILIES]))
                row["rewriting_mean_pct"] = float(np.mean([row[f] for f in c.REWRITING_FAMILIES]))
                rows.append(row)
                print(f"  {strategy:36} seed {seed}  {rule:22} thr {thr:.4f}  val FPR "
                      f"{row['val_fpr_pct']:.2f}%  F1 {row['clean_f1']:.4f}  test FPR "
                      f"{row['clean_fpr_pct']:.2f}%  encoded attacks {row['encoding_mean_pct']:.1f}%")
    return pd.DataFrame(rows)


def save_block(table: pd.DataFrame, filename: str, kind: str) -> None:
    path = c.results_path(filename)
    if os.path.exists(path):
        previous = pd.read_csv(path)
        table = pd.concat([previous[previous["model"] != kind], table], ignore_index=True)
    table.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset analysis and split checks.")
    parser.add_argument("--model", default=config.PRIMARY_MODEL, choices=config.MODELS)
    kind = parser.parse_args().model
    seed = config.SEEDS[0]
    c.set_seed(seed)
    pd.set_option("display.width", 200)

    full = add_descriptions(c.load_source_all().drop_duplicates(subset="payload")
                            .reset_index(drop=True))
    sets = {n: add_descriptions(c.load_data(n)) for n in c.SPLITS}

    print("Part 1 - profile of the dataset before splitting")
    whole = profile(full)
    print(whole.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\nPart 2 - are the three sets alike?")
    table = balance(sets)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    joined = whole.merge(table, on=["section", "item"], how="left", validate="one_to_one")
    assert joined["train"].notna().sum() == len(table), "a row of part 2 has no row in part 1"
    joined.to_csv(c.results_path("split_profile.csv"), index=False)
    worst = table["p_value"].min()
    tests = 7
    print(f"Smallest p-value: {worst:.3f}; with {tests} tests the corrected limit is "
          f"{config.ALPHA / tests:.4f}, so the sets "
          f"{'do not differ' if worst >= config.ALPHA / tests else 'differ'} significantly.")

    print(f"\nPart 3 - similarity between test and train ({kind})")
    table = similarity(sets, kind, seed)
    save_block(table, "split_similarity.csv", kind)
    print(table.drop(columns="model").to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print(f"\nPart 4 - one kind of SQL injection never seen in training ({kind})")
    table = unseen_kind(sets, kind, seed)
    save_block(table, "split_unseen_kind.csv", kind)
    print(table.drop(columns="model").to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print(f"\nPart 5 - the split repeated with seeds {config.SEEDS} ({kind})")
    table = repeats(full, kind, c.load_tampers())
    save_block(table, "split_repeats.csv", kind)
    summary = table.groupby(["threshold_rule", "strategy"])[
        ["threshold", "val_fpr_pct", "clean_f1", "clean_fpr_pct", "encoding_mean_pct"]
    ].agg(["mean", "std", "min", "max"])
    print(summary.round(4).to_string())
    print("\nSaved to results/: split_profile.csv, split_similarity.csv, "
          "split_unseen_kind.csv, split_repeats.csv")


if __name__ == "__main__":
    main()
