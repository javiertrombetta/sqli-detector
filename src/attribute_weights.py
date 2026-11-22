"""Supporting analysis: dataset columns, feature weighting, class weights and starting weights."""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

import common as c
import config

NGRAMS = dict(analyzer="char_wb", ngram_range=config.NGRAM_RANGE, min_df=config.MIN_DF,
              lowercase=False)

LONG_JOIN = 8
LONG_SAMPLE = 1000
REPEAT = 6

WEIGHTINGS = {
    "binary presence":     lambda: CountVectorizer(binary=True, **NGRAMS),
    "raw counts":          lambda: CountVectorizer(**NGRAMS),
    "term frequency":      lambda: TfidfVectorizer(use_idf=False, **NGRAMS),
    "TF-IDF (detector)":   lambda: TfidfVectorizer(**NGRAMS),
    "sublinear TF-IDF":    lambda: TfidfVectorizer(sublinear_tf=True, **NGRAMS),
}


def length_column(texts, scale: float) -> csr_matrix:
    lengths = np.array([len(t) for t in texts], dtype=float)
    return csr_matrix(np.minimum(np.log1p(lengths) / scale, 1.5).reshape(-1, 1))


def scores(model, features) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return expit(model.decision_function(features))


def threshold(y_val, s_val) -> float:
    fpr, _, thresholds = roc_curve(y_val, s_val, drop_intermediate=False)
    idx = np.where(fpr <= config.TARGET_FPR)[0]
    return float(thresholds[idx[-1]]) if len(idx) else 1.0


def evaluate(name: str, part: str, make_features, estimator, sets, variants, harmless) -> dict:
    train, val, test = sets
    estimator.fit(make_features(train["payload"], True), train["y"])
    thr = threshold(val["y"], scores(estimator, make_features(val["payload"], False)))
    s_test = scores(estimator, make_features(test["payload"], False))
    pred = (s_test >= thr).astype(int)
    row = {"part": part, "variant": name,
           "clean_f1": f1_score(test["y"], pred),
           "clean_fpr_pct": 100 * pred[test["y"].values == 0].mean(),
           "clean_detection_pct": 100 * pred[test["y"].values == 1].mean()}
    for family, texts in variants.items():
        row[family] = 100 * (scores(estimator, make_features(texts, False)) >= thr).mean()
    row["encoding_mean_pct"] = float(np.mean([row[f] for f in c.ENCODING_FAMILIES]))
    row["rewriting_mean_pct"] = float(np.mean([row[f] for f in c.REWRITING_FAMILIES]))
    for label, texts in harmless.items():
        row["normal_" + label] = 100 * (scores(estimator, make_features(texts, False)) >= thr).mean()
    row["normal_encoded_flagged_pct"] = float(np.mean([row["normal_" + f]
                                                       for f in c.ENCODING_FAMILIES]))
    return row


def describe_columns() -> pd.DataFrame:
    name, spec = c._active_dataset()
    if name != "httpparams":
        print(f"Column description skipped: it is written for HttpParamsDataset, not '{name}'.")
        return pd.DataFrame()
    folder = c._dataset_folder(name, spec)
    raw = pd.concat([pd.read_csv(os.path.join(folder, spec[k]))
                     for k in ("train_file", "test_file")], ignore_index=True)
    kept = raw[raw["attack_type"].isin(["norm", "sqli"])]
    same = (kept["length"] == kept["payload"].astype(str).str.len()).mean()
    agree = ((kept["attack_type"] == "sqli") == (kept["label"] == "anom")).mean()
    y = (kept["attack_type"] == "sqli").astype(int)
    rows = [
        {"column": "payload", "content": "the request parameter value (text)",
         "role": "input: the only column given to the model", "check": ""},
        {"column": "length", "content": "number of characters of payload",
         "role": "not used: derived from payload",
         "check": f"equal to len(payload) in {100 * same:.2f}% of the rows used; "
                  f"alone it separates the classes with ROC AUC {roc_auc_score(y, kept['length']):.3f}"},
        {"column": "attack_type", "content": "norm, sqli, xss, cmdi or path-traversal",
         "role": "the answer: defines y (sqli = 1, norm = 0); other types are left out",
         "check": f"{len(raw) - len(kept)} rows of other attack types excluded"},
        {"column": "label", "content": "norm or anom",
         "role": "not used: a coarser copy of attack_type",
         "check": f"agrees with attack_type in {100 * agree:.2f}% of the rows used"},
    ]
    print(f"Source dataset: {len(raw)} rows, columns {list(raw.columns)}; "
          f"{len(kept)} rows are normal or SQL injection.")
    return pd.DataFrame(rows)


def starting_weights(sets, params, variants) -> pd.DataFrame:
    train, val, test = sets
    vec = TfidfVectorizer(**NGRAMS)
    features = vec.fit_transform(train["payload"])

    def fit(tol, start_seed=None):
        model = LogisticRegression(C=params["C"], max_iter=20 * config.MAX_ITER, tol=tol,
                                   warm_start=start_seed is not None)
        if start_seed is not None:
            rng = np.random.RandomState(start_seed)
            model.classes_ = np.array([0, 1])
            model.coef_ = rng.normal(0, 1, size=(1, features.shape[1]))
            model.intercept_ = rng.normal(0, 1, size=1)
        return model.fit(features, train["y"])

    def measure(model) -> tuple:
        thr = threshold(val["y"], scores(model, vec.transform(val["payload"])))
        pred = (scores(model, vec.transform(test["payload"])) >= thr).astype(int)
        enc = np.mean([100 * (scores(model, vec.transform(variants[f])) >= thr).mean()
                       for f in c.ENCODING_FAMILIES])
        return f1_score(test["y"], pred), float(enc)

    rows = []
    for label, tol in (("project tolerance 1e-4", 1e-4), ("tight tolerance 1e-8", 1e-8)):
        base = fit(tol)
        for start in (None,) + tuple(config.SEEDS[:3]):
            model = base if start is None else fit(tol, start)
            f1, enc = measure(model)
            rows.append({"tolerance": label,
                         "start": "zeros (scikit-learn)" if start is None else f"random, seed {start}",
                         "iterations": int(model.n_iter_[0]),
                         "largest_weight": float(np.abs(model.coef_).max()),
                         "max_diff_from_zero_start": float(np.abs(model.coef_ - base.coef_).max()),
                         "clean_f1": f1, "encoding_mean_pct": enc})
    table = pd.DataFrame(rows)
    print(f"\nPart 4 - starting weights (logreg, {features.shape[1]} weights)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence for attributes and weights.")
    parser.add_argument("--model", default=config.PRIMARY_MODEL, choices=config.MODELS)
    kind = parser.parse_args().model
    seed = config.SEEDS[0]
    c.set_seed(seed)
    params = c.model_params(kind)

    columns = describe_columns()
    if len(columns):
        columns.to_csv(c.results_path("attribute_columns.csv"), index=False)
        print(columns.to_string(index=False))

    sets = tuple(c.load_data(s) for s in c.SPLITS)
    train, val, test = sets
    attacks = test[test["y"] == 1]["payload"].tolist()
    tampers = c.load_tampers()
    variants = {f: [c.apply_tamper(tampers, f, p) for p in attacks] for f in c.FAMILIES}
    model = lambda **extra: _model(kind, params, seed, **extra)
    normal = test[test["y"] == 0]["payload"].tolist()
    harmless = {f: [c.apply_tamper(tampers, f, p) for p in normal] for f in c.ENCODING_FAMILIES}
    rng = np.random.RandomState(seed)
    harmless["long_flagged_pct"] = [" ".join(rng.choice(normal, LONG_JOIN)) for _ in range(LONG_SAMPLE)]
    harmless["long_nosep_flagged_pct"] = ["".join(rng.choice(normal, LONG_JOIN)) for _ in range(LONG_SAMPLE)]
    harmless["long_comma_flagged_pct"] = [",".join(rng.choice(normal, LONG_JOIN)) for _ in range(LONG_SAMPLE)]
    harmless["repeated_flagged_pct"] = [(p + " ") * REPEAT for p in rng.choice(normal, LONG_SAMPLE)]

    rows = []
    scale = float(np.log1p(train["payload"].str.len().max()))
    tfidf = TfidfVectorizer(**NGRAMS)
    text = lambda t, fit: tfidf.fit_transform(t) if fit else tfidf.transform(t)
    rows.append(evaluate("length only", "1 columns",
                         lambda t, fit: length_column(t, scale), model(), sets, variants, harmless))
    rows.append(evaluate("payload text (detector)", "1 columns", text, model(), sets, variants, harmless))
    rows.append(evaluate("payload text + length", "1 columns",
                         lambda t, fit: hstack([text(t, fit), length_column(t, scale)]).tocsr(),
                         model(), sets, variants, harmless))
    for name, make in WEIGHTINGS.items():
        vec = make()
        rows.append(evaluate(name, "2 weighting",
                             lambda t, fit, v=vec: v.fit_transform(t) if fit else v.transform(t),
                             model(), sets, variants, harmless))
    for name, cw in (("no class weights (detector)", None), ("balanced class weights", "balanced")):
        vec = TfidfVectorizer(**NGRAMS)
        rows.append(evaluate(name, "3 class weights",
                             lambda t, fit, v=vec: v.fit_transform(t) if fit else v.transform(t),
                             model(class_weight=cw), sets, variants, harmless))

    table = pd.DataFrame(rows)
    print(f"\n{kind}, {params}, threshold at {config.TARGET_FPR:.0%} FPR on val, seed {seed}")
    print(table[["part", "variant", "clean_f1", "clean_fpr_pct", "encoding_mean_pct",
                 "rewriting_mean_pct", "normal_encoded_flagged_pct", "normal_long_flagged_pct",
                 "normal_repeated_flagged_pct"]].to_string(index=False,
                                                          float_format=lambda v: f"{v:.4f}"))
    if kind == "logreg":
        start = starting_weights(sets, params, variants)
        start.insert(0, "variant", start.pop("tolerance") + "; " + start.pop("start"))
        start.insert(0, "part", "4 starting weights")
        table = pd.concat([table, start], ignore_index=True)
    table.insert(0, "model", kind)
    path = c.results_path("attribute_weights.csv")
    previous = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    if len(previous):
        table = pd.concat([previous[previous["model"] != kind], table], ignore_index=True)
    table.to_csv(path, index=False)
    print("\nSaved to results/attribute_weights.csv and attribute_columns.csv")


def _model(kind: str, params: dict, seed: int, class_weight=None):
    estimator = c._make_model(kind, params, seed)
    estimator.set_params(class_weight=class_weight)
    return estimator


if __name__ == "__main__":
    main()
