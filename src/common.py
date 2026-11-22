"""Shared functions: data loading, tamper families, normaliser, detector and threshold."""

from __future__ import annotations

import importlib
import json
import os
import random
import re
import sys
import urllib.parse
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.svm import LinearSVC

import config

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")


def _locate(name: str) -> str:
    candidates = [os.environ.get(name.upper() + "_DIR", ""),
                  os.path.join(ROOT, "..", name), os.path.join(ROOT, name)]
    for path in candidates:
        if path and os.path.isdir(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"Folder '{name}' not found. Clone it next to the repository "
                            f"folder, or set the environment variable {name.upper()}_DIR.")


SQLMAP = _locate("sqlmap")
try:
    _DEFAULT_DATASET = _locate(config.DATASET_SPECS["httpparams"]["folder"])
except FileNotFoundError:
    _DEFAULT_DATASET = None

ENCODING_FAMILIES = ["charencode", "chardoubleencode", "base64encode", "percentage", "htmlencode"]
REWRITING_FAMILIES = ["space2comment", "randomcomments", "randomcase"]
FAMILIES = ENCODING_FAMILIES + REWRITING_FAMILIES

NORMALISER_COVERED = ["charencode", "chardoubleencode", "space2comment"]

Tamper = Callable[[str], str]


def results_path(*parts: str) -> str:
    path = os.path.join(RESULTS, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_tampers() -> dict[str, Tamper]:
    sys.path.insert(0, SQLMAP)
    from lib.core.common import setPaths
    from lib.core.option import _setConfAttributes, _setKnowledgeBaseAttributes
    setPaths(os.path.abspath(SQLMAP))
    _setConfAttributes()
    _setKnowledgeBaseAttributes()
    return {family: importlib.import_module(f"tamper.{family}").tamper for family in FAMILIES}


def apply_tamper(tampers: dict[str, Tamper], family: str, payload: str) -> str:
    try:
        out = tampers[family](payload)
        return out if isinstance(out, str) else payload
    except Exception:
        return payload


def _active_dataset() -> tuple[str, dict]:
    name = os.environ.get("DATASET", "httpparams").lower()
    if name not in config.DATASET_SPECS:
        raise SystemExit(f"Unknown DATASET '{name}'. Options: {list(config.DATASET_SPECS)}")
    return name, config.DATASET_SPECS[name]


def _dataset_folder(name: str, spec: dict) -> str:
    if name == "httpparams" and _DEFAULT_DATASET is not None:
        return _DEFAULT_DATASET
    return _locate(spec["folder"])


def _prepare_frame(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    labels = df[spec["label_col"]].astype(str)
    sqli = [str(x) for x in spec["sqli_labels"]]
    normal = [str(x) for x in spec["normal_labels"]]
    mask = labels.isin(sqli + normal)
    return pd.DataFrame({"payload": df.loc[mask, spec["text_col"]].astype(str).values,
                         "label": labels[mask].values,
                         "y": labels[mask].isin(sqli).astype(int).values})


def load_source(which: str) -> pd.DataFrame:
    name, spec = _active_dataset()
    if "single_file" in spec:
        raise ValueError(f"Dataset '{name}' is a single file and has no official train/test split.")
    if which not in ("train", "test"):
        raise ValueError(f"which must be 'train' or 'test', got {which!r}")
    fname = spec["train_file"] if which == "train" else spec["test_file"]
    return _prepare_frame(pd.read_csv(os.path.join(_dataset_folder(name, spec), fname)), spec)


def load_source_all() -> pd.DataFrame:
    name, spec = _active_dataset()
    if "single_file" in spec:
        path = os.path.join(_dataset_folder(name, spec), spec["single_file"])
        return _prepare_frame(pd.read_csv(path), spec)
    return pd.concat([load_source("train"), load_source("test")], ignore_index=True)


DATA = os.path.join(ROOT, "data")
SPLITS = ("train", "val", "test")


def split_path(split: str) -> str:
    return os.path.join(DATA, split, f"{split}.csv")


def load_data(split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = split_path(split)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run 'python split_dataset.py' first to "
                                "create the train, val and test sets.")
    df = pd.read_csv(path)
    df["payload"] = df["payload"].astype(str)
    return df


def normalise(text: str) -> str:
    s = str(text)
    for _ in range(2):
        s = urllib.parse.unquote(s)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _prepare(texts: Iterable[str], normalised: bool) -> list[str]:
    return [normalise(t) for t in texts] if normalised else list(texts)


def model_params(kind: str) -> dict:
    path = os.path.join(RESULTS, "model_selection.csv")
    if os.path.exists(path):
        table = pd.read_csv(path)
        chosen = table[(table["model"] == kind) & (table["selected"])]
        if len(chosen):
            return json.loads(chosen.iloc[0]["params"])
    return dict(config.DEFAULT_PARAMS[kind])


def _make_model(kind: str, params: dict | None = None, seed: int = 0):
    if kind not in config.MODELS:
        raise ValueError(f"Unknown model '{kind}'. Options: {list(config.MODELS)}")
    p = model_params(kind) if params is None else dict(params)
    if kind == "logreg":
        return LogisticRegression(C=p["C"], max_iter=config.MAX_ITER)
    if kind == "linsvc":
        return LinearSVC(C=p["C"])
    return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                  n_jobs=-1, random_state=seed)


def build_detector(texts: Iterable[str], labels: Sequence[int], normalised: bool = False,
                   model: str = "logreg", params: dict | None = None, seed: int = 0):
    vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=config.NGRAM_RANGE,
                                 min_df=config.MIN_DF, lowercase=False)
    features = vectoriser.fit_transform(_prepare(texts, normalised))
    estimator = _make_model(model, params, seed)
    estimator.fit(features, labels)
    return vectoriser, estimator


def score(vectoriser: TfidfVectorizer, model, texts: Iterable[str],
          normalised: bool = False) -> np.ndarray:
    features = vectoriser.transform(_prepare(texts, normalised))
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return expit(model.decision_function(features))


def threshold_at_fpr(vectoriser: TfidfVectorizer, model,
                     X_val: Iterable[str], y_val: Sequence[int], *,
                     target_fpr: float = config.TARGET_FPR, normalised: bool = False) -> float:
    s = score(vectoriser, model, X_val, normalised)
    fpr, _, thresholds = roc_curve(y_val, s, drop_intermediate=False)
    idx = np.where(fpr <= target_fpr)[0]
    return float(thresholds[idx[-1]]) if len(idx) else 1.0
