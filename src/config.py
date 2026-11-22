"""Settings of the experiment."""

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

NGRAM_RANGE: tuple[int, int] = (1, 4)
MIN_DF: int = 2
MAX_ITER: int = 2000

TARGET_FPR: float = 0.01
DEFAULT_THRESHOLD: float = 0.5

N_AUGMENT: int = 800

VALIDITY_SAMPLE: int = 50

ALPHA: float = 0.05
BOOTSTRAP_RESAMPLES: int = 1000

DATASET_SPECS: dict = {
    "httpparams": {
        "folder": "HttpParamsDataset",
        "train_file": "payload_train.csv",
        "test_file": "payload_test.csv",
        "text_col": "payload",
        "label_col": "attack_type",
        "sqli_labels": ["sqli"],
        "normal_labels": ["norm"],
    },
}

SPLIT: tuple[float, float, float] = (0.70, 0.15, 0.15)

MODELS: tuple[str, ...] = ("logreg", "linsvc", "rf")
PRIMARY_MODEL: str = "logreg"

DEFAULT_PARAMS: dict = {
    "logreg": {"C": 1.0},
    "linsvc": {"C": 1.0},
    "rf": {"n_estimators": 200, "max_depth": None},
}
PARAM_GRID: dict = {
    "logreg": {"C": [0.1, 0.3, 1.0, 3.0, 10.0]},
    "linsvc": {"C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    "rf": {"n_estimators": [100, 300], "max_depth": [None]},
}
