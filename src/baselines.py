"""Baseline models for Track B, all scored with the same grouped 5-fold protocol.

Usage: python3 src/baselines.py {constant,ev_rule,logistic,lda,svm}
"""
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = int(os.environ.get("IML_SEED", "0"))  # run_all.py --seed sets this
N_SPLITS = 5
TARGET = "chose_risky"
GROUP = "participant_id"
RAW_FEATURES = ["prob_high", "risky_high_inr", "risky_low_inr", "sure_amount_inr"]
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
INNER_VAL_SHARE = 0.25
THRESHOLD = 0.5

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train.csv"
RESULTS_PATH = ROOT / "experimental_results.csv"

# Exactly the columns of the course's experimental_results_template.csv, in its order.
RESULT_COLUMNS = [
    "run_id", "track", "experiment", "model", "feature_set", "validation_scheme", "n_folds", "seed",
    "hyperparameters", "primary_metric_name", "primary_metric_mean", "primary_metric_sd",
    "secondary_metric_name", "secondary_metric_mean", "secondary_metric_sd", "threshold",
    "runtime_s", "code_reference", "notes",
]
GRID_LABEL = f"grid {C_GRID[0]}-{C_GRID[-1]}"


def load_train():
    # realised_payout_inr is paid after the choice, so it is removed on load.
    return pd.read_csv(TRAIN_PATH).drop(columns=["realised_payout_inr"])


def grouped_folds(train):
    """Yield (train_idx, val_idx), keeping each participant inside a single fold."""
    cv = GroupKFold(n_splits=N_SPLITS)
    for train_idx, val_idx in cv.split(train, train[TARGET], groups=train[GROUP]):
        overlap = set(train[GROUP].iloc[train_idx]) & set(train[GROUP].iloc[val_idx])
        assert not overlap, "a participant appears in both training and validation"
        yield train_idx, val_idx


def score_fold(y_true, p):
    """Score one validation fold from predicted probabilities of chose_risky = 1."""
    return {
        "auc": roc_auc_score(y_true, p),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
        "accuracy": accuracy_score(y_true, p >= THRESHOLD),
    }


def summarise(run_id, experiment, model, feature_set, fold_scores, hyperparameters, code_reference, runtime_s,
              notes="", threshold=f"p>={THRESHOLD}", validation_scheme=f"GroupKFold{N_SPLITS}"):
    """Turn per-fold scores into one results row (mean and SD across folds).

    AUC is the primary metric and log-loss the secondary. Accuracy has no column in the
    template, so it goes at the start of the notes with its threshold.
    """
    scores = pd.DataFrame(fold_scores)
    mean = lambda metric: float(round(scores[metric].mean(), 4))
    sd = lambda metric: float(round(scores[metric].std(), 4))
    accuracy = f"accuracy={mean('accuracy')} (SD {sd('accuracy')}) at {threshold}"
    return {
        "run_id": run_id, "track": "B", "experiment": experiment, "model": model,
        "feature_set": feature_set, "validation_scheme": validation_scheme, "n_folds": N_SPLITS, "seed": SEED,
        "hyperparameters": hyperparameters,
        "primary_metric_name": "roc_auc", "primary_metric_mean": mean("auc"), "primary_metric_sd": sd("auc"),
        "secondary_metric_name": "log_loss", "secondary_metric_mean": mean("log_loss"), "secondary_metric_sd": sd("log_loss"),
        "threshold": threshold, "runtime_s": round(runtime_s, 2), "code_reference": code_reference,
        "notes": "; ".join(part for part in [accuracy, notes] if part),
    }


def log_run(row):
    """Write one row to experimental_results.csv, replacing any earlier row with the same run_id.

    Rows are kept sorted by run_id, so re-running a step leaves one row per run.
    """
    rows = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == RESULT_COLUMNS, "results file has different columns"
            rows = [r for r in reader if r["run_id"] != row["run_id"]]
    rows.append(row)
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["run_id"]))


def c_per_fold(chosen):
    """Short hyperparameter label, e.g. 'C per fold 1/10/10/0.01/100; grid 0.001-100'."""
    return f"C per fold {'/'.join(str(C) for C in chosen)}; {GRID_LABEL}"


def run_constant(train):
    """Trivial baseline: predict the training folds' risky-choice rate for everyone."""
    y = train[TARGET].to_numpy()
    X = np.zeros((len(train), 1))  # DummyClassifier ignores X, so no column is used
    start = time.perf_counter()
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(grouped_folds(train), start=1):
        model = DummyClassifier(strategy="prior").fit(X[train_idx], y[train_idx])
        p = model.predict_proba(X[val_idx])[:, 1]
        scores = score_fold(y[val_idx], p)
        fold_scores.append(scores)
        print(f"fold {fold}: prior={p[0]:.4f} auc={scores['auc']:.4f} log_loss={scores['log_loss']:.4f} accuracy={scores['accuracy']:.4f}")
    return summarise(
        "B01", "baseline", "constant", "none", fold_scores, "strategy=prior", "src/baselines.py#run_constant",
        time.perf_counter() - start,
    )


def ev_gap(df):
    """Expected value of the risky option minus the sure amount, in rupees.

    The endowment is added to both options, so it cancels. Rounding removes floating-point
    noise, so problems with equal expected values stay exactly tied.
    """
    risky_ev = df["prob_high"] * df["risky_high_inr"] + (1 - df["prob_high"]) * df["risky_low_inr"]
    return (risky_ev - df["sure_amount_inr"]).round(6).to_numpy()


def run_ev_rule(train):
    """Transparent benchmark: rank decisions by EV gap; nothing is learned for the AUC.

    The gap is in rupees, so log-loss needs a mapping to probabilities. That mapping is a
    one-feature logistic fit on the training folds only.
    """
    y = train[TARGET].to_numpy()
    gap = ev_gap(train)
    X = gap.reshape(-1, 1)
    start = time.perf_counter()
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(grouped_folds(train), start=1):
        mapping = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=2000))
        mapping.fit(X[train_idx], y[train_idx])
        p = mapping.predict_proba(X[val_idx])[:, 1]
        scores = score_fold(y[val_idx], p)
        scores["auc"] = roc_auc_score(y[val_idx], gap[val_idx])  # the raw rule, not the mapping
        scores["accuracy"] = accuracy_score(y[val_idx], gap[val_idx] > 0)  # a tied gap counts as sure
        fold_scores.append(scores)
        slope = mapping[-1].coef_[0, 0]
        print(f"fold {fold}: slope={slope:.4f} auc={scores['auc']:.4f} log_loss={scores['log_loss']:.4f} accuracy={scores['accuracy']:.4f}")
    return summarise(
        "B02", "baseline", "ev_rule", "ev_gap", fold_scores, "none (rule)", "src/baselines.py#run_ev_rule",
        time.perf_counter() - start, threshold="gap>0",
        notes="AUC and accuracy from raw EV gap; log_loss via unpenalised logistic mapping fitted on training folds",
    )


def make_logistic(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def choose_C(X, y, groups, make_model=make_logistic):
    """Pick C using only the outer training data, holding out whole participants.

    C_GRID runs from strongest to weakest regularisation, and a later C must beat the
    current best to replace it, so a tie keeps the stronger regularisation.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_SHARE, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(X, y, groups=groups))
    assert not set(groups[fit_idx]) & set(groups[val_idx]), "inner split shares a participant"
    best_C, best_auc = None, -np.inf
    for C in C_GRID:
        model = make_model(C).fit(X[fit_idx], y[fit_idx])
        auc = roc_auc_score(y[val_idx], model.decision_function(X[val_idx]))
        if auc > best_auc:
            best_C, best_auc = C, auc
    return best_C, best_auc


def cross_validate_logistic(train, X, splits=None, inner_groups=None):
    """Run the logistic protocol on feature matrix X. Returns one dict per outer fold.

    By default both the outer folds and the inner split are grouped by participant.
    E3 passes its own splits and inner_groups to replace them.
    """
    y = train[TARGET].to_numpy()
    groups = train[GROUP].to_numpy() if inner_groups is None else inner_groups
    folds = []
    for train_idx, val_idx in (grouped_folds(train) if splits is None else splits):
        C, inner_auc = choose_C(X[train_idx], y[train_idx], groups[train_idx])
        model = make_logistic(C).fit(X[train_idx], y[train_idx])
        scores = score_fold(y[val_idx], model.predict_proba(X[val_idx])[:, 1])
        folds.append({"C": C, "inner_auc": inner_auc, "scores": scores, "model": model})
    return folds


def print_logistic_folds(folds):
    for fold, result in enumerate(folds, start=1):
        s = result["scores"]
        print(f"fold {fold}: C={result['C']} inner_auc={result['inner_auc']:.4f} auc={s['auc']:.4f} log_loss={s['log_loss']:.4f} accuracy={s['accuracy']:.4f}")


def timed_logistic(train, X, **kwargs):
    """cross_validate_logistic plus its wall-clock time in seconds."""
    start = time.perf_counter()
    folds = cross_validate_logistic(train, X, **kwargs)
    return folds, time.perf_counter() - start


def logistic_row(folds, runtime_s, run_id, experiment, feature_set, code_reference, notes="", **kwargs):
    """A results row for a cross_validate_logistic run."""
    return summarise(
        run_id, experiment, "logistic_l2", feature_set, [result["scores"] for result in folds],
        c_per_fold([result["C"] for result in folds]), code_reference, runtime_s, notes=notes, **kwargs,
    )


def run_logistic(train):
    """L2-regularised logistic regression on the presented gamble attributes."""
    folds, runtime_s = timed_logistic(train, train[RAW_FEATURES].to_numpy())
    print_logistic_folds(folds)
    return logistic_row(
        folds, runtime_s, "B03", "baseline", "raw4", "src/baselines.py#run_logistic",
        notes="L2; inner GroupShuffleSplit 0.25 by AUC",
    )


def run_lda(train):
    """LDA on the presented gamble attributes: Gaussian classes with a shared covariance."""
    X = train[RAW_FEATURES].to_numpy()
    y = train[TARGET].to_numpy()
    start = time.perf_counter()
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(grouped_folds(train), start=1):
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        model.fit(X[train_idx], y[train_idx])
        scores = score_fold(y[val_idx], model.predict_proba(X[val_idx])[:, 1])
        fold_scores.append(scores)
        print(f"fold {fold}: auc={scores['auc']:.4f} log_loss={scores['log_loss']:.4f} accuracy={scores['accuracy']:.4f}")
    return summarise(
        "B04", "baseline", "lda", "raw4", fold_scores, "solver=svd; shrinkage=None", "src/baselines.py#run_lda",
        time.perf_counter() - start,
    )


def make_svm(C):
    return make_pipeline(StandardScaler(), SVC(kernel="linear", C=C))


def fit_score_mapping(X, y, groups, C):
    """Learn a score-to-probability mapping from scores on participants the SVM did not fit."""
    cv = GroupKFold(n_splits=N_SPLITS)
    scores = cross_val_predict(make_svm(C), X, y, groups=groups, cv=cv, method="decision_function")
    return LogisticRegression(penalty=None, max_iter=2000).fit(scores.reshape(-1, 1), y)


def run_svm(train):
    """Linear SVM on the presented gamble attributes.

    AUC and accuracy use the SVM's own score. Log-loss needs probabilities, which an SVM
    does not produce, so it uses a logistic mapping learned inside the training folds.
    """
    X = train[RAW_FEATURES].to_numpy()
    y = train[TARGET].to_numpy()
    groups = train[GROUP].to_numpy()
    start = time.perf_counter()
    fold_scores, chosen = [], []
    for fold, (train_idx, val_idx) in enumerate(grouped_folds(train), start=1):
        C, inner_auc = choose_C(X[train_idx], y[train_idx], groups[train_idx], make_model=make_svm)
        mapping = fit_score_mapping(X[train_idx], y[train_idx], groups[train_idx], C)
        model = make_svm(C).fit(X[train_idx], y[train_idx])
        svm_score = model.decision_function(X[val_idx])
        scores = score_fold(y[val_idx], mapping.predict_proba(svm_score.reshape(-1, 1))[:, 1])
        scores["auc"] = roc_auc_score(y[val_idx], svm_score)
        scores["accuracy"] = accuracy_score(y[val_idx], svm_score > 0)
        fold_scores.append(scores)
        chosen.append(C)
        print(f"fold {fold}: C={C} inner_auc={inner_auc:.4f} auc={scores['auc']:.4f} log_loss={scores['log_loss']:.4f} accuracy={scores['accuracy']:.4f}")
    return summarise(
        "B05", "baseline", "linear_svm", "raw4", fold_scores, c_per_fold(chosen), "src/baselines.py#run_svm",
        time.perf_counter() - start, threshold="score>0",
        notes=("kernel=linear; inner GroupShuffleSplit 0.25 by AUC; AUC and accuracy from SVM score; "
               "log_loss via logistic mapping on grouped out-of-fold scores"),
    )


RUNNERS = {
    "constant": run_constant, "ev_rule": run_ev_rule, "logistic": run_logistic,
    "lda": run_lda, "svm": run_svm,
}


def print_rows(rows):
    columns = ["run_id", "experiment", "feature_set", "validation_scheme", "primary_metric_mean", "primary_metric_sd",
               "secondary_metric_mean", "secondary_metric_sd", "runtime_s"]
    print(pd.DataFrame(rows)[columns].to_string(index=False))


def main():
    row = RUNNERS[sys.argv[1]](load_train())
    log_run(row)
    print_rows([row])


if __name__ == "__main__":
    main()
