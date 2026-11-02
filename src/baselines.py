"""Baseline models for Track B, all scored with the same grouped 5-fold protocol.

Usage: python3 src/baselines.py {constant,ev_rule,logistic,lda,svm}
"""
import csv
import sys
from datetime import datetime, timezone
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

SEED = 0
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

RESULT_COLUMNS = [
    "timestamp_utc", "experiment", "model", "features", "params", "validation", "seed",
    "auc_mean", "auc_std", "log_loss_mean", "log_loss_std",
    "accuracy_mean", "accuracy_std", "threshold", "notes",
]


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


def summarise(model, fold_scores, features, params="", notes="", threshold=f"p>={THRESHOLD}", experiment="baseline",
              validation=f"GroupKFold({N_SPLITS}) by {GROUP}"):
    """Turn per-fold scores into one results row (mean and SD across folds)."""
    scores = pd.DataFrame(fold_scores)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": experiment,
        "model": model,
        "features": features,
        "params": params,
        "validation": validation,
        "seed": SEED,
        "auc_mean": round(scores["auc"].mean(), 4),
        "auc_std": round(scores["auc"].std(), 4),
        "log_loss_mean": round(scores["log_loss"].mean(), 4),
        "log_loss_std": round(scores["log_loss"].std(), 4),
        "accuracy_mean": round(scores["accuracy"].mean(), 4),
        "accuracy_std": round(scores["accuracy"].std(), 4),
        "threshold": threshold,
        "notes": notes,
    }


def log_run(row):
    """Append one row to experimental_results.csv, writing the header if needed."""
    is_new = not RESULTS_PATH.exists()
    if not is_new:
        with open(RESULTS_PATH, newline="") as f:
            assert next(csv.reader(f)) == RESULT_COLUMNS, "results file has different columns"
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def run_constant(train):
    """Trivial baseline: predict the training folds' risky-choice rate for everyone."""
    y = train[TARGET].to_numpy()
    X = np.zeros((len(train), 1))  # DummyClassifier ignores X, so no column is used
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(grouped_folds(train), start=1):
        model = DummyClassifier(strategy="prior").fit(X[train_idx], y[train_idx])
        p = model.predict_proba(X[val_idx])[:, 1]
        scores = score_fold(y[val_idx], p)
        fold_scores.append(scores)
        print(f"fold {fold}: prior={p[0]:.4f} auc={scores['auc']:.4f} log_loss={scores['log_loss']:.4f} accuracy={scores['accuracy']:.4f}")
    return summarise("constant", fold_scores, features="none", params="strategy=prior")


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
        "ev_rule", fold_scores, features="ev_gap",
        params="auc and accuracy=raw gap; log_loss=unpenalised logistic on gap, fit on training folds",
        threshold="gap>0",
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


def logistic_params(folds, inner=f"GroupShuffleSplit({INNER_VAL_SHARE})"):
    chosen = [result["C"] for result in folds]
    return f"C per fold={chosen}; grid={C_GRID}; inner={inner} by AUC"


def run_logistic(train):
    """L2-regularised logistic regression on the presented gamble attributes."""
    folds = cross_validate_logistic(train, train[RAW_FEATURES].to_numpy())
    print_logistic_folds(folds)
    return summarise(
        "logistic_l2", [result["scores"] for result in folds], features="+".join(RAW_FEATURES),
        params=logistic_params(folds),
    )


def run_lda(train):
    """LDA on the presented gamble attributes: Gaussian classes with a shared covariance."""
    X = train[RAW_FEATURES].to_numpy()
    y = train[TARGET].to_numpy()
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(grouped_folds(train), start=1):
        model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        model.fit(X[train_idx], y[train_idx])
        scores = score_fold(y[val_idx], model.predict_proba(X[val_idx])[:, 1])
        fold_scores.append(scores)
        print(f"fold {fold}: auc={scores['auc']:.4f} log_loss={scores['log_loss']:.4f} accuracy={scores['accuracy']:.4f}")
    return summarise("lda", fold_scores, features="+".join(RAW_FEATURES), params="solver=svd; shrinkage=None")


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
        "linear_svm", fold_scores, features="+".join(RAW_FEATURES),
        params=f"C per fold={chosen}; grid={C_GRID}; inner=GroupShuffleSplit({INNER_VAL_SHARE}) by AUC; log_loss=logistic mapping on grouped out-of-fold scores",
        threshold="score>0",
    )


RUNNERS = {
    "constant": run_constant, "ev_rule": run_ev_rule, "logistic": run_logistic,
    "lda": run_lda, "svm": run_svm,
}


def main():
    name = sys.argv[1]
    row = RUNNERS[name](load_train())
    log_run(row)
    print(pd.DataFrame([row])[["model", "auc_mean", "auc_std", "log_loss_mean", "log_loss_std", "accuracy_mean", "accuracy_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
