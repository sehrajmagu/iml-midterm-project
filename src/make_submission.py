"""Rebuild the final prediction file from scratch: choose C, refit on all training data, predict test once.

Usage: python3 src/make_submission.py

The final model is E2a, chosen by grouped-CV results among the candidates E1 and E2a. No Kaggle
score is used. The folds and the solver have no random step; SEED is recorded in the results row.
"""
import numpy as np
import pandas as pd

from baselines import (
    C_GRID, GROUP, RAW_FEATURES, RESULTS_PATH, ROOT, TARGET, grouped_folds, load_train, log_run,
    make_logistic, score_fold, summarise,
)
from experiments import frame_features, gap_features

ID = "decision_id"
TEST_PATH = ROOT / "data" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "sample_submission.csv"
OUTPUT_PATH = ROOT / "predictions" / "submission.csv"
REQUIRED_TEST_COLUMNS = [ID, GROUP, "problem_domain", "frame"] + RAW_FEATURES
FORBIDDEN_TEST_COLUMNS = [TARGET, "realised_payout_inr"]


def final_features(df):
    """The E2a features: raw columns + ev_gap + domain x frame indicators."""
    return pd.concat([gap_features(df), frame_features(df)], axis=1)


def choose_final_C(train, X):
    """Score every C on the same grouped folds and keep the highest mean AUC.

    C_GRID runs from strongest to weakest regularisation, so an exact tie keeps the stronger one.
    Returns the chosen C, its per-fold scores, and a table of all the grid's results.
    """
    y = train[TARGET].to_numpy()
    splits = list(grouped_folds(train))
    results, best_C, best_scores, best_auc = [], None, None, -np.inf
    for C in C_GRID:
        fold_scores = []
        for train_idx, val_idx in splits:
            model = make_logistic(C).fit(X[train_idx], y[train_idx])
            fold_scores.append(score_fold(y[val_idx], model.predict_proba(X[val_idx])[:, 1]))
        means = pd.DataFrame(fold_scores).mean()
        results.append({"C": C, "auc": means["auc"], "log_loss": means["log_loss"], "accuracy": means["accuracy"]})
        if means["auc"] > best_auc:
            best_C, best_scores, best_auc = C, fold_scores, means["auc"]
    return best_C, best_scores, pd.DataFrame(results)


def check_test_file(test, train):
    """Stop before predicting if test.csv is not what the model expects."""
    missing = [c for c in REQUIRED_TEST_COLUMNS if c not in test.columns]
    assert not missing, f"test.csv is missing columns: {missing}"
    present = [c for c in FORBIDDEN_TEST_COLUMNS if c in test.columns]
    assert not present, f"test.csv has columns it should not have: {present}"
    assert test[REQUIRED_TEST_COLUMNS].notna().all().all(), "test.csv has missing values in required columns"
    assert test[ID].is_unique, "test.csv has repeated decision ids"
    assert not set(test[GROUP]) & set(train[GROUP]), "a test participant also appears in training"
    assert (test["risky_high_inr"] > test["risky_low_inr"]).all(), "prob_high is not P(larger outcome) in test.csv"
    frame_features(test)  # stops on an unexpected domain x frame pair


def build_submission(test, sample, p):
    """Put the predictions into sample_submission.csv's format and row order, then validate."""
    by_id = pd.Series(p, index=test[ID].to_numpy())
    submission = pd.DataFrame({ID: sample[ID], TARGET: by_id.reindex(sample[ID]).to_numpy()})
    assert list(submission.columns) == list(sample.columns), "columns differ from sample_submission.csv"
    assert len(submission) == len(sample) == len(test), "row count differs from sample_submission.csv"
    assert submission[ID].equals(sample[ID]), "ids or row order differ from sample_submission.csv"
    assert set(submission[ID]) == set(test[ID]), "ids differ from test.csv"
    assert submission[TARGET].notna().all(), "predictions contain missing values"
    assert submission[TARGET].between(0, 1).all(), "predictions fall outside [0, 1]"
    return submission


def log_final_once(fold_scores, C):
    """Log the fixed-C CV row only if it is not already there, so rebuilding adds no duplicate."""
    if RESULTS_PATH.exists() and "final" in pd.read_csv(RESULTS_PATH)["experiment"].values:
        return
    log_run(summarise(
        "logistic_l2", fold_scores, experiment="final",
        features="E2a features: raw 4 + ev_gap + domain x frame cells",
        params=f"fixed C={C}; chosen by mean GroupKFold AUC on the same folds; grid={C_GRID}",
        notes="slightly optimistic: C was chosen on these folds; the nested estimate is the E2a row",
    ))


def main():
    train = load_train()
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    check_test_file(test, train)

    X = final_features(train)
    C, fold_scores, grid_results = choose_final_C(train, X.to_numpy())
    print(grid_results.round(4).to_string(index=False))
    print(f"chosen C={C}")
    log_final_once(fold_scores, C)

    model = make_logistic(C).fit(X.to_numpy(), train[TARGET].to_numpy())
    print("final model coefficients (per SD of each feature):")
    print(pd.Series(model[-1].coef_[0], index=X.columns).round(4).to_string())

    p = model.predict_proba(final_features(test).to_numpy())[:, 1]
    submission = build_submission(test, sample, p)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"wrote {OUTPUT_PATH} with {len(submission)} rows")
    print(submission.head().to_string(index=False))
    print(submission[TARGET].describe().round(4).to_string())


if __name__ == "__main__":
    main()
