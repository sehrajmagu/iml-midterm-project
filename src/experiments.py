"""Controlled experiments for Track B. Each one changes one thing from a baseline.

Usage: python3 src/experiments.py {e1,e1_ablation,e2a,e2a_clean}
"""
import sys

import numpy as np
import pandas as pd

from baselines import (
    RAW_FEATURES, cross_validate_logistic, ev_gap, load_train, log_run, logistic_params,
    print_logistic_folds, summarise,
)

WEALTH_FEATURES = ["wealth_if_high", "wealth_if_low", "wealth_if_sure"]
E1_FEATURES = RAW_FEATURES + ["ev_gap"] + WEALTH_FEATURES
GAP_FEATURES = RAW_FEATURES + ["ev_gap"]
FRAME_INVARIANT_FEATURES = ["prob_high", "ev_gap"] + WEALTH_FEATURES
FRAME_FEATURES = ["gain_x_loss_frame", "loss_x_no_frame", "mixed_x_no_frame"]
VALID_CELLS = {("gain", "gain_frame"), ("gain", "loss_frame"), ("loss", "none"), ("mixed", "none")}


def theory_features(df):
    """Add the E1 columns, each computed from that row's presented columns only.

    The probability of the larger outcome is prob_high, because risky_high_inr is always
    the larger outcome. It is already in RAW_FEATURES, so no duplicate column is added.
    """
    assert (df["risky_high_inr"] > df["risky_low_inr"]).all(), "prob_high is not P(larger outcome)"
    out = df[RAW_FEATURES].copy()
    out["ev_gap"] = ev_gap(df)
    out["wealth_if_high"] = df["endowment_inr"] + df["risky_high_inr"]
    out["wealth_if_low"] = df["endowment_inr"] + df["risky_low_inr"]
    out["wealth_if_sure"] = df["endowment_inr"] + df["sure_amount_inr"]
    return out[E1_FEATURES]


def gap_features(df):
    """The raw columns plus ev_gap: no endowment and no frame."""
    out = df[RAW_FEATURES].copy()
    out["ev_gap"] = ev_gap(df)
    return out[GAP_FEATURES]


def frame_features(df):
    """Indicators for the domain x frame cells, with gain x gain_frame as the reference.

    Only four cells exist, so these three columns also represent frame's main effect. A
    separate frame column would be an exact copy or sum of them.
    """
    cells = set(zip(df["problem_domain"], df["frame"]))
    assert cells <= VALID_CELLS, f"unexpected domain x frame cells: {cells - VALID_CELLS}"
    out = pd.DataFrame(index=df.index)
    out["gain_x_loss_frame"] = ((df["problem_domain"] == "gain") & (df["frame"] == "loss_frame")).astype(int)
    out["loss_x_no_frame"] = ((df["problem_domain"] == "loss") & (df["frame"] == "none")).astype(int)
    out["mixed_x_no_frame"] = ((df["problem_domain"] == "mixed") & (df["frame"] == "none")).astype(int)
    return out[FRAME_FEATURES]


def fold_table(folds):
    return pd.DataFrame([result["scores"] for result in folds])


def print_paired_difference(new, baseline):
    """Print new minus baseline for each metric, compared on the same validation folds."""
    diff = fold_table(new) - fold_table(baseline)
    for metric in ["auc", "log_loss"]:
        d = diff[metric]
        print(f"{metric} difference by fold: {np.round(d.to_numpy(), 4)} mean={d.mean():+.4f} sd={d.std():.4f}")


def run_e1(train):
    """E1: the logistic baseline with theory features added. Only the features change."""
    baseline = cross_validate_logistic(train, train[RAW_FEATURES].to_numpy())
    folds = cross_validate_logistic(train, theory_features(train).to_numpy())
    print_logistic_folds(folds)
    print_paired_difference(folds, baseline)
    print("fold 1 coefficients (per SD of each feature):")
    print(pd.Series(folds[0]["model"][-1].coef_[0], index=E1_FEATURES).round(4).to_string())
    return [summarise(
        "logistic_l2", [result["scores"] for result in folds], experiment="E1",
        features="raw 4 + ev_gap + final wealth (high, low, sure); P(larger outcome)=prob_high, no duplicate column",
        params=logistic_params(folds),
    )]


def run_e1_ablation(train):
    """Log the two parts of E1 separately: ev_gap alone, and endowment alone."""
    feature_sets = {
        "raw 4 + ev_gap (no endowment)": gap_features(train),
        "raw 4 + endowment_inr only": train[RAW_FEATURES + ["endowment_inr"]],
    }
    rows = []
    for description, X in feature_sets.items():
        print(description)
        folds = cross_validate_logistic(train, X.to_numpy())
        print_logistic_folds(folds)
        rows.append(summarise(
            "logistic_l2", [result["scores"] for result in folds], experiment="E1-ablation",
            features=description, params=logistic_params(folds),
        ))
    return rows


def run_e2a(train):
    """E2a: add explicit frame information to raw + ev_gap. Only those columns change."""
    gap = gap_features(train)
    with_frame = pd.concat([gap, frame_features(train)], axis=1)
    baseline = cross_validate_logistic(train, gap.to_numpy())
    folds = cross_validate_logistic(train, with_frame.to_numpy())
    print_logistic_folds(folds)
    print_paired_difference(folds, baseline)
    print("fold 1 coefficients (per SD of each feature):")
    print(pd.Series(folds[0]["model"][-1].coef_[0], index=with_frame.columns).round(4).to_string())
    return [summarise(
        "logistic_l2", [result["scores"] for result in folds], experiment="E2a",
        features="raw 4 + ev_gap + frame and domain x frame cells (gain x loss_frame, loss x none, mixed x none); no endowment",
        params=logistic_params(folds),
    )]


def run_e2a_clean(train):
    """Clean frame test: features that do not change with the frame, with vs without a flag."""
    invariant = theory_features(train)[FRAME_INVARIANT_FEATURES]
    with_flag = invariant.assign(loss_frame=(train["frame"] == "loss_frame").astype(int))
    baseline = cross_validate_logistic(train, invariant.to_numpy())
    folds = cross_validate_logistic(train, with_flag.to_numpy())
    print("without flag")
    print_logistic_folds(baseline)
    print("with flag")
    print_logistic_folds(folds)
    print_paired_difference(folds, baseline)
    print("loss_frame coefficient by fold:", [round(result["model"][-1].coef_[0, -1], 4) for result in folds])
    feature_sets = {
        "prob_high + ev_gap + final wealth (high, low, sure); no raw amounts; no frame flag": baseline,
        "prob_high + ev_gap + final wealth (high, low, sure); no raw amounts; plus loss_frame flag": folds,
    }
    return [
        summarise(
            "logistic_l2", [result["scores"] for result in fold_results], experiment="E2a-clean",
            features=description, params=logistic_params(fold_results),
        )
        for description, fold_results in feature_sets.items()
    ]


RUNNERS = {"e1": run_e1, "e1_ablation": run_e1_ablation, "e2a": run_e2a, "e2a_clean": run_e2a_clean}


def main():
    rows = RUNNERS[sys.argv[1]](load_train())
    for row in rows:
        log_run(row)
    print(pd.DataFrame(rows)[["experiment", "features", "auc_mean", "auc_std", "log_loss_mean", "log_loss_std", "accuracy_mean", "accuracy_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
