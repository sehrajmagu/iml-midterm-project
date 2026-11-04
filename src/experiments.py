"""Controlled experiments for Track B. Each one changes one thing from a baseline.

Usage: python3 src/experiments.py {e1,e1_ablation,e2a,e2a_clean,e3}
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from baselines import (
    N_SPLITS, RAW_FEATURES, SEED, cross_validate_logistic, ev_gap, load_train, log_run, logistic_row,
    print_logistic_folds, print_rows, timed_logistic,
)

WEALTH_FEATURES = ["wealth_if_high", "wealth_if_low", "wealth_if_sure"]
E1_FEATURES = RAW_FEATURES + ["ev_gap"] + WEALTH_FEATURES
GAP_FEATURES = RAW_FEATURES + ["ev_gap"]
FRAME_INVARIANT_FEATURES = ["prob_high", "ev_gap"] + WEALTH_FEATURES
FRAME_FEATURES = ["gain_x_loss_frame", "loss_x_no_frame", "mixed_x_no_frame"]
VALID_CELLS = {("gain", "gain_frame"), ("gain", "loss_frame"), ("loss", "none"), ("mixed", "none")}
PERSON_FEATURES = ["age_years", "numeracy_score", "self_reported_risk_taking"]


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
    folds, runtime_s = timed_logistic(train, theory_features(train).to_numpy())
    print_logistic_folds(folds)
    print_paired_difference(folds, baseline)
    print("fold 1 coefficients (per SD of each feature):")
    print(pd.Series(folds[0]["model"][-1].coef_[0], index=E1_FEATURES).round(4).to_string())
    return [logistic_row(
        folds, runtime_s, "B06", "E1", "raw4+ev_gap+wealth", "src/experiments.py#run_e1",
        notes="wealth = endowment + each outcome; P(larger outcome)=prob_high (no duplicate column)",
    )]


def run_e1_ablation(train):
    """Log the two parts of E1 separately: ev_gap alone, and endowment alone."""
    runs = [
        ("B07", "raw4+ev_gap", gap_features(train), "E1 ablation: no endowment"),
        ("B08", "raw4+endowment", train[RAW_FEATURES + ["endowment_inr"]], "E1 ablation"),
    ]
    rows = []
    for run_id, feature_set, X, notes in runs:
        print(feature_set)
        folds, runtime_s = timed_logistic(train, X.to_numpy())
        print_logistic_folds(folds)
        rows.append(logistic_row(
            folds, runtime_s, run_id, "E1-ablation", feature_set, "src/experiments.py#run_e1_ablation", notes=notes,
        ))
    return rows


def run_e2a(train):
    """E2a: add explicit frame information to raw + ev_gap. Only those columns change."""
    gap = gap_features(train)
    with_frame = pd.concat([gap, frame_features(train)], axis=1)
    baseline = cross_validate_logistic(train, gap.to_numpy())
    folds, runtime_s = timed_logistic(train, with_frame.to_numpy())
    print_logistic_folds(folds)
    print_paired_difference(folds, baseline)
    print("fold 1 coefficients (per SD of each feature):")
    print(pd.Series(folds[0]["model"][-1].coef_[0], index=with_frame.columns).round(4).to_string())
    return [logistic_row(
        folds, runtime_s, "B09", "E2a", "raw4+ev_gap+domain_x_frame", "src/experiments.py#run_e2a",
        notes=("indicators gain x loss_frame / loss x none / mixed x none; no endowment; compare with B07; "
               "also the grouped side of E3"),
    )]


def run_e2a_clean(train):
    """Clean frame test: features that do not change with the frame, with vs without a flag."""
    invariant = theory_features(train)[FRAME_INVARIANT_FEATURES]
    with_flag = invariant.assign(loss_frame=(train["frame"] == "loss_frame").astype(int))
    baseline, baseline_runtime_s = timed_logistic(train, invariant.to_numpy())
    folds, runtime_s = timed_logistic(train, with_flag.to_numpy())
    print("without flag")
    print_logistic_folds(baseline)
    print("with flag")
    print_logistic_folds(folds)
    print_paired_difference(folds, baseline)
    print("loss_frame coefficient by fold:", [round(result["model"][-1].coef_[0, -1], 4) for result in folds])
    code_reference = "src/experiments.py#run_e2a_clean"
    return [
        logistic_row(
            baseline, baseline_runtime_s, "B10", "E2a-clean", "frame_invariant", code_reference,
            notes="frame_invariant = prob_high+ev_gap+wealth; no raw amounts; no frame flag",
        ),
        logistic_row(
            folds, runtime_s, "B11", "E2a-clean", "frame_invariant+loss_frame", code_reference, notes="compare with B10",
        ),
    ]


def random_row_folds(train):
    """Shuffled KFold on rows, so one participant's rows can be on both sides of a split."""
    return list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(train))


def run_e3(train):
    """E3: grouped vs random validation on the E2a model. Only the split changes."""
    e2a = pd.concat([gap_features(train), frame_features(train)], axis=1)
    person = "person = age_years+numeracy_score+self_reported_risk_taking"
    random_note = f"row-level KFold shuffle seed {SEED} with row-level inner split"
    # (feature_set, X, grouped run_id or None, grouped notes, random run_id, random notes)
    runs = [
        ("raw4+ev_gap+domain_x_frame", e2a, None, "",
         "B12", f"{random_note}; grouped counterpart is B09"),
        ("raw4+ev_gap+domain_x_frame+person", pd.concat([e2a, train[PERSON_FEATURES]], axis=1), "B13", f"diagnostic; {person}",
         "B14", f"diagnostic; {random_note}; grouped counterpart is B13"),
    ]
    rows = []
    for feature_set, X, grouped_id, grouped_notes, random_id, random_notes in runs:
        print(feature_set)
        grouped, grouped_runtime_s = timed_logistic(train, X.to_numpy())
        # Giving every row its own group turns the inner grouped split into a row-level split.
        random, random_runtime_s = timed_logistic(
            train, X.to_numpy(), splits=random_row_folds(train), inner_groups=np.arange(len(train)),
        )
        print("grouped")
        print_logistic_folds(grouped)
        print("random")
        print_logistic_folds(random)
        difference = fold_table(random).mean() - fold_table(grouped).mean()
        print(f"random minus grouped: auc={difference['auc']:+.4f} log_loss={difference['log_loss']:+.4f}")
        code_reference = "src/experiments.py#run_e3"
        # The grouped run on the E2a features repeats B09 exactly, so it is not logged again.
        if grouped_id:
            rows.append(logistic_row(grouped, grouped_runtime_s, grouped_id, "E3", feature_set, code_reference, notes=grouped_notes))
        rows.append(logistic_row(
            random, random_runtime_s, random_id, "E3", feature_set, code_reference, notes=random_notes,
            validation_scheme=f"RandomKFold{N_SPLITS}",
        ))
    return rows


RUNNERS = {
    "e1": run_e1, "e1_ablation": run_e1_ablation, "e2a": run_e2a, "e2a_clean": run_e2a_clean,
    "e3": run_e3,
}


def main():
    rows = RUNNERS[sys.argv[1]](load_train())
    for row in rows:
        log_run(row)
    print_rows(rows)


if __name__ == "__main__":
    main()
