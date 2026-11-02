"""Error analysis and calibration for the final model, on training data only.

Usage: python3 src/error_analysis.py

Every probability here is out-of-fold: it comes from a model that did not train on that
participant. test.csv is never read. Figures go to figures/ and tables to results/error_analysis/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from baselines import GROUP, ROOT, SEED, TARGET, THRESHOLD, ev_gap, grouped_folds, load_train, make_logistic, score_fold
from make_submission import final_features

FINAL_C = 0.1
N_BINS = 10
N_BOOTSTRAP = 1000
CONFIDENT_HIGH = 0.8
CONFIDENT_LOW = 0.2
FIGURES = ROOT / "figures"
TABLES = ROOT / "results" / "error_analysis"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def out_of_fold(train):
    """Return train with each row's out-of-fold probability and its fold number."""
    X = final_features(train).to_numpy()
    y = train[TARGET].to_numpy()
    p, fold_of = np.zeros(len(train)), np.zeros(len(train), dtype=int)
    for fold, (train_idx, val_idx) in enumerate(grouped_folds(train), start=1):
        model = make_logistic(FINAL_C).fit(X[train_idx], y[train_idx])
        p[val_idx] = model.predict_proba(X[val_idx])[:, 1]
        fold_of[val_idx] = fold
    return train.assign(p=p, fold=fold_of, ev_gap=ev_gap(train))


def subgroup_row(df, label):
    """Score one subgroup in each fold, then report mean and SD across folds, as in the CV runs."""
    scores = pd.DataFrame([score_fold(d[TARGET], d["p"]) for _, d in df.groupby("fold")])
    return {
        "group": label, "rows": len(df), "participants": df[GROUP].nunique(),
        "risky_rate": df[TARGET].mean(), "mean_p": df["p"].mean(),
        "auc": scores["auc"].mean(), "auc_sd": scores["auc"].std(),
        "log_loss": scores["log_loss"].mean(), "log_loss_sd": scores["log_loss"].std(),
        "accuracy": scores["accuracy"].mean(), "accuracy_sd": scores["accuracy"].std(),
    }


def subgroup_table(df):
    gain = df[df["problem_domain"] == "gain"]
    rows = [subgroup_row(df, "all rows")]
    rows += [subgroup_row(d, f"domain: {k}") for k, d in df.groupby("problem_domain")]
    rows += [subgroup_row(d, f"gain problems, frame: {k}") for k, d in gain.groupby("frame")]
    rows += [subgroup_row(d, f"session: {k}") for k, d in df.groupby("session_type")]
    rows += [subgroup_row(d, f"time pressure: {k}") for k, d in df.groupby("time_pressure")]
    return pd.DataFrame(rows)


def participant_table(df):
    """One row per participant, sorted hardest first. Ties in accuracy are broken by log-loss."""
    rows = []
    for person, d in df.groupby(GROUP):
        both_classes = d[TARGET].nunique() == 2
        rows.append({
            GROUP: person,
            "accuracy": accuracy_score(d[TARGET], d["p"] >= THRESHOLD),
            "log_loss": log_loss(d[TARGET], d["p"], labels=[0, 1]),
            "auc": roc_auc_score(d[TARGET], d["p"]) if both_classes else np.nan,
            "risky_rate": d[TARGET].mean(), "mean_p": d["p"].mean(),
            "session_type": d["session_type"].iloc[0], "numeracy_score": d["numeracy_score"].iloc[0],
            "risk_taking": d["self_reported_risk_taking"].iloc[0], "age_years": d["age_years"].iloc[0],
        })
    table = pd.DataFrame(rows)
    return table.sort_values(["accuracy", "log_loss"], ascending=[True, False]).reset_index(drop=True)


def bin_table(df):
    """Equal-width reliability bins, with a Wilson 95% interval on each observed rate."""
    edges = np.linspace(0, 1, N_BINS + 1)
    bins = np.minimum(np.digitize(df["p"], edges[1:-1], right=True), N_BINS - 1)
    rows = []
    for b in range(N_BINS):
        d = df[bins == b]
        n, observed = len(d), d[TARGET].mean() if len(d) else np.nan
        mean_p = d["p"].mean() if n else np.nan
        low, high = wilson_interval(observed, n) if n else (np.nan, np.nan)
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "rows": n, "mean_p": mean_p, "observed": observed,
            "gap": observed - mean_p, "wilson_low": low, "wilson_high": high,
        })
    return pd.DataFrame(rows)


def wilson_interval(rate, n, z=1.96):
    denominator = 1 + z ** 2 / n
    centre = (rate + z ** 2 / (2 * n)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)) / denominator
    return centre - half, centre + half


def expected_calibration_error(df):
    """Average |observed rate - mean probability| across bins, weighted by bin size."""
    table = bin_table(df).dropna(subset=["gap"])
    return (table["rows"] / table["rows"].sum() * table["gap"].abs()).sum()


def calibration_slope(df):
    """Regress the choice on the logit of p. Perfect calibration gives slope 1, intercept 0."""
    logit = np.log(df["p"] / (1 - df["p"])).to_numpy().reshape(-1, 1)
    fit = LogisticRegression(penalty=None, max_iter=2000).fit(logit, df[TARGET])
    return fit.coef_[0, 0], fit.intercept_[0]


def bootstrap_ece(df):
    """Resample whole participants, because rows from one person are correlated."""
    rng = np.random.default_rng(SEED)
    by_person = {person: d for person, d in df.groupby(GROUP)}
    people = list(by_person)
    rows = []
    for _ in range(N_BOOTSTRAP):
        sample = pd.concat([by_person[people[i]] for i in rng.integers(0, len(people), len(people))])
        paid = expected_calibration_error(sample[sample["session_type"] == "paid"])
        hypothetical = expected_calibration_error(sample[sample["session_type"] == "hypothetical"])
        rows.append({
            "all": expected_calibration_error(sample), "paid": paid, "hypothetical": hypothetical,
            "paid_minus_hypothetical": paid - hypothetical,
        })
    return pd.DataFrame(rows).quantile([0.025, 0.975])


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def draw_reliability(top, bottom, df, color, title):
    table = bin_table(df)
    filled = table.dropna(subset=["observed"])
    top.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    top.errorbar(
        filled["mean_p"], filled["observed"],
        yerr=[filled["observed"] - filled["wilson_low"], filled["wilson_high"] - filled["observed"]],
        color=color, ecolor=color, elinewidth=1, linewidth=2, marker="o", markersize=7,
        markeredgecolor=SURFACE, markeredgewidth=1.5, label="Observed rate (95% Wilson interval)",
    )
    top.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    top.set_title(title, loc="left", color=INK, fontsize=11)
    top.set_ylabel("Observed risky-choice rate", color=MUTED)
    top.tick_params(labelbottom=False)
    style_axis(top)
    centres = (np.arange(N_BINS) + 0.5) / N_BINS
    bottom.bar(centres, table["rows"], width=0.09, color=color)
    bottom.set_xlim(0, 1)
    bottom.set_xlabel("Mean predicted probability", color=MUTED)
    bottom.set_ylabel("Rows", color=MUTED)
    style_axis(bottom)
    bottom.grid(False, axis="x")


def plot_calibration(df, path):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.6, 7), height_ratios=[4, 1], sharex=True, facecolor=SURFACE)
    draw_reliability(top, bottom, df, BLUE, f"All out-of-fold predictions (ECE {expected_calibration_error(df):.3f})")
    top.legend(frameon=False, loc="upper left", labelcolor=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_calibration_by_session(df, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), height_ratios=[4, 1], sharex=True, sharey="row", facecolor=SURFACE)
    for col, (session, color) in enumerate([("paid", BLUE), ("hypothetical", ORANGE)]):
        d = df[df["session_type"] == session]
        title = f"{session.capitalize()} sessions (ECE {expected_calibration_error(d):.3f})"
        draw_reliability(axes[0, col], axes[1, col], d, color, title)
    axes[0, 0].legend(frameon=False, loc="upper left", labelcolor=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_auc_by_domain(df, path):
    """One small dot per fold and a larger dot for each domain's mean across folds."""
    per_fold = {
        domain: [roc_auc_score(f[TARGET], f["p"]) for _, f in d.groupby("fold")]
        for domain, d in df.groupby("problem_domain")
    }
    order = sorted(per_fold, key=lambda k: np.mean(per_fold[k]))
    overall = np.mean([roc_auc_score(f[TARGET], f["p"]) for _, f in df.groupby("fold")])
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    for y, domain in enumerate(order):
        ax.scatter(per_fold[domain], [y] * 5, s=28, color=BLUE, alpha=0.35, linewidths=0, zorder=2,
                   label="One fold" if y == 0 else None)
        mean = np.mean(per_fold[domain])
        ax.scatter([mean], [y], s=90, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                   label="Mean of 5 folds" if y == 0 else None)
        ax.annotate(f"{mean:.3f}", (mean, y), xytext=(0, 9), textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.axvline(overall, color=MUTED, linewidth=1, linestyle="--", label=f"All rows ({overall:.3f})")
    ax.set_yticks(range(len(order)), [d.capitalize() for d in order], color=INK)
    ax.set(xlim=(0.5, 1), ylim=(-0.6, len(order) - 0.3))
    ax.set_xlabel("Out-of-fold ROC-AUC (0.5 = no ranking ability)", color=MUTED)
    ax.set_title("AUC of the final model by problem domain", loc="left", color=INK, fontsize=11)
    style_axis(ax)
    ax.grid(False, axis="y")
    ax.legend(frameon=False, loc="lower right", labelcolor=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def confident_rows(df):
    """Add each row's direction, whether it was wrong, and its loss. Keeps confident rows only."""
    confident = df[(df["p"] > CONFIDENT_HIGH) | (df["p"] < CONFIDENT_LOW)].copy()
    confident["direction"] = np.where(confident["p"] > CONFIDENT_HIGH, "p>0.8", "p<0.2")
    confident["wrong"] = (confident["p"] > 0.5).astype(int) != confident[TARGET]
    confident["row_loss"] = -np.log(np.where(confident[TARGET] == 1, confident["p"], 1 - confident["p"]))
    # The person's risky rate on their other 39 decisions, so this row's own label is left out.
    totals = df.groupby(GROUP)[TARGET].agg(["sum", "count"])
    person = confident[GROUP].map(totals["sum"]) - confident[TARGET]
    confident["person_rate_others"] = person / (confident[GROUP].map(totals["count"]) - 1)
    confident["abs_ev_gap"] = confident["ev_gap"].abs()
    return confident


def confident_summary(confident):
    """Compare wrong with right rows inside each direction, so the direction itself is held fixed."""
    columns = ["abs_ev_gap", "person_rate_others", "numeracy_score", "self_reported_risk_taking",
               "age_years", "time_pressure", "response_time_ms", "trial_index", "prob_high"]
    rows = []
    for (direction, wrong), d in confident.groupby(["direction", "wrong"]):
        row = {"direction": direction, "outcome": "wrong" if wrong else "right", "rows": len(d), "mean_p": d["p"].mean()}
        row.update(d[columns].mean().to_dict())
        row["paid_share"] = (d["session_type"] == "paid").mean()
        row["mixed_share"] = (d["problem_domain"] == "mixed").mean()
        row["loss_share"] = (d["problem_domain"] == "loss").mean()
        row["gain_share"] = (d["problem_domain"] == "gain").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240, "display.max_columns", 40)
    df = out_of_fold(load_train())

    subgroups = subgroup_table(df)
    subgroups.to_csv(TABLES / "subgroups.csv", index=False)
    print(subgroups.round(4).to_string(index=False))

    people = participant_table(df)
    people.to_csv(TABLES / "participants.csv", index=False)
    print("hardest 10 participants")
    print(people.head(10).round(4).to_string(index=False))
    print("easiest 10 participants")
    print(people.tail(10).iloc[::-1].round(4).to_string(index=False))
    expected = df.assign(sure=np.maximum(df["p"], 1 - df["p"])).groupby(GROUP)["sure"].mean()
    binomial_sd = np.sqrt((expected * (1 - expected) / 40).mean())
    print(f"participant accuracy: mean={people['accuracy'].mean():.4f} sd={people['accuracy'].std():.4f} "
          f"sd expected from chance alone={binomial_sd:.4f}")

    for name, d in [("all", df), ("paid", df[df["session_type"] == "paid"]), ("hypothetical", df[df["session_type"] == "hypothetical"])]:
        table = bin_table(d)
        table.to_csv(TABLES / f"calibration_{name}.csv", index=False)
        slope, intercept = calibration_slope(d)
        print(f"calibration: {name} rows={len(d)} ECE={expected_calibration_error(d):.4f} "
              f"MCE={table['gap'].abs().max():.4f} slope={slope:.3f} intercept={intercept:.3f}")
        print(table.round(4).to_string(index=False))
    print("bootstrap 95% intervals for ECE, resampling participants")
    print(bootstrap_ece(df).round(4).to_string())

    plot_calibration(df, FIGURES / "calibration.png")
    plot_calibration_by_session(df, FIGURES / "calibration_by_session.png")
    plot_auc_by_domain(df, FIGURES / "auc_by_domain.png")

    confident = confident_rows(df)
    print("confident rows by direction")
    print(confident.groupby("direction").agg(
        rows=("wrong", "size"), wrong=("wrong", "sum"), wrong_rate=("wrong", "mean"), mean_p=("p", "mean"),
    ).round(4).to_string())
    wrong = confident[confident["wrong"]]
    print(f"confidently wrong: {len(wrong)} rows from {wrong[GROUP].nunique()} participants; "
          f"top 10 participants hold {wrong[GROUP].value_counts().head(10).sum()} rows")
    print(pd.crosstab([confident["problem_domain"], confident["frame"]], [confident["direction"], confident["wrong"]]).to_string())
    summary = confident_summary(confident)
    summary.to_csv(TABLES / "confident_summary.csv", index=False)
    print(summary.round(3).T.to_string())
    columns = ["decision_id", GROUP, "problem_domain", "frame", "prob_high", "risky_high_inr", "risky_low_inr",
               "sure_amount_inr", "ev_gap", "p", TARGET, "row_loss", "person_rate_others", "numeracy_score", "session_type", "time_pressure"]
    largest = wrong.sort_values("row_loss", ascending=False)[columns]
    largest.to_csv(TABLES / "confidently_wrong.csv", index=False)
    print("largest failures")
    print(largest.head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
