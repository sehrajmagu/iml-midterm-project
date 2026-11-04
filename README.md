# IML 2026 Track B — Simulated Risky Choice

> **SIMULATED DATA.** Every row was produced by a documented computer simulation written for
> CS-3410 / MAT-3211-1 *Introduction to Machine Learning* (Ashoka University, Monsoon 2026).
> No row is a decision made by a real person, and no human participants were involved. Do not
> cite these data as evidence about human behaviour.

## Task
Predict whether a decision-maker chooses the **risky** option over the **sure** option. Study
whether framing and context features improve prediction, and whether models generalise to
**people they have never seen**.

| File | Rows | Contents |
|---|---:|---|
| `train.csv` | 8,400 | 210 simulated participants × 40 decisions, with `chose_risky` |
| `test.csv` | 3,600 | 90 **different** simulated participants × 40 decisions, features only |
| `sample_submission.csv` | 3,600 | `decision_id,chose_risky` (predicted probability in [0, 1]) |

Primary metric: **ROC-AUC** of the predicted probability (organiser choice; see the course guide).

## Data dictionary

| Column | Type | Unit | Description | Available when predicting? |
|---|---|---|---|---|
| `decision_id` | id | — | Random identifier | yes (do not use as a feature) |
| `participant_id` | category | — | Simulated participant (test participants never appear in train) | yes |
| `trial_index` | int | — | Position of the decision in the session (1–40) | yes |
| `session_type` | category | — | `paid` (one decision paid out) or `hypothetical` | yes |
| `age_years` | int | years | Simulated age | yes |
| `numeracy_score` | int | 0–8 | Simulated numeracy test score | yes |
| `self_reported_risk_taking` | int | 1–10 | Simulated questionnaire item | yes |
| `problem_domain` | category | — | `gain`, `loss`, or `mixed` (wins and losses) | yes |
| `frame` | category | — | `gain_frame` / `loss_frame` (gain problems only) or `none` | yes |
| `time_pressure` | 0/1 | — | Decision made under a countdown | yes |
| `endowment_inr` | float | ₹ | Amount received before the choice | yes |
| `prob_high` | float | — | Probability of the risky option's first outcome | yes |
| `risky_high_inr` | float | ₹ | Risky outcome with probability `prob_high`, **as presented** | yes |
| `risky_low_inr` | float | ₹ | Risky outcome with probability `1 - prob_high`, as presented | yes |
| `sure_amount_inr` | float | ₹ | Sure option, as presented (0 for mixed problems = reject the gamble) | yes |
| `response_time_ms` | float | ms | Time taken to choose | yes (recorded with the choice; justify if used) |
| `realised_payout_inr` | float | ₹ | What the participant was paid **after** choosing (paid sessions; train only) | **no** |
| `chose_risky` | target | 0/1 | 1 if the risky option was chosen (train only) | target |

Final wealth after a choice is `endowment_inr` + the presented outcome. In a **loss frame** the
same final-wealth problem as a gain-frame problem is shown as deductions from a larger
endowment.

## How the data were simulated (summary)
Choices follow cumulative prospect theory (Tversky and Kahneman, 1992). Each participant has
their own curvature, loss aversion and probability-weighting parameters, centred on published
median estimates. Some simulated participants partly adopt the presented frame as their
reference point (Tversky and Kahneman, 1981). Choices are noisy through a logit rule whose noise
depends on numeracy, payment and time pressure, and each participant has an idiosyncratic
preference for risk. Exact parameter distributions are withheld.

## Known pitfalls (by design)
* Rows from the same participant are correlated. A random row-level split puts the same person
  in both training and validation data and overstates performance on new people.
* `realised_payout_inr` is determined by the choice itself. Using it is leakage.
* Expected value alone cannot see the frame, because both frames have identical expected final
  wealth.

Licence: CC BY 4.0. Citation: *IML 2026 Track B simulated risky-choice dataset, Ashoka University (simulated)*.
Kaggle URL: **To be announced.**

## Reproducing the results

Install the pinned environment with `python3 -m pip install -r requirements.txt`, then run
`python3 run_all.py` (optionally `--seed 0`). It regenerates `experimental_results.csv`,
`predictions/submission.csv`, `figures/` and `results/error_analysis/` in about 70 seconds.

**Hidden warnings.** On the development machine (macOS arm64, NumPy 2.0.2), NumPy's matrix
multiply raises `RuntimeWarning: divide by zero / overflow / invalid value encountered in matmul`
thousands of times during scikit-learn fits. `run_all.py` hides only these three messages so the
output is readable; all other warnings still appear. They do not affect the results: refitting the
models with an independent Newton solver gave the same log-loss to within about 0.00001, and the
outputs are identical with and without the filter. The cause was not identified. Running a script
in `src/` directly does not apply the filter, so the warnings appear there.
