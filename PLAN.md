# Project Plan: IML Midterm, Track B (risky choice)

Research question: do theory-motivated features (EV gap, final wealth, probability of the larger outcome, domain x frame interaction) improve a regularised logistic regression's ranking of risky choices for unseen participants?

Deadline: Tue 3 Nov, 23:59 IST. Late work loses 10% per day. Aim to upload by the afternoon of 3 Nov.

## Rules to remember
- Drop realised_payout_inr. Exclude response_time_ms from the main models. participant_id is the group label, never a feature.
- 5-fold GroupKFold by participant. Preprocessing and tuning happen inside training folds only. Test file used once, at the end.
- Metrics: ROC-AUC (primary), log-loss, accuracy at a stated threshold, calibration plot.
- Log every reported run in experimental_results.csv. Fixed seed. One command rebuilds everything.
- Declare AI tools used in the README. I must be able to explain every line.

## Saturday 31 Oct: baselines and first experiments
- [ ] Write run_constant, ev_gap, run_ev_rule, logistic regression (src/baselines.py)
- [ ] Add accuracy to score_fold
- [ ] Add LDA or Gaussian Naive Bayes, and a linear SVM
- [ ] Run all baselines and log results
- [ ] E1: theory features vs raw attributes (logistic regression)
- [ ] E2a: add frame and domain x frame interaction

## Sunday 1 Nov: experiments, analysis, submission
- [ ] E2b: add context variables (session type, time pressure, numeracy, risk taking, age)
- [ ] E3: random KFold vs GroupKFold with the same model
- [ ] Choose final model by CV only; refit on all training data
- [ ] Error analysis: by domain, frame, session type, per participant
- [ ] Calibration plot (reliability), paid vs hypothetical if time
- [ ] Predict test once; make the prediction file in sample_submission.csv format; submit to Kaggle
- [ ] Optional: E5 (response_time_ms as a labelled side experiment), bootstrap intervals over participants

## Monday 2 Nov: writing
- [ ] Literature review: 3 to 6 sources (prospect theory, framing, calibration; use the guide's references)
- [ ] Report, 4 to 6 pages, from the template (or the guide's sections): question; background and related work; data and preprocessing; methods and experimental design; results; error analysis and limitations; conclusion
- [ ] State clearly that the data is simulated
- [ ] README: environment, exact commands, seeds, runtime, map from each table or figure to its code, AI declaration
- [ ] Clean-run check: rebuild everything from scratch with the one command

## Tuesday 3 Nov: final checks and upload
- [ ] Proofread the report; check every figure and table is referenced
- [ ] Confirm the Kaggle final submission is selected
- [ ] Upload to Classroom: report PDF, code, README, prediction file, experimental_results.csv
- [ ] No new experiments on this day

## Missing items to chase
- [ ] Email samarjeet.malik_tf@ashoka.edu.in for the report, README and results CSV templates
- [ ] Confirm the deadline and time in Classroom (3 Nov, 23:59 IST)

## Viva prep (after the date is announced)
- [ ] Be able to explain: why GroupKFold, why each column was dropped, what each feature means, why each baseline, what AUC means, what the results show and their limits

## Open decisions
- [ ] Accuracy threshold: fixed 0.5 (state it), or tuned inside the training folds only
- [ ] Linear SVM has no probabilities: calibrate its scores inside the training fold, or report AUC only for it
- [ ] "Probability of the larger outcome": prob_high already is that for risky_high_inr, so define what the feature adds (e.g. a transform, or probability of the larger final wealth) or drop it
- [ ] E2b adds fingerprint columns (age, numeracy, risk taking, session type are constant per person): run E3 with the same feature sets as E2
- [ ] EV rule log-loss: how to turn the rupee EV gap into a probability (fit on the training fold only), or report AUC only
- [ ] Whether endowment_inr counts as a raw gamble attribute in baseline 3
