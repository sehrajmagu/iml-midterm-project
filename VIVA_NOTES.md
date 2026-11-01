# Viva Notes

## Step 1: Constant baseline (`run_constant`)

**What was done:** Added the shared protocol to `src/baselines.py` (load the training data, 5-fold GroupKFold by participant, score each fold, log the run) and the constant model. In each fold, the model predicts the training folds' risky-choice rate for every validation row.

**Result:** AUC 0.5000 (SD 0.0000), log-loss 0.6927 (SD 0.0012).

**Why:** It sets the floor. A model that uses the features has to beat AUC 0.5 and log-loss of about 0.693 to have learned anything. It also tests the protocol on a model whose answer is known in advance.

**Likely viva question:** Why is the AUC exactly 0.5, and why report log-loss as well?

**Short answer:** AUC measures how well a model ranks risky choices above non-risky ones. A constant gives every row the same score, so all pairs tie, and ties count as half, giving 0.5. Log-loss measures the probability values themselves. For a constant it is about the entropy of the base rate: with a rate near 0.48, it is close to ln 2, or 0.693.

**Likely follow-up:** The starter notebook says the constant's AUC is "not exactly 0.5". Why is yours exactly 0.5?

**Short answer:** The starter pools all out-of-fold predictions before computing one AUC, and each fold has a slightly different constant, so those differences create an artificial ranking. I compute AUC within each fold and then average, so every score within a fold is tied.

## Step 2: Expected-value rule (`ev_gap`, `run_ev_rule`)

**What was done:** `ev_gap` computes `prob_high x risky_high + (1 - prob_high) x risky_low - sure_amount`, in rupees. `run_ev_rule` ranks decisions by that gap to get AUC, with nothing learned. For log-loss only, a one-feature logistic regression fitted on the training folds turns the gap into a probability.

**Result:** AUC 0.7272 (SD 0.0134), log-loss 0.6270 (SD 0.0104). Constant baseline: AUC 0.5000, log-loss 0.6927.

**Why:** It is a transparent benchmark from economic theory, with no fitted parameters behind its ranking. Any learned model should be judged on what it adds beyond this rule.

**Likely viva question:** Why doesn't the EV gap use `endowment_inr`, and what can't the rule see?

**Short answer:** The endowment is added to both options, so it cancels when you subtract them. The same subtraction removes the frame: a gain problem shown in a loss frame has the same final wealth, so it has the same gap. In the training data, gain problems have a mean gap near zero in both frames, yet the risky-choice rate is 0.398 in the gain frame and 0.557 in the loss frame. The rule also has no term for loss aversion or differences between people.

**Likely follow-up:** The rule needs no training, so why is a model fitted at all?

**Short answer:** AUC only needs a ranking, so the raw gap is enough. Log-loss needs probabilities and a rupee gap isn't one. The mapping is fitted on training folds only, so it can't leak validation labels. Log-loss therefore scores "the rule plus a learned mapping", while AUC scores the rule alone.

## Step 3: Logistic regression (`make_logistic`, `choose_C`, `run_logistic`)

**What was done:** Added L2-regularised logistic regression on `prob_high`, `risky_high_inr`, `risky_low_inr` and `sure_amount_inr`, with scaling inside the pipeline. In each outer fold, `choose_C` holds out 25% of the training participants, tries C = 0.001 to 100, and picks the highest AUC. The model is then refitted on the full outer training data with that C and scored on the outer validation fold.

**Result:** AUC 0.7209 (SD 0.0164), log-loss 0.6203 (SD 0.0112). C chosen per fold: 1, 10, 10, 0.01, 100. EV rule: AUC 0.7272, log-loss 0.6270.

**Why:** It is the course's regularised linear baseline and the model the research question is about. Choosing C on separate participants means the outer validation fold plays no part in tuning, so its score stays a fair estimate for unseen people.

**Likely viva question:** What is C, and why tune it with an inner split instead of using the outer validation folds?

**Short answer:** C is the inverse of regularisation strength: small C shrinks the coefficients more. If I picked the C that scored best on the outer folds, those scores would be used for both choosing and reporting, which makes them optimistic. The inner split groups by participant because the final test set contains only new people.

**Likely follow-up:** Why does this model not clearly beat the EV rule, even though it learns from data?

**Short answer:** Expected value multiplies probability by outcome, and a linear model on the raw columns can only add weighted columns, so it cannot form that product. Its AUC is 0.0063 lower than the rule's on average, but the fold-to-fold SD of that difference is 0.0206, so the two are not clearly different. Its log-loss is slightly better, but that comparison is not like for like because the rule's probabilities come from a separate fitted mapping. The theory features in E1 test this directly.

**Likely follow-up:** The chosen C ranges from 0.01 to 100. Is that a problem?

**Short answer:** Mostly it shows that AUC barely changes for C of 0.1 or more, so the choice there is decided by tiny differences. Fold 4 chose 0.01 from a 42-participant inner split, and its coefficients were shrunk to about half the size of the other folds'. Its validation AUC of 0.7126 and log-loss of 0.6305 were both among the weaker results.

## Step 4: Accuracy in `score_fold`

**What was done:** `score_fold` now also reports accuracy, counting a prediction as risky when p >= 0.5. The EV rule instead predicts risky when gap > 0, so tied gaps count as sure. All three baselines were re-run, and the threshold is logged with each run.

**Result (accuracy):** constant 0.5182 (SD 0.0161), EV rule 0.6661 (SD 0.0139), logistic regression 0.6613 (SD 0.0150). AUC and log-loss did not change from earlier runs.

**Why:** The guide requires accuracy at a stated threshold. It is easy to interpret, but it only says which side of the threshold a prediction falls on, so it is a secondary metric.

**Likely viva question:** Why is AUC the primary metric if accuracy is easier to understand?

**Short answer:** Accuracy depends on one threshold and ignores how far each probability is from it. AUC measures the ranking across all thresholds, and the task asks for a predicted probability rather than a yes/no choice. The constant shows why a reference point matters: it gets 51.8% accuracy by always predicting "sure", having learned nothing about individual decisions, and its AUC is 0.5.

**Likely follow-up:** Why 0.5, and why doesn't the EV rule use it?

**Short answer:** 0.5 is the natural cut-off for "more likely than not", and fixing it in advance avoids tuning it on validation data. The EV rule has its own natural cut-off: choose risky if it has the higher expected value, which means gap > 0. Applying 0.5 to its fitted mapping would be a different decision rule, with accuracy 0.6383 rather than 0.6661.

## Step 5: LDA (`run_lda`)

**What was done:** Added Linear Discriminant Analysis (svd solver, no shrinkage) on the same four raw features as logistic regression, using the same 5 folds and fitting the scaler on training folds only. It has nothing to tune, so there is no inner split.

**Result:** AUC 0.7163 (SD 0.0168), log-loss 0.6195 (SD 0.0091), accuracy 0.6554 (SD 0.0159) at p >= 0.5. Compared with logistic regression, its AUC is lower in all 5 folds (mean difference -0.0046, SD 0.0013), and its log-loss is about the same (mean difference -0.0008, SD 0.0031).

**Why:** It is the course's generative classifier. It has the same linear form as logistic regression but estimates it differently, so comparing them shows what the Gaussian assumptions cost or gain.

**Likely viva question:** How does LDA connect to the shared-covariance Gaussian model?

**Short answer:** LDA models how the features are generated within each class: x | y = k ~ N(mu_k, Sigma). Each class k has its own mean mu_k, both classes share one covariance matrix Sigma, and pi_k is the class's share of the training data. Bayes' rule then turns these into P(y = 1 | x).

Because Sigma is shared, the quadratic terms in x cancel when the two class densities are compared, leaving log-odds that are linear in x:

log[P(y=1|x) / P(y=0|x)] = w'x + b

- w = inverse(Sigma) (mu_1 - mu_0)
- b = -(1/2)(mu_1 + mu_0)' inverse(Sigma) (mu_1 - mu_0) + log(pi_1 / pi_0)

The decision boundary, where the log-odds equal 0, is therefore a straight line (a hyperplane in four dimensions). I checked this formula against scikit-learn's output: in each fold, the probabilities agreed to about 1e-14.

**Likely follow-up:** Why LDA rather than Gaussian Naive Bayes?

**Short answer:** Naive Bayes assumes the features are independent within each class, which means a diagonal covariance matrix. Here `sure_amount_inr` is clearly correlated with the outcomes within each class: 0.56 and 0.61 among non-risky choices, and 0.57 and 0.74 among risky choices. Naive Bayes would count that overlapping information more than once, while LDA's full Sigma accounts for it. LDA also has the same linear form as logistic regression, which makes that comparison direct. This choice rests on the correlation structure: I did not run Naive Bayes and compare scores.

**Likely follow-up:** LDA and logistic regression both have linear log-odds. Why are the results different?

**Short answer:** They fit the line differently. LDA estimates class means and a covariance matrix, and gets the line from those, so it relies on the Gaussian assumptions. Logistic regression fits P(y | x) directly and makes no assumption about how x is distributed. Those assumptions do not hold well here: `prob_high` takes only 9 values, and the prob_high and risky_high_inr correlation is -0.17 in one class and +0.19 in the other, so the covariance is not truly shared. That may be why LDA ranks slightly worse in every fold, though I have not tested the cause.

**Likely follow-up:** Does scaling matter for LDA?

**Short answer:** No. Without regularisation, LDA's predictions do not change under rescaling, and I confirmed they matched to about 1e-14 with and without the scaler. It stays in the pipeline to keep the protocol consistent with the other models.

## Step 6: Linear SVM (`make_svm`, `fit_score_mapping`, `run_svm`)

**What was done:** Added a hinge-loss linear SVM (`SVC(kernel="linear")`) on the same four raw features, with scaling in the pipeline. C is chosen by the same inner grouped split as logistic regression, and `choose_C` now accepts either model. AUC and accuracy use the SVM's own score, with risky predicted when score > 0. For log-loss, a logistic mapping is fitted on grouped out-of-fold scores from the training fold.

**Result:** AUC 0.7209 (SD 0.0163), log-loss 0.6185 (SD 0.0099), accuracy 0.6604 (SD 0.0179). C chosen per fold: 100, 10, 100, 100, 10. Its AUC matches logistic regression's (mean difference 0.0000, SD 0.0010).

**Why:** It completes the course's linear baselines with a different loss. Logistic regression, LDA and the SVM all draw a linear boundary on the same features, so similar results suggest the limit is the features, not the choice of linear classifier.

**Likely viva question:** How does an SVM differ from logistic regression, and what does C do here?

**Short answer:** An SVM maximises the margin between the classes and uses hinge loss, so correctly classified points beyond the margin contribute no loss. Logistic loss never reaches exactly zero, and it directly models a probability. In both models, larger C means weaker regularisation. For the SVM, this means misclassified and margin-violating points cost more. About 76% of training rows are support vectors, which fits heavy class overlap in these features.

**Likely follow-up:** An SVM has no probabilities, so how did you get a log-loss?

**Short answer:** I used Platt-style scaling: a one-feature logistic regression that maps SVM scores to probabilities. It must learn from scores on participants the SVM did not fit, because scores on the SVM's own training rows are over-optimistic. So within the training fold, each participant's score comes from a grouped cross-validation model that excluded them. The final SVM is then refitted on the whole training fold. The validation fold is not used at any point.

**Likely follow-up:** What do the baseline results show overall?

**Short answer:** The EV rule has the highest mean AUC (0.7272) with nothing learned for its ranking, but its lead is within fold-to-fold variation. Logistic regression and the SVM tie at 0.7209, and LDA is slightly lower at 0.7163. Linear models on the raw columns cannot form the probability x outcome product that defines expected value, which motivates E1.

## Experiment E1: Theory features (`theory_features`, `run_e1` in `src/experiments.py`)

**What was done:** Used the logistic baseline's protocol and changed only the features: the four raw columns plus `ev_gap`, `wealth_if_high`, `wealth_if_low` and `wealth_if_sure` (`endowment_inr` plus each presented outcome). The probability of the larger outcome is `prob_high`, because `risky_high_inr` is larger than `risky_low_inr` in all 8,400 rows, so no duplicate column was added.

**Result:** AUC 0.7771 (SD 0.0066), log-loss 0.5810 (SD 0.0040), accuracy 0.7027 (SD 0.0102). Against baseline logistic regression, AUC difference +0.0562 (SD 0.0105) and log-loss difference -0.0393 (SD 0.0121), both improving in all 5 folds.

**Why:** A linear model can only add weighted columns, so it cannot compute probability x outcome by itself. Giving it `ev_gap` supplies that product directly, which tests whether theory-motivated features help.

**Expectation for the report:** Adding `ev_gap` should improve AUC over the raw-column logistic baseline, because a linear model cannot form probability x outcome from the raw columns. Write this in your own words. The claim that wealth adds only the endowment came from checking the data, so report it as a finding, not as a prior expectation.

**Likely viva question:** Which feature caused the improvement, and do the coefficients tell you?

**Short answer:** Mostly `ev_gap`. In unlogged diagnostic runs, raw + `ev_gap` alone reached AUC 0.7668 (+0.0459), and raw + wealth reached 0.7285 (+0.0075). The coefficients cannot fully answer this, because the eight columns have rank 6: each wealth column minus its raw column equals the endowment, so the three wealth columns add only one new dimension. The weight can be split among them in many ways, which is why `sure_amount_inr` (-0.4786) and `wealth_if_sure` (-0.4705) look almost equal in fold 1. `ev_gap` has the largest coefficient (+1.19), and it is the one column that is not a linear combination of the others.

**Likely follow-up:** You excluded frame. Is the frame really hidden from this model?

**Short answer:** No. The endowment can be recovered exactly as `wealth_if_sure - sure_amount_inr`, and in gain problems it is 0 for every gain-frame row and at least 200 for every loss-frame row. So the model gets frame information indirectly, and E1 is not a clean "no frame" condition. Raw + `ev_gap` + endowment gave the same AUC as E1 (0.7771), which shows the wealth columns add only the endowment. Going from raw + `ev_gap` to E1 adds about 0.0103 AUC.

**Likely follow-up:** Do these features leak anything?

**Short answer:** No target leakage. They are deterministic functions of columns in the same row, all of which the data dictionary marks as available at prediction time and which are presented before the choice. They use no `chose_risky`, `realised_payout_inr` or `response_time_ms`, and nothing is learned from other rows when computing them, so computing them outside the folds is safe. The scaler and model are still fitted inside the training folds.

## E1 ablation (`run_e1_ablation`)

**What was done:** Logged E1's two parts as separate rows, with experiment `E1-ablation`.

**Result:** Raw + `ev_gap`: AUC 0.7668 (SD 0.0096), log-loss 0.5879 (SD 0.0048), accuracy 0.7043 (SD 0.0083). Raw + endowment: AUC 0.7282 (SD 0.0124), log-loss 0.6146 (SD 0.0078), accuracy 0.6690 (SD 0.0139).

**Why:** E1 showed a combined gain. These rows show that `ev_gap` provides most of it.

## Experiment E2a: Frame information (`frame_features`, `run_e2a`)

**What was done:** Compared raw + `ev_gap` with raw + `ev_gap` + three indicators: `gain_x_loss_frame`, `loss_x_no_frame` and `mixed_x_no_frame`. Gain x gain_frame is the reference cell. The protocol is otherwise unchanged.

**Result:** AUC 0.7776 (SD 0.0059), log-loss 0.5706 (SD 0.0040), accuracy 0.6905 (SD 0.0097). Against raw + `ev_gap`: AUC +0.0108 (SD 0.0071), log-loss -0.0173 (SD 0.0052), both better in all 5 folds. Accuracy fell by 0.0138 (SD 0.0069).

**Why endowment is excluded from E2a:** In gain problems, the endowment is 0 for every gain-frame row and at least 200 for every loss-frame row, so it acts as a frame signal. If it stayed in the baseline, the baseline would already hold that signal, and the experiment would understate what the frame columns add. E1's wealth columns are excluded for the same reason, since they let a linear model recover the endowment.

**Why there is no separate frame column:** `frame` only varies within gain problems, so only four domain x frame cells exist. The three indicators already represent frame and the interaction, and a separate frame column would be an exact copy or sum of them.

**Expectation for the report:** Explicit frame information should improve prediction, mainly in gain problems, because the simulation lets some participants adopt the presented frame as their reference point. Write this in your own words.

**Likely viva question:** E2a improves on raw + `ev_gap`. Does that show the frame matters for prediction?

**Short answer:** No. It shows the added columns help, but they are mostly domain information. Two of the three indicators separate loss and mixed problems, and they carry the larger coefficients (`mixed_x_no_frame` about -0.47 in fold 1, against 0.03 for `gain_x_loss_frame`). In unlogged diagnostics, adding only a frame flag changed AUC by -0.0015, while adding only domain changed it by +0.0118, about the same as E2a.

**Likely follow-up:** Removing the endowment made the baseline frame-blind, right?

**Short answer:** No, and this is the main limitation. In gain problems, loss-framed amounts are presented as deductions, so a logistic model on raw + `ev_gap` separated the two frames perfectly in a diagnostic (AUC 1.0). Excluding the endowment closes one route, but the raw amounts leave another open. So a flat result for the frame flag in this design does not mean the frame has no effect.

**Likely follow-up:** What would a clean frame test look like?

**Short answer:** Use features that do not change with the frame: `prob_high`, `ev_gap` and the three final-wealth columns, without raw amounts. In gain problems, a model on those features could not recover the frame (AUC 0.505). This comparison is now logged as E2a-clean, below.

## Experiment E2a-clean: Loss-frame flag on frame-invariant features (`run_e2a_clean`)

**What was done:** Compared `prob_high` + `ev_gap` + the three wealth columns, with and without a `loss_frame` flag (1 for gain problems shown in the loss frame, otherwise 0). There are no raw amounts or stand-alone endowment. The logistic protocol is unchanged, and two rows were logged. The earlier E2a row was kept as it was.

**Result:** Without the flag: AUC 0.7542 (SD 0.0070), log-loss 0.5984 (SD 0.0050), accuracy 0.6890 (SD 0.0131). With the flag: AUC 0.7542 (SD 0.0090), log-loss 0.5946 (SD 0.0045), accuracy 0.6970 (SD 0.0128). Paired differences: AUC 0.0000 (SD 0.0082), log-loss -0.0038 (SD 0.0039), accuracy +0.0080 (SD 0.0130).

**Why:** E2a could not isolate the frame, because its indicators include domain and its raw amounts already reveal the frame in gain problems. Here, the feature set cannot recover the frame within gain problems (diagnostic AUC 0.505), so the flag is the only new frame information.

**Expectation for the report:** The flag should improve prediction, mainly in gain problems, because some simulated participants partly adopt the presented frame as their reference point. Write this in your own words. This design was chosen after E2a's results, so describe it as a follow-up test, not a prior plan.

**Likely viva question:** What does this experiment show about framing?

**Short answer:** The flag gives no overall ranking gain, and the small log-loss and accuracy gains are within fold-to-fold noise. So I cannot claim that it clearly improves prediction for unseen participants. Within gain problems, an unlogged diagnostic showed AUC +0.0184 (SD 0.0178), better in 4 of 5 folds, which points the expected way but is not clearly beyond noise. This is about predictive gain only. It does not show that the frame has no effect on choices: the raw risky rate is 0.398 in the gain frame and 0.557 in the loss frame.

**Likely follow-up:** The flag's coefficient is positive in all 5 folds (0.154 to 0.190). Is that the frame effect?

**Short answer:** Not purely. The flag's reference group is every other row, including loss and mixed problems, not just gain-frame rows. So the coefficient also reflects differences between gain problems and the other domains. It is also not causal evidence, and the data is simulated.

**Likely follow-up:** Why does this model score lower than E2a?

**Short answer:** Dropping raw amounts removes information, including what they reveal about domain and frame. These models test the frame, so they are not candidates for best predictor. Comparing them with E2a would change more than one thing.
