# Fraud Triage Service (synthetic Razorpay Merchant Risk Desk capstone)

## Overview

A small fraud-triage service for synthetic payment data (all INR, nothing real).
Part A cleans the raw export and builds three features; Part B trains a tuned random forest and picks an alert threshold on the validation split.
Part C retrieves the fraud policy that fits an analyst's question (TF-IDF) and asks the offline `mock_llm` for a JSON case note, checked by three validation gates, retries, a safe default and a business guardrail.
Part D serves it with FastAPI: `/score`, `/case`, `/health`, `/metrics`, input and output guardrails, one log line and one counter update per POST via a single middleware.
The model is trained at start-up (about 80 seconds) and never saved to disk. No API key, account or network is needed.

## Reproduce

Python 3.11+ (built and checked on Python 3.13, pandas 3.0.5, scikit-learn 1.9.1). Run everything from the repository root.

```bash
git clone <this-repo> && cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py            # prints "rows written: 6090"; md5sum transactions_raw.csv = 8716797e8bc1a477df5f52cbc8aae3b8
python prepare.py                  # Part A audit and checkpoints
python train.py                    # Part B tables (about 80 s)
python retrieval.py                # Part C retrieval table
python casenote.py                 # Part C scenario table
rm -f service.log
uvicorn app:app --port 8000        # terminal 1: wait about 80 s for start-up training
python smoke_test.py               # terminal 2: Q1..Q11 table and /metrics (restart the service before every run)
```

## Part A results

| Audit item | Value |
|---|---|
| rows / columns as read | 6,090 / 14 |
| exact duplicate rows | 90 |
| rows after de-duplication | 6,000 |
| missing `amount_inr` (after de-dup) | 76 |
| missing `distance_from_home_km` (after de-dup) | 193 |
| `amount_inr` < 1 or > 1,000,000 (after de-dup) | 5 |
| distinct raw spellings: `merchant_category` / `city` | 24 / 24 |

`clean_transactions` returns 5,919 rows, no duplicates; categories Grocery 1,406, Food Delivery 1,326, Fashion 976, Electronics 845, Utilities 685, Travel 681; 6 cities; payment modes UPI 3,221, Card 1,450, NetBanking 731, Wallet 517; `amount_inr` from 141.37 to 105,883.95; 191 missing distances remain. After `add_features`: `is_night` sums to 365, `high_value` to 17, max `amount_vs_avg_ratio` 13.11. Both functions run on a one-row DataFrame and the input is left unchanged.

Missing `distance_from_home_km` values are not filled here because the fill statistic must be learned from training rows only, otherwise validation and test rows leak into training.

| Leakage audit (cleaned table) | rows | fraud rows |
|---|---|---|
| `chargeback_filed` = 1 | 373 | 304 |
| `chargeback_filed` = 0 | 5,546 | 62 |

Excluded from model inputs: `chargeback_filed`, `txn_id`, `customer_id`, `is_fraud`. `chargeback_filed` is only known after the payment settles and the customer disputes it, so at scoring time it does not exist and it is nearly a copy of the label (304 of 373 chargebacks are fraud). The identifiers are labels of a row, not behaviour: a model can memorise them, and they cannot generalise to new payments or customers.

Model-input columns (before one-hot encoding): `merchant_category, city, payment_mode, amount_inr, hour_of_day, txn_count_last_24h, avg_amount_30d_inr, distance_from_home_km, is_new_device, account_age_days, amount_vs_avg_ratio, is_night, high_value`.

Fraud rows: 366 of 5,919 (6.18%). A rule that predicts "not fraud" for every payment has accuracy 93.82% and recall 0. Accuracy is the wrong headline metric because it is dominated by the 94% of legitimate payments, so a model that catches no fraud looks excellent; recall and precision on the fraud class measure what the risk desk actually cares about.

## Part B results

The 26 input columns, in the pinned order used for training, validation, test and the service:

```
['amount_inr', 'hour_of_day', 'txn_count_last_24h', 'avg_amount_30d_inr', 'distance_from_home_km', 'is_new_device', 'account_age_days', 'amount_vs_avg_ratio', 'is_night', 'high_value', 'merchant_category_Electronics', 'merchant_category_Fashion', 'merchant_category_Food Delivery', 'merchant_category_Grocery', 'merchant_category_Travel', 'merchant_category_Utilities', 'city_Bengaluru', 'city_Chennai', 'city_Delhi', 'city_Kolkata', 'city_Mumbai', 'city_Pune', 'payment_mode_Card', 'payment_mode_NetBanking', 'payment_mode_UPI', 'payment_mode_Wallet']
```

**Split.** Train / validation / test = 3,551 / 1,184 / 1,184 rows, with 220 / 73 / 73 fraud rows. The training median of `distance_from_home_km` is 12.6 and fills the missing values in all three splits.

**Imbalance (logistic regression, validation, threshold 0.5).**

| Variant | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| plain | 0.625 | 0.342 | 0.442 | 1096 | 15 | 48 | 25 |
| class_weight="balanced" | 0.252 | 0.781 | 0.381 | 942 | 169 | 16 | 57 |
| over-sampled | 0.257 | 0.781 | 0.386 | 946 | 165 | 16 | 57 |

The over-sampled training table has 6,662 rows (3,331 non-fraud and 3,331 fraud draws with replacement). A desk that must not miss fraud would prefer a balanced or over-sampled model: recall rises from 0.342 to 0.781 (32 more frauds caught out of 73). The cost is precision falling from 0.625 to about 0.25, i.e. roughly 150 more legitimate payments held for an analyst on validation.

**Cross-validation (5-fold, F1, training split only).** Balanced logistic regression: mean 0.402, std 0.015. Untuned forest: mean 0.453 (std 0.052). GridSearchCV (12 combinations x 5 folds, F1): best `max_depth=None`, `min_samples_leaf=5`, best mean CV F1 0.465 (std 0.042 across folds). No validation or test row was used.

The tuned forest is the final model. It has the best accuracy by CV F1 (0.465 against 0.402), although its fold-to-fold spread (0.042) is wider than logistic regression's (0.015), so it is less stable. It also gives feature importances, which are less directly interpretable than signed logistic-regression coefficients. Relative to logistic regression we give up some stability and the simple coefficient reading, in exchange for a gain of about 0.06 in CV F1.

**Validation threshold table (tuned forest).**

| t | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|
| 0.1 | 0.111 | 0.973 | 0.200 | 638 |
| 0.2 | 0.158 | 0.945 | 0.271 | 437 |
| 0.3 | 0.209 | 0.863 | 0.336 | 302 |
| **0.4** | 0.272 | 0.712 | 0.394 | 191 |
| 0.5 | 0.352 | 0.589 | 0.441 | 122 |
| 0.6 | 0.507 | 0.507 | 0.507 | 73 |
| 0.7 | 0.528 | 0.260 | 0.349 | 36 |
| 0.8 | 0.778 | 0.192 | 0.308 | 18 |
| 0.9 | 0.800 | 0.055 | 0.103 | 5 |

`t_star` = **0.4**, the highest threshold with validation recall at least 0.65 (0.712 at 0.4, 0.589 at 0.5).

**Test split, once, at t_star = 0.4.** Precision 0.281, recall 0.781, F1 0.413, 203 payments flagged. Confusion matrix: TN 965, FP 146, FN 16, TP 57.

*Risk-committee note.* The alert threshold is 0.4, chosen as the highest threshold on the validation grid whose recall is at least 0.65. On the 1,184 test payments the model flagged 203 for review. That holds about 2.6 legitimate payments per fraud caught (146 false alarms for 57 frauds), and 16 of the 73 frauds were missed. A stricter threshold of 0.6 would cut validation flags from 191 to 73 and raise precision from 0.272 to 0.507, but recall would drop from 0.712 to 0.507, roughly 15 more frauds missed out of 73.

**Leakage experiment (validation F1 at 0.5).** Clean tuned forest 0.441; leaky forest with `chargeback_filed` as a 27th and last column 0.797. The leaky score is not a result to be proud of because `chargeback_filed` is only filed after the payment settles, so at decision time the model would never have it. It lets the model read the answer instead of learning fraud behaviour, and would collapse in production.

**Top five feature importances.** `amount_vs_avg_ratio` 0.270, `txn_count_last_24h` 0.178, `amount_inr` 0.134, `account_age_days` 0.084, `distance_from_home_km` 0.071. Fraudsters spend far more than the customer's usual amount and make many payments in a short window before the account is blocked.

## Part C results

**Retrieval (TF-IDF, default settings, cosine similarity).** In-scope top-1 hits: 8 of 8.

| Query | Expected | Retrieved | Score |
|---|---|---|---|
| card payment above INR 50000 from a new device | P1 | P1 | 0.5501 |
| how many UPI payments in 24 hours trigger a velocity alert | P2 | P2 | 0.5622 |
| payment at 3 AM far away from the home city | P3 | P3 | 0.4948 |
| customer reports unauthorised payment and chargeback case | P4 | P4 | 0.5689 |
| electronics merchant payment on an account younger than 30 days | P5 | P5 | 0.5641 |
| which payments are approved automatically without manual review | P6 | P6 | 0.6250 |
| temporary hold on repeated UPI payments within a day | P2 | P2 | 0.4862 |
| what happens after a customer reports an unauthorised payment | P4 | P4 | 0.4867 |
| what is the cafeteria menu today | none | P3 | 0.4251 |
| how do I reset my email password | none | P1 | 0.0000 |

`MIN_SCORE` = **0.45**: the highest out-of-scope score is 0.4251 and the lowest in-scope score is 0.4862, so 0.45 lies above the first and below the second. All eight in-scope queries are answered and both out-of-scope queries return `no_policy_match`.

**Scenario run** (`python casenote.py`, after `mock_llm.reset()`):

| Scenario | Retrieved policy | Attempts | Failures | Status | Safe default | Final action | Override |
|---|---|---|---|---|---|---|---|
| S1 (T900001) | P1 | 1 | [] | ok | no | hold_for_review | no |
| S2 (T900002) | P2 | 2 | [1] | ok | no | hold_for_review | no |
| S3 (T900003) | P3 | 2 | [2] | ok | no | hold_for_review | no |
| S4 (T900004) | P4 | 2 | [3] | ok | no | block_and_notify | no |
| S5 (T900005) | P5 | 3 | [1, 1, 1] | ok | yes | hold_for_review | no |
| S6 (T900006) | P2 | 1 | [] | ok | no | hold_for_review | yes |
| S7 (T900007) | P1 | 2 | [3] | ok | no | hold_for_review | no |
| S8 (T900008) | P4 | 1 | [] | ok | no | block_and_notify | no |
| S9 (T900009) | P3 | 3 | [1, 2] | ok | no | hold_for_review | no |
| S10 (T900010) | P2 | 2 | [3] | ok | no | hold_for_review | no |

Totals: 19 model calls; 3 scenarios valid on the first call (S1, S6, S8); 6 recovered by retry (S2, S3, S4, S7, S9, S10); 1 safe default (S5); 1 guardrail override (S6).

A hosted model samples text, so an identical prompt can return broken JSON, drop a field or pick an invalid value on two calls in a row, and an upgrade or an overloaded server can change its behaviour without notice. Gate 1 to 3 catch malformed output, the retry gives a transient fault a second chance, the safe default stops bad text from reaching an analyst, and the guardrail catches a well-formed note that is still wrong (an approve for a risky payment). A hosted model gives no guarantee that its output is both well-formed and sensible, so the checks are necessary rather than optional.

## Part D results

**Status of this section, stated plainly.** The sandbox this build was produced in could not install `fastapi` (the package index was blocked by the session's network policy), so the server could not be started there. `app.py` and `smoke_test.py` were therefore **not** run end to end against a live uvicorn process, and `service.log` was **not** generated; the table, metrics and log lines below must be produced on your machine with the Reproduce commands and pasted in before submission. What was checked offline is the request logic: `app.py` was imported against a throwaway stub of `fastapi` and the scoring path, guardrails, counters and log levels were driven directly with the eleven smoke requests. That run gave: Q2 p=0.0430 approve; Q3 p=0.9597 hold; Q4 p=0.1765 not flagged, held by the large-payment guardrail; Q5 400 on `amount_inr`; Q7 two attempts, policy P2; Q8 safe default after three attempts; Q9 503; Q10 and Q11 no case note; and metrics `requests_total` 10, `success_total` 7, `error_total` 3, `success_rate` 0.7, `error_rate` 0.3, `human_review_total` 5, `human_review_rate` 0.5, with log levels INFO for Q2, Q3, Q7, Q10, Q11, WARNING for Q4, Q5, Q6, Q8 and ERROR for Q9. The extra check (T900006) gave probability 0.4391, `decision` hold_for_review, `guardrail_override` true, `human_review` true. The real 422 behaviour for Q6 (a missing field) comes from FastAPI itself and is only confirmed by a live run.

Design notes. One HTTP middleware (`observe`) sees every response, including the framework's 422, my own 400 and the 503, and calls `record()`, the single place that updates the counters and writes the one `service.log` line per POST. `guardrail_override` on `/case` is true when any guardrail changed the decision: the approve-to-hold override from Part C, the flagged-approve replacement, or the large-payment hold of an unflagged payment.

To fill in (run on a fresh service start, with `service.log` deleted first):

```
Smoke-test table: <paste output of python smoke_test.py>
GET /metrics JSON: <paste>
service.log: <paste one INFO, one WARNING and one ERROR line>
```

## Limitations

1. The data is synthetic and the fraud label comes from a hand-built logistic rule, so the model rediscovers that rule and real fraud would behave less cleanly.
2. The threshold of 0.4 is tuned on only 73 validation fraud cases, so a different split could move it by one grid step.
3. The retriever is keyword-based TF-IDF and cannot match meaning: the out-of-scope "cafeteria menu" query still scored 0.4251 against P3 on the shared word "the", leaving little room for `MIN_SCORE`.
4. The language model is a scripted stand-in with a fixed list of defects, so the retry and validation logic has not been tested against the variety of failures a hosted model produces.
5. The service keeps counters and the model in memory, trains for about 80 seconds at every start-up, and loses its metrics on restart.
6. Part D was not run against a live server in the build environment (see Part D results), so the smoke-test table and `service.log` still have to be produced and committed.
