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
