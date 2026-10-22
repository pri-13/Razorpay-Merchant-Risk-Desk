# Fraud Triage Service (synthetic Razorpay Merchant Risk Desk capstone)

## Overview

A small fraud-triage service for synthetic payment data (all amounts in INR; nothing here is real Razorpay data).
Part A cleans the raw export and builds three features; Part B trains a tuned random forest and picks an alert threshold on the validation split.
Part C retrieves the fraud policy that fits an analyst's question (TF-IDF) and asks the offline `mock_llm` for a JSON case note, checked by three validation gates, retries, a safe default and a business guardrail.
Part D serves it with FastAPI: `/score`, `/case`, `/health`, `/metrics`, input and output guardrails, and one log line and one counter update per POST through a single middleware.
The model is trained at start-up (about 80 seconds) and never saved to disk. No API key, account or network is needed to run it.

## Reproduce

Python 3.11 or newer. Run every command from the repository root. Replace `<user>/<repo>` with this repository's path.

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>

python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m generate_data      # prints "rows written: 6090"; MD5 of transactions_raw.csv = 8716797e8bc1a477df5f52cbc8aae3b8
python -m prepare            # Part A audit and checkpoints
python -m train              # Part B tables (about 80 s)
python -m retrieval          # Part C retrieval table
python -m casenote           # Part C scenario table
```

Part D needs two terminals, both with the virtual environment active. Delete `service.log` first, because the log file is appended to.

```bash
# Terminal 1 (Windows PowerShell: Remove-Item service.log -ErrorAction SilentlyContinue)
rm -f service.log
python -m uvicorn app:app --port 8000        # wait about 80 s for "Uvicorn running on http://127.0.0.1:8000"

# Terminal 2
python -m smoke_test                         # Q1..Q11 table and /metrics
```

Restart the service before every smoke run: the mock language model remembers call counts per transaction id, so a second run on the same process behaves differently.

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

`clean_transactions` returns 5,919 rows with no duplicates; categories Grocery 1,406, Food Delivery 1,326, Fashion 976, Electronics 845, Utilities 685, Travel 681; 6 cities; payment modes UPI 3,221, Card 1,450, NetBanking 731, Wallet 517; `amount_inr` from 141.37 to 105,883.95; 191 missing distances remain. After `add_features`: `is_night` sums to 365, `high_value` to 17, and the maximum `amount_vs_avg_ratio` is 13.11. Both functions run on a one-row DataFrame, and `clean_transactions` leaves its input unchanged.

Missing `distance_from_home_km` values are not filled in Part A because the fill statistic must be learned from the training rows only, otherwise information from the validation and test rows leaks into training.

| Leakage audit (cleaned table) | rows | fraud rows |
|---|---|---|
| `chargeback_filed` = 1 | 373 | 304 |
| `chargeback_filed` = 0 | 5,546 | 62 |

Excluded from model inputs: `chargeback_filed`, `txn_id`, `customer_id`, `is_fraud`. `chargeback_filed` is only known after the payment settles and the customer disputes it, so it does not exist at decision time, and it is almost a copy of the label (304 of 373 chargebacks are fraud). The identifiers name a row rather than describe behaviour: a model can memorise them, and they carry nothing that generalises to new payments or customers.

Model-input columns (before one-hot encoding): `merchant_category, city, payment_mode, amount_inr, hour_of_day, txn_count_last_24h, avg_amount_30d_inr, distance_from_home_km, is_new_device, account_age_days, amount_vs_avg_ratio, is_night, high_value`.

Fraud rows: 366 of 5,919 (6.18%). A rule that predicts "not fraud" for every payment has accuracy 93.82% and recall 0. Accuracy is the wrong headline metric because the 94% of legitimate payments dominate it, so a model that catches no fraud still looks excellent; recall and precision on the fraud class measure what the risk desk cares about.

## Part B results

The 26 input columns, in the pinned order used for training, validation, test and the service:

```
['amount_inr', 'hour_of_day', 'txn_count_last_24h', 'avg_amount_30d_inr', 'distance_from_home_km', 'is_new_device', 'account_age_days', 'amount_vs_avg_ratio', 'is_night', 'high_value', 'merchant_category_Electronics', 'merchant_category_Fashion', 'merchant_category_Food Delivery', 'merchant_category_Grocery', 'merchant_category_Travel', 'merchant_category_Utilities', 'city_Bengaluru', 'city_Chennai', 'city_Delhi', 'city_Kolkata', 'city_Mumbai', 'city_Pune', 'payment_mode_Card', 'payment_mode_NetBanking', 'payment_mode_UPI', 'payment_mode_Wallet']
```

**Split.** Train / validation / test = 3,551 / 1,184 / 1,184 rows, with 220 / 73 / 73 fraud rows. The training median of `distance_from_home_km` is 12.6, and it fills the missing values in all three splits.

**Imbalance (logistic regression, validation, threshold 0.5).**

| Variant | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|
| plain | 0.625 | 0.342 | 0.442 | 1096 | 15 | 48 | 25 |
| class_weight="balanced" | 0.252 | 0.781 | 0.381 | 942 | 169 | 16 | 57 |
| over-sampled | 0.257 | 0.781 | 0.386 | 946 | 165 | 16 | 57 |

The over-sampled training table has 6,662 rows (3,331 non-fraud rows and 3,331 fraud rows drawn with replacement). A risk desk that must not miss fraud would prefer the balanced or over-sampled model, because recall rises from 0.342 to 0.781 (32 more of the 73 frauds caught). The cost is that precision falls from 0.625 to about 0.25, which means roughly 150 more legitimate payments held for an analyst on the validation split.

**Cross-validation (5-fold, F1, training split only).** Balanced logistic regression: mean 0.402, standard deviation 0.015. Untuned forest: mean 0.453 (std 0.052). GridSearchCV (12 parameter combinations x 5 folds, F1): best `max_depth=None`, `min_samples_leaf=5` (with `n_estimators=200` fixed), best mean CV F1 0.465 (std 0.042 across folds). No validation or test row was used.

The tuned forest is the final model. It is the most accurate by CV F1 (0.465 against 0.402 for logistic regression). It is less stable, since its fold-to-fold spread (0.042) is wider than logistic regression's (0.015), and it is less interpretable, because feature importances do not give a signed, per-feature effect the way coefficients do. Relative to logistic regression we therefore give up some stability and a simple coefficient reading, in exchange for about 0.06 more CV F1.

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

`t_star` = **0.4**, the highest threshold with validation recall at least 0.65 (recall is 0.712 at 0.4 and 0.589 at 0.5).

**Test split, evaluated once at t_star = 0.4.** Precision 0.281, recall 0.781, F1 0.413, 203 payments flagged. Confusion matrix: TN 965, FP 146, FN 16, TP 57.

*Risk-committee note.* The alert threshold is 0.4, chosen as the highest threshold on the validation grid whose recall is at least 0.65. On the 1,184 test payments the model flagged 203 for review. That holds about 2.6 legitimate payments per fraud caught (146 false alarms for 57 frauds), and 16 of the 73 frauds were missed. A stricter threshold of 0.6 would cut the validation flags from 191 to 73 and raise precision from 0.272 to 0.507, but recall would fall from 0.712 to 0.507, roughly 15 more frauds missed out of 73.

**Leakage experiment (validation F1 at 0.5).** Clean tuned forest 0.441; leaky forest with `chargeback_filed` added as a 27th and last column 0.797. The leaky score is not a result to be proud of because a chargeback is only filed after the payment has settled, so the model would never have the column at decision time. It lets the model read the answer instead of learning fraud behaviour, and it would fail in production.

**Top five feature importances.** `amount_vs_avg_ratio` 0.270, `txn_count_last_24h` 0.178, `amount_inr` 0.134, `account_age_days` 0.084, `distance_from_home_km` 0.071. Fraudsters spend far more than the customer's usual amount, and they make many payments in a short window before the account is blocked.

## Part C results

**Retrieval (TF-IDF with default settings, cosine similarity).** In-scope top-1 hits: 8 of 8.

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

`MIN_SCORE` = **0.45**. The highest out-of-scope score is 0.4251 and the lowest in-scope score is 0.4862, so 0.45 lies above the first and below the second. All eight in-scope queries are answered and both out-of-scope queries return `no_policy_match`.

**Scenario run** (`python -m casenote`, after `mock_llm.reset()`):

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

A hosted model samples its text, so an identical prompt can return broken JSON, drop a field or choose an invalid value twice in a row, and a model update or an overloaded server can change its behaviour without notice. Gates 1 to 3 catch malformed output, the retry gives a transient fault another chance, the safe default stops bad text from reaching an analyst, and the guardrail catches a note that is well-formed but wrong (an approve for a risky payment). Because nothing guarantees that a hosted model's output is both well-formed and sensible, these checks are required rather than optional.

## Part D results

The service was started with `python -m uvicorn app:app --port 8000` on a fresh process, and `python -m smoke_test` was run once against it on Windows 11. Result: 11 of 11 PASS.

| Id | Category | Expected status | Observed status | Result |
|---|---|---|---|---|
| Q1 | happy | 200 | 200 | PASS |
| Q2 | happy | 200 | 200 | PASS |
| Q3 | escalation | 200 | 200 | PASS |
| Q4 | escalation | 200 | 200 | PASS |
| Q5 | edge | 400 | 400 | PASS |
| Q6 | edge | 422 | 422 | PASS |
| Q7 | escalation | 200 | 200 | PASS |
| Q8 | escalation | 200 | 200 | PASS |
| Q9 | edge | 503 | 503 | PASS |
| Q10 | happy | 200 | 200 | PASS |
| Q11 | edge | 200 | 200 | PASS |

Per category: happy 3/3, escalation 4/4, edge 4/4, total 11/11. Besides the status code, each check also compares every field listed in the brief's Expected column (for example Q4 `flagged` false with `decision` hold_for_review, Q7 `attempts` 2 and `case_note.policy_id` P2, Q8 `safe_default` true).

`GET /metrics` after that single run:

```json
{
  "requests_total": 10,
  "success_total": 7,
  "error_total": 3,
  "success_rate": 0.7,
  "error_rate": 0.3,
  "avg_latency_ms": 86.696,
  "human_review_total": 5,
  "human_review_rate": 0.5
}
```

`service.log` from the same run holds exactly ten lines, one per POST request and none for GET. Three of them:

```
2026-10-22 17:15:21,540 INFO /score status=200 latency_ms=80.4 txn_id=T800002 fraud_probability=0.9597
2026-10-22 17:15:21,600 WARNING /score status=200 latency_ms=52.9 txn_id=T800003 fraud_probability=0.1765 guardrail_override=true
2026-10-22 17:15:22,066 ERROR /case status=503 latency_ms=112.2 txn_id=T900099 fraud_probability=0.9597
```

The levels over the ten lines are INFO for Q2, Q3, Q7, Q10 and Q11; WARNING for Q4 (large-payment guardrail hold), Q5 (400), Q6 (422) and Q8 (safe default); and ERROR for Q9 (503).

Design notes:

- **One place for logs and counters.** One HTTP middleware (`observe`) sees every response, including the framework's own 422, my 400 and the 503, and calls `record()`, the single function that updates the counters and writes the log line. Routes put `txn_id`, the probability and the flags into `request.state.info`; a rejected request never sets `human_review`, so it never adds to `human_review_total`.
- **Input and output guardrails.** Impossible values (`amount_inr`, `hour_of_day`, `txn_count_last_24h`, `avg_amount_30d_inr`, unknown `merchant_category`) return 400 before any model runs. An unflagged payment of INR 100,000 or more from a new device is held for review (Q4). On `/case`, a flagged payment whose note still says approve is changed to hold_for_review.
- **`guardrail_override` on `/case`.** It is true when any guardrail changed the decision: Part C's approve override, the flagged-approve replacement, or the large-payment hold of an unflagged payment.
- **Language-model outage.** A `ConnectionError` from `mock_llm` is caught in the route and answered with 503 and a plain JSON error; the service keeps answering later requests, and the log line carries no stack trace.

## Limitations

1. The data is synthetic and its fraud label comes from a hand-built logistic rule, so the model rediscovers that rule and real fraud would behave less cleanly.
2. The threshold of 0.4 is tuned on only 73 validation fraud cases, so a different split could move it by a grid step.
3. The retriever is keyword-based TF-IDF and cannot match meaning: the out-of-scope "cafeteria menu" query still scored 0.4251 against P3 on shared common words, which leaves little room for `MIN_SCORE`.
4. The language model is a scripted stand-in with a fixed list of defects, so the retry and validation logic has not been tested against the variety of failures a hosted model produces.
5. The service keeps the model and counters in memory, trains for about 80 seconds at every start-up, and loses its metrics on restart.
6. The smoke test is a single sequential run on one laptop (average latency about 87 ms), not a load test, and the service has no authentication or rate limiting.