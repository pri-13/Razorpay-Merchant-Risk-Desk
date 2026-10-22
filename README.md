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
