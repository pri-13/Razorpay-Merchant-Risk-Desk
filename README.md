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
