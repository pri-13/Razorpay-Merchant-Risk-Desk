# Part D - the Fraud Triage Service (FastAPI)

import logging
import time
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from casenote import run_case
from prepare import add_features
from train import build_final_model

# start-up

# trained once, kept in memory (never saved to disk)
MODEL_BUNDLE = build_final_model()          
MODEL = MODEL_BUNDLE["model"]
COLUMNS = MODEL_BUNDLE["columns"]
MEDIAN = MODEL_BUNDLE["median"]
THRESHOLD = MODEL_BUNDLE["threshold"]

KNOWN_CATEGORIES = {"Grocery", "Electronics", "Travel", "Food Delivery", "Fashion", "Utilities"}
LLM_DOWN_MESSAGE = "case-note service temporarily unavailable, try again later"

app = FastAPI(title="Fraud Triage Service")

class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: Optional[float] = None
    is_new_device: int
    account_age_days: int

class CaseIn(Txn):
    question: str

# logging + metrics
log = logging.getLogger("triage")
log.setLevel(logging.INFO)
log.propagate = False
_handler = logging.FileHandler("service.log")
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_handler)

COUNTERS = {"requests": 0, "success": 0, "error": 0, "latency_ms": 0.0, "human_review": 0}

def record(path, status, ms, info):
    # the single place that updates the counters and writes the one log line per POST
    COUNTERS["requests"] += 1
    COUNTERS["latency_ms"] += ms
    COUNTERS["success" if status < 400 else "error"] += 1
    if info.get("human_review") is True:
        COUNTERS["human_review"] += 1

    msg = f"{path} status={status} latency_ms={ms:.1f}"
    if "txn_id" in info:
        msg += f" txn_id={info['txn_id']} fraud_probability={info['fraud_probability']:.4f}"
    if info.get("safe_default"):
        msg += " safe_default=true"
    if info.get("override"):
        msg += " guardrail_override=true"

    if status >= 500:
        level = logging.ERROR
    elif status >= 400 or info.get("safe_default") or info.get("override"):
        level = logging.WARNING
    else:
        level = logging.INFO
    log.log(level, msg)

@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()

    # routes put txn_id, probability and flags here
    request.state.info = {}
    try:
        # a route, or the framework's own 422 answer
        response = await call_next(request)
    # never leak a stack trace
    except Exception as exc:
        response = JSONResponse(status_code=500, content={"error": "internal error"})
        request.state.info["internal_error"] = type(exc).__name__
    if request.method == "POST":
        record(request.url.path, response.status_code, (time.perf_counter() - t0) * 1000,
               request.state.info)
    return response

# ascoring path
def check_input(t):
    # return a 400 JSONResponse for a well-typed but impossible value, else None
    def bad(field, message):
        return JSONResponse(status_code=400, content={"error": message, "field": field})

    if not 1 <= t.amount_inr <= 1_000_000:
        return bad("amount_inr", "amount_inr must be between 1 and 1,000,000")
    if not 0 <= t.hour_of_day <= 23:
        return bad("hour_of_day", "hour_of_day must be between 0 and 23")
    if t.txn_count_last_24h < 0:
        return bad("txn_count_last_24h", "txn_count_last_24h must not be negative")
    if t.avg_amount_30d_inr <= 0:
        return bad("avg_amount_30d_inr", "avg_amount_30d_inr must be greater than 0")
    if t.merchant_category.strip().title() not in KNOWN_CATEGORIES:
        return bad("merchant_category", "unknown merchant_category")
    return None

def fraud_probability(t):
    df = pd.DataFrame([t.model_dump(exclude={"question"})])
    for col in ("merchant_category", "city"):
        df[col] = df[col].str.strip().str.title()
    df["distance_from_home_km"] = pd.to_numeric(df["distance_from_home_km"], errors="coerce")
    df = add_features(df).drop(columns=["txn_id"])
    X = pd.get_dummies(df, dtype=int).reindex(columns=COLUMNS, fill_value=0)
    X["distance_from_home_km"] = X["distance_from_home_km"].fillna(MEDIAN)
    return float(MODEL.predict_proba(X)[0, 1])

def score_path(t):
    # return (probability, flagged, decision, large_payment_hold)
    p = fraud_probability(t)
    flagged = p >= THRESHOLD
    decision = "hold_for_review" if flagged else "approve"
    large_hold = False
    if not flagged and t.amount_inr >= 100_000 and t.is_new_device == 1:
        # never auto-approve a huge new-device payment
        decision, large_hold = "hold_for_review", True
    return p, flagged, decision, large_hold

# endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(t: Txn, request: Request):
    rejected = check_input(t)
    if rejected is not None:
        return rejected
    p, flagged, decision, large_hold = score_path(t)
    human_review = decision == "hold_for_review"
    request.state.info.update(txn_id=t.txn_id, fraud_probability=p, human_review=human_review,
                              override=large_hold)
    return {"txn_id": t.txn_id, "fraud_probability": round(p, 4), "flagged": flagged,
            "decision": decision, "human_review": human_review}

@app.post("/case")
def case(t: CaseIn, request: Request):
    rejected = check_input(t)
    if rejected is not None:
        return rejected
    p, flagged, decision, large_hold = score_path(t)
    info = request.state.info
    info.update(txn_id=t.txn_id, fraud_probability=p)

    case_note, attempts, safe_default, override = None, 0, False, large_hold
    if flagged:
        try:
            result = run_case(t.txn_id, p, t.amount_inr, t.question)
        except ConnectionError:
            return JSONResponse(status_code=503, content={"error": LLM_DOWN_MESSAGE})
        attempts = result["attempts"]
        if result["status"] == "no_policy_match":
            decision = "hold_for_review"
        else:
            case_note, safe_default = result["case_note"], result["safe_default"]
            override = result["guardrail_override"]
            if case_note["recommended_action"] == "approve":       # a flagged payment is never auto-approved
                case_note["recommended_action"] = "hold_for_review"
                override = True
            decision = case_note["recommended_action"]

    human_review = decision in ("hold_for_review", "block_and_notify") or safe_default
    info.update(human_review=human_review, safe_default=safe_default, override=override)
    return {"txn_id": t.txn_id, "fraud_probability": round(p, 4), "flagged": flagged,
            "decision": decision, "human_review": human_review, "case_note": case_note,
            "attempts": attempts, "safe_default": safe_default, "guardrail_override": override}

@app.get("/metrics")
def metrics():
    n = COUNTERS["requests"]
    return {
        "requests_total": n,
        "success_total": COUNTERS["success"],
        "error_total": COUNTERS["error"],
        "success_rate": round(COUNTERS["success"] / n, 4) if n else 0.0,
        "error_rate": round(COUNTERS["error"] / n, 4) if n else 0.0,
        "avg_latency_ms": round(COUNTERS["latency_ms"] / n, 3) if n else 0.0,
        "human_review_total": COUNTERS["human_review"],
        "human_review_rate": round(COUNTERS["human_review"] / n, 4) if n else 0.0,
    }