# Part D - smoke test: waits for the service, sends Q1..Q11, prints a result table and /metrics

import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"

BENIGN = {"merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI", "amount_inr": 1500,
          "hour_of_day": 13, "txn_count_last_24h": 0, "avg_amount_30d_inr": 1600,
          "distance_from_home_km": 3.0, "is_new_device": 0, "account_age_days": 900}
RISKY = {"merchant_category": "Electronics", "city": "Mumbai", "payment_mode": "Card", "amount_inr": 45000,
         "hour_of_day": 2, "txn_count_last_24h": 9, "avg_amount_30d_inr": 3000,
         "distance_from_home_km": 60.0, "is_new_device": 1, "account_age_days": 40}
BIG = dict(BENIGN, merchant_category="Utilities", payment_mode="NetBanking", amount_inr=150000,
           avg_amount_30d_inr=250000, distance_from_home_km=0.0, is_new_device=1, account_age_days=2200)

def wait_for_service(timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False

def fields_match(body, expected):
    return all(body.get(k) == v for k, v in expected.items())

def main():
    if not wait_for_service():
        print("service did not come up within 120 seconds")
        sys.exit(2)

    no_hour = {k: v for k, v in BENIGN.items() if k != "hour_of_day"}
    q = "card payment above INR 50000 from a new device"

    # id, category, method, path, body, expected status, check(body) -> bool
    tests = [
        ("Q1", "happy", "GET", "/health", None, 200, lambda b: b == {"status": "ok"}),
        ("Q2", "happy", "POST", "/score", dict(BENIGN, txn_id="T800001"), 200,
         lambda b: fields_match(b, {"flagged": False, "decision": "approve", "human_review": False})),
        ("Q3", "escalation", "POST", "/score", dict(RISKY, txn_id="T800002"), 200,
         lambda b: fields_match(b, {"flagged": True, "decision": "hold_for_review", "human_review": True})),
        ("Q4", "escalation", "POST", "/score", dict(BIG, txn_id="T800003"), 200,
         lambda b: fields_match(b, {"flagged": False, "decision": "hold_for_review", "human_review": True})),
        ("Q5", "edge", "POST", "/score", dict(BENIGN, amount_inr=-5, txn_id="T800004"), 400,
         lambda b: "error" in b and b.get("field") == "amount_inr"),
        ("Q6", "edge", "POST", "/score", dict(no_hour, txn_id="T800005"), 422, lambda b: True),
        ("Q7", "escalation", "POST", "/case",
         dict(RISKY, txn_id="T900002", question="more than 6 UPI payments in 24 hours velocity alert"), 200,
         lambda b: fields_match(b, {"flagged": True, "attempts": 2, "safe_default": False,
                                    "human_review": True})
         and (b.get("case_note") or {}).get("policy_id") == "P2"),
        ("Q8", "escalation", "POST", "/case",
         dict(RISKY, txn_id="T900005",
              question="electronics merchant payment on an account younger than 30 days"), 200,
         lambda b: fields_match(b, {"attempts": 3, "safe_default": True, "decision": "hold_for_review",
                                    "human_review": True})),
        ("Q9", "edge", "POST", "/case", dict(RISKY, txn_id="T900099", question=q), 503,
         lambda b: "error" in b),
        ("Q10", "happy", "POST", "/case", dict(BENIGN, txn_id="T800001", question=q), 200,
         lambda b: fields_match(b, {"case_note": None, "attempts": 0, "decision": "approve",
                                    "human_review": False})),
        ("Q11", "edge", "POST", "/case",
         dict(RISKY, txn_id="T800006", question="what is the cafeteria menu today"), 200,
         lambda b: fields_match(b, {"case_note": None, "attempts": 0, "decision": "hold_for_review",
                                    "human_review": True})),
    ]

    print(f"{'id':<5}{'category':<12}{'expected':>9}{'observed':>9}  result")
    passed = {}
    for qid, cat, method, path, body, exp_status, check in tests:
        r = requests.request(method, BASE + path, json=body, timeout=60)
        try:
            data = r.json()
        except ValueError:
            data = {}
        ok = r.status_code == exp_status and check(data)
        passed.setdefault(cat, [0, 0])
        passed[cat][1] += 1
        passed[cat][0] += ok
        print(f"{qid:<5}{cat:<12}{exp_status:>9}{r.status_code:>9}  {'PASS' if ok else 'FAIL'}")

    total = sum(v[0] for v in passed.values())
    print("\nper category:", ", ".join(f"{c} {p}/{n}" for c, (p, n) in passed.items()),
          f"| total {total}/{len(tests)}")
    print("\n/metrics:")
    print(json.dumps(requests.get(f"{BASE}/metrics", timeout=10).json(), indent=2))
    sys.exit(0 if total == len(tests) else 1)

if __name__ == "__main__":
    main()