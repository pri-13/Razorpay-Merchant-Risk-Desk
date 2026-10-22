# Part C (2/2) - prompt, three validation gates, retry, safe default, output guardrail
import json

import mock_llm
from retrieval import MIN_SCORE, retrieve

# case-note rules, defined once in the json schema vocabulary. the gates are driven from this dict.
CASE_NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_team": {"type": "string",
                             "enum": ["card_risk", "upi_risk", "merchant_ops", "customer_care"]},
        "recommended_action": {"type": "string",
                               "enum": ["approve", "hold_for_review", "block_and_notify"]},
        "policy_id": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5", "P6"]},
    },
    "required": ["summary", "priority", "recommended_team", "recommended_action", "policy_id"],
}

SAFE_DEFAULT_SUMMARY = "Automated case note unavailable; manual review required."

def build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text):
    props = CASE_NOTE_SCHEMA["properties"]
    allowed = "\n".join(
        f"- {k}: " + (" | ".join(v["enum"]) if "enum" in v else "free text, one or two sentences")
        for k, v in props.items()
    )
    return (
        "ROLE:\n"
        "You are a fraud-operations assistant on the Merchant Risk Desk. "
        "You write short case notes for analysts.\n\n"
        "TASK:\n"
        "Write a case note for the held payment below, applying the fraud policy provided.\n\n"
        "CONSTRAINTS:\n"
        "- Use only the policy text given below; do not add rules or facts that are not in it.\n"
        "- Reply with one JSON object and nothing else: no prose, no code fences.\n\n"
        "OUTPUT FORMAT:\n"
        "A JSON object with exactly these five keys and allowed values:\n"
        f"{allowed}\n\n"
        "PAYMENT:\n"
        f"TXN_ID: {txn_id}\n"
        f"FRAUD_PROBABILITY: {fraud_probability}\n"
        f"AMOUNT_INR: {amount_inr}\n"
        f"POLICY_ID: {policy_id}\n\n"
        "POLICY TEXT:\n"
        f"{policy_text}\n"
    )

def validate(text, retrieved_policy_id):
    # return (note, gate): gate 0 = passed, else the first failing gate 1, 2, 3
    schema = CASE_NOTE_SCHEMA
    # gate 1: valid json, and a json object
    try:
        note = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, 1
    if not isinstance(note, dict):
        return None, 1
    # gate 2: all required keys present, each a string
    for key in schema["required"]:
        if key not in note or not isinstance(note[key], str):
            return None, 2
    # gate 3: allowed values, and the policy id must be the retrieved one
    for key, spec in schema["properties"].items():
        if "enum" in spec and note[key] not in spec["enum"]:
            return None, 3
    if note["policy_id"] != retrieved_policy_id:
        return None, 3
    return note, 0

def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=3):
    policy_id, policy_text, score = retrieve(question)
    if score < MIN_SCORE:
        return {"status": "no_policy_match", "attempts": 0, "failures": [], "case_note": None,
                "safe_default": False, "guardrail_override": False}

    prompt = build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text)
    failures, note, attempts = [], None, 0
    while attempts < max_attempts:
        attempts += 1
        # ConnectionError propagates to the caller
        text = mock_llm.generate(prompt)
        note, gate = validate(text, policy_id)
        if gate == 0:
            break
        failures.append(gate)
        note = None

    safe_default = note is None
    if safe_default:
        note = {"summary": SAFE_DEFAULT_SUMMARY, "priority": "high", "recommended_team": "customer_care",
                "recommended_action": "hold_for_review", "policy_id": policy_id}

    # output guardrail: if the model recommends "approve" for a high-fraud-probability payment, override to "hold_for_review"
    override = False
    if note["recommended_action"] == "approve" and fraud_probability >= 0.5:
        note["recommended_action"] = "hold_for_review"
        override = True

    return {"status": "ok", "attempts": attempts, "failures": failures, "case_note": note,
            "safe_default": safe_default, "guardrail_override": override}

def main():
    mock_llm.reset()
    with open("case_scenarios.json") as f:
        scenarios = json.load(f)
    print(f"{'scenario':<16}{'policy':>7}{'attempts':>9}  {'failures':<11}{'status':<8}"
          f"{'safe default':>13}  {'final action':<18}{'override':>8}")
    calls = first_ok = recovered = defaults = overrides = 0
    for i, sc in enumerate(scenarios, 1):
        r = run_case(sc["txn_id"], sc["fraud_probability"], sc["amount_inr"], sc["question"])
        pid = r["case_note"]["policy_id"] if r["case_note"] else "-"
        action = r["case_note"]["recommended_action"] if r["case_note"] else "-"
        print(f"{f'S{i} ({sc['txn_id']})':<16}{pid:>7}{r['attempts']:>9}  {str(r['failures']):<11}{r['status']:<8}"
              f"{'yes' if r['safe_default'] else 'no':>13}  {action:<18}{'yes' if r['guardrail_override'] else 'no':>8}")
        calls += r["attempts"]
        first_ok += r["attempts"] == 1 and not r["safe_default"]
        recovered += r["attempts"] > 1 and not r["safe_default"]
        defaults += r["safe_default"]
        overrides += r["guardrail_override"]
    print(f"\nTotals: {calls} model calls; {first_ok} valid on first call; {recovered} recovered by retry; "
          f"{defaults} safe default; {overrides} guardrail override")

if __name__ == "__main__":
    main()