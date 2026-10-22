# Part C (1/2) - TF-IDF policy retrieval with a refusal threshold

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("fraud_policies.json") as f:
    POLICIES = json.load(f)

# default settings, no stop-word removal
_vectorizer = TfidfVectorizer()
_matrix = _vectorizer.fit_transform([p["text"] for p in POLICIES])

# calibrated from the measured scores printed by `python retrieval.py`: highest out-of-scope score 0.4251, lowest in-scope score 0.4862.
MIN_SCORE = 0.45

def retrieve(query):
    # return (policy_id, policy_text, score) of the most similar policy
    sims = cosine_similarity(_vectorizer.transform([query]), _matrix)[0]
    i = int(sims.argmax())
    return POLICIES[i]["policy_id"], POLICIES[i]["text"], float(sims[i])

def main():
    with open("retrieval_queries.json") as f:
        queries = json.load(f)
    print(f"{'query':<66}{'expected':>9}{'retrieved':>10}{'score':>8}")
    hits, in_scope, out_scores, in_scores = 0, 0, [], []
    for q in queries:
        pid, _, score = retrieve(q["query"])
        exp = q["expected_policy"]
        print(f"{q['query']:<66}{str(exp):>9}{pid:>10}{score:>8.4f}")
        if exp is None:
            out_scores.append(score)
        else:
            in_scope += 1
            in_scores.append(score)
            hits += pid == exp
    print(f"\nin-scope top-1 hits: {hits} of {in_scope}")
    print(f"highest out-of-scope score: {max(out_scores):.4f}; lowest in-scope score: {min(in_scores):.4f}")
    print(f"MIN_SCORE = {MIN_SCORE}  (valid interval: > {max(out_scores):.4f} and <= {min(in_scores):.4f})")
    assert max(out_scores) < MIN_SCORE <= min(in_scores)
    answered = sum(retrieve(q["query"])[2] >= MIN_SCORE for q in queries if q["expected_policy"])
    refused = sum(retrieve(q["query"])[2] < MIN_SCORE for q in queries if q["expected_policy"] is None)
    print(f"in-scope answered: {answered}/8; out-of-scope refused: {refused}/2")

if __name__ == "__main__":
    main()