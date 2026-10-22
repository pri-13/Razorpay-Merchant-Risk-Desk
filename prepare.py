# Part A - data foundation: audit, clean, engineer

import pandas as pd

RAW_FILE = "transactions_raw.csv"
MODEL_INPUT_COLUMNS = [
    "merchant_category", "city", "payment_mode", "amount_inr", "hour_of_day",
    "txn_count_last_24h", "avg_amount_30d_inr", "distance_from_home_km",
    "is_new_device", "account_age_days",
]
EXCLUDED_COLUMNS = ["txn_id", "customer_id", "chargeback_filed", "is_fraud"]

def clean_transactions(raw_df):
    # return a cleaned copy. the input dataframe is never modified
    df = raw_df.drop_duplicates().copy()                                   # (a)
    for col in ("merchant_category", "city"):                              # (b)
        df[col] = df[col].str.strip().str.title()
    df = df[df["amount_inr"].notna()]                                      # (c)
    df = df[(df["amount_inr"] >= 1) & (df["amount_inr"] <= 1_000_000)]     # (d)
    return df.reset_index(drop=True)                                       # (e)

def add_features(df):
    # return a copy with amount_vs_avg_ratio, is_night and high_value
    out = df.copy()
    out["amount_vs_avg_ratio"] = out["amount_inr"] / out["avg_amount_30d_inr"]
    out["is_night"] = (out["hour_of_day"] <= 5).astype(int)
    out["high_value"] = (out["amount_inr"] > 50_000).astype(int)
    return out

def audit(raw):
    dedup = raw.drop_duplicates()
    rows = [
        ("rows as read", len(raw)),
        ("columns as read", raw.shape[1]),
        ("exact duplicate rows", int(raw.duplicated().sum())),
        ("rows after de-duplication", len(dedup)),
        ("missing amount_inr (after de-dup)", int(dedup["amount_inr"].isna().sum())),
        ("missing distance_from_home_km (after de-dup)", int(dedup["distance_from_home_km"].isna().sum())),
        ("amount_inr < 1 or > 1,000,000 (after de-dup)",
         int(((dedup["amount_inr"] < 1) | (dedup["amount_inr"] > 1_000_000)).sum())),
        ("distinct raw spellings of merchant_category", raw["merchant_category"].nunique()),
        ("distinct raw spellings of city", raw["city"].nunique()),
    ]
    return rows

def main():
    raw = pd.read_csv(RAW_FILE)
    raw_before = raw.copy()

    print("Audit (raw file)")
    for name, value in audit(raw):
        print(f"{name:<50} {value}")

    clean = clean_transactions(raw)
    assert raw.equals(raw_before), "clean_transactions modified its input"
    feat = add_features(clean)

    print("\nCleaned table:")
    print("rows:", len(clean), "| duplicates:", int(clean.duplicated().sum()))
    print("merchant_category:", clean["merchant_category"].value_counts().to_dict())
    print("distinct cities:", clean["city"].nunique())
    print("payment_mode:", clean["payment_mode"].value_counts().to_dict())
    print(f"amount_inr min {clean['amount_inr'].min():.2f}  max {clean['amount_inr'].max():.2f}")
    print("missing distance_from_home_km:", int(clean["distance_from_home_km"].isna().sum()))

    print("\nEngineered features:")
    print("is_night sum:", int(feat["is_night"].sum()))
    print("high_value sum:", int(feat["high_value"].sum()))
    print(f"max amount_vs_avg_ratio: {feat['amount_vs_avg_ratio'].max():.2f}")

    print("\nLeakage audit (chargeback_filed):")
    for v in (1, 0):
        sub = clean[clean["chargeback_filed"] == v]
        print(f"chargeback_filed = {v}: {len(sub)} rows, {int(sub['is_fraud'].sum())} fraud")
    print("Excluded from model inputs:", ", ".join(["chargeback_filed", "txn_id", "customer_id", "is_fraud"]))
    print("Model-input columns (before one-hot):", MODEL_INPUT_COLUMNS + ["amount_vs_avg_ratio", "is_night", "high_value"])

    n_fraud = int(clean["is_fraud"].sum())
    n = len(clean)
    print("\nWhy accuracy misleads?")
    print(f"fraud rows: {n_fraud} of {n} ({100 * n_fraud / n:.2f}%)")
    print(f"'never fraud' rule: accuracy {100 * (n - n_fraud) / n:.2f}%, recall 0.00")

    # one-row dataframe check (Part D calls both functions on a single request)
    one = add_features(clean_transactions(clean.head(1)))
    print("\none-row dataframe check:", "OK" if len(one) == 1 else "FAILED")
    print("input unchanged by clean_transactions:", raw.equals(raw_before))

if __name__ == "__main__":
    main()