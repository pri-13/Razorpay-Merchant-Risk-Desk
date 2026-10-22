# Part B - fraud model, imbalance treatment and alert threshold

# python train.py prints every Part B table
# build_final_model() repeats load -> clean -> features -> encode -> split -> median fill -> grid search and returns what the service needs in Part D

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prepare import RAW_FILE, add_features, clean_transactions

EXCLUDED = ["txn_id", "customer_id", "chargeback_filed", "is_fraud"]
SEED = 42
GRID = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}
THRESHOLDS = [round(0.1 * k, 1) for k in range(1, 10)]

def load_table():
    # cleaned + engineered table (still has the excluded columns)
    return add_features(clean_transactions(pd.read_csv(RAW_FILE)))

def make_splits(table=None):
    # stratified 60/20/20 split, one-hot encoded, distance filled with the TRAIN median
    # returns a dict with x/y per split, the 26-column list, the training median and the chargeback_filed column per split (for the leakage experiment)

    table = load_table() if table is None else table
    y = table["is_fraud"]
    X = pd.get_dummies(table.drop(columns=EXCLUDED), dtype=int)
    cb = table["chargeback_filed"]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    # learned on training rows only
    median = float(X_tr["distance_from_home_km"].median())
    for part in (X_tr, X_va, X_te):
        part["distance_from_home_km"] = part["distance_from_home_km"].fillna(median)

    return {
        "X_train": X_tr, "X_val": X_va, "X_test": X_te,
        "y_train": y_tr, "y_val": y_va, "y_test": y_te,
        "columns": list(X.columns), "median": median,
        "cb_train": cb.loc[X_tr.index], "cb_val": cb.loc[X_va.index], "cb_test": cb.loc[X_te.index],
    }

def make_forest(**params):
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, **params)

def cv_splitter():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def tune_forest(X_train, y_train):
    grid = GridSearchCV(make_forest(), GRID, scoring="f1", cv=cv_splitter(), refit=True)
    grid.fit(X_train, y_train)
    return grid

def pick_threshold(y_true, proba):
    # highest threshold in 0.1..0.9 with recall >= 0.65 (0.1 if none)
    best = 0.1
    for t in THRESHOLDS:
        if recall_score(y_true, (proba >= t).astype(int), zero_division=0) >= 0.65:
            best = t
    return best

def build_final_model():
    s = make_splits()
    grid = tune_forest(s["X_train"], s["y_train"])
    model = grid.best_estimator_
    threshold = pick_threshold(s["y_val"], model.predict_proba(s["X_val"])[:, 1])
    return {"model": model, "columns": s["columns"], "median": s["median"], "threshold": threshold}

# reporting
def metrics_row(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return (precision_score(y_true, pred, zero_division=0), recall_score(y_true, pred, zero_division=0),
            f1_score(y_true, pred, zero_division=0), int(tn), int(fp), int(fn), int(tp))

def lr_pipeline(**kw):
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, **kw))])

def main():
    s = make_splits()
    Xtr, Xva, Xte = s["X_train"], s["X_val"], s["X_test"]
    ytr, yva, yte = s["y_train"], s["y_val"], s["y_test"]

    print("Input columns (%d, pinned order)-" % len(s["columns"]))
    print(s["columns"])

    print("\nSplit")
    print(f"sizes train/val/test: {len(Xtr)} / {len(Xva)} / {len(Xte)}")
    print(f"fraud rows train/val/test: {int(ytr.sum())} / {int(yva.sum())} / {int(yte.sum())}")
    print(f"training median of distance_from_home_km: {s['median']:.1f}")

    # three ways to treat imbalance
    plain = lr_pipeline().fit(Xtr, ytr)
    balanced = lr_pipeline(class_weight="balanced").fit(Xtr, ytr)
    fraud_rows = Xtr[ytr == 1]
    non_fraud_rows = Xtr[ytr == 0]
    boot = fraud_rows.sample(n=len(non_fraud_rows), replace=True, random_state=SEED)
    X_os = pd.concat([non_fraud_rows, boot])
    y_os = pd.Series([0] * len(non_fraud_rows) + [1] * len(boot), index=X_os.index)
    oversampled = lr_pipeline().fit(X_os, y_os)

    print("\nImbalance treatments (logistic regression, validation, threshold 0.5)")
    print(f"{'variant':<14}{'prec':>7}{'recall':>8}{'F1':>7}{'TN':>6}{'FP':>5}{'FN':>5}{'TP':>5}")
    for name, m in (("plain", plain), ("balanced", balanced), ("over-sampled", oversampled)):
        p, r, f, tn, fp, fn, tp = metrics_row(yva, (m.predict_proba(Xva)[:, 1] >= 0.5).astype(int))
        print(f"{name:<14}{p:>7.3f}{r:>8.3f}{f:>7.3f}{tn:>6}{fp:>5}{fn:>5}{tp:>5}")
    print(f"over-sampled training table: {len(X_os)} rows ({int(y_os.sum())} fraud, {int((y_os == 0).sum())} non-fraud)")

    # cross-validation and tuning (training split only)
    cv = cv_splitter()
    lr_cv = cross_val_score(lr_pipeline(class_weight="balanced"), Xtr, ytr, scoring="f1", cv=cv)
    rf_cv = cross_val_score(make_forest(), Xtr, ytr, scoring="f1", cv=cv)
    print("\n5-fold CV, F1, training split only")
    print(f"balanced logistic regression: mean {lr_cv.mean():.3f}  std {lr_cv.std():.3f}")
    print(f"random forest (untuned):      mean {rf_cv.mean():.3f}  std {rf_cv.std():.3f}")

    grid = tune_forest(Xtr, ytr)
    n_combo = len(grid.cv_results_["params"])
    best_i = grid.best_index_
    print(f"\nGridSearchCV ({n_combo} combinations x 5 folds)")
    print("best params:", grid.best_params_)
    print(f"best mean CV F1: {grid.best_score_:.3f}  (std across folds {grid.cv_results_['std_test_score'][best_i]:.3f})")
    model = grid.best_estimator_

    #threshold on validation
    proba_va = model.predict_proba(Xva)[:, 1]
    t_star = pick_threshold(yva, proba_va)
    print("\nValidation threshold table (tuned forest)")
    print(f"{'t':>4}{'prec':>8}{'recall':>8}{'F1':>7}{'flagged':>9}")
    for t in THRESHOLDS:
        pred = (proba_va >= t).astype(int)
        print(f"{t:>4}{precision_score(yva, pred, zero_division=0):>8.3f}{recall_score(yva, pred):>8.3f}"
              f"{f1_score(yva, pred, zero_division=0):>7.3f}{int(pred.sum()):>9}")
    print(f"t_star (highest t with validation recall >= 0.65): {t_star}")

    # one final test
    pred_te = (model.predict_proba(Xte)[:, 1] >= t_star).astype(int)
    p, r, f, tn, fp, fn, tp = metrics_row(yte, pred_te)
    print(f"\nTest split, once, at t_star = {t_star}")
    print(f"precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}  flagged {int(pred_te.sum())}")
    print(f"confusion matrix: TN {tn}  FP {fp}  FN {fn}  TP {tp}")
    print(f"legitimate payments held per fraud caught: {fp / tp:.2f}  |  frauds missed: {fn}")

    # leakage experiment
    clean_f1 = f1_score(yva, (proba_va >= 0.5).astype(int))
    leaky = make_forest(**grid.best_params_)
    leaky.fit(Xtr.assign(chargeback_filed=s["cb_train"]), ytr)
    leaky_f1 = f1_score(yva, (leaky.predict_proba(Xva.assign(chargeback_filed=s["cb_val"]))[:, 1] >= 0.5).astype(int))
    print("\nLeakage experiment (validation F1 at 0.5)")
    print(f"clean tuned forest: {clean_f1:.3f}   leaky (+chargeback_filed as 27th column): {leaky_f1:.3f}")

    # feature importance
    imp = pd.Series(model.feature_importances_, index=s["columns"]).sort_values(ascending=False)
    print("\nTop 5 feature importances")
    for name, v in imp.head(5).items():
        print(f"{name:<24}{v:.3f}")

    fm = {"model": model, "columns": s["columns"], "median": s["median"], "threshold": t_star}
    assert len(fm["columns"]) == 26
    print("\nbuild_final_model() keys:", sorted(build_final_model().keys()))

if __name__ == "__main__":
    main()