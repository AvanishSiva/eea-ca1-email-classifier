

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from modelling.data_model import Data
from Config import Config
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


def _safe_split(X, y, test_size=0.2):
    y_series = pd.Series(y)
    good_classes = y_series.value_counts()[y_series.value_counts() >= 3].index

    # Need at least 2 classes for classification
    if len(good_classes) < 2:
        return None

    mask = y_series.isin(good_classes)
    X_good = X[mask.values] if hasattr(mask, 'values') else X[mask]
    y_good = y[mask.values] if hasattr(mask, 'values') else y[mask]

    if len(y_good) < 5:
        return None

    actual_test = max(0.2, 1.0 / len(y_good))
    if actual_test >= 1.0:
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_good, y_good, test_size=actual_test, random_state=0, stratify=y_good
        )
        return X_train, X_test, y_train, y_test
    except ValueError:
        return None


def run_hierarchical(X: np.ndarray, df: pd.DataFrame, model_class, model_name: str):
    mask = (df["y3"].notna()) & (df["y3"].astype(str) != "nan") & (df["y3"].astype(str) != "")
    mask &= (df["y4"].notna()) & (df["y4"].astype(str) != "nan") & (df["y4"].astype(str) != "")
    df = df[mask].copy()
    X = X[mask.values]

    results = {"l1_acc": None, "l2_details": [], "l3_details": []}

    df_l1 = df.copy()
    df_l1["y"] = df_l1["y2"]

    data_l1 = Data(X, df_l1)
    if data_l1.X_train is None:
        print(f"  [{model_name}] Not enough data for Level 1")
        return None

    model_l1 = model_class(model_name, data_l1.get_embeddings(), data_l1.get_type())
    model_l1.train(data_l1)
    model_l1.predict(data_l1.X_test)

    acc_l1 = accuracy_score(data_l1.y_test, model_l1.predictions)
    results["l1_acc"] = acc_l1
    print(f"\n  [{model_name}] Level 1 — Classify y2 (full dataset)")
    print(f"  Accuracy: {acc_l1:.4f}")
    print(classification_report(data_l1.y_test, model_l1.predictions, zero_division=0))

    y2_classes = sorted(df["y2"].unique())
    model_count = 1  # model_l1 is #1

    for cls2 in y2_classes:
        subset_mask = df["y2"] == cls2
        X_sub = X[subset_mask.values]
        y_sub = df.loc[subset_mask, "y3"].to_numpy()

        split = _safe_split(X_sub, y_sub)
        if split is None:
            print(f"  [{model_name}] Level 2 — y3 where y2='{cls2}': skipped (insufficient data)")
            continue

        X_train, X_test, y_train, y_test = split
        model_count += 1

        
        model_l2 = model_class(model_name, X_sub, y_sub)
        model_l2.mdl.fit(X_train, y_train)
        preds = model_l2.mdl.predict(X_test)
        model_l2.predictions = preds

        acc_l2 = accuracy_score(y_test, preds)
        results["l2_details"].append({"y2_class": cls2, "accuracy": acc_l2})
        print(f"\n  [{model_name}] Level 2 — y3 where y2='{cls2}' (Model Instance #{model_count})")
        print(f"  Accuracy: {acc_l2:.4f}")
        print(classification_report(y_test, preds, zero_division=0))

        y3_classes = sorted(df.loc[subset_mask, "y3"].unique())
        for cls3 in y3_classes:
            sub3_mask = subset_mask & (df["y3"] == cls3)
            X_sub3 = X[sub3_mask.values]
            y_sub3 = df.loc[sub3_mask, "y4"].to_numpy()

            split3 = _safe_split(X_sub3, y_sub3)
            if split3 is None:
                continue

            X_train3, X_test3, y_train3, y_test3 = split3
            model_count += 1

            model_l3 = model_class(model_name, X_sub3, y_sub3)
            model_l3.mdl.fit(X_train3, y_train3)
            preds3 = model_l3.mdl.predict(X_test3)
            model_l3.predictions = preds3

            acc_l3 = accuracy_score(y_test3, preds3)
            results["l3_details"].append({"y2_class": cls2, "y3_class": cls3, "accuracy": acc_l3})
            print(f"\n  [{model_name}] Level 3 — y4 where y2='{cls2}', y3='{cls3}' (Model Instance #{model_count})")
            print(f"  Accuracy: {acc_l3:.4f}")
            print(classification_report(y_test3, preds3, zero_division=0))

    print(f"\n  [{model_name}] Total model instances created: {model_count}")
    return results
