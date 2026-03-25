import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from modelling.data_model import Data
from Config import Config
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


def build_chained_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["chain_1"] = df["y2"].astype(str)
    df["chain_2"] = df["y2"].astype(str) + " + " + df["y3"].astype(str)
    df["chain_3"] = df["y2"].astype(str) + " + " + df["y3"].astype(str) + " + " + df["y4"].astype(str)
    return df


def extract_chain_level(labels: np.ndarray, n_parts: int) -> np.ndarray:
    return np.array([" + ".join(s.split(" + ")[:n_parts]) for s in labels])


def run_chained(X: np.ndarray, df: pd.DataFrame, model_class, model_name: str):
    
    df = build_chained_labels(df)

    mask = (df["y3"].notna()) & (df["y3"].astype(str) != "nan") & (df["y3"].astype(str) != "")
    mask &= (df["y4"].notna()) & (df["y4"].astype(str) != "nan") & (df["y4"].astype(str) != "")
    df = df[mask].copy()
    X = X[mask.values]

    df["y"] = df["chain_3"]

    data = Data(X, df)

    if data.X_train is None:
        print(f"  [{model_name}] Not enough data for chained classification")
        return None

    model = model_class(model_name, data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)

    true_chain3 = data.y_test
    pred_chain3 = model.predictions

    true_chain1 = extract_chain_level(true_chain3, 1)
    true_chain2 = extract_chain_level(true_chain3, 2)
    pred_chain1 = extract_chain_level(pred_chain3, 1)
    pred_chain2 = extract_chain_level(pred_chain3, 2)

    results = {}
    for level, (y_true, y_pred, desc) in enumerate([
        (true_chain1, pred_chain1, "Level 1 — y2 only"),
        (true_chain2, pred_chain2, "Level 2 — y2 + y3"),
        (true_chain3, pred_chain3, "Level 3 — y2 + y3 + y4"),
    ], start=1):
        acc = accuracy_score(y_true, y_pred)
        print(f"\n  [{model_name}] {desc}")
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, zero_division=0))
        results[f"Level {level}"] = acc

    return results
