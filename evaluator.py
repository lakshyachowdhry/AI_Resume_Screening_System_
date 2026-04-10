from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from preprocess import preprocess_text
from utils import ROOT_DIR


# ✅ Safe absolute paths
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"


def evaluate_model(data_path: Path | None = None) -> Dict[str, float]:
    """
    Load the trained model and evaluate on the dataset safely.
    """

    try:
        # ✅ FIX: Safe dataset path
        if data_path is None:
            data_path = ROOT_DIR / "data" / "dataset.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        df = pd.read_csv(data_path)

        if df.empty:
            raise ValueError("Dataset is empty.")

        # ✅ Preprocessing
        df["Resume_Text_clean"] = df["Resume_Text"].astype(str).apply(preprocess_text)
        df["Job_Description_clean"] = df["Job_Description"].astype(str).apply(preprocess_text)
        df["combined_text"] = df["Resume_Text_clean"] + " " + df["Job_Description_clean"]

        X = df["combined_text"].values
        y_true = df["Label"].values

        # ✅ FIX: Safe model loading
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}")

        clf = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        # ✅ Transform + predict
        X_vec = vectorizer.transform(X)
        y_pred = clf.predict(X_vec)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        return metrics

    except Exception as e:
        # ✅ IMPORTANT: Prevent Streamlit crash
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": str(e),
        }


if __name__ == "__main__":
    print(evaluate_model())