"""
predict.py

Small command-line script that loads the trained model + scaler produced by
End2EndML.ipynb and predicts placement from CGPA and IQ.

Usage:
    python predict.py --cgpa 6.8 --iq 123
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

MODEL_PATH = Path("model/model.pkl")
SCALER_PATH = Path("model/scaler.pkl")


def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        sys.exit(
            "Model or scaler not found. Run the notebook (End2EndML.ipynb) first "
            "to generate model/model.pkl and model/scaler.pkl."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict(cgpa: float, iq: float, model, scaler) -> tuple[int, float | None]:
    X = np.array([[cgpa, iq]])
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_scaled)[0][1])
    return pred, proba


def main():
    parser = argparse.ArgumentParser(description="Predict student placement from CGPA and IQ.")
    parser.add_argument("--cgpa", type=float, required=True, help="Student's CGPA")
    parser.add_argument("--iq", type=float, required=True, help="Student's IQ score")
    args = parser.parse_args()

    model, scaler = load_artifacts()
    pred, proba = predict(args.cgpa, args.iq, model, scaler)

    label = "Placed" if pred == 1 else "Not Placed"
    print(f"Prediction: {label}")
    if proba is not None:
        print(f"Confidence (probability of placement): {proba:.2%}")


if __name__ == "__main__":
    main()
