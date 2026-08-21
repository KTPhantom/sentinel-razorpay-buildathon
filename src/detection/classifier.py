"""
XGBoost classifier with SHAP feature attribution for Kaggle fraud data.

Handles the severe class imbalance (~0.17% fraud) via scale_pos_weight,
then tunes the decision threshold on the validation set to maximise F1
rather than using the default 0.5 cutoff.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import xgboost as xgb

from src.config import (
    KAGGLE_EXPECTED_COLUMNS,
    MODELS_DIR,
    RANDOM_SEED,
    TOP_N_SHAP_FEATURES,
)

FEATURE_COLUMNS = [col for col in KAGGLE_EXPECTED_COLUMNS if col != "Class"]


class FraudClassifier:

    def __init__(
        self,
        model: xgb.XGBClassifier | None = None,
        optimal_threshold: float = 0.5,
        explainer: shap.TreeExplainer | None = None,
    ):
        self.model = model
        self.optimal_threshold = optimal_threshold
        self.explainer = explainer
        self.feature_names = FEATURE_COLUMNS

    @classmethod
    def train(
        cls,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        scale_pos_weight: float | None = None,
        seed: int = RANDOM_SEED,
    ) -> "FraudClassifier":
        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["Class"]
        X_val = val_df[FEATURE_COLUMNS]
        y_val = val_df["Class"]

        if scale_pos_weight is None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = float(neg_count / max(pos_count, 1))

        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_probs = model.predict_proba(X_val)[:, 1]
        best_threshold, _ = cls._tune_threshold(y_val.values, val_probs)

        explainer = shap.TreeExplainer(model)

        instance = cls(model=model, optimal_threshold=best_threshold, explainer=explainer)
        instance.val_threshold_audit: dict[str, Any] = instance.threshold_audit(y_val.values, val_probs)

        return instance

    @staticmethod
    def _tune_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
        """Grid-sweep thresholds 0.05→0.95 and return the one that maximises F1."""
        best_threshold = 0.5
        best_f1 = 0.0
        for t in np.linspace(0.05, 0.95, 91):
            preds = (y_probs >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)
        return round(best_threshold, 3), round(best_f1, 4)

    def threshold_audit(self, y_true: np.ndarray, y_probs: np.ndarray) -> dict[str, Any]:
        """Precision/recall/F1 at the calibrated threshold — useful for training logs."""
        preds = (y_probs >= self.optimal_threshold).astype(int)
        return {
            "threshold": self.optimal_threshold,
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        }

    def predict_single(
        self, txn: dict | pd.Series, top_n_shap: int = TOP_N_SHAP_FEATURES
    ) -> dict[str, Any]:
        """Predict fraud probability and top SHAP attributions for one transaction."""
        if self.model is None or self.explainer is None:
            raise RuntimeError("Model and SHAP explainer must be initialised before prediction.")

        if isinstance(txn, dict):
            row_df = pd.DataFrame([{col: float(txn[col]) for col in self.feature_names}])
        else:
            row_df = pd.DataFrame([txn[self.feature_names].astype(float)])

        risk_score = float(self.model.predict_proba(row_df)[0, 1])
        flag = bool(risk_score >= self.optimal_threshold)

        shap_values = self.explainer.shap_values(row_df)
        if isinstance(shap_values, list):
            sv = np.array(shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            sv = shap_values[0]
        else:
            sv = np.array(shap_values)

        contribs = []
        for feat_name, val, contrib in zip(self.feature_names, row_df.values[0], sv):
            contribs.append({
                "type": "model_feature",
                "name": feat_name,
                "value": round(float(val), 4),
                "contribution": round(float(contrib), 4),
            })

        contribs.sort(key=lambda x: x["contribution"], reverse=True)

        return {
            "risk_score": round(risk_score, 4),
            "flag": flag,
            "threshold_used": self.optimal_threshold,
            "top_shap_features": contribs[:top_n_shap],
        }

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Evaluate on a full DataFrame using the calibrated threshold."""
        X = df[self.feature_names]
        y_true = df["Class"].values
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.optimal_threshold).astype(int)

        report = classification_report(y_true, preds, target_names=["Normal", "Fraud"], output_dict=True)

        return {
            "threshold": self.optimal_threshold,
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
            "classification_report": report,
        }

    def save(self, directory: Path | None = None) -> Path:
        target_dir = directory or MODELS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "xgboost_fraud_model.joblib"
        joblib.dump({
            "model": self.model,
            "optimal_threshold": self.optimal_threshold,
            "explainer": self.explainer,
            "feature_names": self.feature_names,
        }, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "FraudClassifier":
        model_path = path or (MODELS_DIR / "xgboost_fraud_model.joblib")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run scripts/train_classifier.py first."
            )
        data = joblib.load(model_path)
        return cls(
            model=data["model"],
            optimal_threshold=data["optimal_threshold"],
            explainer=data["explainer"],
        )
