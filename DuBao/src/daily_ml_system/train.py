# -*- coding: utf-8 -*-
"""Train ban đầu / retrain: RandomForest, XGBoost, LinearRegression — dự đoán mưa ngày mai."""
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config
from .features import build_training_frame, load_daily_combined, merge_actuals_from_log

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def _metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_all(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Huấn luyện 3 mô hình, lưu bundle triple_models.pkl.
    Trả về (bundle_dict, metrics_per_model).
    """
    path = data_path or config.DAILY_COMBINED
    df = load_daily_combined(path)
    df = merge_actuals_from_log(df, config.PREDICTION_LOG)
    train_df, feature_cols = build_training_frame(df)

    X = train_df[feature_cols].values
    y = train_df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "rf": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=2, random_state=random_state, n_jobs=-1
        ),
        "lr": LinearRegression(),
    }
    if xgb is None:
        raise ImportError("Cần cài xgboost: pip install xgboost")
    models["xgb"] = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        random_state=random_state,
        n_jobs=-1,
    )

    metrics_out: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        metrics_out[name] = _metrics(y_test, pred)
        print(f"[{name}] MAE={metrics_out[name]['mae']:.3f} RMSE={metrics_out[name]['rmse']:.3f} R2={metrics_out[name]['r2']:.4f}")

    os.makedirs(os.path.dirname(config.MODEL_BUNDLE_PATH), exist_ok=True)
    bundle = {
        "models": models,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics_test": metrics_out,
        "base_year": config.BASE_YEAR,
    }
    with open(config.MODEL_BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved: {config.MODEL_BUNDLE_PATH}")
    return bundle, metrics_out


def main():
    train_all()


if __name__ == "__main__":
    main()
