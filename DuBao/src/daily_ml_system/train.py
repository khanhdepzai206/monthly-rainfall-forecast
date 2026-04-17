# -*- coding: utf-8 -*-
"""Train ban đầu / retrain: 2-stage (rain/no-rain + mm) cho 3 mô hình."""
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score
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

def _log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0))

def _expm1(x: np.ndarray) -> np.ndarray:
    return np.expm1(x)


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

    X = train_df[feature_cols].values.astype(np.float64)
    y = train_df["target"].values.astype(np.float64)
    y_has_rain = (y > 0).astype(int)

    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y, y_has_rain, test_size=test_size, random_state=random_state, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if xgb is None:
        raise ImportError("Cần cài xgboost: pip install xgboost")

    # 3 mô hình: xgb / et / rf (mỗi mô hình gồm classifier + regressor)
    models = {
        "xgb": {
            "clf": xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            "reg": xgb.XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
            ),
        },
        "et": {
            "clf": ExtraTreesClassifier(
                n_estimators=400, max_depth=16, min_samples_leaf=2, random_state=random_state, n_jobs=-1
            ),
            "reg": ExtraTreesRegressor(
                n_estimators=400, max_depth=16, min_samples_leaf=2, random_state=random_state, n_jobs=-1
            ),
        },
        "rf": {
            "clf": RandomForestClassifier(
                n_estimators=400, max_depth=16, min_samples_leaf=2, random_state=random_state, n_jobs=-1
            ),
            "reg": RandomForestRegressor(
                n_estimators=400, max_depth=16, min_samples_leaf=2, random_state=random_state, n_jobs=-1
            ),
        },
    }

    # Train/eval
    metrics_out: Dict[str, Dict[str, float]] = {}
    rain_prob_threshold = 0.5
    y_train_log = _log1p(y_train_reg)
    y_test_log = _log1p(y_test_reg)

    for name, pack in models.items():
        clf = pack["clf"]
        reg = pack["reg"]

        clf.fit(X_train_s, y_train_cls)
        cls_pred = clf.predict(X_test_s)
        cls_acc = float(accuracy_score(y_test_cls, cls_pred))
        cls_f1 = float(f1_score(y_test_cls, cls_pred, zero_division=0))

        # regressor chỉ học trên ngày mưa (tập train)
        mask_rain = y_train_cls == 1
        if mask_rain.sum() < 50:
            # fallback nếu quá ít ngày mưa
            reg.fit(X_train_s, y_train_log)
        else:
            reg.fit(X_train_s[mask_rain], y_train_log[mask_rain])

        # dự đoán final mm: nếu prob<threshold => 0, else expm1(reg_pred_log)
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(X_test_s)[:, 1]
        else:
            prob = cls_pred.astype(float)
        reg_pred_log = reg.predict(X_test_s)
        reg_pred_mm = _expm1(reg_pred_log)
        final = np.where(prob >= rain_prob_threshold, reg_pred_mm, 0.0)
        final = np.maximum(final, 0.0)

        m = _metrics(y_test_reg, final)
        m.update({"cls_acc": cls_acc, "cls_f1": cls_f1})
        metrics_out[name] = m
        print(
            f"[{name}] MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} R2={m['r2']:.4f} | cls_acc={cls_acc:.3f} cls_f1={cls_f1:.3f}"
        )

    os.makedirs(os.path.dirname(config.MODEL_BUNDLE_PATH), exist_ok=True)
    bundle = {
        "models": models,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics_test": metrics_out,
        "base_year": config.BASE_YEAR,
        "two_stage": True,
        "rain_prob_threshold": 0.5,
        "target_transform": "log1p",
    }
    with open(config.MODEL_BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved: {config.MODEL_BUNDLE_PATH}")
    return bundle, metrics_out


def main():
    train_all()


if __name__ == "__main__":
    main()
