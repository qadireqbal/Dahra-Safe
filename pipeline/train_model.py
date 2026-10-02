"""
Trains Dhara-Safe's Layer-1 (ARIMA rainfall forecast) and Layer-3 (Random Forest
flood/landslide classifiers), validates both properly, and saves everything app.py
needs for live inference:

  pipeline/models/rf_flood.pkl
  pipeline/models/rf_landslide.pkl
  pipeline/models/arima_orders.json      (best (p,d,q) per state, picked by AIC)
  pipeline/models/feature_columns.json
  pipeline/models/validation_results.json  (2025 backtest metrics — real numbers, not invented)

Run: python -m pipeline.train_model
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              confusion_matrix, roc_auc_score, mean_absolute_error,
                              mean_squared_error)
from statsmodels.tsa.arima.model import ARIMA

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.feature_engineering import FEATURE_COLUMNS, build_features

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
MODEL_DIR = BASE_DIR / "pipeline" / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


# ----------------------------------------------------------------------------
# Layer 1: ARIMA rainfall forecasting, auto order selection + walk-forward CV
# ----------------------------------------------------------------------------

def auto_arima_order(series: pd.Series, p_range=(0, 1, 2), d_range=(0, 1), q_range=(0, 1)):
    """Grid-search small (p,d,q) space, pick lowest AIC. Cheap stand-in for pmdarima.
    Order search runs on the most recent 365 points only, purely for speed; the
    chosen order is then used for the full walk-forward validation below."""
    sample = series.iloc[-365:] if len(series) > 365 else series
    best_aic, best_order = np.inf, (1, 1, 1)
    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(sample, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic, best_order = model.aic, (p, d, q)
                except Exception:
                    continue
    return best_order, best_aic


def walk_forward_validate(series: pd.Series, order, window=120, horizon=1, step=45):
    """
    Slide a training window forward, forecast `horizon` step(s) ahead, compare to
    actual, repeat. This is what spec section 17 asks for instead of a single train/test split.
    Returns RMSE, MAE, NSE over all forecasted points.
    """
    actuals, preds = [], []
    n = len(series)
    for start in range(0, n - window - horizon, step):
        train_slice = series.iloc[start:start + window]
        actual = series.iloc[start + window: start + window + horizon]
        try:
            model = ARIMA(train_slice, order=order).fit()
            forecast = model.forecast(steps=horizon)
            preds.extend(list(forecast.values))
            actuals.extend(list(actual.values))
        except Exception:
            continue

    actuals = np.array(actuals)
    preds = np.array(preds)
    if len(actuals) == 0:
        return {"rmse": None, "mae": None, "nse": None, "n_points": 0}

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))
    denom = np.sum((actuals - actuals.mean()) ** 2)
    nse = float(1 - np.sum((actuals - preds) ** 2) / denom) if denom > 0 else None
    return {"rmse": round(rmse, 3), "mae": round(mae, 3),
            "nse": round(nse, 3) if nse is not None else None, "n_points": int(len(actuals))}


def train_arima_per_state(train_df: pd.DataFrame):
    print("\n=== Layer 1: ARIMA rainfall forecasting (auto-order + walk-forward CV) ===")
    orders = {}
    wf_metrics = {}
    for state, g in train_df.groupby("state"):
        series = g.sort_values("timestamp").groupby("timestamp")["rainfall_24h"].mean()
        series = series.asfreq("D").interpolate()
        if len(series) < 250:
            continue
        order, aic = auto_arima_order(series)
        metrics = walk_forward_validate(series, order)
        orders[state] = {"order": order, "aic": round(float(aic), 1)}
        wf_metrics[state] = metrics
        print(f"  {state:20s} order={order}  AIC={aic:9.1f}  "
              f"RMSE={metrics['rmse']}  MAE={metrics['mae']}  NSE={metrics['nse']}  "
              f"(n={metrics['n_points']})")
    return orders, wf_metrics


# ----------------------------------------------------------------------------
# Layer 3: Random Forest flood / landslide classifiers
# ----------------------------------------------------------------------------

def rows_to_feature_matrix(df: pd.DataFrame):
    rows = df.to_dict("records")
    feats = []
    for r in rows:
        raw = dict(
            rainfall_1h=r["rainfall_1h"], rainfall_3h=r["rainfall_3h"], rainfall_6h=r["rainfall_6h"],
            rainfall_12h=r["rainfall_12h"], rainfall_24h=r["rainfall_24h"], rainfall_72h=r["rainfall_72h"],
            rainfall_forecast_6h=r["rainfall_forecast_6h"], soil_moisture=r["soil_moisture"],
            elevation=r["elevation"], slope=r["slope"], river_level=r["river_level"],
            river_rate_of_rise=r["river_rate_of_rise"], river_distance_km=r["river_distance_km"],
            historical_flood_count=r["historical_flood_count"],
            historical_landslide_count=r["historical_landslide_count"],
            eq_days_since=(0 if pd.notna(r.get("earthquake_magnitude")) else None),
            eq_magnitude=r.get("earthquake_magnitude"), eq_distance_km=r.get("earthquake_distance_km"),
            glacier_distance_km=r.get("glacier_distance_km"), temperature_c=r.get("temperature"),
            is_himalayan=pd.notna(r.get("glacier_distance_km")),
        )
        f = build_features(raw)
        feats.append([f[c] for c in FEATURE_COLUMNS])
    return np.array(feats, dtype=float)


def train_random_forests(train_df: pd.DataFrame):
    print("\n=== Layer 3: Random Forest — flood & landslide classifiers ===")
    X = rows_to_feature_matrix(train_df)
    y_flood = train_df["flood_occurred"].values
    y_slide = train_df["landslide_occurred"].values

    rf_flood = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=8,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(X, y_flood)

    rf_slide = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=8,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(X, y_slide)

    imp_flood = dict(zip(FEATURE_COLUMNS, rf_flood.feature_importances_.round(4).tolist()))
    imp_slide = dict(zip(FEATURE_COLUMNS, rf_slide.feature_importances_.round(4).tolist()))
    print("  Flood feature importance (top 6):",
          sorted(imp_flood.items(), key=lambda kv: -kv[1])[:6])
    print("  Landslide feature importance (top 6):",
          sorted(imp_slide.items(), key=lambda kv: -kv[1])[:6])

    return rf_flood, rf_slide, imp_flood, imp_slide


# ----------------------------------------------------------------------------
# 2025 backtest (spec section 44)
# ----------------------------------------------------------------------------

def backtest_2025(rf_flood, rf_slide, val_df: pd.DataFrame):
    print("\n=== 2025 real-event backtest ===")
    X_val = rows_to_feature_matrix(val_df)
    y_flood = val_df["flood_occurred"].values
    y_slide = val_df["landslide_occurred"].values

    p_flood = rf_flood.predict_proba(X_val)[:, 1]
    p_slide = rf_slide.predict_proba(X_val)[:, 1]
    pred_flood = (p_flood >= 0.5).astype(int)
    pred_slide = (p_slide >= 0.5).astype(int)

    def block(y_true, y_pred, y_prob, label):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None
        result = {
            "events_tested": int(y_true.sum()),
            "correctly_detected": int(tp),
            "missed_events": int(fn),
            "false_alarms": int(fp),
            "true_negatives": int(tn),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
            "roc_auc": round(auc, 3) if auc is not None else None,
        }
        print(f"  {label}: tested={result['events_tested']} detected={result['correctly_detected']} "
              f"missed={result['missed_events']} false_alarms={result['false_alarms']} "
              f"precision={result['precision']} recall={result['recall']} f1={result['f1']} "
              f"auc={result['roc_auc']}")
        return result

    return {
        "flood": block(y_flood, pred_flood, p_flood, "Flood"),
        "landslide": block(y_slide, pred_slide, p_slide, "Landslide"),
    }


def main():
    sklearn_version = sklearn.__version__
    train_df = pd.read_csv(DATA_DIR / "historical_training_data.csv", parse_dates=["timestamp"])
    val_df = pd.read_csv(DATA_DIR / "validation_2025.csv", parse_dates=["timestamp"])

    arima_orders, arima_wf_metrics = train_arima_per_state(train_df)
    rf_flood, rf_slide, imp_flood, imp_slide = train_random_forests(train_df)
    backtest = backtest_2025(rf_flood, rf_slide, val_df)

    joblib.dump(rf_flood, MODEL_DIR / "rf_flood.pkl")
    joblib.dump(rf_slide, MODEL_DIR / "rf_landslide.pkl")
    (MODEL_DIR / "arima_orders.json").write_text(json.dumps(arima_orders, indent=2))
    (MODEL_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    (MODEL_DIR / "feature_importance.json").write_text(
        json.dumps({"flood": imp_flood, "landslide": imp_slide}, indent=2))

    validation_results = {
        "validated_on": "2025 synthetic-but-realistic held-out year (see generate_dataset.py "
                         "docstring — replace with real 2025 IMD/CWC/NDMA event records before "
                         "reporting this number to judges as real-world accuracy)",
        "arima_walk_forward": arima_wf_metrics,
        "random_forest_backtest": backtest,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "trained_with_sklearn": sklearn_version,
        "training_rows": int(len(train_df)),
    }
    (MODEL_DIR / "validation_results.json").write_text(json.dumps(validation_results, indent=2, default=str))

    print(f"\nSaved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
