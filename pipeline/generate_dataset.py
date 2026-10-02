"""
Generates Dhara-Safe's datasets.

*** THIS DATA IS SYNTHETIC / DEMO DATA. ***
Spec section 5 is explicit: production predictions must run on real sources only.
There is no public, pre-cleaned, India-wide, village-level historical flood+landslide
CSV bundled with an API key in this environment, so this script builds a
statistically-realistic stand-in: seasonal monsoon rainfall curves per region,
terrain-driven runoff, and rule-plus-noise flood/landslide occurrence — enough
signal for the ML pipeline (ARIMA/SCS-CN/Random Forest) to genuinely train and
be evaluated on, and enough realism to demo end-to-end.

Before a real deployment, swap this file's output for:
  - IMD/data.gov.in gridded rainfall + CWC gauge archives (rainfall/river columns)
  - NRSC/Bhuvan or NIDM landslide & flood inventories (occurrence labels)
  - SRTM/CartoDEM (elevation/slope/aspect, already reasonably approximated per-location)
and re-run pipeline/train_model.py — nothing else in the pipeline needs to change,
since app.py and train_model.py only depend on the column names in FEATURE_COLUMNS.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.locations import LOCATIONS
from pipeline.hydrology import scs_cn_runoff

OUT_DIR = Path(__file__).resolve().parent.parent / "datasets"
OUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

MASTER_COLUMNS = [
    "timestamp", "location_id", "latitude", "longitude", "state", "district", "block", "village",
    "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_12h", "rainfall_24h", "rainfall_72h",
    "rainfall_forecast_6h", "temperature", "humidity", "pressure", "cloud_cover",
    "soil_moisture", "elevation", "slope", "aspect", "flow_accumulation",
    "river_level", "river_discharge", "river_rate_of_rise", "river_distance_km",
    "earthquake_magnitude", "earthquake_distance_km", "earthquake_depth_km",
    "glacier_distance_km", "snow_cover_pct", "satellite_water_change_pct",
    "historical_flood_count", "historical_flood_severity", "historical_landslide_count",
    "flood_occurred", "landslide_occurred", "flood_risk", "landslide_risk", "risk_level",
]


def seasonal_rainfall_mm(day_of_year: int, is_himalayan: bool, is_northeast: bool) -> float:
    """Mean daily rainfall (mm) driven by the Indian monsoon calendar (very simplified)."""
    # Monsoon core: day-of-year ~152 (Jun 1) to ~273 (Sep 30)
    monsoon = np.exp(-((day_of_year - 200) ** 2) / (2 * 45 ** 2))
    base = 3.0 + 22.0 * monsoon
    if is_northeast:
        base *= 1.6  # Assam/NE gets heavier, more sustained rainfall
    if is_himalayan:
        # extra pre-monsoon (Mar-May) western disturbance bump
        premonsoon = np.exp(-((day_of_year - 90) ** 2) / (2 * 20 ** 2))
        base += 6.0 * premonsoon
    return max(0.0, base)


def gen_series_for_location(loc: dict, start: datetime, end: datetime) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    n = len(days)
    is_himalayan = loc["state"] in {"Uttarakhand", "Himachal Pradesh", "Sikkim", "Jammu and Kashmir",
                                     "Arunachal Pradesh", "Ladakh", "West Bengal"} and loc["elevation_m"] > 600
    is_northeast = loc["state"] in {"Assam", "Arunachal Pradesh", "Sikkim"}

    rainfall_daily = np.zeros(n)
    is_cloudburst = np.zeros(n, dtype=bool)
    soil_moisture = np.zeros(n)
    river_level = np.zeros(n)
    sm = 35.0  # start soil moisture %
    rl_base = 2.0 + loc["elevation_m"] * 0.0005  # arbitrary base river level
    rl = rl_base

    for i, d in enumerate(days):
        doy = d.dayofyear
        mean_rain = seasonal_rainfall_mm(doy, is_himalayan, is_northeast)
        # occasional extreme-rain cluster (cloudburst-like), more likely if flood/landslide prone
        extreme_p = 0.012 if (loc["flood_prone"] or loc["landslide_prone"]) else 0.004
        cloudburst_today = RNG.random() < extreme_p and 150 <= doy <= 280
        if cloudburst_today:
            rain = mean_rain + RNG.gamma(shape=2.0, scale=35.0)
        else:
            rain = max(0.0, RNG.gamma(shape=1.3, scale=max(mean_rain, 0.5) / 1.3))
        rainfall_daily[i] = rain
        is_cloudburst[i] = cloudburst_today

        # soil moisture responds to rainfall, decays otherwise
        sm = sm * 0.92 + rain * 0.9
        sm = float(np.clip(sm, 8, 100))
        soil_moisture[i] = sm

        # river level responds to rainfall + soil saturation (proxy for catchment runoff)
        inflow = rain * 0.03 + max(0, sm - 60) * 0.01
        rl = rl * 0.85 + rl_base * 0.15 + inflow
        rl = float(max(0.3, rl))
        river_level[i] = rl

    df = pd.DataFrame({"date": days, "rainfall_24h": rainfall_daily,
                        "soil_moisture": soil_moisture, "river_level": river_level})
    # rolling aggregates
    df["rainfall_72h"] = df["rainfall_24h"].rolling(3, min_periods=1).sum()

    # Sub-daily split: a STEADY monsoon day spreads its total across 24h, but a
    # cloudburst genuinely concentrates most of the day's rain into 1-6 hours
    # (real events regularly dump 50-100mm+ in a single hour). Using one fixed
    # ratio for both cases meant even our "extreme" synthetic days only ever
    # produced modest rainfall_1h values (~6-12mm) -- nothing close to a real
    # cloudburst -- so the model never learned what genuinely extreme short-
    # window rainfall looks like. This was caught by real-event backtesting
    # (see pipeline/backtest_historical.py) systematically under-predicting.
    ratio_12h = np.where(is_cloudburst, 0.92, 0.60)
    ratio_6h = np.where(is_cloudburst, 0.80, 0.35)
    ratio_3h = np.where(is_cloudburst, 0.62, 0.20)
    ratio_1h = np.where(is_cloudburst, 0.40, 0.08)
    # small per-day random jitter so cloudburst days aren't all identically shaped
    jitter = RNG.uniform(0.85, 1.15, n)
    df["rainfall_12h"] = df["rainfall_24h"] * ratio_12h
    df["rainfall_6h"] = df["rainfall_24h"] * ratio_6h * jitter
    df["rainfall_3h"] = df["rainfall_24h"] * ratio_3h * jitter
    df["rainfall_1h"] = df["rainfall_24h"] * ratio_1h * jitter
    df["rainfall_forecast_6h"] = df["rainfall_6h"].shift(-1).fillna(df["rainfall_6h"].mean()) * (
        1 + RNG.normal(0, 0.15, n))
    df["rainfall_forecast_6h"] = df["rainfall_forecast_6h"].clip(lower=0)
    df["river_rate_of_rise"] = df["river_level"].diff().fillna(0.0)
    df["river_discharge"] = df["river_level"] * (18 + RNG.normal(0, 1.5, n))

    # weather extras (light synthetic realism, not used as model features but kept for the schema)
    seasonal_temp = 24 + 8 * np.sin(2 * np.pi * (df["date"].dt.dayofyear - 60) / 365) \
        - loc["elevation_m"] / 300.0
    df["temperature"] = seasonal_temp + RNG.normal(0, 1.5, n)
    df["humidity"] = np.clip(50 + df["rainfall_24h"] * 1.2 + RNG.normal(0, 5, n), 15, 100)
    df["pressure"] = 1013 - loc["elevation_m"] * 0.11 + RNG.normal(0, 2, n)
    df["cloud_cover"] = np.clip(20 + df["rainfall_24h"] * 2.5 + RNG.normal(0, 8, n), 0, 100)
    df["snow_cover_pct"] = 0.0
    if is_himalayan and loc["elevation_m"] > 2500:
        winter = np.clip(-np.sin(2 * np.pi * (df["date"].dt.dayofyear - 15) / 365), 0, 1)
        df["snow_cover_pct"] = np.clip(winter * 70 + RNG.normal(0, 5, n), 0, 100)
    df["satellite_water_change_pct"] = np.clip(
        (df["river_level"] - df["river_level"].rolling(7, min_periods=1).mean()) * 8, -40, 60)

    # earthquake events: rare, random magnitude/depth/distance, more common in Himalayan seismic belt
    eq_p = 0.006 if is_himalayan else 0.001
    eq_events = RNG.random(n) < eq_p
    df["earthquake_magnitude"] = np.where(eq_events, RNG.uniform(3.5, 6.5, n), np.nan)
    df["earthquake_distance_km"] = np.where(eq_events, RNG.uniform(2, 80, n), np.nan)
    df["earthquake_depth_km"] = np.where(eq_events, RNG.uniform(5, 40, n), np.nan)
    # forward-fill so "days since" style modifiers see recent quakes for a couple weeks
    df["earthquake_magnitude"] = df["earthquake_magnitude"].ffill(limit=14)
    df["earthquake_distance_km"] = df["earthquake_distance_km"].ffill(limit=14)
    df["earthquake_depth_km"] = df["earthquake_depth_km"].ffill(limit=14)

    # static per-location fields
    df["location_id"] = loc["id"]
    df["latitude"] = loc["lat"]
    df["longitude"] = loc["lon"]
    df["state"] = loc["state"]
    df["district"] = loc["district"]
    df["block"] = loc["block"]
    df["village"] = loc["village"]
    df["elevation"] = loc["elevation_m"]
    df["slope"] = loc["slope_deg"]
    df["aspect"] = RNG.uniform(0, 360)
    df["flow_accumulation"] = np.clip(RNG.normal(200 if loc["flood_prone"] else 60, 40, n), 5, None)
    df["river_distance_km"] = loc["river_distance_km"]
    df["glacier_distance_km"] = loc["glacier_distance_km"] if loc["glacier_distance_km"] is not None else np.nan

    # running historical counters (how many floods/landslides *so far* at this point in time)
    df = df.sort_values("date").reset_index(drop=True)

    # --- SCS-CN runoff feeds into occurrence rule ---
    runoff_mm = np.array([
        scs_cn_runoff(r, sm_, loc["slope_deg"]).runoff_mm
        for r, sm_ in zip(df["rainfall_24h"], df["soil_moisture"])
    ])

    flood_potential = (
        0.55 * (df["rainfall_24h"] / 80.0) +
        0.20 * (runoff_mm / 60.0) +
        0.15 * (df["soil_moisture"] / 100.0) +
        0.15 * np.clip(df["river_rate_of_rise"] / 0.5, 0, 1) +
        (0.25 if loc["flood_prone"] else -0.15) +
        (0.15 * (1 - min(loc["river_distance_km"], 5) / 5.0))
    )
    landslide_potential = (
        0.35 * (df["rainfall_24h"] / 80.0) +
        0.25 * (df["soil_moisture"] / 100.0) +
        0.20 * min(loc["slope_deg"], 40) / 40.0 +
        (0.30 if loc["landslide_prone"] else -0.20)
    )
    # earthquake modifier bumps landslide potential on steep terrain
    recent_quake = df["earthquake_magnitude"].notna() & (loc["slope_deg"] > 15)
    landslide_potential = landslide_potential + np.where(recent_quake, 0.15, 0.0)

    flood_p = 1 / (1 + np.exp(-6 * (flood_potential - 1.05)))
    landslide_p = 1 / (1 + np.exp(-6 * (landslide_potential - 0.95)))

    df["flood_occurred"] = (RNG.random(n) < flood_p).astype(int)
    df["landslide_occurred"] = (RNG.random(n) < landslide_p).astype(int) if loc["slope_deg"] > 8 else 0
    df["flood_risk"] = np.round(flood_p * 100, 1)
    df["landslide_risk"] = np.round(landslide_p * 100, 1)

    combined = np.maximum(df["flood_risk"], df["landslide_risk"])
    df["risk_level"] = pd.cut(combined, bins=[-1, 25, 50, 75, 100.1],
                               labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])

    df["historical_flood_count"] = df["flood_occurred"].cumsum().shift(1).fillna(0)
    df["historical_landslide_count"] = df["landslide_occurred"].cumsum().shift(1).fillna(0)
    df["historical_flood_severity"] = np.where(
        df["historical_flood_count"] > 0,
        np.clip(df["historical_flood_count"] / (df.index + 1) * 100, 0, 100), 0)

    df = df.rename(columns={"date": "timestamp"})
    return df[MASTER_COLUMNS]


def main():
    print("Generating per-location series ...")
    train_frames, val_frames = [], []
    for loc in LOCATIONS:
        full = gen_series_for_location(loc, datetime(2022, 1, 1), datetime(2025, 12, 31))
        train = full[full["timestamp"] < "2025-01-01"].copy()
        val = full[full["timestamp"] >= "2025-01-01"].copy()
        train_frames.append(train)
        val_frames.append(val)
        print(f"  {loc['id']:12s} {loc['village']:20s} train={len(train):5d} val={len(val):4d}")

    train_df = pd.concat(train_frames, ignore_index=True)
    val_df = pd.concat(val_frames, ignore_index=True)

    train_path = OUT_DIR / "historical_training_data.csv"
    val_path = OUT_DIR / "validation_2025.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    master = pd.concat(
        [train_df.assign(split="train"), val_df.assign(split="validation")], ignore_index=True
    )
    master_path = OUT_DIR / "dhara_safe_master_dataset.csv"
    master.to_csv(master_path, index=False)

    # a small locations reference CSV too (section-14-adjacent convenience file)
    pd.DataFrame(LOCATIONS).to_csv(OUT_DIR / "locations_india.csv", index=False)

    print(f"\nWrote:\n  {train_path} ({len(train_df):,} rows)\n  {val_path} ({len(val_df):,} rows)"
          f"\n  {master_path} ({len(master):,} rows)\n  {OUT_DIR/'locations_india.csv'} ({len(LOCATIONS)} rows)")
    print(f"\nFlood positive rate (train): {train_df['flood_occurred'].mean():.3%}")
    print(f"Landslide positive rate (train): {train_df['landslide_occurred'].mean():.3%}")
    print("\n*** SYNTHETIC / DEMO DATA — see module docstring before using for real predictions. ***")


if __name__ == "__main__":
    main()
