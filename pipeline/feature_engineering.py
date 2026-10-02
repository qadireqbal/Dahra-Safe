"""
Feature engineering shared by pipeline/train_model.py (training) and app.py (live inference).

Keeping this in one place is what makes training/serving skew impossible: both sides
call build_features() on raw readings and get the same columns in the same order.
"""

from .hydrology import scs_cn_runoff

# Exact column order the ML models are trained on and must be fed at inference time.
FEATURE_COLUMNS = [
    "rainfall_1h", "rainfall_3h", "rainfall_6h", "rainfall_12h", "rainfall_24h", "rainfall_72h",
    "rainfall_forecast_6h",
    "runoff_mm", "runoff_ratio",
    "soil_moisture", "elevation", "slope",
    "river_level", "river_rate_of_rise", "river_distance_km",
    "earthquake_modifier", "glacier_modifier",
    "historical_flood_count", "historical_landslide_count",
]


def earthquake_modifier(days_since_last_quake, magnitude, distance_km, slope_deg):
    """
    Earthquake is a landslide risk MODIFIER, not a direct flood predictor (spec section 11).
    Recent + strong + close + on steep terrain => higher modifier (0-1).
    Decays to ~0 after ~30 days.
    """
    if days_since_last_quake is None or magnitude is None or distance_km is None:
        return 0.0
    if days_since_last_quake > 30 or slope_deg < 10:
        return 0.0
    recency = max(0.0, 1.0 - days_since_last_quake / 30.0)
    strength = max(0.0, (magnitude - 3.0) / 5.0)
    proximity = max(0.0, 1.0 - min(distance_km, 100) / 100.0)
    return round(min(1.0, recency * strength * proximity), 3)


def glacier_modifier(glacier_distance_km, temperature_c, is_himalayan_region):
    """
    Glacier/snow melt only matters near Himalayan terrain (spec section 12).
    Warm temperature + close glacier => higher melt-driven downstream risk (0-1).
    Auto-ignored (returns 0) for flat, non-Himalayan regions.
    """
    if not is_himalayan_region or glacier_distance_km is None:
        return 0.0
    proximity = max(0.0, 1.0 - min(glacier_distance_km, 60) / 60.0)
    melt_factor = max(0.0, min(1.0, (temperature_c - 5) / 20.0)) if temperature_c is not None else 0.3
    return round(proximity * melt_factor, 3)


def build_features(raw: dict) -> dict:
    """
    raw must contain: rainfall_1h/3h/6h/12h/24h/72h, rainfall_forecast_6h, soil_moisture,
    elevation, slope, river_level, river_rate_of_rise, river_distance_km,
    historical_flood_count, historical_landslide_count,
    and optionally: eq_days_since, eq_magnitude, eq_distance_km,
    glacier_distance_km, temperature_c, is_himalayan
    """
    runoff = scs_cn_runoff(
        rainfall_mm=raw.get("rainfall_24h", 0.0),
        soil_moisture_pct=raw.get("soil_moisture", 40.0),
        slope_deg=raw.get("slope", 5.0),
    )

    eq_mod = earthquake_modifier(
        raw.get("eq_days_since"), raw.get("eq_magnitude"), raw.get("eq_distance_km"),
        raw.get("slope", 0.0),
    )
    gl_mod = glacier_modifier(
        raw.get("glacier_distance_km"), raw.get("temperature_c"), raw.get("is_himalayan", False)
    )

    feats = {
        "rainfall_1h": raw.get("rainfall_1h", 0.0),
        "rainfall_3h": raw.get("rainfall_3h", 0.0),
        "rainfall_6h": raw.get("rainfall_6h", 0.0),
        "rainfall_12h": raw.get("rainfall_12h", 0.0),
        "rainfall_24h": raw.get("rainfall_24h", 0.0),
        "rainfall_72h": raw.get("rainfall_72h", 0.0),
        "rainfall_forecast_6h": raw.get("rainfall_forecast_6h", 0.0),
        "runoff_mm": runoff.runoff_mm,
        "runoff_ratio": runoff.runoff_ratio,
        "soil_moisture": raw.get("soil_moisture", 40.0),
        "elevation": raw.get("elevation", 0.0),
        "slope": raw.get("slope", 0.0),
        "river_level": raw.get("river_level", 0.0),
        "river_rate_of_rise": raw.get("river_rate_of_rise", 0.0),
        "river_distance_km": raw.get("river_distance_km", 5.0),
        "earthquake_modifier": eq_mod,
        "glacier_modifier": gl_mod,
        "historical_flood_count": raw.get("historical_flood_count", 0),
        "historical_landslide_count": raw.get("historical_landslide_count", 0),
    }
    feats["_runoff_detail"] = runoff
    return feats


def feature_vector(raw: dict):
    """Return the ordered numeric list the sklearn models expect."""
    feats = build_features(raw)
    return [feats[c] for c in FEATURE_COLUMNS]
