"""
Inverse Distance Weighting (IDW) — a simple, well-established spatial
interpolation method. Given a set of known points with values (here: our
monitored locations' latest risk scores), it estimates the value at any other
point by averaging the known values, weighted by 1/distance^power — nearer
points count more. This is what lets the app say something about risk at a
village that ISN'T one of the monitored points, instead of just showing
nothing for the gaps between them.

This is NOT a replacement for the real ML prediction (Random Forest + SCS-CN)
at monitored points — it is a lightweight, honest estimate for everywhere
else, and every result carries confidence + "based on N monitored points,
nearest is Xkm away" so the UI can be upfront about how rough it is.
"""

import math


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def idw_interpolate(query_lat, query_lon, known_points, power=2, k_nearest=6):
    """
    known_points: list of dicts with 'lat', 'lon', 'value' (e.g. risk_score).
    Uses the k_nearest points only (classic IDW variant — using every
    monitored point in India to estimate a point in, say, Bihar would dilute
    the estimate with irrelevant, far-away readings).

    Returns dict with the interpolated value, a rough confidence (higher when
    points are close and numerous), and the contributing points for transparency.
    """
    if not known_points:
        return None

    dists = []
    for p in known_points:
        d = haversine_km(query_lat, query_lon, p["lat"], p["lon"])
        dists.append((d, p))
    dists.sort(key=lambda x: x[0])
    nearest = dists[:k_nearest]

    if nearest[0][0] < 0.05:
        return {
            "value": nearest[0][1]["value"], "confidence": "exact_match",
            "nearest_km": round(nearest[0][0], 2), "contributing_points": 1,
        }

    weights = [1.0 / (d ** power) for d, _ in nearest]
    total_w = sum(weights)
    value = sum(w * p["value"] for w, (d, p) in zip(weights, nearest)) / total_w

    nearest_km = nearest[0][0]
    if nearest_km < 25:
        confidence = "moderate"
    elif nearest_km < 100:
        confidence = "low"
    else:
        confidence = "very_low"

    return {
        "value": round(value, 1), "confidence": confidence,
        "nearest_km": round(nearest_km, 1), "contributing_points": len(nearest),
        "sources": [{"id": p.get("id"), "distance_km": round(d, 1)} for d, p in nearest],
    }


def make_grid(lat_min, lat_max, lon_min, lon_max, step_deg=1.0):
    """Coarse lat/lon grid over a bounding box. step_deg=1.0 is roughly
    ~111km spacing — coarse on purpose (this is a demo-scale heatmap grid,
    not a survey-grade raster)."""
    cells = []
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            cells.append((round(lat, 2), round(lon, 2)))
            lon += step_deg
        lat += step_deg
    return cells
