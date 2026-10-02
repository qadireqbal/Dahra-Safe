"""
Simplified SCS Curve Number (SCS-CN) rainfall-runoff model.

Reference: USDA NRCS National Engineering Handbook, Part 630, Chapter 10.

Q = (P - Ia)^2 / (P - Ia + S)   for P > Ia, else Q = 0
S = (25400 / CN) - 254           (mm, metric form)
Ia = 0.2 * S                     (initial abstraction, standard assumption)

CN (curve number, 0-100) depends on soil group + land use + antecedent moisture.
Higher CN => less infiltration => more runoff. We estimate CN from soil moisture
(as a proxy for Antecedent Moisture Condition, AMC) and slope (steeper slope drains
faster / stores less water, so we nudge CN up a little for steep terrain).
"""

from dataclasses import dataclass


@dataclass
class RunoffResult:
    curve_number: float
    potential_retention_s_mm: float
    initial_abstraction_mm: float
    runoff_mm: float
    runoff_ratio: float  # runoff / rainfall, 0-1


def estimate_curve_number(soil_moisture_pct: float, slope_deg: float, base_cn: float = 70.0) -> float:
    """
    Estimate an effective CN for the demo model.

    base_cn ~70 is a moderate value for mixed agricultural/village land use (AMC-II).
    Antecedent moisture shifts CN toward AMC-I (dry, lower CN) or AMC-III (wet, higher CN)
    using the standard Hawkins conversion approximation.
    Steep slopes get a small upward nudge because storage capacity is lower.
    """
    if soil_moisture_pct >= 80:
        # AMC-III (wet) approx conversion
        cn = (23 * base_cn) / (10 + 0.13 * base_cn)
    elif soil_moisture_pct <= 30:
        # AMC-I (dry) approx conversion
        cn = (4.2 * base_cn) / (10 - 0.058 * base_cn)
    else:
        # linear interpolation between dry and wet conversions across 30-80%
        wet_cn = (23 * base_cn) / (10 + 0.13 * base_cn)
        dry_cn = (4.2 * base_cn) / (10 - 0.058 * base_cn)
        frac = (soil_moisture_pct - 30) / 50.0
        cn = dry_cn + frac * (wet_cn - dry_cn)

    slope_adj = min(slope_deg, 40) * 0.05  # up to +2 CN at very steep slopes
    cn = cn + slope_adj
    return max(30.0, min(cn, 98.0))


def scs_cn_runoff(rainfall_mm: float, soil_moisture_pct: float, slope_deg: float,
                   base_cn: float = 70.0) -> RunoffResult:
    if rainfall_mm < 0:
        rainfall_mm = 0.0

    cn = estimate_curve_number(soil_moisture_pct, slope_deg, base_cn)
    s = (25400.0 / cn) - 254.0
    ia = 0.2 * s

    if rainfall_mm > ia:
        q = ((rainfall_mm - ia) ** 2) / (rainfall_mm - ia + s)
    else:
        q = 0.0

    ratio = (q / rainfall_mm) if rainfall_mm > 0 else 0.0
    return RunoffResult(
        curve_number=round(cn, 1),
        potential_retention_s_mm=round(s, 1),
        initial_abstraction_mm=round(ia, 1),
        runoff_mm=round(q, 2),
        runoff_ratio=round(min(ratio, 1.0), 3),
    )
