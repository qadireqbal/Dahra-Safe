"""
Monitored location grid for Dhara-Safe.

This is the seed set of village/ward-level monitoring points used by the demo.
Coordinates are approximate (real place coordinates, not surveyed grid centroids).
elevation_m / slope_deg / river_distance_km / glacier_distance_km are reasonable
real-world approximations for the demo ML pipeline — before production use, replace
with actual DEM (elevation/slope) and hydrology-layer (river distance) lookups, e.g.
via a PostGIS raster query against SRTM/CartoDEM, as described in README.md.

historical_flood_prone / historical_landslide_prone are qualitative flags used only
to bias the synthetic historical-dataset generator so the demo model has a learnable
signal. They are NOT a substitute for the real historical disaster inventory
(section 13 of the spec) that a production system must ingest.
"""

import math

LOCATIONS = [
    # id, village/ward, block, district, state, lat, lon, elevation_m, slope_deg,
    # river_distance_km, glacier_distance_km, flood_prone, landslide_prone
    dict(id="BR-PAT-01", village="Digha", block="Patna Sadar", district="Patna", state="Bihar",
         lat=25.5941, lon=85.1376, elevation_m=53, slope_deg=1.5, river_distance_km=0.8,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="BR-DBG-01", village="Kusheshwar Asthan", block="Kusheshwar Asthan", district="Darbhanga", state="Bihar",
         lat=26.2833, lon=86.0333, elevation_m=45, slope_deg=0.8, river_distance_km=1.2,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="BR-MUZ-01", village="Kanti", block="Kanti", district="Muzaffarpur", state="Bihar",
         lat=26.2000, lon=85.2000, elevation_m=52, slope_deg=1.0, river_distance_km=2.0,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="BR-SUP-01", village="Triveniganj", block="Triveniganj", district="Supaul", state="Bihar",
         lat=26.1167, lon=86.6000, elevation_m=45, slope_deg=0.7, river_distance_km=0.5,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="UK-CHM-01", village="Joshimath", block="Joshimath", district="Chamoli", state="Uttarakhand",
         lat=30.5551, lon=79.5644, elevation_m=1875, slope_deg=32, river_distance_km=0.4,
         glacier_distance_km=18, flood_prone=True, landslide_prone=True),
    dict(id="UK-CHM-02", village="Kedarnath", block="Ukhimath", district="Rudraprayag", state="Uttarakhand",
         lat=30.7346, lon=79.0669, elevation_m=3583, slope_deg=38, river_distance_km=0.2,
         glacier_distance_km=6, flood_prone=True, landslide_prone=True),
    dict(id="UK-DDN-01", village="Rishikesh", block="Doiwala", district="Dehradun", state="Uttarakhand",
         lat=30.0869, lon=78.2676, elevation_m=372, slope_deg=12, river_distance_km=0.3,
         glacier_distance_km=45, flood_prone=True, landslide_prone=False),
    dict(id="UK-PTG-01", village="Devprayag", block="Devprayag", district="Tehri Garhwal", state="Uttarakhand",
         lat=30.1462, lon=78.5967, elevation_m=830, slope_deg=29, river_distance_km=0.1,
         glacier_distance_km=40, flood_prone=True, landslide_prone=True),
    dict(id="SK-GTK-01", village="Gangtok", block="Gangtok", district="East Sikkim", state="Sikkim",
         lat=27.3389, lon=88.6065, elevation_m=1650, slope_deg=27, river_distance_km=0.9,
         glacier_distance_km=22, flood_prone=True, landslide_prone=True),
    dict(id="SK-LCH-01", village="Lachen", block="Lachen", district="North Sikkim", state="Sikkim",
         lat=27.7167, lon=88.5500, elevation_m=2750, slope_deg=34, river_distance_km=0.3,
         glacier_distance_km=9, flood_prone=True, landslide_prone=True),
    dict(id="AS-GHY-01", village="Palasbari", block="Palasbari", district="Kamrup", state="Assam",
         lat=26.1500, lon=91.5500, elevation_m=49, slope_deg=1.2, river_distance_km=0.6,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="AS-DBR-01", village="Dibrugarh", block="Dibrugarh", district="Dibrugarh", state="Assam",
         lat=27.4728, lon=94.9120, elevation_m=108, slope_deg=1.0, river_distance_km=0.5,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="HP-KLU-01", village="Kullu", block="Kullu", district="Kullu", state="Himachal Pradesh",
         lat=31.9578, lon=77.1095, elevation_m=1279, slope_deg=30, river_distance_km=0.4,
         glacier_distance_km=25, flood_prone=True, landslide_prone=True),
    dict(id="HP-KNR-01", village="Kinnaur", block="Kalpa", district="Kinnaur", state="Himachal Pradesh",
         lat=31.5350, lon=78.2600, elevation_m=2960, slope_deg=36, river_distance_km=0.5,
         glacier_distance_km=15, flood_prone=True, landslide_prone=True),
    dict(id="KL-WYD-01", village="Wayanad", block="Meppadi", district="Wayanad", state="Kerala",
         lat=11.6050, lon=76.0800, elevation_m=780, slope_deg=28, river_distance_km=0.7,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="KL-IDK-01", village="Munnar", block="Devikulam", district="Idukki", state="Kerala",
         lat=10.0889, lon=77.0595, elevation_m=1600, slope_deg=31, river_distance_km=0.3,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="MH-MUM-01", village="Kurla", block="Kurla", district="Mumbai Suburban", state="Maharashtra",
         lat=19.0728, lon=72.8826, elevation_m=8, slope_deg=1.0, river_distance_km=0.9,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="MH-RTG-01", village="Mahad", block="Mahad", district="Raigad", state="Maharashtra",
         lat=18.0833, lon=73.4167, elevation_m=41, slope_deg=18, river_distance_km=0.4,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="WB-DGJ-01", village="Kalimpong", block="Kalimpong-I", district="Kalimpong", state="West Bengal",
         lat=27.0644, lon=88.4772, elevation_m=1247, slope_deg=29, river_distance_km=0.5,
         glacier_distance_km=30, flood_prone=True, landslide_prone=True),
    dict(id="JK-JMU-01", village="Kishtwar", block="Kishtwar", district="Kishtwar", state="Jammu and Kashmir",
         lat=33.3120, lon=75.7674, elevation_m=1650, slope_deg=33, river_distance_km=0.3,
         glacier_distance_km=12, flood_prone=True, landslide_prone=True),
    dict(id="AR-ARP-01", village="Tawang", block="Tawang", district="Tawang", state="Arunachal Pradesh",
         lat=27.5860, lon=91.8697, elevation_m=3048, slope_deg=35, river_distance_km=0.6,
         glacier_distance_km=8, flood_prone=True, landslide_prone=True),
    dict(id="OD-KEN-01", village="Jagatsinghpur", block="Jagatsinghpur", district="Jagatsinghpur", state="Odisha",
         lat=20.2667, lon=86.1667, elevation_m=6, slope_deg=0.5, river_distance_km=1.5,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="AP-VSK-01", village="Araku Valley", block="Araku Valley", district="Alluri Sitharama Raju", state="Andhra Pradesh",
         lat=18.3273, lon=82.8770, elevation_m=911, slope_deg=26, river_distance_km=0.7,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="LD-LEH-01", village="Leh", block="Leh", district="Leh", state="Ladakh",
         lat=34.1526, lon=77.5771, elevation_m=3500, slope_deg=24, river_distance_km=0.4,
         glacier_distance_km=5, flood_prone=True, landslide_prone=False),
    dict(id="TN-NLG-01", village="Coonoor", block="Coonoor", district="Nilgiris", state="Tamil Nadu",
         lat=11.3530, lon=76.7950, elevation_m=1850, slope_deg=27, river_distance_km=0.6,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),

    # --- expanded set: more states + more depth per state ---
    dict(id="BR-SIT-01", village="Sitamarhi", block="Sitamarhi Sadar", district="Sitamarhi", state="Bihar",
         lat=26.5941, lon=85.4909, elevation_m=59, slope_deg=0.8, river_distance_km=1.0,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="BR-KTH-01", village="Katihar", block="Katihar Sadar", district="Katihar", state="Bihar",
         lat=25.5390, lon=87.5700, elevation_m=32, slope_deg=0.5, river_distance_km=0.6,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="UP-VNS-01", village="Varanasi", block="Varanasi Sadar", district="Varanasi", state="Uttar Pradesh",
         lat=25.3176, lon=82.9739, elevation_m=81, slope_deg=1.0, river_distance_km=0.3,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="UP-PRG-01", village="Prayagraj", block="Prayagraj Sadar", district="Prayagraj", state="Uttar Pradesh",
         lat=25.4358, lon=81.8463, elevation_m=98, slope_deg=0.8, river_distance_km=0.2,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="UK-NTL-01", village="Nainital", block="Nainital", district="Nainital", state="Uttarakhand",
         lat=29.3803, lon=79.4636, elevation_m=2084, slope_deg=30, river_distance_km=0.2,
         glacier_distance_km=60, flood_prone=True, landslide_prone=True),
    dict(id="UK-PTH-01", village="Pithoragarh", block="Pithoragarh", district="Pithoragarh", state="Uttarakhand",
         lat=29.5829, lon=80.2181, elevation_m=1645, slope_deg=28, river_distance_km=0.5,
         glacier_distance_km=35, flood_prone=True, landslide_prone=True),
    dict(id="UK-UTK-01", village="Uttarkashi", block="Uttarkashi", district="Uttarkashi", state="Uttarakhand",
         lat=30.7268, lon=78.4354, elevation_m=1352, slope_deg=31, river_distance_km=0.2,
         glacier_distance_km=25, flood_prone=True, landslide_prone=True),
    dict(id="HP-SML-01", village="Shimla", block="Shimla Rural", district="Shimla", state="Himachal Pradesh",
         lat=31.1048, lon=77.1734, elevation_m=2205, slope_deg=29, river_distance_km=1.5,
         glacier_distance_km=70, flood_prone=True, landslide_prone=True),
    dict(id="HP-MND-01", village="Mandi", block="Sadar", district="Mandi", state="Himachal Pradesh",
         lat=31.7084, lon=76.9319, elevation_m=760, slope_deg=25, river_distance_km=0.3,
         glacier_distance_km=55, flood_prone=True, landslide_prone=True),
    dict(id="JK-SRI-01", village="Srinagar", block="Srinagar", district="Srinagar", state="Jammu and Kashmir",
         lat=34.0837, lon=74.7973, elevation_m=1585, slope_deg=5, river_distance_km=0.4,
         glacier_distance_km=20, flood_prone=True, landslide_prone=False),
    dict(id="JK-ANT-01", village="Anantnag", block="Anantnag", district="Anantnag", state="Jammu and Kashmir",
         lat=33.7311, lon=75.1487, elevation_m=1620, slope_deg=18, river_distance_km=0.3,
         glacier_distance_km=15, flood_prone=True, landslide_prone=True),
    dict(id="AS-JOR-01", village="Jorhat", block="Jorhat", district="Jorhat", state="Assam",
         lat=26.7509, lon=94.2037, elevation_m=116, slope_deg=0.8, river_distance_km=0.5,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="AS-SIL-01", village="Silchar", block="Silchar", district="Cachar", state="Assam",
         lat=24.8333, lon=92.7789, elevation_m=22, slope_deg=0.6, river_distance_km=0.4,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="WB-DRJ-01", village="Darjeeling", block="Darjeeling Sadar", district="Darjeeling", state="West Bengal",
         lat=27.0410, lon=88.2663, elevation_m=2045, slope_deg=32, river_distance_km=0.6,
         glacier_distance_km=45, flood_prone=True, landslide_prone=True),
    dict(id="WB-JLP-01", village="Jalpaiguri", block="Jalpaiguri Sadar", district="Jalpaiguri", state="West Bengal",
         lat=26.5167, lon=88.7333, elevation_m=89, slope_deg=1.0, river_distance_km=0.4,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="KL-KTM-01", village="Kottayam", block="Kottayam", district="Kottayam", state="Kerala",
         lat=9.5916, lon=76.5222, elevation_m=3, slope_deg=1.5, river_distance_km=0.3,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="KL-KZK-01", village="Kozhikode", block="Kozhikode", district="Kozhikode", state="Kerala",
         lat=11.2588, lon=75.7804, elevation_m=1, slope_deg=1.0, river_distance_km=0.6,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="MH-PUN-01", village="Pune (Western Ghats)", block="Mulshi", district="Pune", state="Maharashtra",
         lat=18.5204, lon=73.4189, elevation_m=560, slope_deg=15, river_distance_km=0.8,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="MH-KLH-01", village="Kolhapur", block="Karvir", district="Kolhapur", state="Maharashtra",
         lat=16.7050, lon=74.2433, elevation_m=558, slope_deg=6, river_distance_km=0.3,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="KA-KDG-01", village="Madikeri", block="Madikeri", district="Kodagu", state="Karnataka",
         lat=12.4244, lon=75.7382, elevation_m=1525, slope_deg=26, river_distance_km=0.4,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="KA-CKM-01", village="Chikmagalur", block="Chikmagalur", district="Chikmagalur", state="Karnataka",
         lat=13.3161, lon=75.7720, elevation_m=1108, slope_deg=24, river_distance_km=0.5,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="GA-PNJ-01", village="Panaji", block="Tiswadi", district="North Goa", state="Goa",
         lat=15.4909, lon=73.8278, elevation_m=8, slope_deg=2.0, river_distance_km=0.2,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="SK-NCH-01", village="Namchi", block="Namchi", district="South Sikkim", state="Sikkim",
         lat=27.1667, lon=88.3667, elevation_m=1675, slope_deg=28, river_distance_km=0.5,
         glacier_distance_km=50, flood_prone=True, landslide_prone=True),
    dict(id="AR-ITN-01", village="Itanagar", block="Itanagar", district="Papum Pare", state="Arunachal Pradesh",
         lat=27.0844, lon=93.6053, elevation_m=440, slope_deg=20, river_distance_km=0.5,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="MN-IMP-01", village="Imphal", block="Imphal East", district="Imphal East", state="Manipur",
         lat=24.8170, lon=93.9368, elevation_m=786, slope_deg=3, river_distance_km=0.3,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="ML-SHL-01", village="Shillong", block="East Khasi Hills", district="East Khasi Hills", state="Meghalaya",
         lat=25.5788, lon=91.8933, elevation_m=1496, slope_deg=22, river_distance_km=0.6,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="ML-CRP-01", village="Cherrapunji", block="Sohra", district="East Khasi Hills", state="Meghalaya",
         lat=25.2702, lon=91.7323, elevation_m=1484, slope_deg=30, river_distance_km=0.4,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="MZ-AIZ-01", village="Aizawl", block="Aizawl", district="Aizawl", state="Mizoram",
         lat=23.7271, lon=92.7176, elevation_m=1132, slope_deg=33, river_distance_km=0.7,
         glacier_distance_km=None, flood_prone=False, landslide_prone=True),
    dict(id="NL-KOH-01", village="Kohima", block="Kohima", district="Kohima", state="Nagaland",
         lat=25.6751, lon=94.1086, elevation_m=1444, slope_deg=27, river_distance_km=0.8,
         glacier_distance_km=None, flood_prone=False, landslide_prone=True),
    dict(id="TR-AGT-01", village="Agartala", block="Agartala", district="West Tripura", state="Tripura",
         lat=23.8315, lon=91.2868, elevation_m=13, slope_deg=1.0, river_distance_km=0.4,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="OD-KRP-01", village="Koraput", block="Koraput", district="Koraput", state="Odisha",
         lat=18.8120, lon=82.7100, elevation_m=858, slope_deg=18, river_distance_km=0.6,
         glacier_distance_km=None, flood_prone=True, landslide_prone=True),
    dict(id="TG-BHD-01", village="Bhadrachalam", block="Bhadrachalam", district="Bhadradri Kothagudem", state="Telangana",
         lat=17.6688, lon=80.8933, elevation_m=56, slope_deg=1.2, river_distance_km=0.2,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="CG-JGD-01", village="Jagdalpur", block="Jagdalpur", district="Bastar", state="Chhattisgarh",
         lat=19.0748, lon=82.0230, elevation_m=553, slope_deg=10, river_distance_km=0.5,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="MP-CHN-01", village="Chhindwara", block="Chhindwara", district="Chhindwara", state="Madhya Pradesh",
         lat=22.0574, lon=78.9382, elevation_m=655, slope_deg=8, river_distance_km=0.7,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="PB-RPR-01", village="Rupnagar", block="Rupnagar", district="Rupnagar", state="Punjab",
         lat=30.9668, lon=76.5262, elevation_m=262, slope_deg=3, river_distance_km=0.3,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="WB-KOL-01", village="Kolkata", block="Kolkata", district="Kolkata", state="West Bengal",
         lat=22.5726, lon=88.3639, elevation_m=9, slope_deg=0.5, river_distance_km=0.3,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
    dict(id="WB-HLD-01", village="Haldia", block="Haldia", district="Purba Medinipur", state="West Bengal",
         lat=22.0667, lon=88.0698, elevation_m=4, slope_deg=0.3, river_distance_km=0.1,
         glacier_distance_km=None, flood_prone=True, landslide_prone=False),
]

LOCATIONS_BY_ID = {loc["id"]: loc for loc in LOCATIONS}


def all_states():
    return sorted(set(loc["state"] for loc in LOCATIONS))


def locations_in_state(state_name: str):
    s = state_name.strip().lower()
    return [loc for loc in LOCATIONS if loc["state"].lower() == s]


# ==============================================================================
# GLACIERS — curated from published glaciology sources (RGI/GLIMS-documented
# named glaciers). This is REAL, hand-curated reference data — glacier
# outlines don't move day-to-day, so unlike weather this doesn't need a live
# API; the full Randolph Glacier Inventory (~198,000 glaciers, NSIDC/GLIMS) is
# the authoritative free source but ships as multi-GB shapefiles, overkill
# for a hyper-local demo covering ~25 monitored points. Size class is based on
# each glacier's published length/area in the glaciology literature.
# ==============================================================================

GLACIERS = [
    dict(name="Siachen Glacier", state="Ladakh", lat=35.421, lon=77.109, length_km=76, size="LARGE"),
    dict(name="Gangotri Glacier", state="Uttarakhand", lat=30.993, lon=79.083, length_km=30, size="LARGE"),
    dict(name="Bara Shigri Glacier", state="Himachal Pradesh", lat=32.203, lon=77.607, length_km=28, size="LARGE"),
    dict(name="Drang-Drung Glacier", state="Ladakh", lat=33.980, lon=76.350, length_km=23, size="LARGE"),
    dict(name="Zemu Glacier", state="Sikkim", lat=27.750, lon=88.150, length_km=26, size="MEDIUM"),
    dict(name="East Rathong Glacier", state="Sikkim", lat=27.600, lon=88.150, length_km=8, size="SMALL"),
    dict(name="Milam Glacier", state="Uttarakhand", lat=30.420, lon=80.050, length_km=16, size="MEDIUM"),
    dict(name="Chhota Shigri Glacier", state="Himachal Pradesh", lat=32.280, lon=77.580, length_km=9, size="SMALL"),
    dict(name="Kolahoi Glacier", state="Jammu and Kashmir", lat=34.200, lon=75.300, length_km=4, size="SMALL"),
    dict(name="Pindari Glacier", state="Uttarakhand", lat=30.280, lon=80.020, length_km=3, size="SMALL"),
    dict(name="Chorabari Glacier", state="Uttarakhand", lat=30.750, lon=79.070, length_km=3, size="SMALL"),
]


def nearest_glacier(lat: float, lon: float):
    """Haversine to every curated glacier, returns the closest one + distance."""
    best, best_d = None, float("inf")
    for g in GLACIERS:
        R = 6371
        p1, p2 = math.radians(lat), math.radians(g["lat"])
        dphi, dlmb = math.radians(g["lat"] - lat), math.radians(g["lon"] - lon)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        d = 2 * R * math.asin(math.sqrt(a))
        if d < best_d:
            best, best_d = g, d
    return {**best, "distance_km": round(best_d, 1)} if best else None
