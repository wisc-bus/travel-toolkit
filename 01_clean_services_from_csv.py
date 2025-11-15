"""
01_clean_services_from_csv.py
Clean + dedupe Milwaukee services already pulled from Yelp.

Input:
  data/milwaukee_services_raw.csv  (columns: GEOID,name,rating,lat,lon,category)

Outputs:
  data/outputs/milwaukee_services_clean.csv  (GeoCSV; one row per unique business)
"""

import os
import pandas as pd
import geopandas as gpd

RAW_IN   = os.getenv("SERVICES_RAW", "milwaukee_yelp_services.csv")
OUT_DIR  = "outputs"
OUT_FILE = os.path.join(OUT_DIR, "milwaukee_yelp_services_clean.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Read & sanitize
df = pd.read_csv(RAW_IN, on_bad_lines="skip", dtype={
    "GEOID": str, "name": str, "rating": float, "lat": float, "lon": float, "category": str
}).dropna(subset=["GEOID", "name", "lat", "lon"])

for c in ["GEOID", "name", "category"]:
    df[c] = df[c].astype(str).str.strip()

# 2) Deduplicate by (name, lat, lon); keep best rating, join categories & GEOIDs
agg = (df.groupby(["name", "lat", "lon"], as_index=False)
         .agg({
             "rating": "max",
             "GEOID":  lambda s: ",".join(sorted(set(map(str, s)))),
             "category": lambda s: ",".join(sorted(set([x for x in s if isinstance(x, str) and x]))),
         }))

agg.rename(columns={"GEOID": "geoids_joined", "category": "categories"}, inplace=True)

# 3) GeoDataFrame (WGS84); save as plain CSV (coords preserved)
g = gpd.GeoDataFrame(
    agg,
    geometry=gpd.points_from_xy(agg["lon"], agg["lat"]),
    crs="EPSG:4326"
)

# Optional: drop geometry column before CSV, or keep lat/lon; here we keep lat/lon and drop WKT
g2 = pd.DataFrame(g.drop(columns="geometry"))

g2.to_csv(OUT_FILE, index=False)
print(f"✅ Cleaned {len(g2)} unique businesses → {OUT_FILE}")
