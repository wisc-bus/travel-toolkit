"""
01_clean_services_from_csv.py
Cleans and deduplicates services from a CSV file. Works with any dataset.
Currently using Milwaukee as an example.

Input: CSV with columns GEOID, name, rating, lat, lon, category
Output: Cleaned CSV with one row per unique business
"""

import os
import pandas as pd
import geopandas as gpd

# using milwaukee as example, change these for other datasets
RAW_IN   = os.getenv("SERVICES_RAW", "milwaukee_yelp_services.csv")
OUT_DIR  = "outputs"
OUT_FILE = os.path.join(OUT_DIR, "milwaukee_yelp_services_clean.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# read and clean the data
df = pd.read_csv(RAW_IN, on_bad_lines="skip", dtype={
    "GEOID": str, "name": str, "rating": float, "lat": float, "lon": float, "category": str
}).dropna(subset=["GEOID", "name", "lat", "lon"])

for c in ["GEOID", "name", "category"]:
    df[c] = df[c].astype(str).str.strip()

# deduplicate by name/location, keep best rating
agg = (df.groupby(["name", "lat", "lon"], as_index=False)
         .agg({
             "rating": "max",
             "GEOID":  lambda s: ",".join(sorted(set(map(str, s)))),
             "category": lambda s: ",".join(sorted(set([x for x in s if isinstance(x, str) and x]))),
         }))

agg.rename(columns={"GEOID": "geoids_joined", "category": "categories"}, inplace=True)

# convert to geodataframe and save
g = gpd.GeoDataFrame(
    agg,
    geometry=gpd.points_from_xy(agg["lon"], agg["lat"]),
    crs="EPSG:4326"
)

g2 = pd.DataFrame(g.drop(columns="geometry"))
g2.to_csv(OUT_FILE, index=False)
print(f"Cleaned {len(g2)} unique businesses to {OUT_FILE}")
