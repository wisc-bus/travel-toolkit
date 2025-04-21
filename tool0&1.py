import geopandas as gpd
import requests
import pandas as pd
import time
import os

# Load shapefile 
block_groups = gpd.read_file("tl_2024_55_bg/tl_2024_55_bg.shp").to_crs(epsg=4326)

# Reproject to a local projected CRS 
block_groups_proj = block_groups.to_crs(epsg=3071)

# Compute centroids 
block_groups_proj['centroid'] = block_groups_proj.geometry.centroid

# transform centroids back to WGS84 for yelp
centroids_wgs84 = block_groups_proj.set_geometry('centroid').to_crs(epsg=4326)
block_groups['lat'] = centroids_wgs84.geometry.y
block_groups['lon'] = centroids_wgs84.geometry.x

# Yelp API 
YELP_API_KEY = "bDAon8ZKyViqMGdQp07QOu9X9trtnPOcJGpUxX02u1proTwYqM6DmJqOCzIrHac1e2A-WhrYHfuMQJ3fX7tNpY4icycxFX_VJEYqvCuZm4ptZdhGlYaPpKlXXLH8Z3Yx"
headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
yelp_url = "https://api.yelp.com/v3/businesses/search"

yelp_categories = [
    "grocery",
    "restaurant",
    "pharmacy",
    "clinic",
    "hospital",
    "school",
    "gasstation",
    "bank",
    "hair",
    "gym",
    "coffee",
    "daycare",
    "laundry"
]
# Tool 1 (get services per block centroid) 
services = []

N = 50  # Number of block groups sampled

for idx, row in block_groups.iterrows():
    for category in yelp_categories:
        print(f"🔍 GEOID {row['GEOID']} — {category}")
        
        params = {
            "term": category,
            "latitude": row["lat"],
            "longitude": row["lon"],
            "radius": 1000,
            "limit": 5
        }
        try:
            response = requests.get(yelp_url, headers=headers, params=params)
            if response.status_code == 200:
                businesses = response.json().get("businesses", [])
                for b in businesses:
                    services.append({
                        "GEOID": row["GEOID"],
                        "category": category,
                        "name": b["name"],
                        "rating": b["rating"],
                        "lat": b["coordinates"]["latitude"],
                        "lon": b["coordinates"]["longitude"]
                    })
            else:
                print(f"Yelp error {response.status_code} for GEOID {row['GEOID']}")
        except Exception as e:
            print(f"Request failed for GEOID {row['GEOID']}: {e}")

        time.sleep(0.5) 

    if idx >= N - 1:
        break

# Save to csv
services_df = pd.DataFrame(services)
services_df.to_csv("milwaukee_multiservice_yelp.csv", index=False)
print(f"Done. Saved {len(services_df)} results to milwaukee_multiservice_yelp.csv.")
