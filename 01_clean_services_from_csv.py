"""
01_clean_services_from_csv.py
Cleans and deduplicates services from a CSV file.

This script expects a CSV file with service/business data (e.g., from Yelp, Google Places, etc.)
that may contain duplicate entries for the same business. It deduplicates by name and location,
keeping the best rating when multiple entries exist for the same business.

Expected input CSV columns:
    - GEOID: Geographic identifier (string)
    - name: Business/service name (string)
    - rating: Rating value (float)
    - lat: Latitude (float)
    - lon: Longitude (float)
    - category: Service category (string)

Output: Cleaned CSV with one row per unique business (deduplicated by name and location)
"""

import argparse
import os
import pandas as pd
import geopandas as gpd


def clean_services(input_file, output_file):
    """
    Clean and deduplicate services from a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    # read and clean the data
    df = pd.read_csv(input_file, on_bad_lines="skip", dtype={
        "GEOID": str, "name": str, "rating": float, "lat": float, "lon": float, "category": str
    }).dropna(subset=["GEOID", "name", "lat", "lon"])
    
    for c in ["GEOID", "name", "category"]:
        df[c] = df[c].astype(str).str.strip()
    
    # Deduplicate by name/location. Multiple entries may exist for the same business
    # (e.g., from different data sources, different time periods, or data collection errors).
    # When duplicates exist, we keep the maximum rating to ensure we retain the best
    # representation of the business quality.
    agg = (df.groupby(["name", "lat", "lon"], as_index=False)
             .agg({
                 "rating": "max",
                 "GEOID":  lambda s: ",".join(sorted(set(map(str, s)))),
                 "category": lambda s: ",".join(sorted(set([x for x in s if isinstance(x, str) and x]))),
             }))
    
    agg.rename(columns={"GEOID": "geoids_joined", "category": "categories"}, inplace=True)
    
    # Convert to geodataframe using WGS84 coordinate reference system (EPSG:4326).
    # EPSG:4326 is the standard latitude/longitude coordinate system used by GPS
    # and most mapping applications (e.g., Google Maps, OpenStreetMap).
    g = gpd.GeoDataFrame(
        agg,
        geometry=gpd.points_from_xy(agg["lon"], agg["lat"]),
        crs="EPSG:4326"
    )
    
    g2 = pd.DataFrame(g.drop(columns="geometry"))
    g2.to_csv(output_file, index=False)
    print(f"Cleaned {len(g2)} unique businesses to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean and deduplicate services from a CSV file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.getenv("SERVICES_RAW", "services.csv"),
        help="Input CSV file path (default: SERVICES_RAW env var or 'services.csv')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: outputs/<input_basename>_clean.csv)"
    )
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if args.output is None:
        input_basename = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join("outputs", f"{input_basename}_clean.csv")
    
    clean_services(args.input, args.output)


if __name__ == "__main__":
    main()
