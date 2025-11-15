# 04_postsolve_metrics.py
import os, pandas as pd, geopandas as gpd

OUT_DIR = "outputs"
BG_SHP  = "tl_2024_55_bg/tl_2024_55_bg.shp"
SERV    = os.path.join(OUT_DIR, "milwaukee_yelp_services_clean.csv")
W_CSV   = os.path.join(OUT_DIR, "W.csv")
T_CSV   = os.path.join(OUT_DIR, "T.csv")
SEL_R   = os.path.join(OUT_DIR, "selected_routes.csv")
SEL_S   = os.path.join(OUT_DIR, "selected_services.csv")
ASSIGN  = os.path.join(OUT_DIR, "block_assignments.csv")

# Load
blocks = gpd.read_file(BG_SHP).to_crs(epsg=4326)[["GEOID","geometry"]]
W      = pd.read_csv(W_CSV, dtype={"GEOID":str})
T      = pd.read_csv(T_CSV, dtype={"GEOID":str})
sel_r  = pd.read_csv(SEL_R, dtype=str)["route_id"].tolist()
sel_s  = pd.read_csv(SEL_S)
svc_all= pd.read_csv(SERV)

# KPIs
n_blocks_walk = W["GEOID"].nunique()
n_routes_all  = W["route_id"].nunique()
n_blocks_T    = T["GEOID"].nunique()

kpis = {
  "blocks_with_walk_access": n_blocks_walk,
  "routes_in_W": n_routes_all,
  "blocks_with_service_reachability": n_blocks_T,
  "selected_routes": len(sel_r),
  "selected_services": len(sel_s),
}

# Avg min walk among assigned blocks
assign = pd.read_csv(ASSIGN, dtype={"GEOID":str})
Wmin   = (W.merge(assign, on=["GEOID","route_id"], how="inner")
            .groupby("GEOID")["min_dist_m"].min().reset_index())
kpis["assigned_blocks"] = assign["GEOID"].nunique()
kpis["avg_min_walk_m"]  = float(Wmin["min_dist_m"].mean()) if len(Wmin) else None
kpis["median_min_walk_m"]= float(Wmin["min_dist_m"].median()) if len(Wmin) else None

# Category mix of selected services (if categories present)
svc_sel = svc_all.reset_index().rename(columns={"index":"service_idx"})
svc_sel = svc_sel.merge(sel_s[["service_idx"]], on="service_idx", how="inner")
if "categories" in svc_sel.columns:
    cats = {}
    for row in svc_sel["categories"].fillna(""):
        for c in [x.strip() for x in row.split(",") if x.strip()]:
            cats[c] = cats.get(c, 0) + 1
    cat_df = pd.DataFrame(sorted(cats.items(), key=lambda x: -x[1]), columns=["category","count"])
    cat_df.to_csv(os.path.join(OUT_DIR, "selected_services_category_mix.csv"), index=False)

# Save KPI summary
with open(os.path.join(OUT_DIR, "solution_summary_readable.txt"), "w") as f:
    for k,v in kpis.items(): f.write(f"{k}: {v}\n")
print("KPIs:", kpis)

# Geo exports for quick mapping
assign_g = blocks.merge(assign, on="GEOID", how="left")
assign_g.to_file(os.path.join(OUT_DIR, "assigned_blocks.geojson"), driver="GeoJSON")

svc_geo = gpd.GeoDataFrame(
    svc_sel,
    geometry=gpd.points_from_xy(svc_sel["lon"], svc_sel["lat"]),
    crs="EPSG:4326"
)
svc_geo.to_file(os.path.join(OUT_DIR, "selected_services.geojson"), driver="GeoJSON")
print("Wrote GeoJSONs to outputs/")
