"""
02_build_W_T_from_gtfs.py
Builds optimizer inputs from block groups, GTFS, and cleaned services.
Works with any dataset. Currently using Milwaukee/Racine as examples.

Inputs: block groups shapefile, GTFS directory, cleaned services CSV
Outputs: W.csv, T.csv, route_neighbors.csv
"""

import os
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from collections import deque

# config - change these for different datasets
BG_SHP   = os.getenv("BG_SHP", "tl_2024_55_bg/tl_2024_55_bg.shp")
GTFS_DIR = os.getenv("GTFS_DIR", "./racine_gtfs/")
SERV_CSV = os.getenv("SERVICES_CLEAN", "outputs/milwaukee_yelp_services_clean.csv")
OUT_DIR  = "outputs"

# distances in meters
D_WALK   = int(os.getenv("D_WALK", "800"))
D_SVC    = int(os.getenv("D_SVC",  "200"))
D_XFER   = int(os.getenv("D_XFER", "100"))
MAX_HOPS = int(os.getenv("MAX_HOPS","2"))

# local coordinate system - adjust EPSG for your region
LOCAL_EPSG = int(os.getenv("LOCAL_EPSG","3071"))

os.makedirs(OUT_DIR, exist_ok=True)

# load block groups
blocks = gpd.read_file(BG_SHP).to_crs(epsg=LOCAL_EPSG)
blocks["rep_pt"] = blocks.geometry.representative_point()
g_blocks = gpd.GeoDataFrame(blocks[["GEOID", "rep_pt"]], geometry="rep_pt", crs=blocks.crs)

# load services
services = pd.read_csv(SERV_CSV, dtype={"name":str, "lat":float, "lon":float})
g_services = gpd.GeoDataFrame(
    services,
    geometry=gpd.points_from_xy(services["lon"], services["lat"]),
    crs="EPSG:4326"
).to_crs(epsg=LOCAL_EPSG)
g_services = g_services.reset_index().rename(columns={"index": "service_idx"})

# load GTFS files
routes = pd.read_csv(os.path.join(GTFS_DIR, "routes.txt"), dtype=str)
trips = pd.read_csv(os.path.join(GTFS_DIR, "trips.txt"), dtype=str)
stop_times = pd.read_csv(os.path.join(GTFS_DIR, "stop_times.txt"), dtype=str)
stops = pd.read_csv(os.path.join(GTFS_DIR, "stops.txt"),
                    dtype={"stop_id":str, "stop_lat":float, "stop_lon":float})

# map routes to stops
trip2route = trips.set_index("trip_id")["route_id"].to_dict()
stop_times["route_id"] = stop_times["trip_id"].map(trip2route)
route_stops = (stop_times.groupby("route_id")["stop_id"]
                         .apply(lambda s: set(s.dropna().astype(str))).to_dict())

g_stops = gpd.GeoDataFrame(
    stops.assign(stop_id=stops["stop_id"].astype(str)),
    geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
    crs="EPSG:4326"
).to_crs(epsg=LOCAL_EPSG)

# find route neighbors via transfer clusters
buf = g_stops.copy()
buf["geometry"] = buf.buffer(D_XFER)
clusters = buf.dissolve().explode(index_parts=False).reset_index(drop=True)
c_sidx = clusters.sindex

stop2cluster = {}
for i, row in g_stops.iterrows():
    hits = list(c_sidx.intersection(row.geometry.bounds))
    cid = None
    for h in hits:
        if clusters.loc[h, "geometry"].contains(row.geometry):
            cid = h
            break
    stop2cluster[row["stop_id"]] = cid

route_clusters = {r: {stop2cluster.get(s) for s in st if stop2cluster.get(s) is not None}
                  for r, st in route_stops.items()}

# routes are neighbors if they share any cluster
neighbors = {r: set() for r in route_clusters}
routes_list = list(route_clusters.keys())
for i in range(len(routes_list)):
    r1 = routes_list[i]
    for j in range(i+1, len(routes_list)):
        r2 = routes_list[j]
        if len(route_clusters[r1].intersection(route_clusters[r2])) > 0:
            neighbors[r1].add(r2); neighbors[r2].add(r1)

# save neighbor list
edges = []
for r, nbrs in neighbors.items():
    for n in nbrs:
        edges.append({"route_id": r, "neighbor_id": n})
pd.DataFrame(edges).to_csv(os.path.join(OUT_DIR, "route_neighbors.csv"), index=False)

# build W: walkable blocks to routes
sidx = g_stops.sindex
stop_geom = g_stops.set_index("stop_id")["geometry"].to_dict()
route_stop_points = {r: [stop_geom[s] for s in st if s in stop_geom] for r, st in route_stops.items()}

W_rows = []
for row in g_blocks.itertuples(index=False):
    b_geo = row.rep_pt
    b_id  = row.GEOID

    near_idx = list(sidx.query(b_geo.buffer(D_WALK)))
    cand_stop_ids = set(g_stops.iloc[near_idx]["stop_id"].astype(str))

    for r, r_stops in route_stops.items():
        if not (cand_stop_ids & r_stops):
            continue
        dmin = min(b_geo.distance(pt) for pt in route_stop_points[r]) if route_stop_points[r] else float("inf")
        if dmin <= D_WALK:
            W_rows.append({"GEOID": b_id, "route_id": r, "min_dist_m": float(dmin)})

W_df = pd.DataFrame(W_rows)
W_out = os.path.join(OUT_DIR, "W.csv")
W_df.to_csv(W_out, index=False)
print(f"W: {len(W_df)} rows to {W_out}")

# build T: blocks to services
st_sidx = g_stops.sindex
svc_to_routes = []
for srow in g_services.itertuples(index=True):
    spt = srow.geometry
    idxs = list(st_sidx.query(spt.buffer(D_SVC)))
    near_stop_ids = set(g_stops.iloc[idxs]["stop_id"].astype(str))
    r_near = [r for r, st in route_stops.items() if near_stop_ids & st]
    svc_to_routes.append((srow.Index, set(r_near)))

routes_by_block = {}
for row in W_df.itertuples(index=False):
    routes_by_block.setdefault(row.GEOID, set()).add(row.route_id)

def closure_within_hops(start_routes, max_hops):
    seen = set(start_routes)
    q = deque([(r, 0) for r in start_routes])
    while q:
        r, d = q.popleft()
        if d == max_hops:
            continue
        for nxt in neighbors.get(r, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return seen

T_rows = []
for b, seed in routes_by_block.items():
    if not seed:
        continue
    reachable = set()
    for r in seed:
        reachable |= closure_within_hops([r], MAX_HOPS)
    for si, rset in svc_to_routes:
        if rset & reachable:
            T_rows.append({"GEOID": b, "service_idx": int(si)})

T_df = pd.DataFrame(T_rows)
T_out = os.path.join(OUT_DIR, "T.csv")
T_df.to_csv(T_out, index=False)
print(f"T: {len(T_df)} rows to {T_out}")


print("All done.")
