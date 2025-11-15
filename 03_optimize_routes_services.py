"""
03_optimize_routes_services.py
--------------------------------
Solve a maximum-coverage style MILP to pick bus routes and service locations.

Inputs (CSV):
  - data/outputs/W.csv                columns: GEOID,route_id,min_dist_m
  - data/outputs/T.csv                columns: GEOID,service_idx
  - data/outputs/milwaukee_services_clean.csv  (from step 01)
  - data/outputs/route_neighbors.csv (optional) columns: route_id,neighbor_id
  - (optional) data/block_population.csv columns: GEOID,pop

Outputs:
  - data/outputs/selected_routes.csv      route_id
  - data/outputs/selected_services.csv    service_idx,name,lat,lon
  - data/outputs/block_assignments.csv    GEOID,route_id
  - data/outputs/solution_summary.txt

Usage:
  python 03_optimize_routes_services.py \
      --W data/outputs/W.csv \
      --T data/outputs/T.csv \
      --services data/outputs/milwaukee_services_clean.csv \
      --neighbors data/outputs/route_neighbors.csv \
      --pop data/block_population.csv \
      --R_max 15 --S_max 60 \
      --alpha 1.0 --beta 0.7 --lam 0.001 --mu 0.05 \
      --single_assignment
"""

import os
import argparse
import pandas as pd
from collections import defaultdict

# ---- MILP (PuLP) ----
import pulp


def load_data(args):
    W = pd.read_csv(args.W, dtype={"GEOID": str, "route_id": str})
    if "min_dist_m" not in W.columns:
        raise ValueError("W.csv must contain column 'min_dist_m'.")

    T = pd.read_csv(args.T, dtype={"GEOID": str, "service_idx": int})
    services = pd.read_csv(args.services)
    services = services.reset_index(drop=True)  # service_idx matches this index

    # population (optional): default to 1 if not provided
    if args.pop and os.path.exists(args.pop):
        pop_df = pd.read_csv(args.pop, dtype={"GEOID": str})
        pop_map = dict(zip(pop_df["GEOID"], pop_df["pop"]))
    else:
        # fallback pop=1
        pop_map = defaultdict(lambda: 1)

    # neighbors (optional)
    neighbors = defaultdict(set)
    if args.neighbors and os.path.exists(args.neighbors):
        nbr = pd.read_csv(args.neighbors, dtype={"route_id": str, "neighbor_id": str})
        for _, row in nbr.iterrows():
            neighbors[row["route_id"]].add(row["neighbor_id"])

    return W, T, services, pop_map, neighbors


def solve_milp(W, T, services, pop_map, neighbors, args):
    # Sets
    B = sorted(set(W["GEOID"]) | set(T["GEOID"]))
    R = sorted(W["route_id"].unique())
    S = list(range(len(services)))  # service_idx aligns to cleaned file row index

    # Fast lookups
    # Walk pairs with distance
    W_pairs = [(row.GEOID, row.route_id) for _, row in W.iterrows()]
    d_map = {(row.GEOID, row.route_id): float(row.min_dist_m) for _, row in W.iterrows()}
    # T pairs
    T_pairs = [(row.GEOID, int(row.service_idx)) for _, row in T.iterrows()]

    # Build index helpers
    routes_by_b = defaultdict(list)
    for b, r in W_pairs:
        routes_by_b[b].append(r)

    services_by_b = defaultdict(list)
    for b, s in T_pairs:
        services_by_b[b].append(s)

    # ------------------ Model ------------------
    m = pulp.LpProblem("TransitCoverageMILP", pulp.LpMaximize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", R, lowBound=0, upBound=1, cat="Binary")            # select route
    y = pulp.LpVariable.dicts("y", S, lowBound=0, upBound=1, cat="Binary")            # select service
    z = pulp.LpVariable.dicts("z", W_pairs, lowBound=0, upBound=1, cat="Binary")      # assign b->r (walkable)
    u = pulp.LpVariable.dicts("u", B, lowBound=0, upBound=1, cat="Binary")            # block covered by any route
    v = pulp.LpVariable.dicts("v", B, lowBound=0, upBound=1, cat="Binary")            # block has any reachable service

    # Linearize service access v_b via w_{b,s}
    w = pulp.LpVariable.dicts("w", T_pairs, lowBound=0, upBound=1, cat="Binary")      # if b uses service s

    # Optional connectivity bonus variables q_r (if neighbors provided)
    use_connectivity = len(neighbors) > 0 and args.mu > 0
    if use_connectivity:
        q = pulp.LpVariable.dicts("q", R, lowBound=0, upBound=1, cat="Binary")
    else:
        q = {}

    # ------------------ Constraints ------------------

    # Link z to x (you can only assign b to r if route r is selected)
    for (b, r) in W_pairs:
        m += z[(b, r)] <= x[r]

    # Block coverage u_b if any z_{b,r} is 1
    for b in B:
        if routes_by_b[b]:
            m += u[b] <= pulp.lpSum(z[(b, r)] for r in routes_by_b[b])
        else:
            m += u[b] == 0  # no walkable route known

    # Single assignment (optional): at most one assigned route per block
    if args.single_assignment:
        for b in B:
            if routes_by_b[b]:
                m += pulp.lpSum(z[(b, r)] for r in routes_by_b[b]) <= 1

    # Service linking: w_{b,s} <= y_s
    for (b, s) in T_pairs:
        m += w[(b, s)] <= y[s]

    # v_b <= sum_s w_{b,s}
    for b in B:
        if services_by_b[b]:
            m += v[b] <= pulp.lpSum(w[(b, s)] for s in services_by_b[b])
        else:
            m += v[b] == 0

    # Budgets
    m += pulp.lpSum(x[r] for r in R) <= args.R_max
    m += pulp.lpSum(y[s] for s in S) <= args.S_max

    # Connectivity (optional): q_r <= x_r and q_r <= sum neighbors selected
    if use_connectivity:
        for r in R:
            m += q[r] <= x[r]
            if len(neighbors[r]) > 0:
                m += q[r] <= pulp.lpSum(x[rp] for rp in neighbors[r])
            else:
                m += q[r] <= 0  # no neighbors → can't be connected

    # ------------------ Objective ------------------
    # Maximize: sum_b pop_b (alpha * u_b + beta * v_b)
    #           - lam * sum_{b,r} (d_br * z_{b,r})
    #           + mu * sum_r q_r   (if connectivity enabled)
    alpha, beta, lam, mu = args.alpha, args.beta, args.lam, (args.mu if use_connectivity else 0.0)
    # population map fallback
    def pop(b):
        return float(pop_map[b]) if b in pop_map else 1.0

    obj_coverage = pulp.lpSum(pop(b) * (alpha * u[b] + beta * v[b]) for b in B)
    obj_walkpen  = pulp.lpSum(lam * d_map[(b, r)] * z[(b, r)] for (b, r) in W_pairs)
    obj_connect  = pulp.lpSum(mu * q[r] for r in R) if use_connectivity else 0

    m += obj_coverage - obj_walkpen + obj_connect

    # ------------------ Solve ------------------
    if args.solver == "gurobi":
        try:
            solver = pulp.GUROBI(timeLimit=args.timelimit, msg=True)
        except Exception:
            print("Gurobi not available in PuLP; falling back to CBC.")
            solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=args.timelimit)
    else:
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=args.timelimit)

    m.solve(solver)

    status = pulp.LpStatus[m.status]
    print(f"Solver status: {status}; Objective = {pulp.value(m.objective):.3f}")

    # ------------------ Extract solution ------------------
    sel_routes = [r for r in R if pulp.value(x[r]) > 0.5]
    sel_services = [s for s in S if pulp.value(y[s]) > 0.5]
    assignments = [(b, r) for (b, r) in W_pairs if pulp.value(z[(b, r)]) > 0.5]

    return status, sel_routes, sel_services, assignments


def write_outputs(status, sel_routes, sel_services, assignments, services, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({"route_id": sel_routes}).to_csv(
        os.path.join(out_dir, "selected_routes.csv"), index=False
    )

    svc_out = services.loc[sel_services, ["name", "lat", "lon"]].copy()
    svc_out.insert(0, "service_idx", sel_services)
    svc_out.to_csv(os.path.join(out_dir, "selected_services.csv"), index=False)

    pd.DataFrame(assignments, columns=["GEOID", "route_id"]).to_csv(
        os.path.join(out_dir, "block_assignments.csv"), index=False
    )

    with open(os.path.join(out_dir, "solution_summary.txt"), "w") as f:
        f.write(f"Status: {status}\n")
        f.write(f"Selected routes: {len(sel_routes)}\n")
        f.write(f"Selected services: {len(sel_services)}\n")
        f.write(f"Assigned blocks: {len(assignments)}\n")

    print(f"✅ Wrote outputs to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Optimize bus routes & services (MILP).")
    p.add_argument("--W", default="outputs/W.csv")
    p.add_argument("--T", default="outputs/T.csv")
    p.add_argument("--services", default="outputs/milwaukee_yelp_services_clean.csv")
    p.add_argument("--neighbors", default="outputs/route_neighbors.csv")
    p.add_argument("--pop", default="")  # optional GEOID,pop CSV

    p.add_argument("--R_max", type=int, default=15)
    p.add_argument("--S_max", type=int, default=60)

    p.add_argument("--alpha", type=float, default=1.0)   # weight for route coverage
    p.add_argument("--beta", type=float, default=0.7)    # weight for service access
    p.add_argument("--lam", type=float, default=0.001)   # penalty per meter of walking
    p.add_argument("--mu", type=float, default=0.05)     # connectivity bonus (needs neighbors)

    p.add_argument("--single_assignment", action="store_true", help="At most one route per block")
    p.add_argument("--solver", choices=["cbc", "gurobi"], default="cbc")
    p.add_argument("--timelimit", type=int, default=600)
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    W, T, services, pop_map, neighbors = load_data(args)
    status, sel_routes, sel_services, assignments = solve_milp(
        W, T, services, pop_map, neighbors, args
    )
    write_outputs(status, sel_routes, sel_services, assignments, services, args.out_dir)
