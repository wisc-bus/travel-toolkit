"""
03_optimize_routes_services.py
Solves MILP to pick bus routes and service locations.
Works with any dataset. Currently using Milwaukee as example.
"""

import os
import argparse
import pandas as pd
from collections import defaultdict
import pulp


def load_data(args):
    W = pd.read_csv(args.W, dtype={"GEOID": str, "route_id": str})
    if "min_dist_m" not in W.columns:
        raise ValueError("W.csv must contain column 'min_dist_m'.")

    T = pd.read_csv(args.T, dtype={"GEOID": str, "service_idx": int})
    services = pd.read_csv(args.services)
    services = services.reset_index(drop=True)  # service_idx matches this index

    # population defaults to 1 if not provided
    if args.pop and os.path.exists(args.pop):
        pop_df = pd.read_csv(args.pop, dtype={"GEOID": str})
        pop_map = dict(zip(pop_df["GEOID"], pop_df["pop"]))
    else:
        pop_map = defaultdict(lambda: 1)

    # load neighbors 
    neighbors = defaultdict(set)
    if args.neighbors and os.path.exists(args.neighbors):
        nbr = pd.read_csv(args.neighbors, dtype={"route_id": str, "neighbor_id": str})
        for _, row in nbr.iterrows():
            neighbors[row["route_id"]].add(row["neighbor_id"])

    return W, T, services, pop_map, neighbors


def solve_milp(W, T, services, pop_map, neighbors, args):
    B = sorted(set(W["GEOID"]) | set(T["GEOID"]))
    R = sorted(W["route_id"].unique())
    S = list(range(len(services)))

    W_pairs = [(row.GEOID, row.route_id) for _, row in W.iterrows()]
    d_map = {(row.GEOID, row.route_id): float(row.min_dist_m) for _, row in W.iterrows()}
    T_pairs = [(row.GEOID, int(row.service_idx)) for _, row in T.iterrows()]
    routes_by_b = defaultdict(list)
    for b, r in W_pairs:
        routes_by_b[b].append(r)

    services_by_b = defaultdict(list)
    for b, s in T_pairs:
        services_by_b[b].append(s)

    m = pulp.LpProblem("TransitCoverageMILP", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", R, lowBound=0, upBound=1, cat="Binary")
    y = pulp.LpVariable.dicts("y", S, lowBound=0, upBound=1, cat="Binary")
    z = pulp.LpVariable.dicts("z", W_pairs, lowBound=0, upBound=1, cat="Binary")
    u = pulp.LpVariable.dicts("u", B, lowBound=0, upBound=1, cat="Binary")
    v = pulp.LpVariable.dicts("v", B, lowBound=0, upBound=1, cat="Binary")
    w = pulp.LpVariable.dicts("w", T_pairs, lowBound=0, upBound=1, cat="Binary")

    use_connectivity = len(neighbors) > 0 and args.mu > 0
    if use_connectivity:
        q = pulp.LpVariable.dicts("q", R, lowBound=0, upBound=1, cat="Binary")
    else:
        q = {}

    # constraints
    for (b, r) in W_pairs:
        m += z[(b, r)] <= x[r]

    for b in B:
        if routes_by_b[b]:
            m += u[b] <= pulp.lpSum(z[(b, r)] for r in routes_by_b[b])
        else:
            m += u[b] == 0

    if args.single_assignment:
        for b in B:
            if routes_by_b[b]:
                m += pulp.lpSum(z[(b, r)] for r in routes_by_b[b]) <= 1

    for (b, s) in T_pairs:
        m += w[(b, s)] <= y[s]

    for b in B:
        if services_by_b[b]:
            m += v[b] <= pulp.lpSum(w[(b, s)] for s in services_by_b[b])
        else:
            m += v[b] == 0

    m += pulp.lpSum(x[r] for r in R) <= args.R_max
    m += pulp.lpSum(y[s] for s in S) <= args.S_max

    if use_connectivity:
        for r in R:
            m += q[r] <= x[r]
            if len(neighbors[r]) > 0:
                m += q[r] <= pulp.lpSum(x[rp] for rp in neighbors[r])
            else:
                m += q[r] <= 0

    # objective
    alpha, beta, lam, mu = args.alpha, args.beta, args.lam, (args.mu if use_connectivity else 0.0)
    def pop(b):
        return float(pop_map[b]) if b in pop_map else 1.0

    obj_coverage = pulp.lpSum(pop(b) * (alpha * u[b] + beta * v[b]) for b in B)
    obj_walkpen  = pulp.lpSum(lam * d_map[(b, r)] * z[(b, r)] for (b, r) in W_pairs)
    obj_connect  = pulp.lpSum(mu * q[r] for r in R) if use_connectivity else 0

    m += obj_coverage - obj_walkpen + obj_connect

    # solve
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

    print(f"Wrote outputs to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Optimize bus routes & services (MILP).")
    p.add_argument("--W", default="outputs/W.csv")
    p.add_argument("--T", default="outputs/T.csv")
    # using milwaukee as example, change for other datasets
    p.add_argument("--services", default="outputs/milwaukee_yelp_services_clean.csv")
    p.add_argument("--neighbors", default="outputs/route_neighbors.csv")
    p.add_argument("--pop", default="")

    p.add_argument("--R_max", type=int, default=15)
    p.add_argument("--S_max", type=int, default=60)

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.7)
    p.add_argument("--lam", type=float, default=0.001)
    p.add_argument("--mu", type=float, default=0.05)

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
