#!/usr/bin/env python3
"""
CHI2 Exp 06 -- robustness battery for the depth gradient (audited outcome).

  1. leave-one-database-out Cochran-Armitage trend z
  2. percentile bootstrap CIs for per-stratum error rates (10,000 resamples)
  3. AIC race among single-structural-predictor logistic models
  4. depth x question-length interaction test

Outcome: SQL error in the research-audit round (134/498), matching the
logistic analyses in the paper. Usage:
  python3 06_robustness.py
"""
import importlib.util, json, math, os

import numpy as np
import sqlglot

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("psim", os.path.join(HERE, "04_policy_sim.py"))
psim = importlib.util.module_from_spec(spec); spec.loader.exec_module(psim)

DATA = os.path.join(HERE, "data", "text_to_sql_benchmarks", "data",
                    "arcwise_plat_full_with_diff.json")


def ca_z(ns, ks, scores=(0, 1, 2)):
    N, K = sum(ns), sum(ks)
    p = K / N
    num = sum(k * s for k, s in zip(ks, scores)) - p * sum(n * s for n, s in zip(ns, scores))
    s1 = sum(n * s for n, s in zip(ns, scores))
    s2 = sum(n * s * s for n, s in zip(ns, scores))
    return num / math.sqrt(p * (1 - p) * (s2 - s1 * s1 / N))


def fit_ll(X, y):
    w = psim.logistic_fit(X, y)
    eta = X @ w
    mu = 1 / (1 + np.exp(-eta))
    ll = float(np.sum(y * np.log(mu + 1e-12) + (1 - y) * np.log(1 - mu + 1e-12)))
    return w, ll


def main():
    raw = json.load(open(DATA))
    rows = [psim.feats(e) for e in raw]
    dbs = [e["db_id"] for e in raw]
    err = np.array([r["err"] for r in rows], float)
    bucket = np.array([0 if r["d_demand"] <= 1 else (1 if r["d_demand"] == 2 else 2)
                       for r in rows])
    out = {}

    # 1. leave-one-database-out trend
    loo = {}
    for db in sorted(set(dbs)):
        keep = [i for i, d in enumerate(dbs) if d != db]
        ns = [sum(1 for i in keep if bucket[i] == b) for b in range(3)]
        ks = [sum(1 for i in keep if bucket[i] == b and err[i]) for b in range(3)]
        loo[db] = round(ca_z(ns, ks), 2)
    out["leave_one_db_out_z"] = loo
    print("leave-one-db-out z:", min(loo.values()), "-", max(loo.values()), loo)

    # 2. bootstrap CIs per stratum (percentile, 10k)
    rng = np.random.default_rng(11)
    B = 10_000
    cis = {}
    for b in range(3):
        idx = np.where(bucket == b)[0]
        stats = [float(err[rng.choice(idx, len(idx))].mean()) for _ in range(B)]
        lo, hi = np.percentile(stats, [2.5, 97.5])
        cis[f"stratum{b}"] = {"rate": round(100 * float(err[idx].mean()), 1),
                              "ci95": [round(100 * lo, 1), round(100 * hi, 1)]}
    out["bootstrap_ci"] = cis
    print("bootstrap CIs:", cis)

    # 3. AIC race, one structural predictor each (+intercept, +qlen? no: pure)
    qlen = np.array([r["qlen"] for r in rows], float)
    preds = {"d_demand": np.array([r["d_demand"] for r in rows], float),
             "ntab": np.array([r["ntab"] for r in rows], float),
             "njoin": np.array([r["njoin"] for r in rows], float)}
    aic = {}
    for name, v in preds.items():
        X = np.column_stack([np.ones(len(err)), v])
        _, ll = fit_ll(X, err)
        aic[name] = round(2 * 2 - 2 * ll, 1)
    out["aic_single_predictor"] = aic
    print("AIC race:", aic)

    # 4. depth x qlen interaction (Wald z on interaction term, standardized)
    zq = (qlen - qlen.mean()) / qlen.std()
    zb = (bucket - bucket.mean()) / bucket.std()
    X = np.column_stack([np.ones(len(err)), zb, zq, zb * zq])
    w, _ = fit_ll(X, err)
    eta = X @ w
    mu = 1 / (1 + np.exp(-eta))
    W = mu * (1 - mu)
    cov = np.linalg.inv(X.T @ (X * W[:, None]))
    z_int = w[3] / math.sqrt(cov[3, 3])
    p_int = math.erfc(abs(z_int) / math.sqrt(2))
    out["interaction_depth_x_qlen"] = {"z": round(float(z_int), 2),
                                       "p_two_sided": round(float(p_int), 3)}
    print(f"depth x qlen interaction: z={z_int:.2f} p={p_int:.3f}")

    json.dump(out, open(os.path.join(HERE, "results", "robustness.json"), "w"), indent=1)
    print("wrote results/robustness.json")


if __name__ == "__main__":
    main()
