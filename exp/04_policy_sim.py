#!/usr/bin/env python3
"""
CHI2 Exp 04 -- policy simulation: routing review budget by observable features.

Question: if a benchmark team can double-review only a fraction k of examples,
how many of the 134 real annotation errors does each routing policy catch?

Policies (all evaluated against ground-truth error labels):
  random        expectation = k (analytic) -- the current practice
  qlen          rank by question length (observable, naive baseline)
  observable    5-fold CV logistic score on PRE-REVIEW observables only:
                original-SQL depth/tables/joins, question & evidence length,
                and linguistic demand markers in the question
  oracle-depth  rank by TRUE demanded depth (corrected SQL) -- ceiling, not
                a deployable policy (labeled as oracle)

Side finding (F5 candidate): can demanded depth be predicted from the
question text alone? We report the marker->demanded-depth association.

Usage: python3 04_policy_sim.py
"""
import json, math, os, random, re
import numpy as np
import sqlglot
from sqlglot import exp as E

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "text_to_sql_benchmarks", "data",
                    "arcwise_plat_full_with_diff.json")

MARKERS = re.compile(
    r"\b(ratio|percentage|proportion|average|mean|among|of (?:those|these|the \w+ that)"
    r"|more than the|less than the|higher than the|lower than the|compared? (?:to|with)"
    r"|difference between|highest|lowest|most|least|maximum|minimum|top \d+"
    r"|per |respectively|second[- ](?:highest|largest|most))\b", re.I)

def sel_depth(t):
    def d(node, dd=0):
        best = dd + (1 if isinstance(node, E.Select) else 0)
        for c in node.args.values():
            for x in (c if isinstance(c, list) else [c]):
                if isinstance(x, E.Expression):
                    best = max(best, d(x, dd + (1 if isinstance(node, E.Select) else 0)))
        return best
    return d(t)

def feats(e):
    orig = e.get("original_SQL") or e["SQL"]
    to = sqlglot.parse_one(orig, read="sqlite")
    tc = sqlglot.parse_one(e["SQL"], read="sqlite")
    q = e["question"]; ev = e.get("evidence") or ""
    return {
        "err": 1 if "original_SQL" in e else 0,
        "d_demand": sel_depth(tc),
        "d_written": sel_depth(to),
        "ntab": len({x.name for x in to.find_all(E.Table)}),
        "njoin": len(list(to.find_all(E.Join))),
        "qlen": len(q.split()), "elen": len(ev.split()),
        "markers": len(MARKERS.findall(q)),
        "m_ev": len(MARKERS.findall(ev)),
    }

def logistic_fit(X, y, iters=80):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ w
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        try:
            w_new = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6*np.eye(X.shape[1]),
                                    X.T @ (W * eta + (y - mu)))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new; break
        w = w_new
    return w

def catch_curve(order, err, ks):
    """order: example indices ranked most-suspicious first."""
    total = err.sum()
    caught = np.cumsum(err[order])
    n = len(err)
    return [100 * caught[max(1, int(k * n)) - 1] / total for k in ks]

def main(seeds=20):
    rows = [feats(e) for e in json.load(open(DATA))]
    err = np.array([r["err"] for r in rows], float)
    n = len(rows)
    OBS = ["d_written", "ntab", "njoin", "qlen", "elen", "markers", "m_ev"]
    Xo = np.array([[1] + [r[k] for k in OBS] for r in rows], float)
    Xo[:, 1:] = (Xo[:, 1:] - Xo[:, 1:].mean(0)) / (Xo[:, 1:].std(0) + 1e-9)

    ks = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    # F5: is demanded depth predictable from the question alone?
    dd = np.array([r["d_demand"] for r in rows])
    mk = np.array([r["markers"] for r in rows])
    from scipy import stats
    rho, p = stats.spearmanr(mk, dd)
    deep = dd >= 2
    print(f"F5 问题文本标记 vs 任务要求深度: Spearman rho={rho:.3f} p={p:.1e}")
    print(f"   标记>=2 的问题中深度>=2 的比例: "
          f"{100*np.mean(dd[mk>=2]>=2):.1f}%  vs 标记==0: {100*np.mean(dd[mk==0]>=2):.1f}%")

    # oracle and simple rankings (deterministic; random tie-break averaged over seeds)
    res = {k: {p: [] for p in ("random", "qlen", "observable", "oracle")} for k in ks}
    rng = np.random.default_rng(7)
    for seed in range(seeds):
        tie = rng.random(n)
        # random policy
        order_r = np.argsort(tie)
        # qlen
        order_q = np.lexsort((tie, -np.array([r["qlen"] for r in rows], float)))
        # oracle demanded depth
        order_o = np.lexsort((tie, -dd.astype(float)))
        # observable model: 5-fold CV scores
        idx = rng.permutation(n)
        score = np.zeros(n)
        for f in range(5):
            te = idx[f::5]; tr = np.setdiff1d(idx, te)
            w = logistic_fit(Xo[tr], err[tr])
            score[te] = Xo[te] @ w
        order_m = np.lexsort((tie, -score))
        for k in ks:
            for name, o in (("random", order_r), ("qlen", order_q),
                            ("observable", order_m), ("oracle", order_o)):
                res[k][name].append(catch_curve(o, err, [k])[0])
    print(f"\n复核预算 -> 逮住 134 个真实错误的百分比（{seeds} 次重复的均值±std）")
    print(f"{'budget':>7} {'random':>14} {'qlen':>14} {'observable':>14} {'oracle':>14}")
    summary = {}
    for k in ks:
        line = [f"{100*k:5.0f}%"]
        summary[k] = {}
        for name in ("random", "qlen", "observable", "oracle"):
            v = np.array(res[k][name])
            line.append(f"{v.mean():6.1f}±{v.std():4.1f}")
            summary[k][name] = [round(v.mean(), 1), round(v.std(), 1)]
        print(" ".join(f"{s:>14}" for s in line))
    out = {"summary": summary, "F5": {"spearman_rho": round(rho, 3), "p": float(p)},
           "n": n, "n_err": int(err.sum()), "obs_features": OBS, "seeds": seeds}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "policy_sim.json"), "w"), indent=1)
    print("\nwrote results/policy_sim.json")

if __name__ == "__main__":
    main()
