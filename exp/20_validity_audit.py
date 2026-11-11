#!/usr/bin/env python3
"""
ICDE Exp 20 -- what would a database reviewer attack first?

  A. Is every recorded "error" a real change? (revert / cosmetic audit)
  B. Does the gradient survive once mechanical off-diagonal cases are removed?
  C. Is SELECT nesting an artifact of how the query happens to be written?
     (alternative depth measures, including an irreducible depth that does not
      count blocks a standard rewrite could flatten)
  D. Primary-outcome regression with database fixed effects and cluster-robust
     standard errors, plus a model race on the primary outcome.

Usage: python3 20_validity_audit.py
"""
import importlib.util, json, math, os, re, zipfile
from collections import Counter
import numpy as np
import sqlglot
from sqlglot import exp as E

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("a5", os.path.join(HERE, "05_audit_survival.py"))
a5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a5)
UIUC = a5.UIUC


def p(sql):
    return sqlglot.parse_one(sql, read="sqlite")


def select_depth(node, d=0):
    here = d + (1 if isinstance(node, E.Select) else 0)
    best = here
    for c in node.args.values():
        for x in (c if isinstance(c, list) else [c]):
            if isinstance(x, E.Expression):
                best = max(best, select_depth(x, here))
    return best


def is_irreducible(sel):
    """A nested block a standard unnesting rewrite cannot simply merge into its
    parent: it aggregates, groups, limits, dedups, or uses a window."""
    if not isinstance(sel, E.Select):
        return True                      # set operations cannot be merged
    if sel.args.get("group") or sel.args.get("limit") or sel.args.get("distinct"):
        return True
    for x in sel.expressions:
        if x.find(E.AggFunc) or x.find(E.Window):
            return True
    return False


def irreducible_depth(node, d=0, top=True):
    here = d
    if isinstance(node, E.Select):
        here = d + 1 if (top or is_irreducible(node)) else d
    best = here
    for c in node.args.values():
        for x in (c if isinstance(c, list) else [c]):
            if isinstance(x, E.Expression):
                best = max(best, irreducible_depth(x, here, top and not isinstance(node, E.Select)))
    return best


def agg_depth(node):
    """Longest chain of aggregating blocks (a sum of counts, a max of sums)."""
    def rec(n, d):
        here = d
        if isinstance(n, E.Select) and (n.args.get("group") or any(x.find(E.AggFunc) for x in n.expressions)):
            here = d + 1
        best = here
        for c in n.args.values():
            for x in (c if isinstance(c, list) else [c]):
                if isinstance(x, E.Expression):
                    best = max(best, rec(x, here))
        return best
    return rec(node, 0)


def bucket(d):
    return 0 if d <= 1 else (1 if d == 2 else 2)


def trend(rows, dkey, ykey="err"):
    ns = [sum(r[dkey] == b for r in rows) for b in range(3)]
    ks = [sum(r[dkey] == b and r[ykey] for r in rows) for b in range(3)]
    use = [i for i in range(3) if ns[i]]
    z, pv = a5.ca_trend([ns[i] for i in use], [ks[i] for i in use], scores=use)
    return {"n": ns, "k": ks, "rate": [round(100 * k / n, 1) if n else None for k, n in zip(ks, ns)],
            "z": round(z, 2), "p": float(f"{pv:.2e}")}


def logit(X, y, cluster=None, iters=100):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = np.clip(1 / (1 + np.exp(-X @ w)), 1e-9, 1 - 1e-9)
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        w += step
        if np.abs(step).max() < 1e-8:
            break
    mu = np.clip(1 / (1 + np.exp(-X @ w)), 1e-9, 1 - 1e-9)
    ll = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    Hinv = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-6 * np.eye(X.shape[1]))
    if cluster is None:
        cov = Hinv
    else:
        meat = np.zeros_like(Hinv)
        G = sorted(set(cluster))
        for g in G:
            i = [k for k, c in enumerate(cluster) if c == g]
            s = (X[i] * (y[i] - mu[i])[:, None]).sum(0)
            meat += np.outer(s, s)
        cov = Hinv @ meat @ Hinv * len(G) / (len(G) - 1)
    return w, np.sqrt(np.diag(cov)), ll


def main():
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        mini = {e["question_id"]: e for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    t1 = {e["question_id"]: e["SQL"] for e in json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    uiuc = json.load(open(UIUC))
    rows = []
    for e in uiuc:
        q = int(e["question_id"])
        s0, s1, s2 = mini[q]["SQL"], t1[q], e["SQL"]
        err1 = a5.norm(s0) != a5.norm(s1)
        err2_flag = "original_SQL" in e
        err2_norm = a5.norm(s1) != a5.norm(s2)
        t2 = p(s2); t0 = p(s0)
        rows.append({
            "qid": q, "db": e["db_id"], "difficulty": mini[q].get("difficulty"),
            "s0": s0, "s1": s1, "s2": s2,
            "err1": err1, "err2": err2_flag, "err2_norm": err2_norm,
            "err": err1 or err2_flag,
            "reverted": (err1 or err2_flag) and a5.norm(s0) == a5.norm(s2),
            "dd": bucket(select_depth(t2)), "dw": bucket(select_depth(t0)),
            "irr": bucket(irreducible_depth(t2)), "agg": min(agg_depth(t2), 2),
            "dd_raw": select_depth(t2), "irr_raw": irreducible_depth(t2),
            "ntab": len({t.name for t in t2.find_all(E.Table)}),
            "njoin": len(list(t2.find_all(E.Join))),
            "nsel": len(list(t2.find_all(E.Select))),
            "has_cte": bool(t2.find(E.With)), "has_union": bool(t2.find(E.Union)),
            "has_window": bool(t2.find(E.Window)),
            "qlen": len(e["question"].split()), "elen": len((e.get("evidence") or "").split()),
        })
    out = {}

    # A
    a = {"n": len(rows), "any": sum(r["err"] for r in rows),
         "reverted_to_T0": sum(r["reverted"] for r in rows),
         "round2_flag_but_T1_eq_T2_after_normalising": sum(r["err2"] and not r["err2_norm"] for r in rows),
         "round2_unflagged_but_text_differs": sum((not r["err2"]) and r["err2_norm"] for r in rows)}
    a["reverted_qids"] = [r["qid"] for r in rows if r["reverted"]]
    a["cosmetic_round2_qids"] = [r["qid"] for r in rows if r["err2"] and not r["err2_norm"]]
    out["A_error_definition"] = a
    print("A", a)

    # B headline + diagonal only
    out["B_headline"] = trend(rows, "dd")
    diag = [r for r in rows if r["dd"] == r["dw"]]
    out["B_diagonal_only"] = trend(diag, "dd")
    off = [r for r in rows if r["dd"] != r["dw"]]
    out["B_offdiag"] = {"n": len(off), "err": sum(r["err"] for r in off),
                        "shallower": sum(r["dw"] < r["dd"] for r in off), "deeper": sum(r["dw"] > r["dd"] for r in off)}
    print("B headline", out["B_headline"]); print("B diagonal", out["B_diagonal_only"]); print("B off", out["B_offdiag"])

    # C alternatives
    out["C_irreducible"] = trend(rows, "irr")
    out["C_agg_chain"] = trend(rows, "agg")
    diag_i = [r for r in rows if r["dd"] == r["dw"]]
    out["C_irreducible_diag"] = trend(diag_i, "irr")
    out["C_feature_counts"] = {"cte": sum(r["has_cte"] for r in rows), "union": sum(r["has_union"] for r in rows),
                               "window": sum(r["has_window"] for r in rows),
                               "demand_nest": sum(r["dd"] > 0 for r in rows),
                               "demand_nest_but_reducible": sum(r["dd"] > 0 and r["irr"] == 0 for r in rows)}
    print("C irreducible", out["C_irreducible"]); print("C agg", out["C_agg_chain"]); print("C counts", out["C_feature_counts"])

    # D regression
    y = np.array([r["err"] for r in rows], float)
    dbs = [r["db"] for r in rows]
    dep = np.array([r["dd"] for r in rows], float)
    ntab = np.array([r["ntab"] for r in rows], float)
    njoin = np.array([r["njoin"] for r in rows], float)
    qlen = np.array([r["qlen"] for r in rows], float) / 10
    elen = np.array([r["elen"] for r in rows], float) / 10
    fe = np.column_stack([[1.0 if d == g else 0.0 for d in dbs] for g in sorted(set(dbs))])
    def rep(names, w, se):
        return {n: {"OR": round(math.exp(b), 2), "p": float(f"{math.erfc(abs(b/s)/math.sqrt(2)):.1e}")}
                for n, b, s in zip(names, w, se)}
    X = np.column_stack([dep, ntab, qlen, elen])
    Xc = np.column_stack([np.ones(len(y)), X])
    names = ["depth", "tables", "qlen10", "elen10"]
    w, se, ll = logit(Xc, y)
    out["D_plain"] = rep(names, w[1:], se[1:])
    w, se, ll = logit(Xc, y, cluster=dbs)
    out["D_cluster_robust"] = rep(names, w[1:], se[1:])
    Xfe = np.column_stack([fe, X])
    w, se, ll = logit(Xfe, y, cluster=dbs)
    out["D_db_fixed_effects_cluster"] = rep(names, w[fe.shape[1]:], se[fe.shape[1]:])
    print("D plain", out["D_plain"]); print("D cluster", out["D_cluster_robust"]); print("D FE", out["D_db_fixed_effects_cluster"])

    aic = {}
    for nm, v in (("depth", dep), ("irreducible depth", np.array([r["irr"] for r in rows], float)),
                  ("tables", ntab), ("joins", njoin), ("select blocks", np.array([r["nsel"] for r in rows], float)),
                  ("question length", qlen), ("evidence length", elen)):
        _, _, ll = logit(np.column_stack([np.ones(len(y)), v]), y)
        aic[nm] = round(4 - 2 * ll, 1)
    _, _, ll0 = logit(np.ones((len(y), 1)), y)
    aic["intercept only"] = round(2 - 2 * ll0, 1)
    out["D_aic_primary"] = aic
    print("D AIC", aic)

    # LOO on primary
    loo = {}
    for db in sorted(set(dbs)):
        keep = [r for r in rows if r["db"] != db]
        loo[db] = trend(keep, "dd")["z"]
    out["D_loo_primary_z"] = loo
    print("D LOO", min(loo.values()), max(loo.values()))
    # within-db stratified (Cochran-Mantel-Haenszel, flat vs nested)
    num = var = 0.0
    wins = 0; per = {}
    for db in sorted(set(dbs)):
        s = [r for r in rows if r["db"] == db]
        a_ = sum(r["dd"] > 0 and r["err"] for r in s); n1 = sum(r["dd"] > 0 for r in s)
        c_ = sum(r["dd"] == 0 and r["err"] for r in s); n0 = sum(r["dd"] == 0 for r in s)
        T = n1 + n0; m = a_ + c_
        num += a_ - n1 * m / T
        var += n1 * n0 * m * (T - m) / (T * T * (T - 1))
        wins += (a_ / n1 > c_ / n0) if n1 and n0 else 0
        per[db] = {"nested_n": n1, "nested_err": a_, "flat_n": n0, "flat_err": c_}
    out["D_cmh"] = {"chi2": round(num * num / var, 2), "z": round(num / math.sqrt(var), 2),
                    "p_one_sided": float(f"{0.5*math.erfc(abs(num/math.sqrt(var))/math.sqrt(2)):.1e}"), "dbs_with_positive_gap": int(wins)}
    out["D_per_db_primary"] = per
    print("D CMH", out["D_cmh"]); print(per)

    json.dump(out, open(os.path.join(HERE, "results", "validity_audit.json"), "w"), indent=1)
    json.dump([{k: v for k, v in r.items()} for r in rows], open(os.path.join(HERE, "results", "example_table.json"), "w"), indent=0)

if __name__ == "__main__":
    main()
