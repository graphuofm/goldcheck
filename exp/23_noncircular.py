#!/usr/bin/env python3
"""
ICDE Exp 23 -- what survives once nothing is measured on the corrected query?

Every predictor here is computed from information that exists BEFORE the review
(the question text and the gold query under review). Corpora:
  mini : Mini-Dev T0 -> T2 (498), the set the lexicon and lints came from
  dev  : Dev 2024 -> 2025 review, 1,036 examples outside Mini-Dev, first analysed
         after lexicon and lints were frozen (held-out, different reviewers)

Also: direction of structural change for depth, tables and joins measured
symmetrically, and a review-routing model fit on mini and scored on dev.

Usage: python3 23_noncircular.py
"""
import importlib.util, json, math, os, re, zipfile
from math import comb
import numpy as np
import sqlglot
from sqlglot import exp as E

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
a5 = _load("a5", "05_audit_survival.py")
v = _load("v20", "20_validity_audit.py")
psim = _load("psim", "04_policy_sim.py")
EXT = os.path.join(HERE, "data_ext")
SUPER = re.compile(r"\b(highest|lowest|most|least|maximum|minimum|top \d+|largest|smallest|biggest|oldest|youngest|latest|earliest|longest|shortest|fastest)\b", re.I)


def lints(sql, tree, q):
    """Query smells taken from the Mini-Dev mistake catalogue; no reference answer needed."""
    has_join = bool(tree.find(E.Join))
    has_limit_order = any(s.args.get("limit") and s.args.get("order") for s in tree.find_all(E.Select))
    count_nd = any(not c.find(E.Distinct) and not c.find(E.Star) for c in tree.find_all(E.Count)) and has_join
    up = sql.upper()
    year_sub = bool(re.search(r"STRFTIME\(\s*'%Y'.*?\)\s*-\s*STRFTIME\(\s*'%Y'", up.replace('"', "'"), re.S))
    flat = v.select_depth(tree) <= 1
    super_flat = flat and bool(SUPER.search(q))
    division = bool(tree.find(E.Div))
    no_null_guard_order = has_limit_order and not tree.find(E.Is)
    return {"limit_order": has_limit_order, "count_nodistinct_join": count_nd, "year_subtract": year_sub,
            "superlative_flat": super_flat, "division": division, "order_no_nullguard": no_null_guard_order}


def row(old, new, db, err):
    t = sqlglot.parse_one(old["SQL"], read="sqlite")
    tn = sqlglot.parse_one(new, read="sqlite")
    q = old["question"]; ev = old.get("evidence") or ""
    r = {"db": db, "err": err, "q": q,
         "wd": v.select_depth(t), "dd": v.select_depth(tn),
         "wt": len({x.name for x in t.find_all(E.Table)}), "dt": len({x.name for x in tn.find_all(E.Table)}),
         "wj": len(list(t.find_all(E.Join))), "dj": len(list(tn.find_all(E.Join))),
         "qlen": len(q.split()), "elen": len(ev.split()), "mk": len(psim.MARKERS.findall(q)), "mk_ev": len(psim.MARKERS.findall(ev)),
         "difficulty": old.get("difficulty")}
    r.update(lints(old["SQL"], t, q))
    return r


def binom(a, b):
    n = a + b
    return float(f"{sum(comb(n, i) for i in range(a, n + 1)) / 2 ** n:.1e}") if n else None


def trend(rows, f, levels):
    ns = [sum(f(r) == l for r in rows) for l in levels]
    ks = [sum(f(r) == l and r["err"] for r in rows) for l in levels]
    use = [i for i in range(len(levels)) if ns[i]]
    z, p = a5.ca_trend([ns[i] for i in use], [ks[i] for i in use], scores=use)
    return {"n": ns, "rate": [round(100 * k / n, 1) if n else None for k, n in zip(ks, ns)], "z": round(z, 2), "p": float(f"{p:.1e}")}


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    rk = np.empty(len(s)); order = np.argsort(s); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    P, N = y.sum(), (1 - y).sum()
    return float((rk[y == 1].sum() - P * (P + 1) / 2) / (P * N))


FEATS = ["wd", "wt", "wj", "qlen", "elen", "mk", "mk_ev", "limit_order", "count_nodistinct_join", "year_subtract", "superlative_flat", "division", "order_no_nullguard"]
def mat(rows, feats, mu=None, sd=None):
    X = np.array([[float(r[f]) for f in feats] for r in rows])
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    return np.column_stack([np.ones(len(X)), (X - mu) / sd]), mu, sd


def catch(y, score, budgets, seeds=200):
    y = np.asarray(y, float); n = len(y); rng = np.random.default_rng(3); out = {}
    for b in budgets:
        k = max(1, int(b * n)); vals = []
        for _ in range(seeds):
            order = np.lexsort((rng.random(n), -np.asarray(score)))
            vals.append(100 * y[order[:k]].sum() / y.sum())
        out[str(b)] = round(float(np.mean(vals)), 1)
    return out


def describe(rows):
    res = {"n": len(rows), "err": sum(r["err"] for r in rows)}
    for nm, w, d in (("depth", "wd", "dd"), ("tables", "wt", "dt"), ("joins", "wj", "dj")):
        add = sum(r[w] < r[d] for r in rows); rem = sum(r[w] > r[d] for r in rows)
        res["dir_" + nm] = {"add": add, "remove": rem, "p": binom(add, rem)}
    nested = [r for r in rows if r["dd"] >= 2]; flat = [r for r in rows if r["dd"] <= 1]
    res["miss_rate_nested_written_shallower"] = {"k": sum(r["wd"] < r["dd"] for r in nested), "n": len(nested), "pct": round(100 * sum(r["wd"] < r["dd"] for r in nested) / len(nested), 1)}
    res["reverse_flat_written_deeper"] = {"k": sum(r["wd"] > r["dd"] for r in flat), "n": len(flat), "pct": round(100 * sum(r["wd"] > r["dd"] for r in flat) / len(flat), 1)}
    # symmetric gradients: everything on the query under review
    res["by_written_depth"] = trend(rows, lambda r: min(r["wd"], 3) - 1, [0, 1, 2])
    res["by_written_tables"] = trend(rows, lambda r: min(r["wt"], 4), [1, 2, 3, 4])
    res["by_written_joins"] = trend(rows, lambda r: min(r["wj"], 2), [0, 1, 2])
    res["by_markers"] = trend(rows, lambda r: min(r["mk"], 3), [0, 1, 2, 3])
    res["by_qlen_quartile"] = trend(rows, lambda r: int(np.searchsorted(np.percentile([x["qlen"] for x in rows], [25, 50, 75]), r["qlen"], side="left")), [0, 1, 2, 3])
    res["by_label"] = trend(rows, lambda r: r["difficulty"], ["simple", "moderate", "challenging"])
    res["lints"] = {f: {"n": sum(bool(r[f]) for r in rows), "rate_on": round(100 * sum(r["err"] for r in rows if r[f]) / max(1, sum(bool(r[f]) for r in rows)), 1),
                        "rate_off": round(100 * sum(r["err"] for r in rows if not r[f]) / max(1, sum(not r[f] for r in rows)), 1)} for f in FEATS[7:]}
    Y = np.array([r["err"] for r in rows], float); dbs = [r["db"] for r in rows]
    X, _, _ = mat(rows, ["wd", "wt", "qlen", "mk"])
    w, se, _ = v.logit(X, Y, cluster=dbs)
    res["joint_written_cluster"] = {f: {"OR_perSD": round(math.exp(b), 2), "p": float(f"{math.erfc(abs(b/s)/math.sqrt(2)):.1e}")} for f, b, s in zip(["wd", "wt", "qlen", "mk"], w[1:], se[1:])}
    fe = np.column_stack([[1.0 if d == g else 0.0 for d in dbs] for g in sorted(set(dbs))])
    w, se, _ = v.logit(np.column_stack([fe, X[:, 1:]]), Y, cluster=dbs)
    res["joint_written_db_FE"] = {f: {"OR_perSD": round(math.exp(b), 2), "p": float(f"{math.erfc(abs(b/s)/math.sqrt(2)):.1e}")} for f, b, s in zip(["wd", "wt", "qlen", "mk"], w[fe.shape[1]:], se[fe.shape[1]:])}
    aic = {}
    for f in ["wd", "wt", "wj", "qlen", "elen", "mk"]:
        A, _, _ = mat(rows, [f]); _, _, ll = v.logit(A, Y); aic[f] = round(4 - 2 * ll, 1)
    _, _, ll = v.logit(np.ones((len(Y), 1)), Y); aic["intercept"] = round(2 - 2 * ll, 1)
    res["aic_written"] = aic
    return res


def main():
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    t1 = {e["question_id"]: e["SQL"] for e in json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    mini = []
    for e in json.load(open(a5.UIUC)):
        q = int(e["question_id"])
        err = a5.norm(t0[q]["SQL"]) != a5.norm(t1[q]) or "original_SQL" in e
        mini.append(row(t0[q], e["SQL"], e["db_id"], bool(err)))
    old = {e["question_id"]: e for e in json.load(open(f"{EXT}/bird_dev_20240627/dev.json"))}
    raw = open(f"{EXT}/bird_dev_20251106/dev.json").read().strip()
    newl = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]
    tab = {r["q"]: r for r in json.load(open(os.path.join(HERE, "results", "full_dev_table.json")))}
    dev_all = []; 
    for e in newl:
        q = e["question_id"]
        r = row(old[q], e["SQL"], e["db_id"], bool(tab[q]["txt"])); r["out"] = bool(tab[q]["out"]); r["mini"] = tab[q]["mini"]
        dev_all.append(r)
    dev = [r for r in dev_all if not r["mini"]]
    out = {"mini": describe(mini), "dev_heldout": describe(dev), "dev_all": describe(dev_all)}
    dev_o = [{**r, "err": r["out"]} for r in dev]
    out["dev_heldout_output_changed"] = describe(dev_o)

    # forward test: fit on mini, score on held-out dev
    Y = np.array([r["err"] for r in mini], float)
    Xtr, mu, sd = mat(mini, FEATS)
    w, _, _ = v.logit(Xtr, Y)
    Xte, _, _ = mat(dev, FEATS, mu, sd)
    ys = [r["err"] for r in dev]; yo = [r["out"] for r in dev]
    s_full = Xte @ w
    Xq, muq, sdq = mat(mini, ["qlen", "mk"]); wq, _, _ = v.logit(Xq, Y); s_q = mat(dev, ["qlen", "mk"], muq, sdq)[0] @ wq
    pol = {"random": np.zeros(len(dev)), "question length": np.array([r["qlen"] for r in dev], float),
           "markers": np.array([r["mk"] for r in dev], float), "written tables": np.array([r["wt"] for r in dev], float),
           "written depth": np.array([r["wd"] for r in dev], float),
           "question model": s_q, "question+lints model": s_full}
    B = [0.05, 0.1, 0.2, 0.3, 0.5]
    out["forward"] = {nm: {"auc_text_changed": round(auc(ys, s), 3), "auc_output_changed": round(auc(yo, s), 3),
                           "catch_text": catch(ys, s, B), "catch_output": catch(yo, s, B)} for nm, s in pol.items()}
    out["forward_coef_perSD"] = {f: round(math.exp(b), 2) for f, b in zip(FEATS, w[1:])}
    # in-sample guard for mini (leave one database out)
    s = np.zeros(len(mini)); dbs = np.array([r["db"] for r in mini])
    for db in sorted(set(dbs)):
        tr = dbs != db
        wl, _, _ = v.logit(Xtr[tr], Y[tr]); s[~tr] = Xtr[~tr] @ wl
    out["mini_leave_one_db_auc"] = round(auc(Y, s), 3)
    json.dump(out, open(os.path.join(HERE, "results", "noncircular.json"), "w"), indent=1, default=float)
    for k in ("mini", "dev_heldout"):
        d = out[k]
        print("==", k, {x: d[x] for x in ("n", "err", "dir_depth", "dir_tables", "dir_joins", "miss_rate_nested_written_shallower", "reverse_flat_written_deeper")})
        for x in ("by_written_depth", "by_written_tables", "by_written_joins", "by_markers", "by_qlen_quartile", "by_label", "joint_written_cluster", "joint_written_db_FE", "aic_written", "lints"):
            print("  ", x, d[x])
    print("== output-changed held-out", {x: out["dev_heldout_output_changed"][x] for x in ("by_markers", "by_written_tables", "by_written_depth", "by_qlen_quartile", "by_label")})
    print("== forward"); [print("  ", k, x) for k, x in out["forward"].items()]
    print("coef", out["forward_coef_perSD"], "mini LODO auc", out["mini_leave_one_db_auc"])

if __name__ == "__main__":
    main()
