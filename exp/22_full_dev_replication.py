#!/usr/bin/env python3
"""
ICDE Exp 22 -- full BIRD Dev replication (2024-06-27 gold vs 2025-11-06 review).

Three definitions of "the review changed this gold", from loosest to strictest:
  text    : normalised SQL text differs
  output  : both run and the result sets differ on the official database
Reports the depth and breadth gradients for all examples and for the 1,038 that
are NOT in Mini-Dev (a held-out replication), a joint regression, and the same
for the conservative output-changed outcome.

Usage: python3 22_full_dev_replication.py
"""
import importlib.util, json, math, os, zipfile
import numpy as np
import sqlglot
from sqlglot import exp as E

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
a5 = _load("a5", "05_audit_survival.py")
v = _load("v20", "20_validity_audit.py")
e7 = _load("e7", "07_execution_consequences.py")
EXT = os.path.join(HERE, "data_ext")


def run(db, sql):
    st, rows, _ = e7.run_query(os.path.join(e7.DBDIR, db + ".sqlite"), sql)
    return st, (frozenset(map(tuple, rows)) if st == "ok" and rows is not None else None)


def grad(rows, key, y):
    ns = [sum(r[key] == b for r in rows) for b in range(3)]
    ks = [sum(r[key] == b and r[y] for r in rows) for b in range(3)]
    z, p = a5.ca_trend(ns, ks)
    return {"n": ns, "k": ks, "rate": [round(100 * k / n, 1) for k, n in zip(ks, ns)], "z": round(z, 2), "p": float(f"{p:.1e}")}


def grid(rows, key, cuts, y):
    out = {}
    for i, c in enumerate(cuts):
        s = [r for r in rows if (r[key] >= c if i == len(cuts) - 1 else r[key] == c)]
        out[str(c) + ("+" if i == len(cuts) - 1 else "")] = {"n": len(s), "rate": round(100 * sum(r[y] for r in s) / max(1, len(s)), 1)}
    return out


def models(rows, y):
    Y = np.array([r[y] for r in rows], float)
    dbs = [r["db"] for r in rows]
    dep = np.array([r["d"] for r in rows], float)
    tab = np.array([r["ntab"] for r in rows], float)
    jn = np.array([r["njoin"] for r in rows], float)
    q = np.array([r["qlen"] for r in rows], float)
    X = np.column_stack([np.ones(len(Y)), dep, tab, q / 10])
    w, se, _ = v.logit(X, Y, cluster=dbs)
    names = ["depth", "tables", "qlen10"]
    res = {n: {"OR": round(math.exp(b), 2), "p": float(f"{math.erfc(abs(b/s)/math.sqrt(2)):.1e}")} for n, b, s in zip(names, w[1:], se[1:])}
    Z = np.column_stack([np.ones(len(Y)), (dep - dep.mean()) / dep.std(), (tab - tab.mean()) / tab.std(), (q - q.mean()) / q.std()])
    w, se, _ = v.logit(Z, Y, cluster=dbs)
    res["per_SD"] = {n: round(math.exp(b), 2) for n, b in zip(names, w[1:])}
    aic = {}
    for nm, a in (("depth", dep), ("tables", tab), ("joins", jn), ("qlen", q)):
        _, _, ll = v.logit(np.column_stack([np.ones(len(Y)), a]), Y)
        aic[nm] = round(4 - 2 * ll, 1)
    res["aic"] = aic
    return res


def main():
    e7.ensure_dbs()
    old = {e["question_id"]: e for e in json.load(open(f"{EXT}/bird_dev_20240627/dev.json"))}
    raw = open(f"{EXT}/bird_dev_20251106/dev.json").read().strip()
    newl = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]
    new = {e["question_id"]: e for e in newl}
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"] for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    rows = []; skipped = 0
    for q, e in new.items():
        if q not in old:
            skipped += 1; continue
        try:
            tn, to = sqlglot.parse_one(e["SQL"], read="sqlite"), sqlglot.parse_one(old[q]["SQL"], read="sqlite")
        except Exception:
            skipped += 1; continue
        chg = a5.norm(old[q]["SQL"]) != a5.norm(e["SQL"])
        d, w = v.select_depth(tn), v.select_depth(to)
        st_o = st_n = "same"; out_diff = False
        if chg:
            st_o, ro = run(e["db_id"], old[q]["SQL"]); st_n, rn = run(e["db_id"], e["SQL"])
            out_diff = st_n == "ok" and (st_o != "ok" or ro != rn)
        rows.append({"q": q, "db": e["db_id"], "txt": chg, "out": out_diff, "st_o": st_o, "st_n": st_n,
                     "d": v.bucket(d), "w": v.bucket(w), "d_raw": d, "w_raw": w, "irr": v.bucket(v.irreducible_depth(tn)),
                     "dw_irr": v.bucket(v.irreducible_depth(to)),
                     "ntab": len({t.name for t in tn.find_all(E.Table)}), "njoin": len(list(tn.find_all(E.Join))),
                     "ntab_old": len({t.name for t in to.find_all(E.Table)}),
                     "qlen": len(e["question"].split()), "mini": q in t0, "difficulty": e.get("difficulty"),
                     "qchg": old[q]["question"] != e["question"], "echg": (old[q].get("evidence") or "") != (e.get("evidence") or "")})
    out = {"n": len(rows), "skipped": skipped, "text_changed": sum(r["txt"] for r in rows), "output_changed": sum(r["out"] for r in rows),
           "old_failed": sum(r["st_o"] not in ("ok", "same") for r in rows), "new_failed": sum(r["st_n"] not in ("ok", "same") for r in rows),
           "question_changed": sum(r["qchg"] for r in rows), "evidence_changed": sum(r["echg"] for r in rows), "mini_overlap": sum(r["mini"] for r in rows)}
    print(out)
    for tag, sub in (("all", rows), ("heldout_not_in_minidev", [r for r in rows if not r["mini"]]), ("in_minidev", [r for r in rows if r["mini"]])):
        res = {}
        for y in ("txt", "out"):
            res[y] = {"depth": grad(sub, "d", y), "irreducible": grad(sub, "irr", y),
                      "written_depth": grad(sub, "w", y),
                      "diag": grad([r for r in sub if r["d"] == r["w"]], "d", y),
                      "tables": grid(sub, "ntab", [1, 2, 3, 4], y), "tables_old": grid(sub, "ntab_old", [1, 2, 3, 4], y), "joins": grid(sub, "njoin", [0, 1, 2], y),
                      "models": models(sub, y)}
            add = sum(r["w_raw"] < r["d_raw"] for r in sub if r[y]); rem = sum(r["w_raw"] > r["d_raw"] for r in sub if r[y])
            res[y]["direction"] = {"add": add, "remove": rem}
        out[tag] = res
        print("==", tag, json.dumps(res, default=str)[:2300])
    # within 4+ tables: which databases and how much does the table-count group change
    four = [r for r in rows if r["ntab"] >= 4]
    from collections import Counter
    print("4+ tables by db", Counter(r["db"] for r in four).most_common(12))
    print("4+ tables: rate by depth", grad(four, "d", "txt")["rate"], " ntab_old>=4:", sum(r["ntab_old"] >= 4 for r in rows), "new>=4 & old<4:", sum(r["ntab"] >= 4 and r["ntab_old"] < 4 for r in rows))
    print("per-db change rate", {db: round(100 * sum(r["txt"] for r in rows if r["db"] == db) / sum(1 for r in rows if r["db"] == db), 1) for db in sorted({r["db"] for r in rows})})
    json.dump(out, open(os.path.join(HERE, "results", "full_dev_replication.json"), "w"), indent=1, default=str)
    json.dump(rows, open(os.path.join(HERE, "results", "full_dev_table.json"), "w"), default=str)

if __name__ == "__main__":
    main()
