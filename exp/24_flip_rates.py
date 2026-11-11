#!/usr/bin/env python3
"""
ICDE Exp 24 -- is "corrections add depth" more than a base-rate effect?

A flat query can only gain depth and flat queries are the large majority, so raw
add:remove counts are not evidence of a directional bias. This reports change
rates per starting query, for depth, tables and joins, in both corpora, and tests
them against the equal-rate null. Also: breadth direction without the two
databases whose schemas drove most table additions.

Usage: python3 24_flip_rates.py
"""
import importlib.util, json, os, sys, zipfile
from math import comb
import sqlglot
from sqlglot import exp as E
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
a5 = _load("a5", "05_audit_survival.py"); v = _load("v20", "20_validity_audit.py")
EXT = os.path.join(HERE, "data_ext")

def m(sql):
    t = sqlglot.parse_one(sql, read="sqlite")
    return {"d": v.select_depth(t), "t": len({x.name for x in t.find_all(E.Table)}), "j": len(list(t.find_all(E.Join)))}

def flips(rows):
    o = {}
    flat = [r for r in rows if r["old"]["d"] == 1]; nest = [r for r in rows if r["old"]["d"] >= 2]
    up = sum(r["new"]["d"] > 1 for r in flat); down = sum(r["new"]["d"] < r["old"]["d"] for r in nest)
    _, p = stats.fisher_exact([[up, len(flat) - up], [down, len(nest) - down]])
    o["depth"] = {"flat_n": len(flat), "flat_gain": up, "flat_gain_pct": round(100 * up / len(flat), 1),
                  "nested_n": len(nest), "nested_lose": down, "nested_lose_pct": round(100 * down / max(1, len(nest)), 1), "fisher_p": round(float(p), 3),
                  "add_total": sum(r["new"]["d"] > r["old"]["d"] for r in rows), "remove_total": sum(r["new"]["d"] < r["old"]["d"] for r in rows)}
    # among queries that were actually changed
    cf = [r for r in flat if r["err"]]; cn = [r for r in nest if r["err"]]
    o["depth_given_changed"] = {"flat_changed": len(cf), "gain": sum(r["new"]["d"] > 1 for r in cf), "nested_changed": len(cn), "lose": sum(r["new"]["d"] < r["old"]["d"] for r in cn),
                                "any_depth_change_nested": sum(r["new"]["d"] != r["old"]["d"] for r in cn)}
    for k, nm in (("t", "tables"), ("j", "joins")):
        add = sum(r["new"][k] > r["old"][k] for r in rows); rem = sum(r["new"][k] < r["old"][k] for r in rows)
        n = add + rem
        o[nm] = {"add": add, "remove": rem, "p_one_sided": float(f"{sum(comb(n, i) for i in range(add, n + 1)) / 2 ** n:.1e}") if n else None}
    return o

def main():
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    mini = [{"db": e["db_id"], "old": m(t0[int(e["question_id"])]["SQL"]), "new": m(e["SQL"]),
             "err": a5.norm(t0[int(e["question_id"])]["SQL"]) != a5.norm(e["SQL"])} for e in json.load(open(a5.UIUC))]
    old = {e["question_id"]: e for e in json.load(open(f"{EXT}/bird_dev_20240627/dev.json"))}
    raw = open(f"{EXT}/bird_dev_20251106/dev.json").read().strip()
    newl = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]
    dev = [{"db": e["db_id"], "mini": e["question_id"] in t0, "old": m(old[e["question_id"]]["SQL"]), "new": m(e["SQL"]),
            "err": a5.norm(old[e["question_id"]]["SQL"]) != a5.norm(e["SQL"])} for e in newl]
    held = [r for r in dev if not r["mini"]]
    out = {"mini": flips(mini), "dev_all": flips(dev), "dev_heldout": flips(held)}
    from collections import Counter
    add_by_db = Counter(r["db"] for r in dev if r["new"]["t"] > r["old"]["t"])
    out["dev_table_add_by_db"] = add_by_db.most_common()
    out["dev_excl_financial_schools"] = flips([r for r in dev if r["db"] not in ("financial", "california_schools")])
    print(json.dumps(out, indent=0))
    json.dump(out, open(os.path.join(HERE, "results", "flip_rates.json"), "w"), indent=1)
main()
