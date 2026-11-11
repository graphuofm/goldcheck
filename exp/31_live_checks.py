#!/usr/bin/env python3
"""
ICDE Exp 31 -- three more "the official answer is an accident" checks, run on the
official database for both BIRD Dev releases (2024-06-27 and 2025-11-06).

  int_div   : a division whose result changes when the numerator is cast to REAL
              (SQLite integer division silently truncates, e.g. 3/4 = 0)
  pk_dup    : COUNT(t.pk) without DISTINCT over a JOIN, where adding DISTINCT
              changes the result (the same entity is counted several times)
  empty_null: the gold answer is empty or contains only NULLs (any system that
              returns nothing is scored correct)

Each variant is compared against the parser round-trip of the gold query, and a
query is skipped if the round-trip itself changes the official result.
Output: results/live_checks.json
Usage: python3 31_live_checks.py
"""
import json, multiprocessing as mp, os, sqlite3, threading
import sqlglot
from sqlglot import exp as E

HERE = os.path.dirname(os.path.abspath(__file__))
DBDIR = os.path.join(HERE, "data", "dev_databases"); EXT = os.path.join(HERE, "data_ext")
CHECKS = ["int_div", "pk_dup", "empty_null"]


def load(path):
    raw = open(path).read().strip()
    return json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]


def rows(con, sql, secs=30):
    t = threading.Timer(secs, con.interrupt); t.start()
    try:
        return con.execute(sql).fetchall()
    except Exception:
        return None
    finally:
        t.cancel()


def as_set(r):
    return None if r is None else frozenset(tuple(round(x, 6) if isinstance(x, float) else x for x in row) for row in r)


def pk_cols(con, table, cache):
    if table not in cache:
        info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
        pks = [i[1].lower() for i in info if i[5] > 0]
        cache[table] = pks[0] if len(pks) == 1 else None
    return cache[table]


def check(job):
    db, qid, sql, tag = job
    res = {"qid": qid, "db": db, "tag": tag, "flags": [], "status": "ok"}
    con = sqlite3.connect(f"file:{os.path.join(DBDIR, db + '.sqlite')}?mode=ro", uri=True)
    con.create_function("IIF", 3, lambda c, x, y: x if c else y)
    orig = rows(con, sql)
    if orig is None:
        res["status"] = "gold_fails"; con.close(); return res
    if not orig or all(v is None for r in orig for v in r):
        res["flags"].append("empty_null")
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        base_sql = tree.sql(dialect="sqlite")
    except Exception:
        res["status"] = "parse_fail"; con.close(); return res
    base = as_set(rows(con, base_sql))
    if base != as_set(orig):
        res["status"] = "roundtrip_differs"; con.close(); return res
    # int_div: try each division separately
    for i, d in enumerate(list(tree.find_all(E.Div))):
        t2 = tree.copy()
        d2 = list(t2.find_all(E.Div))[i]
        d2.set("this", E.Cast(this=d2.this.copy(), to=E.DataType.build("REAL")))
        v = as_set(rows(con, t2.sql(dialect="sqlite")))
        if v is not None and v != base:
            res["flags"].append("int_div"); break
    # pk_dup: COUNT(alias.pk) with a JOIN in the same SELECT
    cache = {}
    for i, c in enumerate(list(tree.find_all(E.Count))):
        arg = c.this
        if not isinstance(arg, E.Column):
            continue
        sel = c.find_ancestor(E.Select)
        if sel is None or not sel.args.get("joins"):
            continue
        amap = {t.alias_or_name.lower(): t.name for t in sel.find_all(E.Table)}
        tb = amap.get(arg.table.lower()) if arg.table else (list(amap.values())[0] if len(amap) == 1 else None)
        if tb is None:
            continue
        try:
            if pk_cols(con, tb, cache) != arg.name.lower():
                continue
        except Exception:
            continue
        t2 = tree.copy()
        c2 = list(t2.find_all(E.Count))[i]
        c2.set("this", E.Distinct(expressions=[c2.this.copy()]))
        v = as_set(rows(con, t2.sql(dialect="sqlite")))
        if v is not None and v != base:
            res["flags"].append("pk_dup"); break
    con.close()
    return res


def main():
    old = {e["question_id"]: e for e in load(f"{EXT}/bird_dev_20240627/dev.json")}
    new = {e["question_id"]: e for e in load(f"{EXT}/bird_dev_20251106/dev.json")}
    jobs = [(e["db_id"], q, e["SQL"], "v2024") for q, e in old.items()] + \
           [(e["db_id"], q, e["SQL"], "v2025") for q, e in new.items()]
    with mp.Pool(10) as pool:
        out = pool.map(check, jobs, chunksize=8)
    by = {(r["tag"], r["qid"]): r for r in out}
    summ = {}
    for tag in ("v2024", "v2025"):
        s = [r for r in out if r["tag"] == tag]
        summ[tag] = {"n": len(s), **{st: sum(r["status"] == st for r in s) for st in sorted({r["status"] for r in s})},
                     **{c: sum(c in r["flags"] for r in s) for c in CHECKS}}
    for c in CHECKS:
        t = {"flagged_2024": 0, "fixed_text_changed": 0, "still_flagged_text_changed": 0, "still_flagged_text_same": 0,
             "new_in_2025": 0}
        for q in old:
            a, b = c in by[("v2024", q)]["flags"], c in by[("v2025", q)]["flags"]
            ch = old[q]["SQL"].strip() != new[q]["SQL"].strip()
            if a:
                t["flagged_2024"] += 1
                if not b:
                    t["fixed_text_changed" if ch else "fixed_text_same"] = t.get("fixed_text_changed" if ch else "fixed_text_same", 0) + 1
                else:
                    t["still_flagged_text_changed" if ch else "still_flagged_text_same"] += 1
            elif b:
                t["new_in_2025"] += 1
        summ[f"transitions_{c}"] = t
    summ["v2025_examples"] = [{"qid": r["qid"], "db": r["db"], "flags": r["flags"], "question": new[r["qid"]]["question"],
                               "sql": new[r["qid"]]["SQL"]} for r in out if r["tag"] == "v2025" and r["flags"]]
    print(json.dumps({k: v for k, v in summ.items() if k != "v2025_examples"}, indent=1))
    json.dump({"summary": summ, "rows": out}, open(os.path.join(HERE, "results", "live_checks.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
