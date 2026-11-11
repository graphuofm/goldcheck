#!/usr/bin/env python3
"""
ICDE Exp 40 -- is the answer of a tie / NULL-top gold query decided by the engine?

Run each gold query unchanged under four conditions that keep the data identical:
  reverse_scan : SQLite PRAGMA reverse_unordered_selects=ON (scan order reversed)
  rebuilt_rev  : every referenced table copied to a TEMP table in reverse rowid order (no indexes)
  rebuilt_same : same, original rowid order (no indexes): isolates the effect of indexes / plan
  duckdb       : DuckDB 1.1.3 reading the same SQLite file (another engine)
and record whether the result set differs from the official SQLite result.
Groups: Dev 2025 and Dev 2024 queries with a live tie or NULL-top finding, and as a control all top-k
queries whose cut has no tie ("pinned").
For a tie query, a changed result that is still a subset of the tie-complete result means the engine
picked another tied row.
Output: results/tie_engines.json
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import sqlglot
from sqlglot import exp as E
import goldcheck as G

HERE = os.path.dirname(os.path.abspath(__file__))
DBDIR = os.path.join(HERE, "data", "dev_databases")
EXT = os.path.join(HERE, "data", "duckdb_ext")
GC = json.load(open(os.path.join(HERE, "results", "goldcheck.json")))
VERS = {"dev_2024": os.path.join(HERE, "data_ext", "bird_dev_20240627", "dev.json"),
        "dev_2025": os.path.join(HERE, "data_ext", "bird_dev_20251106", "dev.json")}


def shadow(con, sql, reverse):
    names = {t.name for t in sqlglot.parse_one(sql, read="sqlite").find_all(E.Table)}
    real = {r[0].lower(): r[0] for r in con.execute("SELECT name FROM main.sqlite_master WHERE type='table'")}
    for n in names:
        t = real.get(n.lower())
        if t is None:
            continue
        cols = [i[1] for i in con.execute(f"PRAGMA main.table_info({G.q(t)})")]
        con.execute(f"CREATE TEMP TABLE {G.q(t)} AS SELECT * FROM main.{G.q(t)} WHERE 0")
        con.execute(f"INSERT INTO temp.{G.q(t)} ({', '.join(map(G.q, cols))}) SELECT {', '.join(map(G.q, cols))} "
                    f"FROM main.{G.q(t)} ORDER BY rowid {'DESC' if reverse else 'ASC'}")


def duck(db, sql):
    import duckdb
    try:
        c = duckdb.connect(config={"extension_directory": EXT})
        c.execute("LOAD sqlite")
        c.execute(f"ATTACH '{os.path.join(DBDIR, db + '.sqlite')}' AS d (TYPE sqlite, READ_ONLY)")
        c.execute("USE d")
        return c.execute(sql).fetchall()
    except Exception:
        return None


def one(args):
    ver, qid, db, sql, group = args
    path = os.path.join(DBDIR, db + ".sqlite")
    out = {"ver": ver, "qid": qid, "group": group}
    con = G.connect(path)
    try:
        base = G.as_set(G.run(con, sql, 60))
        tree = sqlglot.parse_one(sql, read="sqlite")
        k, ords = G._topk_parts(tree)
        tc = G.tie_complete(tree, k, ords)
        tcs = G.as_set(G.run(con, tc, 60)) if tc else None
    finally:
        con.close()
    res = {}
    con = G.connect(path)
    try:
        con.execute("PRAGMA reverse_unordered_selects=ON")
        res["reverse_scan"] = G.as_set(G.run(con, sql, 60))
    finally:
        con.close()
    for name, rev in (("rebuilt_rev", True), ("rebuilt_same", False)):
        con = G.connect(path)
        try:
            shadow(con, sql, rev)
            res[name] = G.as_set(G.run(con, sql, 120))
        except Exception:
            res[name] = None
        finally:
            con.close()
    d = duck(db, sql)
    res["duckdb"] = G.as_set(d) if d is not None else None
    for name, r in res.items():
        if r is None:
            out[name] = "fails"
        elif r == base:
            out[name] = "same"
        elif tcs is not None and r <= tcs:
            out[name] = "other_tied"
        else:
            out[name] = "changed"
    return out


def main():
    jobs = []
    for ver, path in VERS.items():
        data = {e["question_id"]: e for e in json.load(open(path))}
        for r in GC["rows"]:
            if r["ver"] != ver or "topk" not in r:
                continue
            f = r["flags"]
            g = "tie" if "topk_tie" in f else ("null_top" if "topk_null" in f else
                                                ("control" if r["topk"].get("status") == "pinned" else None))
            if g:
                e = data[r["qid"]]
                jobs.append((ver, r["qid"], e["db_id"], e["SQL"], g))
    with ProcessPoolExecutor(10) as ex:
        rows = list(ex.map(one, jobs, chunksize=2))
    summ = {}
    for ver in VERS:
        for g in ("tie", "null_top", "control"):
            R = [r for r in rows if r["ver"] == ver and r["group"] == g]
            s = {"n": len(R)}
            for v in ("reverse_scan", "rebuilt_rev", "rebuilt_same", "duckdb"):
                s[v] = {c: sum(r[v] == c for r in R) for c in ("same", "other_tied", "changed", "fails")}
            s["any_variant_differs"] = sum(any(r[v] in ("other_tied", "changed") for v in
                                               ("reverse_scan", "rebuilt_rev", "rebuilt_same", "duckdb")) for r in R)
            s["any_sqlite_variant_differs"] = sum(any(r[v] in ("other_tied", "changed") for v in
                                                      ("reverse_scan", "rebuilt_rev", "rebuilt_same")) for r in R)
            summ[f"{ver}|{g}"] = s
            print(ver, g, json.dumps(s))
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(HERE, "results", "tie_engines.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
