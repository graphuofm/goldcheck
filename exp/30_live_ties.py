#!/usr/bin/env python3
"""
ICDE Exp 30 -- is the "unpinned top-k" problem real in the CURRENT official release?

For every gold query in BIRD Dev (2024 and 2025-11-06 versions) whose outermost
SELECT has ORDER BY ... LIMIT k, we re-run it on the official database with the
ORDER BY keys appended to the output and the LIMIT removed, then look at the
ordering values around the cut:
  live_tie   : row k and row k+1 have the same ordering key -> the official answer
               depends on which tied row SQLite happens to return (arbitrary)
  null_top   : the first ordering key is NULL -> the "top" answer is a missing value
  pinned     : the cut is strict and non-NULL -> the data pins the answer, but a
               single tie would unpin it (latent)
Output: results/live_ties.json
Usage: python3 30_live_ties.py
"""
import json, multiprocessing as mp, os, sqlite3, threading
import sqlglot
from sqlglot import exp as E

HERE = os.path.dirname(os.path.abspath(__file__))
DBDIR = os.path.join(HERE, "data", "dev_databases"); EXT = os.path.join(HERE, "data_ext")


def load(path):
    raw = open(path).read().strip()
    return json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]


def probe_sql(sql):
    """Return (probe_sql, k, n_keys) or None if the outer query has no ORDER BY ... LIMIT."""
    t = sqlglot.parse_one(sql, read="sqlite")
    if not isinstance(t, E.Select):
        return None
    order, limit = t.args.get("order"), t.args.get("limit")
    if order is None or limit is None:
        return None
    try:
        k = int(limit.expression.name)
    except Exception:
        return None
    if t.args.get("offset") is not None:
        return None
    aliases = {s.alias: s.this for s in t.expressions if isinstance(s, E.Alias)}
    keys = []
    for o in order.expressions:
        e = o.this.copy()
        if isinstance(e, E.Column) and not e.table and e.name in aliases:
            e = aliases[e.name].copy()
        keys.append(e)
    p = t.copy()
    p.set("limit", None)
    p.set("expressions", list(p.expressions) + [E.alias_(kk, f"__k{i}") for i, kk in enumerate(keys)])
    return p.sql(dialect="sqlite"), k, len(keys)


def classify(job):
    db, qid, sql, tag = job
    r = {"qid": qid, "db": db, "tag": tag, "status": "no_topk"}
    try:
        pr = probe_sql(sql)
    except Exception:
        r["status"] = "parse_fail"; return r
    if pr is None:
        return r
    psql, k, nk = pr
    r["k"] = k
    con = sqlite3.connect(f"file:{os.path.join(DBDIR, db + '.sqlite')}?mode=ro", uri=True)
    con.create_function("IIF", 3, lambda c, x, y: x if c else y)
    t = threading.Timer(30, con.interrupt); t.start()
    try:
        rows = con.execute(psql).fetchmany(max(k + 1, 5000))
    except Exception as ex:
        r["status"] = "probe_fail"; r["err"] = str(ex)[:80]; return r
    finally:
        t.cancel(); con.close()
    keys = [tuple(row[-nk:]) for row in rows]
    if not keys:
        r["status"] = "empty"
    elif keys[0][0] is None:
        r["status"] = "null_top"
    elif len(keys) > k and keys[k - 1] == keys[k]:
        cut = keys[k - 1]
        tied = {tuple(row[:-nk]) for row, kk in zip(rows, keys) if kk == cut}
        above = {tuple(row[:-nk]) for row, kk in zip(rows[:k], keys[:k]) if kk != cut}
        r["n_tied"] = sum(kk == cut for kk in keys)
        # harmful only if the tied rows produce different outputs AND some of them are cut off
        r["status"] = "live_tie" if len(tied - above) > 1 else "tie_harmless"
    else:
        r["status"] = "pinned"
    return r


def main():
    old = {e["question_id"]: e for e in load(f"{EXT}/bird_dev_20240627/dev.json")}
    new = {e["question_id"]: e for e in load(f"{EXT}/bird_dev_20251106/dev.json")}
    jobs = [(e["db_id"], q, e["SQL"], "v2024") for q, e in old.items()] + \
           [(e["db_id"], q, e["SQL"], "v2025") for q, e in new.items()]
    with mp.Pool(10) as pool:
        out = pool.map(classify, jobs, chunksize=8)
    by = {(r["tag"], r["qid"]): r for r in out}
    summ = {}
    for tag in ("v2024", "v2025"):
        s = [r for r in out if r["tag"] == tag]
        summ[tag] = {st: sum(r["status"] == st for r in s) for st in sorted({r["status"] for r in s})}
        summ[tag]["n"] = len(s)
    # what did the 2025 review do with the 2024 live problems?
    trans = {}
    for q in old:
        a, b = by[("v2024", q)]["status"], by[("v2025", q)]["status"]
        if a in ("live_tie", "null_top"):
            changed = old[q]["SQL"].strip() != new[q]["SQL"].strip()
            key = f"{a}->{b}" + (" (text changed)" if changed else " (text same)")
            trans[key] = trans.get(key, 0) + 1
    summ["transitions_of_2024_problems"] = dict(sorted(trans.items(), key=lambda x: -x[1]))
    summ["v2025_live_examples"] = [{"qid": r["qid"], "db": r["db"], "status": r["status"], "sql": new[r["qid"]]["SQL"],
                                    "question": new[r["qid"]]["question"]}
                                   for r in out if r["tag"] == "v2025" and r["status"] in ("live_tie", "null_top")][:400]
    print(json.dumps({k: v for k, v in summ.items() if k != "v2025_live_examples"}, indent=1))
    json.dump({"summary": summ, "rows": out}, open(os.path.join(HERE, "results", "live_ties.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
