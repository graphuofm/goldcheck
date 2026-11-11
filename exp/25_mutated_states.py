#!/usr/bin/env python3
"""
ICDE Exp 25 -- execution match depends on the database state. Make it visible.

For every old/corrected gold pair whose outputs are IDENTICAL on the official
database, run both queries on systematically perturbed copies of the tables they
reference. A difference on any perturbed state is a concrete counterexample: the
two queries are not equivalent, and the official state merely hid it.

Perturbations are schema-agnostic and need no reference answer:
  duplicate : repeat a few rows of each referenced table
  null      : set referenced columns to NULL in a few rows
  tie       : copy the extreme ORDER BY value into other rows
  date      : move date-like referenced values to the birthday boundary
  delete    : remove a random 20% of rows

Control: unchanged golds vs their own parser-regenerated text must never differ.
Usage: python3 25_mutated_states.py
"""
import importlib.util, json, multiprocessing as mp, os, random, re, sqlite3, sys, threading, zipfile, datetime
import sqlglot
from sqlglot import exp as E

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
a5 = _load("a5", "05_audit_survival.py")
DBDIR = os.path.join(HERE, "data", "dev_databases"); EXT = os.path.join(HERE, "data_ext")
OPS = ["duplicate", "null", "tie", "date", "delete"]
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def norm_rows(rows):
    return frozenset(tuple(round(x, 6) if isinstance(x, float) else x for x in r) for r in rows)


def run(con, sql, secs=20):
    t = threading.Timer(secs, con.interrupt); t.start()
    try:
        return norm_rows(con.execute(sql).fetchall())
    except Exception:
        return None
    finally:
        t.cancel()


def refs(sql):
    t = sqlglot.parse_one(sql, read="sqlite")
    return {x.name for x in t.find_all(E.Table)}, {c.name for c in t.find_all(E.Column)}, \
           {c.name for o in t.find_all(E.Order) for c in o.find_all(E.Column)}


def q(n):
    return '"' + n.replace('"', '""') + '"'


def apply_op(con, op, tables, cols, order_cols, rng):
    for tb in tables:
        info = con.execute(f"PRAGMA temp.table_info({q(tb)})").fetchall()
        names = [i[1] for i in info]
        ids = [r[0] for r in con.execute(f"SELECT rowid FROM temp.{q(tb)} ORDER BY RANDOM() LIMIT 200").fetchall()]
        if not ids:
            continue
        pick = lambda k: rng.sample(ids, min(k, len(ids)))
        low = {n.lower(): n for n in names}
        here = [low[c.lower()] for c in cols if c.lower() in low]
        if op == "duplicate":
            for rid in pick(5):
                con.execute(f"INSERT INTO temp.{q(tb)} SELECT * FROM temp.{q(tb)} WHERE rowid=?", (rid,))
        elif op == "null":
            for c in here:
                for rid in pick(5):
                    con.execute(f"UPDATE temp.{q(tb)} SET {q(c)}=NULL WHERE rowid=?", (rid,))
        elif op == "tie":
            for c in [low[x.lower()] for x in order_cols if x.lower() in low]:
                for fn in ("MAX", "MIN"):
                    val = con.execute(f"SELECT {fn}({q(c)}) FROM temp.{q(tb)}").fetchone()[0]
                    for rid in pick(3):
                        con.execute(f"UPDATE temp.{q(tb)} SET {q(c)}=? WHERE rowid=?", (val, rid))
        elif op == "date":
            bd = (datetime.date.today() + datetime.timedelta(days=1)).strftime("1980-%m-%d")
            for c in here:
                sample = con.execute(f"SELECT {q(c)} FROM temp.{q(tb)} WHERE {q(c)} IS NOT NULL LIMIT 5").fetchall()
                if sample and all(isinstance(s[0], str) and DATE.match(s[0]) for s in sample):
                    for rid in pick(10):
                        con.execute(f"UPDATE temp.{q(tb)} SET {q(c)}=? WHERE rowid=?", (bd, rid))
        elif op == "delete":
            total = con.execute(f"SELECT COUNT(*) FROM temp.{q(tb)}").fetchone()[0]
            con.execute(f"DELETE FROM temp.{q(tb)} WHERE rowid IN (SELECT rowid FROM temp.{q(tb)} ORDER BY RANDOM() LIMIT ?)", (int(total * 0.2),))


def test_pair(p):
    db, a, b, tag, qid = p
    res = {"qid": qid, "db": db, "tag": tag, "exposed": {}, "base_ok": False, "baseline_equal": None}
    try:
        ta, ca, oa = refs(a); tb_, cb, ob = refs(b)
    except Exception:
        return res
    con = sqlite3.connect(f"file:{os.path.join(DBDIR, db + '.sqlite')}?mode=ro", uri=True, timeout=10, isolation_level=None)
    con.create_function("IIF", 3, lambda c, x, y: x if c else y)
    real = {r[0].lower(): r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    tables = [real[t.lower()] for t in ta | tb_ if t.lower() in real]
    try:
        for t in tables:
            con.execute(f"CREATE TEMP TABLE {q(t)} AS SELECT * FROM main.{q(t)}")
    except Exception as ex:
        res["err"] = str(ex)[:100]; return res
    ra, rb = run(con, a), run(con, b)
    res["base_ok"] = ra is not None and rb is not None
    res["baseline_equal"] = res["base_ok"] and ra == rb
    cols = ca | cb; order = oa | ob
    for op in OPS:
        hit = False
        for seed in range(3):
            rng = random.Random(hash((qid, op, seed)) & 0xffffff)
            con.execute("SAVEPOINT s")
            try:
                con.execute("SELECT 1")
                apply_op(con, op, tables, cols, order, rng)
                x, y = run(con, a), run(con, b)
                if x is not None and y is not None and x != y:
                    hit = True
            except Exception:
                pass
            con.execute("ROLLBACK TO s"); con.execute("RELEASE s")
            if hit:
                break
        res["exposed"][op] = hit
    con.close()
    return res


def main():
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    uiuc = {int(e["question_id"]): e for e in json.load(open(a5.UIUC))}
    pairs = []
    pairs_ctrl = []
    con_cache = {}
    def same(db, a, b):
        c = con_cache.setdefault(db, sqlite3.connect(f"file:{os.path.join(DBDIR, db + '.sqlite')}?mode=ro", uri=True))
        c.create_function("IIF", 3, lambda c_, x, y: x if c_ else y)
        ra, rb = run(c, a, 60), run(c, b, 60)
        return ra is not None and rb is not None and ra == rb
    for q_, e in uiuc.items():
        a, b = t0[q_]["SQL"], e["SQL"]
        if a5.norm(a) != a5.norm(b) and same(e["db_id"], a, b):
            pairs.append((e["db_id"], a, b, "mini", q_))
    old = {e["question_id"]: e for e in json.load(open(f"{EXT}/bird_dev_20240627/dev.json"))}
    raw = open(f"{EXT}/bird_dev_20251106/dev.json").read().strip()
    newl = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]
    rng = random.Random(5)
    for e in newl:
        q_ = e["question_id"]; a, b = old[q_]["SQL"], e["SQL"]
        if a5.norm(a) != a5.norm(b):
            if q_ not in t0 and same(e["db_id"], a, b):
                pairs.append((e["db_id"], a, b, "dev_heldout", q_))
        elif rng.random() < 0.08:
            try:
                regen = sqlglot.parse_one(a, read="sqlite").sql(dialect="sqlite")
                pairs_ctrl.append((e["db_id"], a, regen, "control", q_))
            except Exception:
                pass
    print("pairs", len(pairs), "control", len(pairs_ctrl), flush=True)
    with mp.Pool(10) as pool:
        out = pool.map(test_pair, pairs + pairs_ctrl, chunksize=2)
    summ = {}
    for tag in ("mini", "dev_heldout", "control"):
        s = [r for r in out if r["tag"] == tag and r["base_ok"] and r["baseline_equal"]]
        d = {"n_tested": len(s), "n_all": sum(r["tag"] == tag for r in out), "any_exposed": sum(any(r["exposed"].values()) for r in s)}
        for op in OPS:
            d[op] = sum(r["exposed"].get(op, False) for r in s)
        d["pct_any"] = round(100 * d["any_exposed"] / max(1, len(s)), 1)
        summ[tag] = d
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": out}, open(os.path.join(HERE, "results", "mutated_states.json"), "w"), indent=0)

if __name__ == "__main__":
    main()
