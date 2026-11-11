#!/usr/bin/env python3
"""
goldcheck -- find gold SQL queries whose official result is an accident of the
database state, and return an executable witness for each finding.

A finding is reported only with evidence produced by running SQL:

  topk_tie      the query cuts a ranked list with ORDER BY ... LIMIT k, and on the
                official database rows tied with the k-th row are cut off although
                they produce different output. Witness: the tie-complete rewrite
                (RANK() <= k) returns a different result on the official database.
  topk_null     the first ranked row has a NULL sort key (the "top" is a missing value).
  topk_latent   the official data happen to have no tie at the cut, but setting ONE
                cell (the sort key of the first row below the cut) to the boundary
                value makes the gold and its tie-complete rewrite disagree.
                Witness: (table, rowid, column, old value, new value) and both results.
  count_fanout  COUNT(t.pk) without DISTINCT in a SELECT with joins, and adding
                DISTINCT changes the official result: the same entity is counted
                several times. Witness: both values.
  empty_answer  the gold answer is empty or contains only NULLs, so a system that
                returns nothing is scored correct.

All edits happen in TEMP shadow tables on a read-only connection and are rolled
back. The original database files are never written.
"""
import sqlite3, threading, time
import sqlglot
from sqlglot import exp as E

LIVE = ("topk_tie", "topk_null", "count_fanout", "empty_answer")
ALL = LIVE + ("topk_latent",)


def connect(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None, timeout=10)
    con.create_function("IIF", 3, lambda c, x, y: x if c else y)   # SQLite < 3.32 lacks IIF
    return con


def run(con, sql, secs=30, limit=None):
    t = threading.Timer(secs, con.interrupt); t.start()
    try:
        cur = con.execute(sql)
        return cur.fetchmany(limit) if limit else cur.fetchall()
    except Exception:
        return None
    finally:
        t.cancel()


def as_set(rows):
    if rows is None:
        return None
    return frozenset(tuple(round(x, 6) if isinstance(x, float) else x for x in r) for r in rows)


def q(name):
    return '"' + name.replace('"', '""') + '"'


# ---------------------------------------------------------------- top-k ------
def _topk_parts(tree):
    if not isinstance(tree, E.Select) or tree.args.get("offset") is not None:
        return None
    order, limit = tree.args.get("order"), tree.args.get("limit")
    if order is None or limit is None:
        return None
    try:
        k = int(limit.expression.name)
    except Exception:
        return None
    aliases = {s.alias: s.this for s in tree.expressions if isinstance(s, E.Alias)}
    ords = []
    for o in order.expressions:
        o2 = o.copy(); e = o2.this
        if isinstance(e, E.Column) and not e.table and e.name in aliases:
            o2.set("this", aliases[e.name].copy())
        ords.append(o2)
    return k, ords


def tie_complete(tree, k, ords):
    """Rewrite ORDER BY o LIMIT k as RANK() OVER (ORDER BY o) <= k. None if unsupported."""
    if tree.args.get("distinct") or any(isinstance(x, E.Star) for x in tree.expressions):
        return None
    sub = tree.copy(); sub.set("order", None); sub.set("limit", None)
    outs = [E.alias_((x.this if isinstance(x, E.Alias) else x).copy(), f"__c{i}") for i, x in enumerate(sub.expressions)]
    ostr = ", ".join(o.sql(dialect="sqlite") for o in ords)
    rank = sqlglot.parse_one(f"SELECT RANK() OVER (ORDER BY {ostr}) AS __r", read="sqlite").expressions[0]
    sub.set("expressions", outs + [rank])
    cols = ", ".join(f"__c{i}" for i in range(len(outs)))
    return f"SELECT {cols} FROM ({sub.sql(dialect='sqlite')}) WHERE __r <= {k}"


def _probe(tree, ords, extra=()):
    p = tree.copy(); p.set("limit", None)
    p.set("expressions", list(p.expressions) + [E.alias_(o.this.copy(), f"__k{i}") for i, o in enumerate(ords)] + list(extra))
    return p.sql(dialect="sqlite")


def _key_table(tree, ords):
    """All sort keys plain columns of one base table -> (alias, table, [columns])."""
    amap = {t.alias_or_name.lower(): t.name for t in tree.find_all(E.Table)}
    cols, al = [], None
    for o in ords:
        c = o.this
        if not isinstance(c, E.Column):
            return None
        a = c.table.lower() if c.table else (list(amap)[0] if len(amap) == 1 else None)
        if a is None or a not in amap or (al is not None and a != al):
            return None
        al = a; cols.append(c.name)
    return al, amap[al], cols


def check_topk(con, tree, out, base, latent=True):
    parts = _topk_parts(tree)
    if parts is None:
        return
    k, ords = parts
    out["topk"] = {"k": k}
    rows = run(con, _probe(tree, ords), limit=k + 1)
    if rows and rows[0][len(rows[0]) - len(ords)] is None:
        out["flags"].append("topk_null"); out["topk"]["first_key"] = None
    v = tie_complete(tree, k, ords)
    if v is None:
        out["topk"]["status"] = "unsupported_rewrite"; return
    vres = as_set(run(con, v))
    if vres is None:
        out["topk"]["status"] = "rewrite_fails"; return
    if vres != base:
        if vres >= base:
            out["flags"].append("topk_tie")
            out["topk"]["status"] = "live_tie"
            out["topk"]["tie_complete_rows"] = len(vres); out["topk"]["gold_rows"] = len(base)
            out["topk"]["example_cut_off"] = [list(r) for r in list(vres - base)[:3]]
        else:
            out["topk"]["status"] = "rewrite_mismatch"
        return
    out["topk"]["status"] = "pinned"
    if not latent:
        return
    kt = _key_table(tree, ords)
    if kt is None:
        out["topk"]["latent"] = "key_not_plain_column"; return
    alias, table, kcols = kt
    real = {r[0].lower(): r[0] for r in con.execute("SELECT name FROM main.sqlite_master WHERE type='table'").fetchall()}
    if table.lower() not in real:
        out["topk"]["latent"] = "key_not_base_column"; return
    table = real[table.lower()]
    rows = run(con, _probe(tree, ords, [E.alias_(E.column("rowid", table=alias), "__rid")]), limit=k + 6)
    if rows is None:
        out["topk"]["latent"] = "probe_fails"; return
    if len(rows) <= k:
        out["topk"]["latent"] = "nothing_below_cut"; return
    nk = len(ords)
    boundary = rows[k - 1][-1 - nk:-1]
    try:
        info = con.execute(f"PRAGMA main.table_info({q(table)})").fetchall()
        names = [i[1] for i in info]
        con.execute(f"CREATE TEMP TABLE {q(table)} AS SELECT * FROM main.{q(table)} WHERE 0")
        con.execute(f"INSERT INTO temp.{q(table)} (rowid, {', '.join(q(n) for n in names)}) "
                    f"SELECT rowid, {', '.join(q(n) for n in names)} FROM main.{q(table)}")
    except Exception as ex:
        out["topk"]["latent"] = "shadow_failed"; return
    try:
        real = {n.lower(): n for n in names}
        for cand in rows[k:k + 6]:
            rid = cand[-1]
            con.execute("SAVEPOINT w")
            try:
                sets = ", ".join(f"{q(real[c.lower()])} = ?" for c in kcols)
                old = con.execute(f"SELECT {', '.join(q(real[c.lower()]) for c in kcols)} FROM temp.{q(table)} WHERE rowid = ?", (rid,)).fetchone()
                con.execute(f"UPDATE temp.{q(table)} SET {sets} WHERE rowid = ?", (*boundary, rid))
                g2, v2 = as_set(run(con, tree.sql(dialect="sqlite"))), as_set(run(con, v))
                if g2 is not None and v2 is not None and g2 != v2:
                    out["flags"].append("topk_latent")
                    out["topk"]["latent"] = "witness"
                    out["topk"]["witness"] = {"table": table, "rowid": rid, "columns": kcols, "old": list(old or []),
                                              "new": list(boundary), "gold_rows": len(g2), "tie_complete_rows": len(v2)}
                    return
            finally:
                con.execute("ROLLBACK TO w"); con.execute("RELEASE w")
        out["topk"]["latent"] = "no_witness"
    finally:
        con.execute(f"DROP TABLE temp.{q(table)}")


# ---------------------------------------------------------------- COUNT fan-out
def check_fanout(con, tree, out, base):
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
        if tb not in cache:
            try:
                pks = [r[1].lower() for r in con.execute(f"PRAGMA main.table_info({q(tb)})").fetchall() if r[5] > 0]
            except Exception:
                pks = []
            cache[tb] = pks[0] if len(pks) == 1 else None
        if cache[tb] != arg.name.lower():
            continue
        t2 = tree.copy(); c2 = list(t2.find_all(E.Count))[i]
        c2.set("this", E.Distinct(expressions=[c2.this.copy()]))
        v = run(con, t2.sql(dialect="sqlite"))
        if v is not None and as_set(v) != base:
            out["flags"].append("count_fanout")
            out["fanout"] = {"column": f"{tb}.{arg.name}", "gold": [list(r) for r in list(base)[:3]],
                             "with_distinct": [list(r) for r in v[:3]]}
            return


# ---------------------------------------------------------------- driver ------
def check(db_path, sql, latent=True):
    t0 = time.perf_counter()
    out = {"flags": [], "status": "ok"}
    con = connect(db_path)
    try:
        orig = run(con, sql)
        if orig is None:
            out["status"] = "gold_fails"; return out
        if not orig or all(v is None for r in orig for v in r):
            out["flags"].append("empty_answer")
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
            base = as_set(run(con, tree.sql(dialect="sqlite")))
        except Exception:
            out["status"] = "parse_fail"; return out
        if base != as_set(orig):
            out["status"] = "roundtrip_differs"; return out
        check_topk(con, tree, out, base, latent)
        check_fanout(con, tree, out, base)
        return out
    finally:
        con.close()
        out["secs"] = round(time.perf_counter() - t0, 3)
