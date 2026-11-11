#!/usr/bin/env python3
"""
ICDE Exp 35 -- baselines that need no GPU, compared with GoldCheck on the same
releases and against the same human reviews.

  static     : read the SQL only. Flag a top-level ORDER BY ... LIMIT, or a COUNT(col)
               without DISTINCT in a SELECT with joins. Same constructs as GoldCheck,
               no execution, no evidence.
  run_once   : what an annotator sees when running the gold once. Flag if the query
               fails, returns nothing, returns duplicate rows, or returns a NULL.
  random_edit: test-suite style search for latent top-k witnesses. For the same
               top-k queries on which GoldCheck attempted a targeted witness, change
               the sort key of RANDOM rows of the key table to the boundary value
               (one cell per try) and count how many tries find a witness.
Output: results/baselines.json
"""
import json, os, random, sqlite3, time
import sqlglot
from sqlglot import exp as E
import goldcheck as G

HERE = os.path.dirname(os.path.abspath(__file__))
DBDIR = os.path.join(HERE, "data", "dev_databases")
GC = json.load(open(os.path.join(HERE, "results", "goldcheck.json")))
spec = __import__("importlib.util").util.spec_from_file_location("r32", os.path.join(HERE, "32_run_goldcheck.py"))
r32 = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(r32)
REVIEWS = (("mini_T0", "mini_T1"), ("mini_T1", "mini_T2"), ("dev_2024", "dev_2025"))
LIVE = ("topk_tie", "topk_null", "count_fanout", "empty_answer")


def static_flag(sql):
    try:
        t = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return False
    if isinstance(t, E.Select) and t.args.get("order") is not None and t.args.get("limit") is not None:
        return True
    for c in t.find_all(E.Count):
        if isinstance(c.this, E.Column):
            s = c.find_ancestor(E.Select)
            if s is not None and s.args.get("joins"):
                return True
    return False


def run_once_flag(db, sql):
    con = G.connect(os.path.join(DBDIR, db + ".sqlite"))
    try:
        r = G.run(con, sql, 60)
    finally:
        con.close()
    if r is None or not r:
        return True
    return len(set(r)) < len(r) or any(v is None for row in r for v in row)


def random_edit(args):
    db, sql, seed = args
    """Return the number of random one-cell tries (<=100) until a witness, or None."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    k, ords = G._topk_parts(tree)
    v = G.tie_complete(tree, k, ords)
    alias, table, kcols = G._key_table(tree, ords)
    con = G.connect(os.path.join(DBDIR, db + ".sqlite"))
    try:
        real = {r[0].lower(): r[0] for r in con.execute("SELECT name FROM main.sqlite_master WHERE type='table'")}
        table = real[table.lower()]
        rows = G.run(con, G._probe(tree, ords), limit=k)
        boundary = rows[k - 1][-len(ords):]
        names = [i[1] for i in con.execute(f"PRAGMA main.table_info({G.q(table)})")]
        con.execute(f"CREATE TEMP TABLE {G.q(table)} AS SELECT * FROM main.{G.q(table)} WHERE 0")
        con.execute(f"INSERT INTO temp.{G.q(table)} (rowid, {', '.join(G.q(n) for n in names)}) SELECT rowid, "
                    f"{', '.join(G.q(n) for n in names)} FROM main.{G.q(table)}")
        rids = [r[0] for r in con.execute(f"SELECT rowid FROM temp.{G.q(table)}")]
        lowmap = {n.lower(): n for n in names}
        rng = random.Random(seed)
        gsql = tree.sql(dialect="sqlite")
        for t in range(1, 101):
            rid = rng.choice(rids)
            con.execute("SAVEPOINT w")
            try:
                con.execute(f"UPDATE temp.{G.q(table)} SET {', '.join(G.q(lowmap[c.lower()]) + '=?' for c in kcols)} WHERE rowid=?",
                            (*boundary, rid))
                a, b = G.as_set(G.run(con, gsql, 10)), G.as_set(G.run(con, v, 10))
                if a is not None and b is not None and a != b:
                    return t
            finally:
                con.execute("ROLLBACK TO w"); con.execute("RELEASE w")
        return None
    except Exception:
        return "error"
    finally:
        con.close()


def rate(flagged, ch, v, n, i):
    qs = list(flagged)
    k = sum(ch[f"{v}|{n}|{q}"][i] for q in qs)
    return k, len(qs)


def main():
    V = r32.versions()
    ch = GC["changes"]
    by = {(r["ver"], r["qid"]): r for r in GC["rows"]}
    import multiprocessing as mp
    out = {"agreement": {}, "random_edit": {}}
    with mp.Pool(10) as pool:
        for v, n in REVIEWS:
            qs = sorted(V[v])
            st = dict(zip(qs, pool.map(static_flag, [V[v][q]["SQL"] for q in qs])))
            ro = dict(zip(qs, pool.starmap(run_once_flag, [(V[v][q]["db_id"], V[v][q]["SQL"]) for q in qs])))
            gc = {q: any(f in by[(v, q)]["flags"] for f in LIVE) for q in qs}
            tot = sum(ch[f"{v}|{n}|{q}"][1] for q in qs)
            res = {}
            for name, fl in (("static", st), ("run_once", ro), ("goldcheck_live", gc)):
                F = [q for q in qs if fl[q]]; U = [q for q in qs if not fl[q]]
                k1, n1 = rate(F, ch, v, n, 1); k0, n0 = rate(U, ch, v, n, 1)
                res[name] = {"flagged": n1, "out_changed_flagged": k1, "pct_flagged": round(100 * k1 / max(n1, 1), 1),
                             "pct_unflagged": round(100 * k0 / max(n0, 1), 1), "lift": round((k1 / max(n1, 1)) / max(k0 / max(n0, 1), 1e-9), 2),
                             "recall_out": f"{k1}/{tot}", "recall_pct": round(100 * k1 / max(tot, 1), 1)}
            out["agreement"][f"{v}->{n}"] = res
        # random edits vs targeted, on Dev 2025 queries where GoldCheck attempted a witness
        cand = [r for r in GC["rows"] if r["ver"] == "dev_2025" and r.get("topk", {}).get("latent") in ("witness", "no_witness")]
        d25 = V["dev_2025"]
        t0 = time.time()
        tries = pool.map(random_edit, [(d25[r["qid"]]["db_id"], d25[r["qid"]]["SQL"], r["qid"]) for r in cand])
        secs = time.time() - t0
    targ = {r["qid"]: r["topk"]["latent"] == "witness" for r in cand}
    rr = dict(zip([r["qid"] for r in cand], tries))
    ok = [t for t in tries if isinstance(t, int)]
    out["random_edit"] = {"attempted": len(cand), "targeted_found": sum(targ.values()),
                          "random_found_within_1": sum(t <= 1 for t in ok), "random_found_within_10": sum(t <= 10 for t in ok),
                          "random_found_within_100": len(ok), "random_errors": sum(t == "error" for t in tries),
                          "random_found_but_targeted_missed": sum(isinstance(rr[q], int) and not targ[q] for q in rr),
                          "targeted_found_but_random_missed": sum(targ[q] and not isinstance(rr[q], int) for q in rr),
                          "median_tries_when_found": sorted(ok)[len(ok) // 2] if ok else None, "wall_secs": round(secs, 1)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "results", "baselines.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
