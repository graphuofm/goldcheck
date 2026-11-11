#!/usr/bin/env python3
"""
ICDE Exp 39 -- downstream effect of state-dependent gold queries on real predictions.

For every prediction file results/sar_local/pred_<tag>_dev_2025.jsonl (38_predict_sql.py), run each
prediction on the official database and compare it with the gold (BIRD execution match) and, on
queries with a live GoldCheck finding, with the other reading:
  tie         : prediction == tie-complete result ("all tied rows")   or
                a different choice of k tied rows ("other tied row")
  fan-out     : prediction == result with COUNT(DISTINCT pk)
  empty answer: prediction returns nothing / only NULL and is therefore scored correct
Then re-score with the reading-tolerant rule of Section X (accept the gold or the other reading).
Output: results/pred_scores.json
"""
import glob, json, os, re, sqlite3
from concurrent.futures import ProcessPoolExecutor
import sqlglot
from sqlglot import exp as E
import goldcheck as G

HERE = os.path.dirname(os.path.abspath(__file__))
DBDIR = os.path.join(HERE, "data", "dev_databases")
GC = json.load(open(os.path.join(HERE, "results", "goldcheck.json")))
D25 = {e["question_id"]: e for e in json.load(open(os.path.join(HERE, "data_ext", "bird_dev_20251106", "dev.json")))}
ROWS = {r["qid"]: r for r in GC["rows"] if r["ver"] == "dev_2025"}
LIVE = ("topk_tie", "topk_null", "count_fanout", "empty_answer")


def distinct_reading(con, tree):
    """SQL of the gold with the first fan-out COUNT(pk) made DISTINCT (same rule as goldcheck.check_fanout)."""
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
        pks = [r[1].lower() for r in con.execute(f"PRAGMA main.table_info({G.q(tb)})").fetchall() if r[5] > 0]
        if len(pks) != 1 or pks[0] != arg.name.lower():
            continue
        t2 = tree.copy(); c2 = list(t2.find_all(E.Count))[i]
        c2.set("this", E.Distinct(expressions=[c2.this.copy()]))
        return t2.sql(dialect="sqlite")
    return None


def empty(rows):
    return rows is not None and (not rows or all(v is None for r in rows for v in r))


def score_one(args):
    qid, psql = args
    e = D25[qid]; flags = ROWS[qid]["flags"]
    con = G.connect(os.path.join(DBDIR, e["db_id"] + ".sqlite"))
    try:
        g = G.run(con, e["SQL"], 60)
        p = G.run(con, psql, 30) if psql else None
        gs, ps = G.as_set(g), G.as_set(p)
        res = {"qid": qid, "correct": ps is not None and ps == gs, "pred_fails": p is None, "flags": flags}
        if not any(f in flags for f in LIVE) or p is None:
            return res
        tree = sqlglot.parse_one(e["SQL"], read="sqlite")
        if "topk_tie" in flags:
            k, ords = G._topk_parts(tree)
            vs = G.as_set(G.run(con, G.tie_complete(tree, k, ords), 60))
            if ps == gs:
                res["tie"] = "gold"
            elif ps == vs:
                res["tie"] = "all_tied"
            elif vs is not None and ps <= vs and len(ps) == len(gs):
                res["tie"] = "other_tied"
            else:
                res["tie"] = "other"
        if "count_fanout" in flags:
            ds = distinct_reading(con, tree)
            dv = G.as_set(G.run(con, ds, 60)) if ds else None
            res["fanout"] = "gold" if ps == gs else ("distinct" if dv is not None and ps == dv else "other")
        if "empty_answer" in flags:
            res["empty"] = "pred_empty" if empty(p) else "pred_nonempty"
        res["tolerant"] = res["correct"] or res.get("tie") in ("all_tied", "other_tied") or res.get("fanout") == "distinct"
        return res
    except Exception as ex:
        return {"qid": qid, "correct": False, "pred_fails": True, "flags": flags, "error": str(ex)[:100]}
    finally:
        con.close()


def main():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "results", "sar_local", "pred_*_dev_2025.jsonl"))):
        tag = os.path.basename(f)[5:-len("_dev_2025.jsonl")]
        preds = {}
        for l in open(f):
            r = json.loads(l)
            if "error" not in r:
                preds[r["qid"]] = r["sql"]
        if len(preds) < len(D25):
            print(tag, "incomplete:", len(preds)); continue
        with ProcessPoolExecutor(10) as ex:
            S = list(ex.map(score_one, sorted(preds.items()), chunksize=8))
        live = [s for s in S if any(x in s["flags"] for x in LIVE)]
        m = {"n": len(S), "acc": round(100 * sum(s["correct"] for s in S) / len(S), 2),
             "acc_tolerant": round(100 * sum(s.get("tolerant", s["correct"]) for s in S) / len(S), 2),
             "pred_fails": sum(s["pred_fails"] for s in S),
             "live_n": len(live), "live_correct": sum(s["correct"] for s in live),
             "live_tolerant": sum(s.get("tolerant", s["correct"]) for s in live)}
        for key, cls in (("tie", ("gold", "all_tied", "other_tied", "other")), ("fanout", ("gold", "distinct", "other")),
                         ("empty", ("pred_empty", "pred_nonempty"))):
            xs = [s[key] for s in S if key in s]
            m[key] = {c: xs.count(c) for c in cls}
        m["empty_scored_correct"] = sum(s["correct"] for s in S if "empty" in s)
        m["by_class_correct"] = {c: [sum(s["correct"] for s in S if c in s["flags"]), sum(1 for s in S if c in s["flags"])]
                                 for c in LIVE}
        m["examples_all_tied"] = [s["qid"] for s in S if s.get("tie") == "all_tied"][:10]
        m["examples_distinct"] = [s["qid"] for s in S if s.get("fanout") == "distinct"][:10]
        out[tag] = m
        print(tag, json.dumps({k: v for k, v in m.items() if not k.startswith("examples")}))
    json.dump(out, open(os.path.join(HERE, "results", "pred_scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
