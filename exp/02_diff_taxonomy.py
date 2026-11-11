#!/usr/bin/env python3
"""
CHI2 Exp 02 -- what did the corrections DO, structurally?

For each of the 134 SQL-level corrections (original_SQL -> SQL), diff the
ASTs and count change categories. The headline statistic: how many
corrections ADDED nesting depth (the expert wrote too shallow) versus
removed it (wrote too deep) versus left depth unchanged.

Categories (aligned with UIUC's E1-E4 where possible):
  depth+ / depth-      nesting depth increased / decreased by the fix
  table+ / table-      tables added / removed (join structure wrong)
  distinct+            DISTINCT added (dedup fix -- E2 signature)
  agg_chg              aggregation functions changed (SUM/COUNT/AVG/MIN/MAX set)
  where_chg            WHERE predicate set changed (filter logic)
  order_limit_chg      ORDER BY / LIMIT changed (output spec)
  select_chg           projection changed (wrong column reported)

Usage: python3 02_diff_taxonomy.py
"""
import json, os
from collections import Counter
import sqlglot
from sqlglot import exp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "text_to_sql_benchmarks", "data",
                    "arcwise_plat_full_with_diff.json")
AGGS = (exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max)

def parse(sql):
    return sqlglot.parse_one(sql, read="sqlite")

def sel_depth(t):
    def d(node, dd=0):
        best = dd + (1 if isinstance(node, exp.Select) else 0)
        for c in node.args.values():
            for x in (c if isinstance(c, list) else [c]):
                if isinstance(x, exp.Expression):
                    best = max(best, d(x, dd + (1 if isinstance(node, exp.Select) else 0)))
        return best
    return d(t)

def features(t):
    return {
        "depth": sel_depth(t),
        "tables": frozenset(x.name.lower() for x in t.find_all(exp.Table)),
        "distinct": len(list(t.find_all(exp.Distinct))),
        "aggs": Counter(type(x).__name__ for x in t.find_all(*AGGS)),
        "wheres": frozenset(w.this.sql().lower() for w in t.find_all(exp.Where)),
        "order": " ".join(o.sql().lower() for o in t.find_all(exp.Order)),
        "limit": " ".join(l.sql().lower() for l in t.find_all(exp.Limit)),
        "select": frozenset(s.sql().lower() for sel in t.find_all(exp.Select)
                            for s in sel.expressions),
    }

def main():
    d = json.load(open(DATA))
    cats = Counter()
    depth_delta = Counter()
    per = []
    n = fail = 0
    for e in d:
        if "original_SQL" not in e:
            continue
        n += 1
        try:
            fo, fc = features(parse(e["original_SQL"])), features(parse(e["SQL"]))
        except Exception:
            fail += 1
            continue
        c = []
        dd = fc["depth"] - fo["depth"]
        if dd > 0: c.append("depth+")
        if dd < 0: c.append("depth-")
        depth_delta[dd] += 1
        if fc["tables"] - fo["tables"]: c.append("table+")
        if fo["tables"] - fc["tables"]: c.append("table-")
        if fc["distinct"] > fo["distinct"]: c.append("distinct+")
        if fo["aggs"] != fc["aggs"]: c.append("agg_chg")
        if fo["wheres"] != fc["wheres"]: c.append("where_chg")
        if (fo["order"], fo["limit"]) != (fc["order"], fc["limit"]): c.append("order_limit_chg")
        if fo["select"] != fc["select"]: c.append("select_chg")
        if not c: c.append("cosmetic_only")
        cats.update(c)
        per.append({"qid": e["question_id"], "cats": c, "depth_delta": dd})

    ok = n - fail
    print(f"SQL corrections analyzed: {ok}/{n} (parse fail {fail})\n")
    print("修正动作分布（一条修正可属多类）:")
    for k, v in cats.most_common():
        print(f"  {k:16s} {v:3d}  ({100*v/ok:.0f}%)")
    print("\n修正引起的嵌套深度变化 Δdepth:")
    for k in sorted(depth_delta):
        print(f"  Δ={k:+d}: {depth_delta[k]:3d}")
    up = sum(v for k, v in depth_delta.items() if k > 0)
    dn = sum(v for k, v in depth_delta.items() if k < 0)
    print(f"\n加深(专家写浅了): {up} ({100*up/ok:.0f}%)   "
          f"变浅(写深了): {dn} ({100*dn/ok:.0f}%)   持平: {ok-up-dn}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"n": ok, "cats": dict(cats), "depth_delta": {str(k): v for k, v in depth_delta.items()},
               "per_example": per},
              open(os.path.join(HERE, "results", "diff_taxonomy.json"), "w"),
              indent=1, ensure_ascii=False)
    print("\nwrote results/diff_taxonomy.json")

if __name__ == "__main__":
    main()
