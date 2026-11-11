#!/usr/bin/env python3
"""
CHI2 Exp 01 -- GO/NO-GO: do expert annotation errors track query structure?

Data: uiuc-kang-lab/text_to_sql_benchmarks (CC BY-SA 4.0),
  data/arcwise_plat_full_with_diff.json -- 498 BIRD Mini-Dev examples.
  An example carries `original_SQL` iff the SQL annotation was corrected
  (=the expert's gold SQL was wrong). 134/498 such SQL-level errors.

Hypothesis (the CHI1 thesis on a third substrate): errors concentrate on
queries whose *schema footprint* is larger -- more tables to join, deeper
nesting -- i.e. deeper hidden-dependency structure.

Kill condition: if the error rate is flat across complexity buckets,
this substrate dies here.

Usage: python3 01_error_vs_complexity.py
"""
import json, os, sys
from collections import Counter
import sqlglot
from sqlglot import exp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "text_to_sql_benchmarks", "data",
                    "arcwise_plat_full_with_diff.json")

def complexity(sql):
    """Structural footprint of the *corrected* (best-truth) SQL."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    tables = {t.name for t in tree.find_all(exp.Table)}
    n_joins = len(list(tree.find_all(exp.Join)))
    n_selects = len(list(tree.find_all(exp.Select)))
    # nesting depth of SELECTs
    def depth(node, d=0):
        best = d + (1 if isinstance(node, exp.Select) else 0)
        for c in node.args.values():
            for x in (c if isinstance(c, list) else [c]):
                if isinstance(x, exp.Expression):
                    best = max(best, depth(x, d + (1 if isinstance(node, exp.Select) else 0)))
        return best
    return {"n_tables": len(tables), "n_joins": n_joins,
            "n_selects": n_selects, "sel_depth": depth(tree)}

def main():
    d = json.load(open(DATA))
    rows, failed = [], 0
    for e in d:
        try:
            c = complexity(e["SQL"])
        except Exception:
            failed += 1
            continue
        rows.append({
            **c,
            "sql_err": "original_SQL" in e,
            "any_err": any(k in e for k in ("original_SQL", "original_question", "original_evidence")),
        })
    n = len(rows)
    print(f"parsed {n}/{len(d)} (failed {failed})")
    print(f"SQL-level error rate overall: {100*sum(r['sql_err'] for r in rows)/n:.1f}%")
    print(f"any-level error rate overall: {100*sum(r['any_err'] for r in rows)/n:.1f}%\n")

    def table(metric, buckets, label):
        print(f"-- {label} --")
        out = []
        for name, pred in buckets:
            sel = [r for r in rows if pred(r[metric])]
            if not sel:
                continue
            se = 100*sum(r["sql_err"] for r in sel)/len(sel)
            ae = 100*sum(r["any_err"] for r in sel)/len(sel)
            print(f"  {name:14s} n={len(sel):3d}  SQL错误率 {se:5.1f}%   任意错误率 {ae:5.1f}%")
            out.append((name, len(sel), se, ae))
        return out

    res = {}
    res["tables"] = table("n_tables",
        [("1 表", lambda v: v == 1), ("2 表", lambda v: v == 2),
         ("3 表", lambda v: v == 3), ("4+ 表", lambda v: v >= 4)],
        "按引用表数（schema 足迹）")
    res["depth"] = table("sel_depth",
        [("无嵌套", lambda v: v <= 1), ("1 层子查询", lambda v: v == 2),
         ("2+ 层", lambda v: v >= 3)],
        "按 SELECT 嵌套深度")
    res["joins"] = table("n_joins",
        [("0 join", lambda v: v == 0), ("1 join", lambda v: v == 1),
         ("2+ join", lambda v: v >= 2)],
        "按 join 数")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"n": n, "buckets": res},
              open(os.path.join(HERE, "results", "error_vs_complexity.json"), "w"),
              indent=1, ensure_ascii=False)
    print("\nwrote results/error_vs_complexity.json")

if __name__ == "__main__":
    main()
