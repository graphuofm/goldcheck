#!/usr/bin/env python3
"""
CHI2 Exp 15 -- 说人话的名字：换一个数据库，"越深越错"还成立吗？（逐库分解）

反驳目标："梯度是个别数据库造成的"。做法：498 题按 db_id 分组，
每库内部算 不嵌套 vs 要求嵌套 的 SQL 错误率；
报告每库 n、嵌套占比、两档错误率、库内梯度方向。
输出 results/per_database.json。
用法: python3 15_per_database.py
"""
import json, os
from collections import defaultdict
import sqlglot
from sqlglot import exp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "text_to_sql_benchmarks", "data",
                    "arcwise_plat_full_with_diff.json")


def sel_depth(sql):
    tree = sqlglot.parse_one(sql, read="sqlite")

    def depth(node, d=0):
        best = d + (1 if isinstance(node, exp.Select) else 0)
        for c in node.args.values():
            for x in (c if isinstance(c, list) else [c]):
                if isinstance(x, exp.Expression):
                    best = max(best, depth(x, d + (1 if isinstance(node, exp.Select) else 0)))
        return best
    return depth(tree)


def main():
    d = json.load(open(DATA))
    per = defaultdict(list)
    for e in d:
        try:
            dep = sel_depth(e["SQL"])
        except Exception:
            continue
        per[e.get("db_id", "?")].append({"deep": dep >= 2,
                                         "err": "original_SQL" in e})
    out = []
    for db, rows in sorted(per.items(), key=lambda kv: -len(kv[1])):
        shallow = [r for r in rows if not r["deep"]]
        deep = [r for r in rows if r["deep"]]
        se = 100 * sum(r["err"] for r in shallow) / len(shallow) if shallow else None
        de = 100 * sum(r["err"] for r in deep) / len(deep) if deep else None
        out.append({"db": db, "n": len(rows), "n_deep": len(deep),
                    "shallow_err": None if se is None else round(se, 1),
                    "deep_err": None if de is None else round(de, 1)})
        print(f"{db:28s} n={len(rows):3d} deep={len(deep):2d} "
              f"shallow_err={se if se is None else round(se,1)} deep_err={de if de is None else round(de,1)}")
    both = [r for r in out if r["shallow_err"] is not None and r["deep_err"] is not None and r["n_deep"] >= 3]
    up = sum(1 for r in both if r["deep_err"] > r["shallow_err"])
    print(f"\n库内梯度（deep>=3 的库）: {up}/{len(both)} 库 深题错误率更高")
    json.dump({"per_db": out, "dbs_with_gradient": up, "dbs_compared": len(both)},
              open(os.path.join(HERE, "results", "per_database.json"), "w"), indent=1)
    print("DONE exp15")


if __name__ == "__main__":
    main()
