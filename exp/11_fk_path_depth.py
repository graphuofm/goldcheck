#!/usr/bin/env python3
"""
CHI2 Exp 11 -- 说人话的名字：错误多，是因为题目要算得深，
还是因为数据库本身长得复杂？

"深"有两种来源。一种是任务性质：正确答案需要先算中间结果再算最终结果
（查询嵌套）。另一种是环境性质：答案要用的表在数据库里离得远，
要沿外键链走好几步才能连起来（外键路径长）。
这两种"深"可能被混在一起。这个实验把它们分开：

  1. 从官方 dev_tables.json 建每个库的外键图（表 = 点，外键 = 边）。
  2. 对每道题，取正确答案用到的表集合，算连接这些表所需的
     最长最短路径（斯坦纳树的直径近似：表对之间最短路的最大值）。
  3. 把"查询嵌套深度"和"外键路径深度"放进同一个逻辑回归，
     看哪个真正预测标注错误。

用法: python3 11_fk_path_depth.py
"""
import json, math, os, zipfile
from collections import defaultdict, deque

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("a5", os.path.join(HERE, "05_audit_survival.py"))
a5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a5)
spec2 = importlib.util.spec_from_file_location("psim", os.path.join(HERE, "04_policy_sim.py"))
psim = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(psim)
import numpy as np
import sqlglot
from sqlglot import exp as E


def fk_graphs():
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        tabs = json.loads(z.read("minidev/MINIDEV/dev_tables.json"))
    graphs = {}
    for db in tabs:
        names = [t.lower() for t in db["table_names_original"]]
        g = defaultdict(set)
        for a, b in db["foreign_keys"]:
            ta = names[db["column_names_original"][a][0]]
            tb = names[db["column_names_original"][b][0]]
            if ta != tb:
                g[ta].add(tb); g[tb].add(ta)
        graphs[db["db_id"]] = (set(names), g)
    return graphs


def pair_dist(g, a, b):
    if a == b:
        return 0
    seen = {a}
    q = deque([(a, 0)])
    while q:
        x, d = q.popleft()
        for y in g[x]:
            if y == b:
                return d + 1
            if y not in seen:
                seen.add(y); q.append((y, d + 1))
    return None            # 不连通


def main():
    graphs = fk_graphs()
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e["SQL"]
              for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    t1 = {e["question_id"]: e["SQL"] for e in
          json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    uiuc = json.load(open(a5.UIUC))

    rows = []
    for e in uiuc:
        qid = int(e["question_id"])
        tree = sqlglot.parse_one(e["SQL"], read="sqlite")
        tables = {t.name.lower() for t in tree.find_all(E.Table)}
        names, g = graphs[e["db_id"]]
        tables = tables & names
        dists = []
        tl = sorted(tables)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                d = pair_dist(g, tl[i], tl[j])
                if d is not None:
                    dists.append(d)
        fk_depth = max(dists) if dists else 0
        dd = a5.sel_depth(tree)
        err = (a5.norm(t0[qid]) != a5.norm(t1[qid])) or ("original_SQL" in e)
        rows.append({"qid": qid, "fk": fk_depth,
                     "nest": 0 if dd <= 1 else (1 if dd == 2 else 2),
                     "ql": len(e["question"].split()), "err": float(err)})

    import collections
    print("外键路径深度分布:", dict(collections.Counter(r["fk"] for r in rows)))
    for f in sorted({r["fk"] for r in rows}):
        R = [r for r in rows if r["fk"] == f]
        print(f"  外键深度{f}: n={len(R)} 错误率 {100*sum(r['err'] for r in R)/len(R):.1f}%")

    y = np.array([r["err"] for r in rows])
    X = np.array([[1, r["nest"], r["fk"], r["ql"]] for r in rows], float)
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
    w = psim.logistic_fit(X, y)
    mu = 1 / (1 + np.exp(-(X @ w)))
    W = mu * (1 - mu)
    cov = np.linalg.inv(X.T @ (X * W[:, None]))
    names = ["截距", "嵌套深度", "外键路径深度", "题长"]
    out = {"coef": {}}
    for i, nm in enumerate(names):
        z = w[i] / math.sqrt(cov[i, i])
        p = math.erfc(abs(z) / math.sqrt(2))
        out["coef"][nm] = {"beta_std": round(float(w[i]), 3),
                           "z": round(float(z), 2), "p": round(float(p), 4)}
        print(f"  {nm:8s} 标准化β={w[i]:+.3f}  z={z:+.2f}  p={p:.4f}")
    out["fk_dist"] = dict(collections.Counter(r["fk"] for r in rows))
    json.dump(out, open(os.path.join(HERE, "results", "fk_path_depth.json"), "w"),
              indent=1, ensure_ascii=False)
    print("wrote results/fk_path_depth.json")


if __name__ == "__main__":
    main()
