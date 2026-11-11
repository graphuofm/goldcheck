#!/usr/bin/env python3
"""
CHI2 Exp 14 -- 说人话的名字：排行榜在深题上的名次，还可信吗？

材料：UIUC 仓库里 15 个 text-to-SQL 系统在 BIRD Dev-100 上的逐题预测
（CHESS 被剔除：其 results 文件与 gold 逐字相同，是占位拷贝而非预测），
修正前(dev_old.json)与修正后(dev.json)各一份；以及两版 gold。
我们在官方数据库上把每个预测真的执行、与对应 gold 的执行结果比对
（BIRD 的执行匹配口径），然后：

  1. 每个系统算两个总分（按旧 gold / 按新 gold）；
  2. 按"修正后 gold 的要求深度"分层，看每层里系统排名在两版 gold 之间
     变动多大（Kendall tau、平均名次位移）；
  3. 直接量化：深层的分数差异有多少来自 gold 而不是系统。

用法: python3 14_leaderboard_depth.py
"""
import json, os, urllib.request
from collections import defaultdict

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("a5", os.path.join(HERE, "05_audit_survival.py"))
a5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a5)
spec2 = importlib.util.spec_from_file_location("e7x", os.path.join(HERE, "07_execution_consequences.py"))
e7x = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(e7x)
import sqlglot

AGENTS = ["Alpha-SQL", "CodeS", "Contextual-SQL", "DAIL-SQL", "DIN-SQL",
          "DTS-SQL", "E-SQL", "GSR", "GenaSQL", "MAC-SQL", "OmniSQL",
          "OpenSearch-SQL", "RSL-SQL", "TA-SQL", "csc_sql"]
RAW = "https://raw.githubusercontent.com/uiuc-kang-lab/text_to_sql_benchmarks/main"
ADIR = os.path.join(HERE, "data", "agent_preds")
os.makedirs(ADIR, exist_ok=True)


def get_preds(agent, which):
    fn = os.path.join(ADIR, f"{agent}_{which}.json")
    if not os.path.exists(fn):
        url = f"{RAW}/text_to_sql_agents/{agent}/results/{which}.json"
        try:
            urllib.request.urlretrieve(url, fn)
        except Exception as e:
            print(f"  {agent}/{which}: 下载失败 {e}")
            return None
    try:
        return json.load(open(fn))
    except Exception:
        return None


def parse_entry(v):
    sql = v.split("\t----- bird -----")[0].strip()
    db = v.split("\t")[-1].strip()
    return sql, db


CACHE = {}
def run(db, sql):
    key = (db, sql)
    if key not in CACHE:
        st, rows, _ = e7x.run_query(os.path.join(e7x.DBDIR, db + ".sqlite"), sql)
        CACHE[key] = (st, frozenset(map(tuple, rows)) if st == "ok" and rows is not None else None)
    return CACHE[key]


def main():
    e7x.ensure_dbs()
    gold_new = {k: parse_entry(v) for k, v in
                json.load(open(os.path.join(HERE, "data", "dev100", "dev.json"))).items()}
    gold_old = {k: parse_entry(v) for k, v in
                json.load(open(os.path.join(HERE, "data", "dev100", "dev_old.json"))).items()}
    qids = sorted(gold_new, key=int)
    # 深度分层（按修正后 gold）
    bucket = {}
    for q in qids:
        sql, db = gold_new[q]
        try:
            dd = a5.sel_depth(sqlglot.parse_one(sql, read="sqlite"))
        except Exception:
            dd = 1
        bucket[q] = 0 if dd <= 1 else (1 if dd == 2 else 2)
    import collections
    print("深度分层:", dict(collections.Counter(bucket.values())),
          "(0=不嵌套,1=需嵌套)")

    # gold 执行缓存
    for gset in (gold_new, gold_old):
        for q, (sql, db) in gset.items():
            run(db, sql)

    scores = {}          # agent -> {"old": {qid: 0/1}, "new": {...}}
    for ag in AGENTS:
        po = get_preds(ag, "dev_old")
        pn = get_preds(ag, "dev")
        if not po or not pn:
            continue
        rec = {"old": {}, "new": {}}
        for q in qids:
            for tag, preds, gold in (("old", po, gold_old), ("new", pn, gold_new)):
                if q not in preds:
                    rec[tag][q] = 0
                    continue
                psql, pdb = parse_entry(preds[q])
                gsql, gdb = gold[q]
                st_p, res_p = run(gdb, psql)
                st_g, res_g = run(gdb, gsql)
                rec[tag][q] = int(st_p == "ok" and st_g == "ok" and res_p == res_g)
        scores[ag] = rec
        print(f"  {ag:16s} 按旧gold {sum(rec['old'].values())}/100"
              f"  按新gold {sum(rec['new'].values())}/100", flush=True)

    # 分层排名对比
    def ranks(tag, stratum):
        sc = {ag: sum(v for q, v in scores[ag][tag].items()
                      if bucket[q] == stratum) for ag in scores}
        order = sorted(sc, key=lambda a: -sc[a])
        return {a: i for i, a in enumerate(order)}, sc

    out = {"agents": {ag: {"old_total": sum(s["old"].values()),
                           "new_total": sum(s["new"].values())}
                      for ag, s in scores.items()}}
    for stratum, name in ((0, "不嵌套"), (1, "一层嵌套"), (2, "两层以上")):
        r_old, sc_old = ranks("old", stratum)
        r_new, sc_new = ranks("new", stratum)
        moves = [abs(r_old[a] - r_new[a]) for a in scores]
        n = len(scores)
        conc = disc = 0
        ags = list(scores)
        for i in range(n):
            for j in range(i + 1, n):
                d1 = r_old[ags[i]] - r_old[ags[j]]
                d2 = r_new[ags[i]] - r_new[ags[j]]
                if d1 * d2 > 0:
                    conc += 1
                elif d1 * d2 < 0:
                    disc += 1
        tau = (conc - disc) / max(1, conc + disc)
        nq = sum(1 for q in qids if bucket[q] == stratum)
        print(f"{name} (题数{nq}): 平均名次位移 {sum(moves)/n:.2f}, 最大 {max(moves)}, "
              f"Kendall tau(旧vs新 gold 排名) = {tau:.3f}")
        out[name] = {"n_questions": nq, "mean_rank_move": round(sum(moves)/n, 2),
                     "max_rank_move": max(moves), "kendall_tau": round(tau, 3),
                     "scores_old": sc_old, "scores_new": sc_new}
    json.dump(out, open(os.path.join(HERE, "results", "leaderboard_depth.json"), "w"),
              indent=1, ensure_ascii=False)
    print("wrote results/leaderboard_depth.json")


if __name__ == "__main__":
    main()
