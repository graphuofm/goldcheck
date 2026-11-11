#!/usr/bin/env python3
"""
CHI2 Exp 12 -- 说人话的名字：不用人工，几条便宜的检查规则能拦住多少错答案？

想法：标注平台完全可以在专家提交 gold 时自动跑一遍查询，用几条谁都写得出的
规则把可疑的标注拦下来复查。这个实验测这些规则在真实数据上的效果。

规则（对全部 498 条原始 gold 执行后判断）：
  R1 跑不通       查询在官方库上报错
  R2 结果为空     执行成功但一行都没有
  R3 结果有重复行  同样一行出现两次（漏 DISTINCT 的典型症状）
  R4 结果含 NULL   输出里有空值（漏守卫的典型症状）

对每条规则报：命中多少条、命中里多少是真错误（查准）、
201 个真错误里逮住多少（查全）。再报四条规则并联的效果。

用法: python3 12_cheap_checks.py
"""
import json, os, zipfile
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("a5", os.path.join(HERE, "05_audit_survival.py"))
a5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a5)
spec2 = importlib.util.spec_from_file_location("e7x", os.path.join(HERE, "07_execution_consequences.py"))
e7x = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(e7x)


def main():
    e7x.ensure_dbs()
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        mini = json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))
    t1 = {e["question_id"]: e["SQL"] for e in
          json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    uiuc = json.load(open(a5.UIUC))
    errset = set()
    for e in uiuc:
        qid = int(e["question_id"])
        if a5.norm(t1[qid]) != a5.norm({m["question_id"]: m["SQL"] for m in mini}[qid]) \
           or "original_SQL" in e:
            errset.add(qid)
    # (recompute properly below to avoid rebuilding dict per loop)
    t0 = {m["question_id"]: m["SQL"] for m in mini}
    errset = set()
    for e in uiuc:
        qid = int(e["question_id"])
        if a5.norm(t0[qid]) != a5.norm(t1[qid]) or "original_SQL" in e:
            errset.add(qid)

    rows = []
    for i, m in enumerate(mini):
        qid = m["question_id"]
        dbf = os.path.join(e7x.DBDIR, m["db_id"] + ".sqlite")
        status, res, ncols = e7x.run_query(dbf, m["SQL"])
        flags = {
            "R1_error": status != "ok",
            "R2_empty": status == "ok" and len(res) == 0,
            "R3_dupes": status == "ok" and res is not None and
                        len(res) != len({tuple(r) for r in res}),
            "R4_null": status == "ok" and res is not None and
                       any(v is None for r in res[:5000] for v in r),
        }
        rows.append({"qid": qid, "err": qid in errset, **flags})
        if (i + 1) % 100 == 0:
            print(f"{i+1}/498", flush=True)

    n_err = sum(r["err"] for r in rows)
    out = {"n": len(rows), "n_err": n_err, "rules": {}}
    names = {"R1_error": "跑不通", "R2_empty": "结果为空",
             "R3_dupes": "结果有重复行", "R4_null": "结果含NULL"}
    union = [False] * len(rows)
    print(f"\n498 条原始 gold，其中真错误 {n_err}")
    for k, nm in names.items():
        hit = [r for r in rows if r[k]]
        tp = sum(r["err"] for r in hit)
        out["rules"][k] = {"名称": nm, "命中": len(hit), "真错误": tp,
                           "查准": round(tp / len(hit), 3) if hit else None,
                           "查全": round(tp / n_err, 3)}
        print(f"  {nm:8s} 命中 {len(hit):3d}  查准 {tp}/{len(hit)}"
              f" = {100*tp/max(1,len(hit)):.0f}%  查全 {tp}/{n_err} = {100*tp/n_err:.0f}%")
        for i, r in enumerate(rows):
            union[i] = union[i] or r[k]
    tp = sum(r["err"] for i, r in enumerate(rows) if union[i])
    nh = sum(union)
    out["union"] = {"命中": nh, "真错误": tp,
                    "查准": round(tp / nh, 3), "查全": round(tp / n_err, 3)}
    print(f"  四条并联   命中 {nh:3d}  查准 {tp}/{nh} = {100*tp/nh:.0f}%"
          f"  查全 {tp}/{n_err} = {100*tp/n_err:.0f}%")
    json.dump(out, open(os.path.join(HERE, "results", "cheap_checks.json"), "w"),
              indent=1, ensure_ascii=False)
    print("wrote results/cheap_checks.json")


if __name__ == "__main__":
    main()
