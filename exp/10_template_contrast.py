#!/usr/bin/env python3
"""
CHI2 Exp 10 -- 说人话的名字：同一类题里，要求套层的那些是不是错得更多？

背景：有人会质疑"深题错得多，也许只是因为深题的题材本身难"。
这个实验把题目按"证据注释的模板家族"分组——同一家族的题共享同一个领域
公式（比如同一句年龄怎么算的说明），题材和领域几乎相同——
然后只在家族内部比较：要求嵌套的题 vs 不要求嵌套的题，错误率差多少。

家族的定义：把 evidence 文本归一化（去掉具体数字、具体值），
取前 8 个词做指纹；同指纹且 >=4 道题的算一个家族。
配对检验：对每个同时含有深题和浅题的家族，比较两组错误率；
汇总用分层（Mantel-Haenszel 风格）的比值比。

用法: python3 10_template_contrast.py
"""
import json, math, os, re, zipfile
from collections import defaultdict

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("a5", os.path.join(HERE, "05_audit_survival.py"))
a5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a5)
import sqlglot


def fingerprint(ev):
    t = ev.lower()
    t = re.sub(r"'[^']*'", "V", t)
    t = re.sub(r"\d+(\.\d+)?", "N", t)
    t = re.sub(r"[^a-z ]", " ", t)
    words = t.split()
    return " ".join(words[:8]) if len(words) >= 4 else None


def main():
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e["SQL"]
              for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    t1 = {e["question_id"]: e["SQL"] for e in
          json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    uiuc = json.load(open(a5.UIUC))

    fam = defaultdict(list)
    for e in uiuc:
        qid = int(e["question_id"])
        fp = fingerprint(e.get("evidence") or "")
        if not fp:
            continue
        dd = a5.sel_depth(sqlglot.parse_one(e["SQL"], read="sqlite"))
        err = (a5.norm(t0[qid]) != a5.norm(t1[qid])) or ("original_SQL" in e)
        fam[(e["db_id"], fp)].append({"qid": qid, "deep": dd >= 2, "err": err})

    fams = {k: v for k, v in fam.items() if len(v) >= 4}
    mixed = {k: v for k, v in fams.items()
             if any(r["deep"] for r in v) and any(not r["deep"] for r in v)}
    print(f"家族(>=4题) {len(fams)} 个，其中同时含深浅题的 {len(mixed)} 个")

    num = den = 0.0        # Mantel-Haenszel
    kd = nd = ks = ns = 0
    for k, v in mixed.items():
        a = sum(r["deep"] and r["err"] for r in v)       # deep err
        b = sum(r["deep"] and not r["err"] for r in v)
        c = sum((not r["deep"]) and r["err"] for r in v)
        d = sum((not r["deep"]) and not r["err"] for r in v)
        n = a + b + c + d
        num += a * d / n
        den += b * c / n
        kd += a; nd += a + b; ks += c; ns += c + d
    print(f"家族内合并：深题 {kd}/{nd} = {100*kd/nd:.1f}% 错 vs "
          f"浅题 {ks}/{ns} = {100*ks/ns:.1f}% 错")
    or_mh = num / den if den else float("inf")
    print(f"Mantel-Haenszel 合并比值比 = {or_mh:.2f}")

    out = {"n_families": len(fams), "n_mixed": len(mixed),
           "deep": [kd, nd], "shallow": [ks, ns],
           "or_mh": round(or_mh, 2) if den else None,
           "mixed_families": [
               {"db": k[0], "fp": k[1], "n": len(v),
                "deep_err": [sum(r['deep'] and r['err'] for r in v),
                             sum(r['deep'] for r in v)],
                "shallow_err": [sum((not r['deep']) and r['err'] for r in v),
                                sum(not r['deep'] for r in v)]}
               for k, v in sorted(mixed.items(), key=lambda x: -len(x[1]))]}
    json.dump(out, open(os.path.join(HERE, "results", "template_contrast.json"), "w"),
              indent=1, ensure_ascii=False)
    for m in out["mixed_families"][:8]:
        print(f"  {m['db'][:20]:20s} n={m['n']:2d} 深 {m['deep_err'][0]}/{m['deep_err'][1]}"
              f" 浅 {m['shallow_err'][0]}/{m['shallow_err'][1]}  | {m['fp'][:40]}")
    print("wrote results/template_contrast.json")


if __name__ == "__main__":
    main()
