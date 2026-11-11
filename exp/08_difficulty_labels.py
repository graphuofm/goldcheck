#!/usr/bin/env python3
"""
CHI2 Exp 08 (proposal E2-4) -- do the benchmark's own difficulty labels
capture demanded depth, and where do errors sit in the label x depth grid?

BIRD assigns each example a human difficulty label (simple / moderate /
challenging). Three questions:
  1. label vs demanded depth: cross-tab (do labels encode structure?)
  2. error rate per label (do labels predict where gold is wrong?)
  3. the blind cell: error rate on examples labeled simple/moderate that
     nonetheless demand nesting.

Usage: python3 08_difficulty_labels.py
"""
import json, os, zipfile
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("a5", os.path.join(HERE, "05_audit_survival.py"))
a5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a5)
import sqlglot

ORDER = ["simple", "moderate", "challenging"]


def main():
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        mini = json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))
    t0 = {e["question_id"]: e["SQL"] for e in mini}
    lab = {e["question_id"]: e["difficulty"] for e in mini}
    t1 = {e["question_id"]: e["SQL"] for e in
          json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    uiuc = json.load(open(a5.UIUC))

    rows = []
    for e in uiuc:
        qid = int(e["question_id"])
        dd = a5.sel_depth(sqlglot.parse_one(e["SQL"], read="sqlite"))
        b = 0 if dd <= 1 else (1 if dd == 2 else 2)
        err = (a5.norm(t0[qid]) != a5.norm(t1[qid])) or ("original_SQL" in e)
        rows.append({"qid": qid, "label": lab[qid], "bucket": b, "err": err})

    # 1. label x depth cross-tab
    print("标签 × 要求深度（行=标签，列=深度桶 0/1/2）")
    grid = {}
    for L in ORDER:
        cnt = [sum(1 for r in rows if r["label"] == L and r["bucket"] == b)
               for b in range(3)]
        grid[L] = cnt
        n = sum(cnt)
        print(f"  {L:12s} {cnt}  需嵌套比例 {100*(cnt[1]+cnt[2])/n:.1f}%")

    # 2. error rate per label
    print("每个标签的错误率（vs 原始 gold）:")
    per_label = {}
    for L in ORDER:
        R = [r for r in rows if r["label"] == L]
        k = sum(r["err"] for r in R)
        per_label[L] = [k, len(R)]
        print(f"  {L:12s} {k}/{len(R)} = {100*k/len(R):.1f}%")

    # 3. blind cells: labeled simple/moderate but demanding nesting
    blind = [r for r in rows if r["label"] in ("simple", "moderate") and r["bucket"] >= 1]
    kb = sum(r["err"] for r in blind)
    print(f"盲区格（标 simple/moderate 但要求嵌套）: n={len(blind)}, "
          f"错误率 {kb}/{len(blind)} = {100*kb/max(1,len(blind)):.1f}%")
    ch_flat = [r for r in rows if r["label"] == "challenging" and r["bucket"] == 0]
    kc = sum(r["err"] for r in ch_flat)
    print(f"反向格（标 challenging 但无嵌套）: n={len(ch_flat)}, "
          f"错误率 {kc}/{len(ch_flat)} = {100*kc/max(1,len(ch_flat)):.1f}%")

    # depth gradient WITHIN each label (is depth informative beyond the label?)
    print("标签内的深度梯度:")
    within = {}
    for L in ORDER:
        rates = []
        for b in range(3):
            R = [r for r in rows if r["label"] == L and r["bucket"] == b]
            rates.append([sum(r["err"] for r in R), len(R)])
        within[L] = rates
        print(f"  {L:12s} " + " / ".join(
            f"{100*k/n:.0f}%({k}/{n})" if n else "--" for k, n in rates))

    json.dump({"grid": grid, "per_label": per_label,
               "blind_cell": [kb, len(blind)], "challenging_flat": [kc, len(ch_flat)],
               "within_label_gradient": within},
              open(os.path.join(HERE, "results", "difficulty_labels.json"), "w"),
              indent=1)
    print("wrote results/difficulty_labels.json")


if __name__ == "__main__":
    main()
