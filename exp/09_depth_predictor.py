#!/usr/bin/env python3
"""
CHI2 Exp 09 (proposal E2-6) -- how well can "this task demands nesting" be
predicted from the question text alone, with a deployable model?

Model: logistic regression on question-side features only (demand markers,
question length, evidence length, marker counts in evidence). Validation:
leave-one-database-out (the model never sees the target database), so the
score reflects transfer to a new schema, not memorized vocabulary.

Reported: ROC AUC, average precision, and precision/recall at the review
budgets used by the routing simulation (top-10%, top-20%).

Usage: python3 09_depth_predictor.py
"""
import json, os
import importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("psim", os.path.join(HERE, "04_policy_sim.py"))
psim = importlib.util.module_from_spec(spec); spec.loader.exec_module(psim)


def auc(y, s):
    order = np.argsort(-s)
    y = y[order]
    P, N = y.sum(), (1 - y).sum()
    tp = fp = 0
    area = 0.0
    prev = None
    for yi, si in zip(y, s[order]):
        if si != prev:
            prev = si
        if yi:
            tp += 1
        else:
            fp += 1
            area += tp
    return area / (P * N)


def main():
    raw = json.load(open(psim.DATA))
    rows = [psim.feats(e) for e in raw]
    dbs = np.array([e["db_id"] for e in raw])
    y = np.array([1.0 if r["d_demand"] >= 2 else 0.0 for r in rows])
    FE = ["markers", "qlen", "elen", "m_ev"]
    X = np.array([[1] + [r[k] for k in FE] for r in rows], float)
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / (X[:, 1:].std(0) + 1e-9)

    score = np.zeros(len(y))
    for db in sorted(set(dbs)):
        te = dbs == db
        w = psim.logistic_fit(X[~te], y[~te])
        score[te] = X[te] @ w
    a = auc(y, score)
    # average precision
    order = np.argsort(-score)
    ys = y[order]
    tp = 0
    ap = 0.0
    for i, yi in enumerate(ys, 1):
        if yi:
            tp += 1
            ap += tp / i
    ap /= y.sum()
    print(f"n={len(y)}, 正类(需嵌套)={int(y.sum())} ({100*y.mean():.1f}%)")
    print(f"留一库 AUC = {a:.3f}   平均查准(AP) = {ap:.3f}  (基线={y.mean():.3f})")
    out = {"n": len(y), "pos": int(y.sum()), "auc_lodo": round(a, 3),
           "ap_lodo": round(ap, 3), "base_rate": round(float(y.mean()), 3),
           "features": FE, "at_budget": {}}
    for k in (0.10, 0.20):
        top = order[: int(k * len(y))]
        prec = y[top].mean()
        rec = y[top].sum() / y.sum()
        out["at_budget"][f"{int(k*100)}%"] = {"precision": round(float(prec), 3),
                                              "recall": round(float(rec), 3)}
        print(f"top-{int(k*100)}% 预算: 查准 {prec:.3f} 查全 {rec:.3f}")
    json.dump(out, open(os.path.join(HERE, "results", "depth_predictor.json"), "w"),
              indent=1)
    print("wrote results/depth_predictor.json")


if __name__ == "__main__":
    main()
