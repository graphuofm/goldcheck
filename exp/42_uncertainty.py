#!/usr/bin/env python3
"""
ICDE Exp 42 -- uncertainty and out-of-sample checks asked for by reviewers.
  * lift of each method with a 95% bootstrap interval (2,000 resamples of queries), per review;
  * the same on held-out Dev only (the 1,036 Dev examples outside Mini-Dev, never inspected when
    the classes were chosen), i.e. an out-of-sample test;
  * Cohen's kappa of the two annotators with a 95% bootstrap interval.
Output: results/uncertainty.json
"""
import json, os, random, importlib.util
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("b35", os.path.join(HERE, "35_baselines.py"))
import sys
b35 = importlib.util.module_from_spec(spec); sys.modules["b35"] = b35; spec.loader.exec_module(b35)
GC = b35.GC; CH = GC["changes"]; LIVE = b35.LIVE
BY = {(r["ver"], r["qid"]): r for r in GC["rows"]}
V = b35.r32.versions()
REV = (("mini_T0", "mini_T1"), ("mini_T1", "mini_T2"), ("dev_2024", "dev_2025"))
rng = random.Random(20261111)


def lift(fl, ch):
    F = [c for f, c in zip(fl, ch) if f]; U = [c for f, c in zip(fl, ch) if not f]
    if not F or not U or sum(U) == 0:
        return None
    return (sum(F) / len(F)) / (sum(U) / len(U))


def boot(fl, ch, B=2000):
    n = len(fl); xs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        v = lift([fl[i] for i in idx], [ch[i] for i in idx])
        if v is not None:
            xs.append(v)
    xs.sort()
    return [round(xs[int(0.025 * len(xs))], 2), round(xs[int(0.975 * len(xs)) - 1], 2)]


def main():
    out = {"lift": {}, "heldout": {}}
    mini_ids = set(V["mini_T0"])
    with Pool(10) as pool:
        for v, n in REV:
            qs = sorted(V[v])
            ch = [CH[f"{v}|{n}|{q}"][1] for q in qs]
            st = pool.map(b35.static_flag, [V[v][q]["SQL"] for q in qs])
            ro = pool.starmap(b35.run_once_flag, [(V[v][q]["db_id"], V[v][q]["SQL"]) for q in qs])
            gc = [any(f in BY[(v, q)]["flags"] for f in LIVE) for q in qs]
            res = {}
            for name, fl in (("static", st), ("run_once", ro), ("goldcheck", gc)):
                res[name] = {"flagged": sum(fl), "changed_flagged": sum(c for f, c in zip(fl, ch) if f),
                             "unflagged": len(fl) - sum(fl), "changed_unflagged": sum(c for f, c in zip(fl, ch) if not f),
                             "lift": round(lift(fl, ch), 2), "ci95": boot(fl, ch)}
            out["lift"][f"{v}->{n}"] = res
            if v == "dev_2024":
                keep = [i for i, q in enumerate(qs) if q not in mini_ids]
                ho = {}
                for name, fl in (("static", st), ("run_once", ro), ("goldcheck", gc)):
                    f2 = [fl[i] for i in keep]; c2 = [ch[i] for i in keep]
                    ho[name] = {"n": len(keep), "flagged": sum(f2), "changed_flagged": sum(c for f, c in zip(f2, c2) if f),
                                "unflagged": len(f2) - sum(f2), "changed_unflagged": sum(c for f, c in zip(f2, c2) if not f),
                                "lift": round(lift(f2, c2), 2), "ci95": boot(f2, c2)}
                out["heldout"] = ho
    A = json.load(open(os.path.join(HERE, "results", "annotation_agreement.json")))
    # kappa CI from the returned sheets
    spec2 = importlib.util.spec_from_file_location("sa", os.path.join(HERE, "..", "review", "score_annotation.py"))
    sa = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(sa)
    (n1, p1), (n2, p2) = sa.FILES
    X, Y = sa.read(p1), sa.read(p2)
    keys = [k for k in X if k in Y]
    a = [X[k][0] for k in keys]; b = [Y[k][0] for k in keys]
    ks = []
    for _ in range(2000):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        k = sa.kappa([a[i] for i in idx], [b[i] for i in idx], sa.CHOICES)
        if k is not None:
            ks.append(k)
    ks.sort()
    out["kappa"] = {"kappa": A["kappa_4"], "agreement": A["agreement"], "n": len(keys),
                    "ci95": [round(ks[int(0.025 * len(ks))], 2), round(ks[int(0.975 * len(ks)) - 1], 2)]}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    json.dump(out, open(os.path.join(HERE, "results", "uncertainty.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
