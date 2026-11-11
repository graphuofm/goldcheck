#!/usr/bin/env python3
"""
ICDE Exp 27 -- two closing checks.
  A. Pure gold effect: the SAME prediction file graded against both gold versions.
     (Exp 14 graded old predictions on old gold and new predictions on new gold, so
      it mixed a gold change with a prediction change.)
  B. How much of the error is about the schema rather than the query's shape?
     Nested-model comparison: database identity vs pre-review predictors.

Usage: python3 27_gold_effect_and_schema.py
"""
import importlib.util, json, math, os, zipfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
L = _load("L14", "14_leaderboard_depth.py"); v = _load("v20", "20_validity_audit.py")
n = _load("n23", "23_noncircular.py")

def tau(a, b):
    k = list(a); c = d = 0
    for i in range(len(k)):
        for j in range(i + 1, len(k)):
            x = a[k[i]] - a[k[j]]; y = b[k[i]] - b[k[j]]
            c += x * y > 0; d += x * y < 0
    return (c - d) / (len(k) * (len(k) - 1) / 2)

def main():
    L.e7x.ensure_dbs()
    gn = {k: L.parse_entry(x) for k, x in json.load(open(os.path.join(HERE, "data", "dev100", "dev.json"))).items()}
    go = {k: L.parse_entry(x) for k, x in json.load(open(os.path.join(HERE, "data", "dev100", "dev_old.json"))).items()}
    qs = sorted(gn, key=int); res = {}; same_text = {}
    for which in ("dev_old", "dev"):
        sc = {}
        for ag in L.AGENTS:
            p = L.get_preds(ag, which)
            if not p: continue
            sc[ag] = {"old": {}, "new": {}}
            for q in qs:
                db = gn[q][1]
                for tag, g in (("old", go), ("new", gn)):
                    ok = 0
                    if q in p:
                        s, _ = L.parse_entry(p[q]); a, ra = L.run(db, s); b, rb = L.run(db, g[q][0]); ok = int(a == "ok" and b == "ok" and ra == rb)
                    sc[ag][tag][q] = ok
        ags = list(sc)
        A = {a: sum(sc[a]["old"].values()) for a in ags}; B = {a: sum(sc[a]["new"].values()) for a in ags}
        rng = np.random.default_rng(2); boots = []
        for _ in range(2000):
            pk = rng.choice(qs, len(qs))
            boots.append(tau({a: sum(sc[a]["old"][q] for q in pk) for a in ags}, {a: sum(sc[a]["new"][q] for q in pk) for a in ags}))
        ra = {a: i for i, a in enumerate(sorted(ags, key=lambda a: (-A[a], a)))}; rb = {a: i for i, a in enumerate(sorted(ags, key=lambda a: (-B[a], a)))}
        res[which] = {"tau": round(tau(A, B), 2), "ci95": [round(float(x), 2) for x in np.percentile(boots, [2.5, 97.5])], "scores_old_gold": A, "scores_new_gold": B,
                      "mean_abs_shift": round(float(np.mean([abs(A[a] - B[a]) for a in ags])), 1), "max_abs_shift": max(abs(A[a] - B[a]) for a in ags),
                      "mean_rank_move": round(float(np.mean([abs(ra[a] - rb[a]) for a in ags])), 1), "max_rank_move": max(abs(ra[a] - rb[a]) for a in ags),
                      "top_old": sorted(ags, key=lambda a: -A[a])[:3], "top_new": sorted(ags, key=lambda a: -B[a])[:3]}
    same = 0; tot = 0
    for ag in L.AGENTS:
        po, pn = L.get_preds(ag, "dev_old"), L.get_preds(ag, "dev")
        for q in qs:
            if q in po and q in pn:
                tot += 1; same += L.parse_entry(po[q])[0] == L.parse_entry(pn[q])[0]
    res["prediction_text_identical_old_vs_new"] = {"same": same, "total": tot}
    print(json.dumps(res, indent=0))
    out = {"gold_effect": res}
    # B
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    t1 = {e["question_id"]: e["SQL"] for e in json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    mini = [n.row(t0[int(e["question_id"])], e["SQL"], e["db_id"], bool(n.a5.norm(t0[int(e["question_id"])]["SQL"]) != n.a5.norm(t1[int(e["question_id"])]) or "original_SQL" in e)) for e in json.load(open(n.a5.UIUC))]
    old = {e["question_id"]: e for e in json.load(open(f"{n.EXT}/bird_dev_20240627/dev.json"))}
    raw = open(f"{n.EXT}/bird_dev_20251106/dev.json").read().strip()
    newl = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]
    tab = {r["q"]: r for r in json.load(open(os.path.join(HERE, "results", "full_dev_table.json")))}
    dev = [n.row(old[e["question_id"]], e["SQL"], e["db_id"], bool(tab[e["question_id"]]["txt"])) for e in newl if not tab[e["question_id"]]["mini"]]
    devo = [{**r, "err": bool(tab[e["question_id"]]["out"])} for r, e in zip(dev, [e for e in newl if not tab[e["question_id"]]["mini"]])]
    for tag, rows in (("mini", mini), ("dev_heldout", dev), ("dev_heldout_output", devo)):
        Y = np.array([r["err"] for r in rows], float); dbs = [r["db"] for r in rows]; G = sorted(set(dbs))
        fe = np.column_stack([[1.0 if d == g else 0.0 for d in dbs] for g in G])
        X, _, _ = n.mat(rows, ["wd", "wt", "wj", "qlen", "elen", "mk"])
        Xl, _, _ = n.mat(rows, n.FEATS)
        fit = lambda A: v.logit(A, Y)[2]
        ll0 = fit(np.ones((len(Y), 1))); llq = fit(X); lll = fit(Xl); lld = fit(fe); llb = fit(np.column_stack([fe, X[:, 1:]]))
        r2 = lambda ll: round(1 - ll / ll0, 3)
        per = {g: {"n": sum(d == g for d in dbs), "rate": round(100 * Y[[i for i, d in enumerate(dbs) if d == g]].mean(), 1)} for g in G}
        out[tag] = {"mcfadden_r2": {"pre_review_shape": r2(llq), "shape_plus_lints": r2(lll), "database_identity": r2(lld), "both": r2(llb)},
                    "aic": {"intercept": round(2 - 2 * ll0, 1), "shape": round(2 * X.shape[1] - 2 * llq, 1), "database": round(2 * len(G) - 2 * lld, 1), "both": round(2 * (len(G) + 6) - 2 * llb, 1)}, "per_db": per}
        print(tag, out[tag]["mcfadden_r2"], out[tag]["aic"], {k: x["rate"] for k, x in per.items()})
    rows = json.load(open(os.path.join(HERE, "results", "full_dev_table.json")))
    h = [r for r in rows if not r["mini"]]
    out["dev_exec"] = {"all_text": sum(r["txt"] for r in rows), "all_output": sum(r["out"] for r in rows), "all_identical": sum(r["txt"] and not r["out"] for r in rows),
                       "held_text": sum(r["txt"] for r in h), "held_output": sum(r["out"] for r in h), "held_identical": sum(r["txt"] and not r["out"] for r in h)}
    print(out["dev_exec"])
    json.dump(out, open(os.path.join(HERE, "results", "gold_effect_schema.json"), "w"), indent=1, default=float)
main()
