#!/usr/bin/env python3
"""
ICDE Exp 26 -- how much of a system ranking depends on which gold file is used?

Dev-100 (sampled and corrected by Jin et al.), fifteen systems, execution match.
Per-question scores against each gold are stored so that bootstrap intervals over
questions can be computed for Kendall tau, stratified two ways: by depth of the
corrected gold, and by demand markers in the question (not based on the correction).
Also tests system disagreement with the gold as an audit signal, and consensus
depth (the median depth the systems wrote) as a demand measure independent of it.

Usage: python3 26_leaderboard_ci.py
"""
import importlib.util, json, os
import numpy as np, sqlglot
HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
L = _load("L14", "14_leaderboard_depth.py"); v = _load("v20", "20_validity_audit.py"); psim = _load("psim", "04_policy_sim.py")
n23 = _load("n23", "23_noncircular.py") if False else None
EXT = os.path.join(HERE, "data_ext")

def tau(a, b):
    ags = list(a); c = d = 0
    for i in range(len(ags)):
        for j in range(i + 1, len(ags)):
            x = a[ags[i]] - a[ags[j]]; y = b[ags[i]] - b[ags[j]]
            if x * y > 0: c += 1
            elif x * y < 0: d += 1
    n = len(ags) * (len(ags) - 1) / 2
    return (c - d) / n, ((c - d) / (c + d) if c + d else 1.0)

def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float); P = y.sum(); N = len(y) - P
    return float(sum(((s[y == 1] > t).sum() + 0.5 * (s[y == 1] == t).sum()) for t in s[y == 0]) / (P * N))

def main():
    L.e7x.ensure_dbs()
    gn = {k: L.parse_entry(v_) for k, v_ in json.load(open(os.path.join(HERE, "data", "dev100", "dev.json"))).items()}
    go = {k: L.parse_entry(v_) for k, v_ in json.load(open(os.path.join(HERE, "data", "dev100", "dev_old.json"))).items()}
    dev24 = {str(e["question_id"]): e for e in json.load(open(f"{EXT}/bird_dev_20240627/dev.json"))}
    qids = sorted(gn, key=int)
    sc = {}; pdepth = {q: [] for q in qids}; disagree = {q: 0 for q in qids}
    for ag in L.AGENTS:
        po, pn = L.get_preds(ag, "dev_old"), L.get_preds(ag, "dev")
        if not po or not pn: continue
        sc[ag] = {"old": {}, "new": {}, "new_gold_old_pred": {}}
        for q in qids:
            db = gn[q][1]
            for tag, preds, gold in (("old", po, go), ("new", pn, gn)):
                ok = 0
                if q in preds:
                    s_, _ = L.parse_entry(preds[q]); a, ra = L.run(db, s_); b, rb = L.run(db, gold[q][0])
                    ok = int(a == "ok" and b == "ok" and ra == rb)
                sc[ag][tag][q] = ok
            if q in po:
                s_, _ = L.parse_entry(po[q])
                try: pdepth[q].append(v.select_depth(sqlglot.parse_one(s_, read="sqlite")))
                except Exception: pass
                disagree[q] += 1 - sc[ag]["old"][q]
        print(ag, sum(sc[ag]["old"].values()), sum(sc[ag]["new"].values()), flush=True)
    ags = list(sc)
    rows = []
    for q in qids:
        a, b = go[q][0], gn[q][0]; db = gn[q][1]
        tchg = L.a5.norm(a) != L.a5.norm(b)
        s1, r1 = L.run(db, a); s2, r2 = L.run(db, b); ochg = s1 != s2 or r1 != r2
        dd = v.select_depth(sqlglot.parse_one(b, read="sqlite")); wd = v.select_depth(sqlglot.parse_one(a, read="sqlite"))
        qt = dev24.get(q, {}).get("question", "")
        rows.append({"q": q, "txt": tchg, "out": ochg, "dd": v.bucket(dd), "wd": wd, "mk": min(len(psim.MARKERS.findall(qt)), 2),
                     "cons": float(np.median(pdepth[q])) if pdepth[q] else 1.0, "dis": disagree[q] / len(ags)})
    out = {"n_systems": len(ags), "n_q": len(rows), "text_changed": sum(r["txt"] for r in rows), "output_changed": sum(r["out"] for r in rows)}
    rng = np.random.default_rng(17)
    def strat(key, levels):
        o = {}
        for lv in levels:
            qs = [r["q"] for r in rows if r[key] == lv]
            if not qs: continue
            sa = {a: sum(sc[a]["old"][q] for q in qs) for a in ags}; sb = {a: sum(sc[a]["new"][q] for q in qs) for a in ags}
            tb, tx = tau(sa, sb)
            boots = []
            for _ in range(2000):
                pick = rng.choice(qs, len(qs))
                boots.append(tau({a: sum(sc[a]["old"][q] for q in pick) for a in ags}, {a: sum(sc[a]["new"][q] for q in pick) for a in ags})[1])
            ra = {a: i for i, a in enumerate(sorted(ags, key=lambda a: -sa[a]))}; rb = {a: i for i, a in enumerate(sorted(ags, key=lambda a: -sb[a]))}
            o[str(lv)] = {"questions": len(qs), "tau_ties_excluded": round(tx, 2), "tau_b_like": round(tb, 2), "ci95": [round(float(x), 2) for x in np.percentile(boots, [2.5, 97.5])],
                          "mean_move": round(float(np.mean([abs(ra[a] - rb[a]) for a in ags])), 1), "changed_share": round(np.mean([r["out"] for r in rows if r[key] == lv]) * 100, 0),
                          "mean_abs_score_shift_pct": round(float(np.mean([abs(sa[a] - sb[a]) for a in ags])) / len(qs) * 100, 1)}
        return o
    out["by_corrected_depth"] = strat("dd", [0, 1, 2]); out["by_markers"] = strat("mk", [0, 1, 2]); 
    for r in rows: r["all"] = 0
    out["overall"] = strat("all", [0])
    # consensus-depth and disagreement signals
    for r in rows: r["c"] = 0 if r["cons"] <= 1 else 1
    for y in ("txt", "out"):
        ys = [r[y] for r in rows]
        out[f"signals_{y}"] = {"auc_disagreement": round(auc(ys, [r["dis"] for r in rows]), 3), "auc_consensus_depth": round(auc(ys, [r["cons"] for r in rows]), 3),
                               "auc_consensus_minus_written": round(auc(ys, [r["cons"] - r["wd"] for r in rows]), 3), "auc_markers": round(auc(ys, [r["mk"] for r in rows]), 3),
                               "rate_by_consensus_flat_vs_nested": [round(100 * np.mean([r[y] for r in rows if r["c"] == c]), 1) for c in (0, 1)],
                               "n_by_consensus": [sum(r["c"] == c for r in rows) for c in (0, 1)],
                               "rate_by_disagree_tercile": [round(100 * np.mean([r[y] for r in rows if lo <= r["dis"] < hi]), 1) for lo, hi in ((0, .2), (.2, .6), (.6, 1.01))],
                               "n_by_disagree_tercile": [sum(lo <= r["dis"] < hi for r in rows) for lo, hi in ((0, .2), (.2, .6), (.6, 1.01))]}
    # score-level: systems penalised by the wrong gold
    out["per_system"] = {a: {"old": sum(sc[a]["old"].values()), "new": sum(sc[a]["new"].values())} for a in ags}
    print(json.dumps({k: v_ for k, v_ in out.items() if k != "per_system"}, indent=0))
    json.dump(out, open(os.path.join(HERE, "results", "leaderboard_ci.json"), "w"), indent=1, default=float)
main()
