#!/usr/bin/env python3
"""
ICDE Exp 32 -- run goldcheck on every public version of the BIRD gold and compare
its findings with what later human reviews changed.

Versions: Mini-Dev T0 (BIRD), T1 (Arcwise), T2 (Jin et al. audit);
          Dev 2024-06-27 and Dev 2025-11-06 (maintainers' review).
Reference "a later review changed this gold": T0 -> T1 or T2; T1 -> T2; Dev 2024 -> 2025.
Change = text differs after sqlglot normalization; strict = output on the official DB differs.
Output: results/goldcheck.json
Usage: python3 32_run_goldcheck.py
"""
import importlib.util, json, multiprocessing as mp, os, zipfile, math
import goldcheck as G

HERE = os.path.dirname(os.path.abspath(__file__))
DBDIR = os.path.join(HERE, "data", "dev_databases"); EXT = os.path.join(HERE, "data_ext")
spec = importlib.util.spec_from_file_location("a5", os.path.join(HERE, "05_audit_survival.py"))
a5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a5)


def load(path):
    raw = open(path).read().strip()
    return json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]


def versions():
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    t1 = {e["question_id"]: e for e in json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    t2 = {int(e["question_id"]): e for e in json.load(open(a5.UIUC))}
    d24 = {e["question_id"]: e for e in load(f"{EXT}/bird_dev_20240627/dev.json")}
    d25 = {e["question_id"]: e for e in load(f"{EXT}/bird_dev_20251106/dev.json")}
    return {"mini_T0": t0, "mini_T1": t1, "mini_T2": t2, "dev_2024": d24, "dev_2025": d25}


def job(args):
    ver, qid, db, sql = args
    r = G.check(os.path.join(DBDIR, db + ".sqlite"), sql)
    r.update({"ver": ver, "qid": qid, "db": db})
    return r


def same_output(db, a, b):
    con = G.connect(os.path.join(DBDIR, db + ".sqlite"))
    try:
        x, y = G.as_set(G.run(con, a, 60)), G.as_set(G.run(con, b, 60))
        return x is not None and x == y
    finally:
        con.close()


def chg_job(args):
    db, a, b = args
    if a5.norm(a) == a5.norm(b):
        return (False, False)
    return (True, not same_output(db, a, b))


def fisher_ratio(k1, n1, k0, n0):
    return {"flagged": f"{k1}/{n1}", "flagged_pct": round(100 * k1 / max(n1, 1), 1),
            "unflagged": f"{k0}/{n0}", "unflagged_pct": round(100 * k0 / max(n0, 1), 1),
            "lift": round((k1 / max(n1, 1)) / max(k0 / max(n0, 1), 1e-9), 2)}


def main():
    V = versions()
    jobs = [(v, qid, e["db_id"], e["SQL"]) for v, d in V.items() for qid, e in d.items()]
    with mp.Pool(10) as pool:
        res = pool.map(job, jobs, chunksize=4)
        later = {"mini_T0": ["mini_T1", "mini_T2"], "mini_T1": ["mini_T2"], "dev_2024": ["dev_2025"]}
        cj, keys = [], []
        for v, nxt in later.items():
            for qid, e in V[v].items():
                for n in nxt:
                    cj.append((e["db_id"], e["SQL"], V[n][qid]["SQL"])); keys.append((v, n, qid))
        ch = dict(zip(keys, pool.map(chg_job, cj, chunksize=8)))
    by = {(r["ver"], r["qid"]): r for r in res}
    summary = {}
    for v, d in V.items():
        rs = [by[(v, q)] for q in d]
        s = {"n": len(rs), "status": {}, "flags": {f: sum(f in r["flags"] for r in rs) for f in G.ALL},
             "any_live": sum(any(f in r["flags"] for f in G.LIVE) for r in rs),
             "any": sum(bool(r["flags"]) for r in rs)}
        for r in rs:
            s["status"][r["status"]] = s["status"].get(r["status"], 0) + 1
        tk = [r["topk"] for r in rs if "topk" in r]
        s["topk"] = {"n": len(tk)}
        for t in tk:
            s["topk"][t.get("status", "?")] = s["topk"].get(t.get("status", "?"), 0) + 1
            if "latent" in t:
                s["topk"]["latent_" + t["latent"]] = s["topk"].get("latent_" + t["latent"], 0) + 1
        secs = sorted(r["secs"] for r in rs)
        s["secs"] = {"total": round(sum(secs), 1), "median": secs[len(secs) // 2], "p95": secs[int(0.95 * len(secs))], "max": secs[-1]}
        summary[v] = s
    # agreement with later reviews
    agree = {}
    for v, nxt in later.items():
        for n in nxt:
            for kind in ("live", "any"):
                fl = lambda r: any(f in r["flags"] for f in (G.LIVE if kind == "live" else G.ALL))
                for strict in (False, True):
                    k1 = n1 = k0 = n0 = 0
                    for q in V[v]:
                        c = ch[(v, n, q)][1 if strict else 0]
                        if fl(by[(v, q)]):
                            n1 += 1; k1 += c
                        else:
                            n0 += 1; k0 += c
                    agree[f"{v}->{n}|{kind}|{'output' if strict else 'text'}"] = fisher_ratio(k1, n1, k0, n0)
            # recall of the review's changes and per-flag fate
            tot = sum(ch[(v, n, q)][1] for q in V[v])
            hit = sum(ch[(v, n, q)][1] and any(f in by[(v, q)]["flags"] for f in G.LIVE) for q in V[v])
            agree[f"{v}->{n}|recall_of_output_changes_live"] = f"{hit}/{tot}"
            fate = {}
            for f in G.ALL:
                flagged = [q for q in V[v] if f in by[(v, q)]["flags"]]
                fixed = sum(f not in by[(n, q)]["flags"] for q in flagged)
                fate[f] = {"flagged": len(flagged), "gone_in_later": fixed,
                           "later_text_changed": sum(ch[(v, n, q)][0] for q in flagged)}
            agree[f"{v}->{n}|fate"] = fate
    print(json.dumps(summary, indent=1)); print(json.dumps(agree, indent=1))
    json.dump({"summary": summary, "agreement": agree, "rows": res,
               "changes": {f"{a}|{b}|{q}": c for (a, b, q), c in ch.items()}},
              open(os.path.join(HERE, "results", "goldcheck.json"), "w"), indent=0, default=str)


if __name__ == "__main__":
    main()
