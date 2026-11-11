#!/usr/bin/env python3
"""
CHI2 Exp 05 -- two-tier audit analysis (three-way gold comparison).

Tiers of gold SQL for the same 498 BIRD Mini-Dev examples:
  T0  BIRD official (un-audited)            data/minidev.zip :: mini_dev_sqlite.json
  T1  Arcwise community correction          data/arcwise_orig.json  (== UIUC original_SQL)
  T2  UIUC second audit (final)             text_to_sql_benchmarks arcwise_plat_full_with_diff.json

Questions answered:
  A. Error rate vs demanded depth measured against the ORIGINAL gold (T0 vs T2):
     the full error population, before any audit -- headline gradient.
  B. Of those original errors, does survival past audit round 1 (Arcwise) depend
     on depth?  (If flat: audits catch shallow and deep errors alike; the depth
     gradient is a property of error PRODUCTION, not of audit blindness.)
  C. Round-level view: both audit rounds independently show the same gradient.

Usage: python3 05_audit_survival.py
"""
import importlib.util, json, math, os, re, zipfile

import sqlglot
from sqlglot import exp as E

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("psim", os.path.join(HERE, "04_policy_sim.py"))
psim = importlib.util.module_from_spec(spec); spec.loader.exec_module(psim)
sel_depth = psim.sel_depth

UIUC = os.path.join(HERE, "data", "text_to_sql_benchmarks", "data",
                    "arcwise_plat_full_with_diff.json")


def norm(sql):
    """Canonical form for equality testing across gold versions."""
    try:
        t = sqlglot.parse_one(sql, read="sqlite")
        s = t.sql(dialect="sqlite", normalize=True)
    except Exception:
        s = sql
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


def ca_trend(ns, ks, scores=(0, 1, 2)):
    """Cochran-Armitage trend z (one-sided p via normal)."""
    N = sum(ns); K = sum(ks)
    if not (0 < K < N):
        return 0.0, 1.0
    p = K / N
    num = sum(k * s for k, s in zip(ks, scores)) - p * sum(n * s for n, s in zip(ns, scores))
    s1 = sum(n * s for n, s in zip(ns, scores))
    s2 = sum(n * s * s for n, s in zip(ns, scores))
    var = p * (1 - p) * (s2 - s1 * s1 / N)
    z = num / math.sqrt(var)
    pval = 0.5 * math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def fisher(a, b, c, d):
    """One-sided Fisher exact P(>=a) for table [[a,b],[c,d]]."""
    from math import comb
    n = a + b + c + d
    row1, col1 = a + b, a + c
    denom = comb(n, col1)
    p = 0.0
    for x in range(a, min(row1, col1) + 1):
        p += comb(row1, x) * comb(n - row1, col1 - x) / denom
    return p


def main():
    # T0 original gold
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e["SQL"]
              for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    # T1 Arcwise
    t1 = {e["question_id"]: e["SQL"] for e in
          json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    # T2 UIUC final (+ per-example demanded depth from final SQL)
    uiuc = json.load(open(UIUC))
    assert len(t0) == len(t1) == len(uiuc) == 498

    rows = []
    for e in uiuc:
        qid = int(e["question_id"])
        err1 = norm(t0[qid]) != norm(t1[qid])          # caught by round 1
        err2 = "original_SQL" in e                      # caught by round 2
        dd = sel_depth(sqlglot.parse_one(e["SQL"], read="sqlite"))
        b = 0 if dd <= 1 else (1 if dd == 2 else 2)     # sel_depth>=1 always
        rows.append({"qid": qid, "err1": err1, "err2": err2,
                     "err_any": err1 or err2, "bucket": b})

    n_err1 = sum(r["err1"] for r in rows)
    n_err2 = sum(r["err2"] for r in rows)
    n_both = sum(r["err1"] and r["err2"] for r in rows)
    n_any = sum(r["err_any"] for r in rows)
    print(f"round1 caught {n_err1} | round2 caught {n_err2} | both {n_both} "
          f"| any {n_any}/498 = {100*n_any/498:.1f}%")

    out = {"caught1": n_err1, "caught2": n_err2, "both": n_both, "any": n_any}

    labels = ["0 nesting", "1 level", "2+ levels"]
    for key, name in (("err_any", "A_vs_original_gold"), ("err1", "C_round1_only"),
                      ("err2", "C_round2_only")):
        ns = [sum(r["bucket"] == b for r in rows) for b in range(3)]
        ks = [sum(r["bucket"] == b and r[key] for r in rows) for b in range(3)]
        z, p = ca_trend(ns, ks)
        pf = fisher(ks[2], ns[2] - ks[2], ks[0], ns[0] - ks[0])
        rates = [100 * k / n for k, n in zip(ks, ns)]
        print(f"{name}: " + " / ".join(f"{r:.1f}% ({k}/{n})" for r, k, n in zip(rates, ks, ns))
              + f"  CA z={z:.2f} p={p:.1e}  Fisher(2+ vs 0) p={pf:.1e}")
        out[name] = {"n": ns, "k": ks, "rate": [round(r, 1) for r in rates],
                     "CA_z": round(z, 2), "CA_p": float(f"{p:.2e}"),
                     "fisher_p": float(f"{pf:.2e}")}

    # B: survival past round 1, among original errors
    ns = [sum(r["bucket"] == b and r["err_any"] for r in rows) for b in range(3)]
    ks = [sum(r["bucket"] == b and r["err_any"] and not r["err1"] for r in rows)
          for b in range(3)]
    z, p = ca_trend(ns, ks)
    rates = [100 * k / n for k, n in zip(ks, ns)]
    print("B_survival_past_round1 (among errors): "
          + " / ".join(f"{r:.1f}% ({k}/{n})" for r, k, n in zip(rates, ks, ns))
          + f"  CA z={z:.2f} p(two-sided)={2*p:.2f}")
    out["B_survival"] = {"n": ns, "k": ks, "rate": [round(r, 1) for r in rates],
                         "CA_z": round(z, 2), "CA_p_two_sided": round(2 * p, 3)}

    # demanded x written depth cross-tab (F4 decomposition)
    ct = {}
    for e, r in zip(uiuc, rows):
        qid = int(e["question_id"])
        try:
            dw = sel_depth(sqlglot.parse_one(t0[qid], read="sqlite"))
        except Exception:
            continue
        bw = 0 if dw <= 1 else (1 if dw == 2 else 2)
        key = f"dd{r['bucket']}_dw{bw}_{'err' if r['err_any'] else 'ok'}"
        ct[key] = ct.get(key, 0) + 1
    out["depth_crosstab"] = ct
    deep = [r for e, r in zip(uiuc, rows) if r["bucket"] >= 1]
    flat_written = sum(v for k, v in ct.items() if k.startswith(("dd1_dw0", "dd2_dw0")))
    print(f"tasks demanding nesting: {len(deep)}; original written flat: {flat_written}")

    # per-database any-error rates (trap narrative; F6/landscape section)
    per = {}
    for e, r in zip(uiuc, rows):
        db = e["db_id"]
        k, n = per.get(db, (0, 0))
        per[db] = (k + r["err_any"], n + 1)
    out["per_db_any"] = {db: {"k": k, "n": n, "rate": round(100 * k / n, 1)}
                         for db, (k, n) in sorted(per.items(), key=lambda x: -x[1][0]/x[1][1])}
    for db, v in out["per_db_any"].items():
        print(f"  {db:28s} {v['k']:3d}/{v['n']:3d} = {v['rate']}%")

    json.dump(out, open(os.path.join(HERE, "results", "audit_survival.json"), "w"),
              indent=1)
    print("wrote results/audit_survival.json")


if __name__ == "__main__":
    main()
