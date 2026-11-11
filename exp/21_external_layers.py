#!/usr/bin/env python3
"""
ICDE Exp 21 -- do new, independent correction releases reproduce the gradient?

  L1. Full BIRD Dev: 2024-06-27 gold vs the maintainers' 2025-11-06 review.
      Larger sample, different reviewers, deep strata no longer tiny.
  L2. Mini-Dev 2026 update (released 2026-10-28, after our analysis was fixed):
      a prospective test. Among the examples our final T2 gold keeps, do the
      maintainers' new changes also concentrate on deep tasks?

Downloaded JSON is treated as data only.
Usage: python3 -I 21_external_layers.py
"""
import sys
pass
import importlib.util, json, math, os, zipfile
import sqlglot
from sqlglot import exp as E

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
a5 = _load("a5", "05_audit_survival.py")
v = _load("v20", "20_validity_audit.py")
EXT = os.path.join(HERE, "data_ext")


def safe(sql):
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def gradient(rows, key="d", y="err"):
    ns = [sum(r[key] == b for r in rows) for b in range(3)]
    ks = [sum(r[key] == b and r[y] for r in rows) for b in range(3)]
    z, p = a5.ca_trend(ns, ks)
    return {"n": ns, "k": ks, "rate": [round(100 * k / n, 1) if n else None for k, n in zip(ks, ns)],
            "z": round(z, 2), "p": float(f"{p:.2e}")}


def direction(rows):
    add = sum(r["w_raw"] < r["d_raw"] for r in rows if r["err"])
    rem = sum(r["w_raw"] > r["d_raw"] for r in rows if r["err"])
    from math import comb
    n = add + rem
    p = sum(comb(n, i) for i in range(add, n + 1)) / 2 ** n if n else 1.0
    return {"add_depth": add, "remove_depth": rem, "binom_p_one_sided": float(f"{p:.2e}")}


def cmh(rows):
    num = var = 0.0; pos = 0; dbs = sorted({r["db"] for r in rows})
    for db in dbs:
        s = [r for r in rows if r["db"] == db]
        a = sum(r["d"] > 0 and r["err"] for r in s); n1 = sum(r["d"] > 0 for r in s)
        c = sum(r["d"] == 0 and r["err"] for r in s); n0 = sum(r["d"] == 0 for r in s)
        T = n1 + n0; m = a + c
        if n1 == 0 or n0 == 0 or T < 2:
            continue
        num += a - n1 * m / T
        var += n1 * n0 * m * (T - m) / (T * T * (T - 1))
        pos += a / n1 > c / n0
    if var == 0:
        return None
    z = num / math.sqrt(var)
    return {"z": round(z, 2), "p": float(f"{0.5*math.erfc(abs(z)/math.sqrt(2)):.1e}"), "dbs_positive": int(pos), "dbs": len(dbs)}


def build(old, new):
    rows, bad = [], 0
    for q, e in new.items():
        if q not in old:
            continue
        tn, to = safe(e["SQL"]), safe(old[q]["SQL"])
        if tn is None or to is None:
            bad += 1
            continue
        d, w = v.select_depth(tn), v.select_depth(to)
        rows.append({"q": q, "db": e["db_id"], "err": a5.norm(old[q]["SQL"]) != a5.norm(e["SQL"]),
                     "d": v.bucket(d), "w": v.bucket(w), "d_raw": d, "w_raw": w,
                     "irr": v.bucket(v.irreducible_depth(tn)),
                     "ntab": len({t.name for t in tn.find_all(E.Table)}),
                     "njoin": len(list(tn.find_all(E.Join))),
                     "difficulty": e.get("difficulty"),
                     "qchg": old[q].get("question") != e.get("question"),
                     "echg": (old[q].get("evidence") or "") != (e.get("evidence") or "")})
    return rows, bad


def jl(fn):
    return [json.loads(l) for l in open(fn) if l.strip()]


def main():
    out = {}
    old = {e["question_id"]: e for e in json.load(open(f"{EXT}/bird_dev_20240627/dev.json"))}
    raw = open(f"{EXT}/bird_dev_20251106/dev.json").read().strip()
    newl = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]
    new = {e["question_id"]: e for e in newl}
    rows, bad = build(old, new)
    print("L1 sizes old/new/matched/unparsed", len(old), len(new), len(rows), bad)
    L1 = {"n": len(rows), "unparsed": bad, "changed": sum(r["err"] for r in rows),
          "gradient": gradient(rows), "gradient_irreducible": gradient(rows, "irr"),
          "diag": gradient([r for r in rows if r["d"] == r["w"]]),
          "direction": direction(rows), "cmh": cmh(rows),
          "question_changed": sum(r["qchg"] for r in rows), "evidence_changed": sum(r["echg"] for r in rows)}
    for nm, key, bs in (("tables", "ntab", [1, 2, 3, 4]), ("joins", "njoin", [0, 1, 2])):
        L1[nm] = {}
        for b in bs:
            s = [r for r in rows if (r[key] >= b if b == bs[-1] else r[key] == b)]
            L1[nm][str(b)] = {"n": len(s), "rate": round(100 * sum(r["err"] for r in s) / max(1, len(s)), 1)}
    out["L1_full_dev"] = L1
    print(json.dumps(L1, indent=0)[:1500])

    # L2: T2 vs 2026 Mini-Dev
    uiuc = {int(e["question_id"]): e for e in json.load(open(a5.UIUC))}
    m26 = {e["question_id"]: e for e in jl(f"{EXT}/bird_minidev_2026/mini_dev_sqlite.jsonl")}
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    print("L2 sizes m26", len(m26), "overlap T2", len(set(m26) & set(uiuc)), "overlap T0", len(set(m26) & set(t0)))
    print("L2 sample keys", list(next(iter(m26.values())).keys()))
    same_q = {q for q in set(m26) & set(uiuc) if m26[q]["question"].strip() == uiuc[q]["question"].strip()}
    print("L2 same question text as T2:", len(same_q))
    # (a) maintainers' own lineage: T0 -> 2026, depth on 2026 gold
    rows_a, bad_a = build(t0, m26)
    La = {"n": len(rows_a), "unparsed": bad_a, "changed": sum(r["err"] for r in rows_a),
          "gradient": gradient(rows_a), "gradient_irreducible": gradient(rows_a, "irr"), "direction": direction(rows_a), "cmh": cmh(rows_a)}
    print("L2a T0->2026", La)
    # (b) prospective: T2 frozen; depth measured on T2 before seeing 2026
    rows_b, bad_b = build({q: uiuc[q] for q in uiuc}, m26)
    pro = []
    for r in rows_b:
        d_t2 = v.bucket(v.select_depth(safe(uiuc[r["q"]]["SQL"])))
        pro.append({**r, "dfrozen": d_t2})
    Lb = {"n": len(pro), "changed": sum(r["err"] for r in pro), "by_frozen_T2_depth": gradient(pro, "dfrozen"),
          "direction": direction(pro)}
    print("L2b T2->2026 prospective", Lb)
    # (c) does 2026 agree with the audit? among T0 errors per our table
    agree = {"T0_eq_T2_and_2026_changed_from_T0": 0, "T0_ne_T2_and_2026_eq_T2": 0, "T0_ne_T2_and_2026_eq_T0": 0, "T0_ne_T2_and_2026_differs_both": 0, "T0_eq_T2_and_2026_eq": 0}
    for q in set(m26) & set(uiuc) & set(t0):
        a, b, c = a5.norm(t0[q]["SQL"]), a5.norm(uiuc[q]["SQL"]), a5.norm(m26[q]["SQL"])
        if a == b:
            agree["T0_eq_T2_and_2026_eq" if c == a else "T0_eq_T2_and_2026_changed_from_T0"] += 1
        elif c == b:
            agree["T0_ne_T2_and_2026_eq_T2"] += 1
        elif c == a:
            agree["T0_ne_T2_and_2026_eq_T0"] += 1
        else:
            agree["T0_ne_T2_and_2026_differs_both"] += 1
    print("L2c", agree)
    out["L2_minidev_2026"] = {"T0_to_2026": La, "prospective_T2_to_2026": Lb, "agreement": agree,
                              "same_question_as_T2": len(same_q), "ids_m26": len(m26), "ids_overlap_T2": len(set(m26) & set(uiuc))}
    json.dump(out, open(os.path.join(HERE, "results", "external_layers.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
