#!/usr/bin/env python3
"""ICDE Exp 28 -- execution outcomes under the benchmark's own set-comparison rule,
for both corpora; lints against the output-changed outcome; per-database stability
across the two review passes; churn of the nested subset.
Usage: python3 28_final_numbers.py"""
import importlib.util, json, os, sqlite3, threading, zipfile
from collections import Counter
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
a5 = _load("a5", "05_audit_survival.py"); n = _load("n23", "23_noncircular.py")
DBDIR = os.path.join(HERE, "data", "dev_databases"); EXT = os.path.join(HERE, "data_ext")


def ex(db, sql):
    c = sqlite3.connect(f"file:{os.path.join(DBDIR, db + '.sqlite')}?mode=ro", uri=True)
    c.create_function("IIF", 3, lambda a, x, y: x if a else y)
    t = threading.Timer(60, c.interrupt); t.start()
    try:
        cur = c.execute(sql); rows = cur.fetchall(); return rows, len(cur.description or [])
    except Exception:
        return None, 0
    finally:
        t.cancel(); c.close()


def outcome(db, a, b):
    ra, ca = ex(db, a); rb, cb = ex(db, b)
    if ra is None:
        return "old query fails", False
    if rb is None:
        return "new query fails", False
    plaus = len(ra) > 0 and ca == cb
    sa, sb = {tuple(r) for r in ra}, {tuple(r) for r in rb}
    if sa == sb:
        same_bag = sorted(map(repr, ra)) == sorted(map(repr, rb))
        return ("identical rows" if same_bag else "identical set, different multiplicity"), plaus
    if ca != cb:
        return "different columns", plaus
    if len(sa) != len(sb):
        return "different row count", plaus
    return "same shape, different values", plaus


def main():
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    t1 = {e["question_id"]: e["SQL"] for e in json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    mp = []
    for e in json.load(open(a5.UIUC)):
        q = int(e["question_id"])
        if a5.norm(t0[q]["SQL"]) != a5.norm(t1[q]) or "original_SQL" in e:
            mp.append((e["db_id"], t0[q]["SQL"], e["SQL"]))
    old = {e["question_id"]: e for e in json.load(open(f"{EXT}/bird_dev_20240627/dev.json"))}
    raw = open(f"{EXT}/bird_dev_20251106/dev.json").read().strip()
    newl = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]
    tab = {r["q"]: r for r in json.load(open(os.path.join(HERE, "results", "full_dev_table.json")))}
    dp = [(e["db_id"], old[e["question_id"]]["SQL"], e["SQL"]) for e in newl
          if tab[e["question_id"]]["txt"] and not tab[e["question_id"]]["mini"]]
    out = {}
    for tag, ps in (("mini", mp), ("dev_heldout", dp)):
        res = [outcome(*p) for p in ps]
        c = Counter(r[0] for r in res); pl = sum(r[1] for r in res)
        out[tag] = {"n": len(ps), "outcomes": dict(c), "plausible": pl, "plausible_pct": round(100 * pl / len(ps), 1)}
        print(tag, out[tag])

    dev = [n.row(old[e["question_id"]], e["SQL"], e["db_id"], bool(tab[e["question_id"]]["out"]))
           for e in newl if not tab[e["question_id"]]["mini"]]
    def lint_table(rows, feats):
        return {f: {"n": sum(bool(r[f]) for r in rows),
                    "on": round(100 * sum(r["err"] for r in rows if r[f]) / max(1, sum(bool(r[f]) for r in rows)), 1),
                    "off": round(100 * sum(r["err"] for r in rows if not r[f]) / max(1, sum(not r[f] for r in rows)), 1)} for f in feats}
    out["lints_output_changed_heldout"] = lint_table(dev, n.FEATS[7:])
    out["lints_output_excl_two_schemas"] = lint_table([r for r in dev if r["db"] not in ("financial", "california_schools")], n.FEATS[7:])
    print(out["lints_output_changed_heldout"]); print(out["lints_output_excl_two_schemas"])

    S = json.load(open(os.path.join(HERE, "results", "gold_effect_schema.json")))
    a = S["mini"]["per_db"]; b = S["dev_heldout"]["per_db"]; k = sorted(a)
    rho, p = stats.spearmanr([a[x]["rate"] for x in k], [b[x]["rate"] for x in k])
    out["per_db_spearman_mini_vs_dev"] = {"rho": round(float(rho), 2), "p": round(float(p), 3)}
    print(out["per_db_spearman_mini_vs_dev"])

    rows = list(tab.values())
    out["subset_churn"] = {"nested_new": sum(r["d"] > 0 for r in rows),
                           "nested_new_flat_before": sum(r["d"] > 0 and r["w"] == 0 for r in rows),
                           "nested_old": sum(r["w"] > 0 for r in rows),
                           "nested_old_flat_now": sum(r["w"] > 0 and r["d"] == 0 for r in rows)}
    print(out["subset_churn"])
    json.dump(out, open(os.path.join(HERE, "results", "final_numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
