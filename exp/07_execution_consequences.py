#!/usr/bin/env python3
"""
CHI2 Exp 07 (proposal E2-1) -- what does an erroneous gold query LOOK like
when it runs?

For each of the 201 examples whose original gold differs from the final
corrected gold, execute both queries on the official database and compare
the outputs. The question behind the experiment: does a wrong gold query
produce output that a reader could tell is wrong, or does it look fine?

Output categories per example (T0 = original gold, T2 = corrected gold):
  identical-output   same result set        -> error invisible even at runtime
  t0-error           T0 fails to execute    -> loudly broken
  t0-empty           T0 returns no rows, T2 returns rows
  shape-differs      different column count
  rowcount-differs   same columns, different number of rows
  values-differ      same shape and count, different values

"Plausible output" = T0 executes, returns >=1 row, and has the same column
count as T2 -- i.e. nothing about the output signals a problem.

Databases: extracted on first run from data/minidev.zip (1.5 GB unpacked;
logged in DATA.md, delete after the experiment family is done).

Usage: python3 07_execution_consequences.py
"""
import json, os, re, sqlite3, threading, zipfile
from collections import Counter, defaultdict

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("a5", os.path.join(HERE, "05_audit_survival.py"))
a5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a5)

DBDIR = os.path.join(HERE, "data", "dev_databases")
TIMEOUT_S = 60


def ensure_dbs():
    if os.path.isdir(DBDIR) and len(os.listdir(DBDIR)) >= 11:
        return
    os.makedirs(DBDIR, exist_ok=True)
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        for n in z.namelist():
            if n.endswith(".sqlite"):
                out = os.path.join(DBDIR, n.split("/")[-1])
                if not os.path.exists(out):
                    with open(out, "wb") as fh:
                        fh.write(z.read(n))
                print("extracted", os.path.basename(out), flush=True)


def run_query(dbfile, sql):
    """Returns (status, rows, ncols). status: ok | error | timeout."""
    con = sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True, timeout=10)
    # local sqlite is 3.31; BIRD golds use IIF (3.32+) -- shim it
    con.create_function("IIF", 3, lambda c, a, b: a if c else b)
    timer = threading.Timer(TIMEOUT_S, con.interrupt)
    timer.start()
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(200000)
        ncols = len(cur.description) if cur.description else 0
        return "ok", rows, ncols
    except sqlite3.OperationalError as e:
        return ("timeout" if "interrupted" in str(e) else "error"), None, 0
    except Exception:
        return "error", None, 0
    finally:
        timer.cancel()
        con.close()


def canon(rows):
    """Order-insensitive canonical form; floats rounded so 2.0 == 2.000001."""
    out = []
    for r in rows:
        out.append(tuple(round(v, 6) if isinstance(v, float) else v for v in r))
    return sorted(out, key=repr)


def classify(qid, db, sql0, sql2):
    dbf = os.path.join(DBDIR, db + ".sqlite")
    s0, r0, c0 = run_query(dbf, sql0)
    s2, r2, c2 = run_query(dbf, sql2)
    rec = {"qid": qid, "db": db, "t0_status": s0, "t2_status": s2,
           "t0_rows": len(r0) if r0 is not None else None,
           "t2_rows": len(r2) if r2 is not None else None,
           "t0_cols": c0, "t2_cols": c2}
    if s0 != "ok":
        rec["category"] = "t0-" + s0
    elif s2 != "ok":
        rec["category"] = "t2-" + s2      # corrected gold itself fails here
    elif canon(r0) == canon(r2):
        rec["category"] = "identical-output"
    elif len(r0) == 0 and len(r2) > 0:
        rec["category"] = "t0-empty"
    elif c0 != c2:
        rec["category"] = "shape-differs"
    elif len(r0) != len(r2):
        rec["category"] = "rowcount-differs"
    else:
        rec["category"] = "values-differ"
    rec["plausible"] = int(s0 == "ok" and r0 and c0 == c2)
    return rec


def main():
    ensure_dbs()
    uiuc = json.load(open(a5.UIUC))
    import sqlglot
    with zipfile.ZipFile(os.path.join(HERE, "data", "minidev.zip")) as z:
        t0 = {e["question_id"]: e["SQL"]
              for e in json.loads(z.read("minidev/MINIDEV/mini_dev_sqlite.json"))}
    t1 = {e["question_id"]: e["SQL"] for e in
          json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}

    outp = os.path.join(HERE, "results", "exec_consequences.jsonl")
    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            done.add(json.loads(line)["qid"])
    out = open(outp, "a")
    n = 0
    for e in uiuc:
        qid = int(e["question_id"])
        err1 = a5.norm(t0[qid]) != a5.norm(t1[qid])
        err2 = "original_SQL" in e
        if not (err1 or err2) or qid in done:
            continue
        dd = a5.sel_depth(sqlglot.parse_one(e["SQL"], read="sqlite"))
        rec = classify(qid, e["db_id"], t0[qid], e["SQL"])
        rec["depth_bucket"] = 0 if dd <= 1 else (1 if dd == 2 else 2)
        rec["err1"], rec["err2"] = int(err1), int(err2)
        out.write(json.dumps(rec) + "\n"); out.flush()
        n += 1
        if n % 20 == 0:
            print(f"{n} done", flush=True)
    out.close()

    rows = [json.loads(l) for l in open(outp)]
    cats = Counter(r["category"] for r in rows)
    print("\n类别分布:", dict(cats.most_common()))
    print("貌似合理(plausible)比例:", round(sum(r["plausible"] for r in rows) / len(rows), 3))
    byd = defaultdict(lambda: [0, 0])
    for r in rows:
        byd[r["depth_bucket"]][0] += r["plausible"]; byd[r["depth_bucket"]][1] += 1
    for b in sorted(byd):
        k, m = byd[b]
        print(f"  深度桶{b}: plausible {k}/{m} = {100*k/m:.1f}%")
    ident = defaultdict(lambda: [0, 0])
    for r in rows:
        ident[r["depth_bucket"]][0] += r["category"] == "identical-output"
        ident[r["depth_bucket"]][1] += 1
    print("identical-output 按深度:",
          {b: f"{v[0]}/{v[1]}" for b, v in sorted(ident.items())})
    summ = {"n": len(rows), "categories": dict(cats),
            "plausible_rate": round(sum(r["plausible"] for r in rows) / len(rows), 3),
            "plausible_by_depth": {b: v for b, v in sorted(byd.items())},
            "identical_by_depth": {b: v for b, v in sorted(ident.items())}}
    json.dump(summ, open(os.path.join(HERE, "results", "exec_consequences.json"), "w"),
              indent=1)
    print("wrote results/exec_consequences.json")


if __name__ == "__main__":
    main()
