#!/usr/bin/env python3
"""
ICDE Exp 41 -- GoldCheck in batch on the BIRD training split (9,428 gold queries, 69 databases).
Training gold queries are what fine-tuned text-to-SQL systems learn from (downstream use 2 of
Section II). Data: .cache/bird_train/train (see DATA.md); example ids are positions in train.json.
Output: results/goldcheck_train.json (per-query flags, timing, summary)
"""
import collections, json, os, statistics, time
from concurrent.futures import ProcessPoolExecutor
import goldcheck as G

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "data", "bird_train")
LIVE = ("topk_tie", "topk_null", "count_fanout", "empty_answer")


def one(args):
    i, db, sql = args
    path = os.path.join(TRAIN, "train_databases", db, db + ".sqlite")
    t0 = time.time()
    try:
        r = G.check(path, sql)
    except Exception as ex:
        r = {"flags": [], "status": "error:" + str(ex)[:80]}
    r.update({"idx": i, "db": db, "wall": round(time.time() - t0, 3)})
    return r


def main():
    data = json.load(open(os.path.join(TRAIN, "train.json")))
    jobs = [(i, e["db_id"], e["SQL"]) for i, e in enumerate(data)]
    t0 = time.time()
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(one, jobs, chunksize=4))
    wall = time.time() - t0
    live = [r for r in rows if any(f in r["flags"] for f in LIVE)]
    cls = collections.Counter(f for r in rows for f in r["flags"])
    st = collections.Counter(r["status"].split(":")[0] for r in rows)
    perdb = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        perdb[r["db"]][0] += 1; perdb[r["db"]][1] += r in live
    secs = sorted(r.get("secs", r["wall"]) for r in rows)
    summ = {"n": len(rows), "databases": len(perdb), "live_queries": len(live),
            "live_pct": round(100 * len(live) / len(rows), 2), "flags": dict(cls), "status": dict(st),
            "topk_queries": sum("topk" in r for r in rows),
            "median_ms": round(1000 * statistics.median(secs), 1), "p95_s": secs[int(0.95 * len(secs))],
            "max_s": secs[-1], "cpu_s": round(sum(secs)), "wall_s": round(wall),
            "databases_with_live": sum(v[1] > 0 for v in perdb.values()),
            "top_databases": sorted(((k, v[0], v[1]) for k, v in perdb.items()), key=lambda x: -x[2])[:8]}
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": rows}, open(os.path.join(HERE, "results", "goldcheck_train.json"), "w"))


if __name__ == "__main__":
    main()
