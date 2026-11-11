#!/usr/bin/env python3
"""
ICDE Exp 36 -- analyse the SAR-Agent re-runs (34_sar_local.py; local 3B, iTiger 7B/32B/120B)
against GoldCheck findings and the maintainers' review, and compute the other numbers
of Sections VI-C..VI-F that are not produced by earlier scripts.

Inputs : results/sar_local/*.jsonl, results/goldcheck.json, results/live_ties.json
Output : results/sar_analysis.json (and a readable summary on stdout)
"""
import glob, json, os, re, statistics, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SAR = os.path.join(RES, "sar_local")
GC = json.load(open(os.path.join(RES, "goldcheck.json")))
CH = GC["changes"]
LIVE = ("topk_tie", "topk_null", "count_fanout", "empty_answer")
BY = {(r["ver"], r["qid"]): r for r in GC["rows"]}
# a mechanism is "named" if the explanation contains one of these patterns
NAMED = {"topk_tie": r"\btie|tied|same (height|value|score)|multiple (rows|records|schools|players)|more than one",
         "topk_null": r"null|none|missing",
         "count_fanout": r"distinct|duplicate|double.count|multiple times",
         "empty_answer": r"empty|no (rows|data|record|result)|returns? none|null"}
MODELS = [("qwen3b", "Qwen2.5-3B-Instruct", "Quadro RTX 6000"), ("qwen7b", "Qwen2.5-7B-Instruct", "H100"),
          ("qwen32b", "Qwen2.5-Coder-32B-Instruct", "H100"), ("gptoss120b", "gpt-oss-120b", "H100")]


def load(path):
    if not os.path.exists(path):
        return None
    return {r["qid"]: r for r in (json.loads(l) for l in open(path) if l.strip()) if "error" not in r}


def changed(v, q):
    n = {"dev_2024": "dev_2025"}[v]
    return CH[f"{v}|{n}|{q}"][1]


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    pa, pb = sum(x == "no" for x in a) / n, sum(x == "no" for x in b) / n
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 2) if pe < 1 else None


def main_run(tag):
    R = load(os.path.join(SAR, f"{tag}_dev_2024_ids_dev2024_eval300.jsonl"))
    if R is None:
        return None
    live = lambda q: any(f in BY[("dev_2024", q)]["flags"] for f in LIVE)
    out = {"n": len(R), "verdicts": dict(collections.Counter(str(r.get("correct")) for r in R.values()))}
    per = {}
    for c in LIVE:
        qs = [q for q in R if c in BY[("dev_2024", q)]["flags"]]
        per[c] = {"n": len(qs), "judged_wrong": sum(R[q].get("correct") == "no" for q in qs),
                  "named": sum(bool(re.search(NAMED[c], R[q].get("analysis") or "", re.I)) for q in qs),
                  "named_and_wrong": sum(bool(re.search(NAMED[c], R[q].get("analysis") or "", re.I))
                                         and R[q].get("correct") == "no" for q in qs)}
    out["per_class"] = per
    out["findings_total"] = sum(p["n"] for p in per.values())
    out["named_total"] = sum(p["named"] for p in per.values())
    L = [q for q in R if live(q)]; O = [q for q in R if not live(q)]
    out["live_queries"] = len(L); out["live_judged_wrong"] = sum(R[q].get("correct") == "no" for q in L)
    strata = {}
    for name, S in (("live", L), ("others", O)):
        F = [q for q in S if R[q].get("correct") == "no"]; U = [q for q in S if R[q].get("correct") != "no"]
        kf, ku = sum(changed("dev_2024", q) for q in F), sum(changed("dev_2024", q) for q in U)
        strata[name] = {"wrong": len(F), "wrong_changed": kf, "correct_or_none": len(U), "correct_changed": ku,
                        "lift": round((kf / max(len(F), 1)) / max(ku / max(len(U), 1), 1e-9), 2)}
    out["agreement"] = strata
    secs = sorted(r["secs"] for r in R.values())
    out["cost"] = {"median_s": statistics.median(secs), "mean_s": round(statistics.mean(secs), 1),
                   "p95_s": secs[int(0.95 * len(secs))], "max_s": secs[-1], "gpu_hours": round(sum(secs) / 3600, 2),
                   "mean_tokens_in": round(statistics.mean(r["tokens_in"] for r in R.values())),
                   "mean_tokens_out": round(statistics.mean(r["tokens_out"] for r in R.values())),
                   "median_steps": statistics.median(r["steps"] for r in R.values()),
                   "mean_steps": round(statistics.mean(r["steps"] for r in R.values()), 1),
                   "no_verdict": sum(r.get("correct") is None for r in R.values()),
                   "hit_30_steps": sum(not r["terminated"] for r in R.values()),
                   "any_bad_format": sum(r["bad_format"] > 0 for r in R.values())}
    for q in (802, 1203):
        if q in R:
            out[f"q{q}"] = {"correct": R[q].get("correct"), "analysis": (R[q].get("analysis") or "")[:400]}
    return out


def rerun(tag, rtag):
    A = load(os.path.join(SAR, f"{tag}_dev_2024_ids_dev2024_eval300.jsonl"))
    B = load(os.path.join(SAR, f"{rtag}_dev_2024_ids_dev2024_rerun50.jsonl"))
    if A is None or B is None:
        return None
    qs = [q for q in B if q in A]
    both = [q for q in qs if A[q].get("correct") and B[q].get("correct")]
    return {"n": len(qs), "same_verdict": sum(A[q].get("correct") == B[q].get("correct") for q in qs),
            "both_verdict": len(both), "kappa": kappa([A[q]["correct"] for q in both], [B[q]["correct"] for q in both])}


def dev2025(tag):
    R = load(os.path.join(SAR, f"{tag}_dev_2025_ids_dev2025_findings65.jsonl"))
    if R is None:
        return None
    per = {}
    for c in LIVE:
        qs = [q for q in R if c in BY[("dev_2025", q)]["flags"]]
        per[c] = {"n": len(qs), "judged_wrong": sum(R[q].get("correct") == "no" for q in qs),
                  "named": sum(bool(re.search(NAMED[c], R[q].get("analysis") or "", re.I)) for q in qs)}
    return {"n": len(R), "verdicts": dict(collections.Counter(str(r.get("correct")) for r in R.values())), "per_class": per}


def hint_cmp(tag):
    """Same model on the 105 live Dev 2024 queries, without (round 1) and with GoldCheck evidence (round 2)."""
    A = load(os.path.join(SAR, f"{tag}_dev_2024_ids_dev2024_eval300.jsonl"))
    H = load(os.path.join(SAR, f"{tag}_hint_dev_2024_ids_dev2024_live105.jsonl"))
    if A is None or H is None:
        return None
    qs = [q for q in H if q in A]
    out = {}
    for name, R in (("plain", A), ("hint", H)):
        per = {}
        for c in LIVE:
            cq = [q for q in qs if c in BY[("dev_2024", q)]["flags"]]
            per[c] = {"n": len(cq), "judged_wrong": sum(R[q].get("correct") == "no" for q in cq),
                      "named_and_wrong": sum(bool(re.search(NAMED[c], R[q].get("analysis") or "", re.I))
                                             and R[q].get("correct") == "no" for q in cq)}
        F = [q for q in qs if R[q].get("correct") == "no"]; U = [q for q in qs if R[q].get("correct") != "no"]
        out[name] = {"per_class": per, "named_total": sum(p["named_and_wrong"] for p in per.values()),
                     "judged_wrong": len(F), "wrong_changed": sum(changed("dev_2024", q) for q in F),
                     "correct_or_none": len(U), "correct_changed": sum(changed("dev_2024", q) for q in U),
                     "ambiguous_yes": sum(R[q].get("ambiguous") == "yes" for q in qs),
                     "mean_steps": round(statistics.mean(R[q]["steps"] for q in qs), 2)}
    out["n"] = len(qs)
    return out


def other_numbers():
    """Numbers of Sections VI-C/VI-F computed from goldcheck.json and live_ties.json."""
    d25 = {e["question_id"]: e for e in json.load(open(os.path.join(HERE, "data_ext", "bird_dev_20251106", "dev.json")))}
    rows = [r for r in GC["rows"] if r["ver"] == "dev_2025"]
    live = [r for r in rows if any(f in r["flags"] for f in LIVE)]
    perdb = collections.defaultdict(lambda: [0, 0])
    diff = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        e = d25[r["qid"]]; f = r in live
        perdb[e["db_id"]][0] += 1; perdb[e["db_id"]][1] += f
        diff[e["difficulty"]][0] += 1; diff[e["difficulty"]][1] += f
    ties = sorted((r["topk"]["gold_rows"], r["topk"]["tie_complete_rows"]) for r in rows if "topk_tie" in r["flags"])
    fan = [r for r in rows if "count_fanout" in r["flags"]]
    ratio = [r for r in fan if re.search(r"percent|ratio|average", d25[r["qid"]]["question"], re.I)]
    above100 = [r["qid"] for r in ratio if any(isinstance(v, (int, float)) and v > 100
                                               for row in r["fanout"]["with_distinct"] for v in row)]
    perclass = {}
    for v, n in (("mini_T0", "mini_T1"), ("mini_T1", "mini_T2"), ("dev_2024", "dev_2025")):
        perclass[f"{v}->{n}"] = {c: [sum(CH[f"{v}|{n}|{r['qid']}"][1] for r in GC["rows"] if r["ver"] == v and c in r["flags"]),
                                     sum(1 for r in GC["rows"] if r["ver"] == v and c in r["flags"])]
                                 for c in LIVE + ("topk_latent",)}
    LT = json.load(open(os.path.join(RES, "live_ties.json")))["rows"]
    harmless = [r["qid"] for r in LT if r["tag"] == "v2024" and r["status"] == "tie_harmless"]
    return {"per_database_2025": dict(perdb), "difficulty_2025": dict(diff),
            "tie_sizes_2025": ties, "fanout_ratio_2025": [r["qid"] for r in ratio], "fanout_ratio_above100": above100,
            "per_class_changed_of_flagged": perclass,
            "harmless_ties_2024": {"n": len(harmless), "out_changed": sum(changed("dev_2024", q) for q in harmless)}}


def main():
    out = {"models": {}, "rerun": {}, "dev2025": {}}
    for tag, name, gpu in MODELS:
        m = main_run(tag)
        if m:
            info = os.path.join(SAR, f"run_{tag}_dev_2024.json")
            m["name"] = name; m["gpu"] = [json.loads(l) for l in open(info) if l.strip()][-1] if os.path.exists(info) else gpu
            out["models"][tag] = m
        r = rerun(tag, f"{tag}_rerun")
        if r:
            out["rerun"][tag] = r
        d = dev2025(tag)
        if d:
            out["dev2025"][tag] = d
    out["hint"] = {t: hint_cmp(t) for t in ("qwen32b", "gptoss120b") if hint_cmp(t)}
    out["other"] = other_numbers()
    json.dump(out, open(os.path.join(RES, "sar_analysis.json"), "w"), indent=1, default=str)
    for tag, m in out["models"].items():
        a = m["agreement"]
        print(f"{tag:11s} n={m['n']} verdicts={m['verdicts']} live_wrong={m['live_judged_wrong']}/{m['live_queries']} "
              f"named={m['named_total']}/{m['findings_total']} | others lift={a['others']['lift']} "
              f"({a['others']['wrong_changed']}/{a['others']['wrong']} vs {a['others']['correct_changed']}/{a['others']['correct_or_none']}) "
              f"| live: {a['live']['wrong_changed']}/{a['live']['wrong']} vs {a['live']['correct_changed']}/{a['live']['correct_or_none']}")
        print("            per class:", {c: (p["judged_wrong"], p["named"], p["n"]) for c, p in m["per_class"].items()})
        print("            cost:", m["cost"])
    print("rerun:", out["rerun"]); print("dev2025:", out["dev2025"])
    for t, h in out["hint"].items():
        for k in ("plain", "hint"):
            x = h[k]
            print(t, k, "wrong", x["judged_wrong"], f"chg {x['wrong_changed']}/{x['judged_wrong']} vs {x['correct_changed']}/{x['correct_or_none']}",
                  "named", x["named_total"], {c: (v["judged_wrong"], v["named_and_wrong"], v["n"]) for c, v in x["per_class"].items()},
                  "amb", x["ambiguous_yes"], "steps", x["mean_steps"])


if __name__ == "__main__":
    main()
