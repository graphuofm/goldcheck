#!/usr/bin/env python3
"""
ICDE Exp 37 -- inputs for the second iTiger round.
  results/sar_local/hints_dev_2024.json, hints_dev_2025.json : GoldCheck evidence per flagged query,
      appended to the SAR-Agent prompt by 34_sar_local.py --hints (the "GoldCheck first, LLM second" run)
  results/sar_local/ids_dev2024_live105.txt : the 105 Dev 2024 queries with a live finding
  results/sar_local/ids_dev2024_all.txt, ids_dev2025_all.txt : every Dev question
Evidence is stated as facts observed by execution; it never says which reading is right.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "sar_local")
GC = json.load(open(os.path.join(HERE, "results", "goldcheck.json")))
LIVE = ("topk_tie", "topk_null", "count_fanout", "empty_answer")


def short(rows, n=3):
    return "; ".join(", ".join(str(v) for v in r) for r in rows[:n])


def hint(r):
    parts = []
    t = r.get("topk", {})
    if "topk_tie" in r["flags"]:
        parts.append(f"The gold query keeps the top {t['k']} row(s) with ORDER BY ... LIMIT {t['k']}, but other rows have "
                     f"the same sort-key value as the last kept row and are cut off although their output differs. "
                     f"Keeping every tied row gives {t['tie_complete_rows']} rows instead of {t['gold_rows']}. "
                     f"Examples of tied rows that are cut off: {short(t.get('example_cut_off', []))}.")
    if "topk_null" in r["flags"]:
        parts.append("The first row of the ranking has a NULL value in the sort key, so the 'top' row is a row "
                     "whose ranked value is missing.")
    if "count_fanout" in r["flags"]:
        f = r["fanout"]
        parts.append(f"COUNT({f['column']}) is computed over a join without DISTINCT, and the join repeats "
                     f"{f['column'].split('.')[0]} rows. The gold query returns {short(f['gold'])}; with "
                     f"COUNT(DISTINCT {f['column']}) the same query returns {short(f['with_distinct'])}.")
    if "empty_answer" in r["flags"]:
        parts.append("On the official database the gold query returns no rows or only NULL, so a prediction that "
                     "returns nothing is scored correct.")
    return " ".join(parts)


def main():
    for v in ("dev_2024", "dev_2025"):
        rows = [r for r in GC["rows"] if r["ver"] == v and any(f in r["flags"] for f in LIVE)]
        json.dump({str(r["qid"]): hint(r) for r in rows}, open(os.path.join(OUT, f"hints_{v}.json"), "w"), indent=1)
        print(v, len(rows), "hints")
        allq = sorted(r["qid"] for r in GC["rows"] if r["ver"] == v)
        open(os.path.join(OUT, f"ids_{v.replace('_', '')}_all.txt"), "w").write("\n".join(map(str, allq)) + "\n")
        print(v, len(allq), "questions")
        if v == "dev_2024":
            open(os.path.join(OUT, "ids_dev2024_live105.txt"), "w").write("\n".join(str(r["qid"]) for r in rows) + "\n")
    # the 65 of Dev 2025 must match ids_dev2025_findings65.txt used in round one
    a = set(json.load(open(os.path.join(OUT, "hints_dev_2025.json"))))
    b = set(open(os.path.join(OUT, "ids_dev2025_findings65.txt")).read().split())
    assert a == b, (len(a), len(b))


if __name__ == "__main__":
    main()
