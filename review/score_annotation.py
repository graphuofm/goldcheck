#!/usr/bin/env python3
"""Read the two returned annotation sheets (review/returned/) back: per-rater counts, raw agreement,
Cohen's kappa (four labels, and error vs. not-error), and the list of disagreements for adjudication.
Output: ../exp/results/annotation_agreement.json"""
import collections, json, os
from openpyxl import load_workbook

HERE = os.path.dirname(os.path.abspath(__file__))
CHOICES = ["标准答案对", "标准答案错", "两种都说得通", "看不懂或都不对"]


FILES = (("Jiacheng Ding", os.path.join(HERE, "returned", "annotation_rater2_Jiacheng Ding.xlsx")),
         ("Cong Guo", os.path.join(HERE, "returned", "annotation_rater1_Cong.xlsx")))


def read(path):
    s = load_workbook(path, data_only=True)["标注"]
    out = {}
    for r in range(4, s.max_row + 1):
        if s.cell(r, 2).value is None:
            continue
        out[(int(s.cell(r, 2).value), s.cell(r, 3).value)] = (s.cell(r, 9).value, s.cell(r, 10).value)
    return out


def kappa(a, b, labels):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def main():
    (n1, p1), (n2, p2) = FILES
    A, B = read(p1), read(p2)
    keys = [k for k in A if A[k][0] in CHOICES and k in B and B[k][0] in CHOICES]
    missing = [k for k in A if k not in keys]
    a = [A[k][0] for k in keys]; b = [B[k][0] for k in keys]
    err = lambda x: "错" if x == "标准答案错" else "非错"
    out = {"raters": [n1, n2], "n_both": len(keys), "missing": [list(k) for k in missing],
           "counts": {n1 or "rater1": dict(collections.Counter(a)), n2 or "rater2": dict(collections.Counter(b))},
           "agreement": round(sum(x == y for x, y in zip(a, b)) / max(len(keys), 1), 3),
           "kappa_4": kappa(a, b, CHOICES),
           "kappa_error": kappa([err(x) for x in a], [err(x) for x in b], ["错", "非错"]),
           "by_class": {}, "disagreements": []}
    for cls in sorted({k[1] for k in keys}):
        ks = [k for k in keys if k[1] == cls]
        out["by_class"][cls] = {"n": len(ks), "agree": sum(A[k][0] == B[k][0] for k in ks),
                                "both_error": sum(A[k][0] == B[k][0] == "标准答案错" for k in ks),
                                "both_ok": sum(A[k][0] == B[k][0] == "标准答案对" for k in ks)}
    for k in keys:
        if A[k][0] != B[k][0]:
            out["disagreements"].append({"qid": k[0], "class": k[1], "rater1": A[k], "rater2": B[k]})
    print(json.dumps(out, ensure_ascii=False, indent=1))
    json.dump(out, open(os.path.join(HERE, "..", "exp", "results", "annotation_agreement.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
